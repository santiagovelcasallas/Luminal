"""llama.cpp + imatrix rival: HF -> GGUF bf16 (round trip checked) -> imatrix on the shared calibration text ->
llama-quantize per format -> dequantize to F32 with llama-quantize -> load the 196 linears into the bench."""
import os
import subprocess
import sys

import torch

FORMATS = ["IQ2_XXS", "IQ2_XS", "IQ2_M", "Q2_K", "IQ3_XXS", "IQ3_M", "Q3_K_M", "IQ4_XS", "Q4_K_M"]
HANDOVER_REF = {"IQ2_XXS": (2.11, 111.7, 452.6), "IQ2_XS": (2.34, 54.3, 208.3), "IQ2_M": (2.76, 20.7, 82.4),
                "Q2_K": (2.95, 16.05, 53.0), "IQ3_XXS": (3.01, 15.37, 49.5), "IQ3_M": (3.67, 11.33, 33.4),
                "Q3_K_M": (3.87, 11.06, 31.4), "IQ4_XS": (4.25, 10.04, 29.25), "Q4_K_M": (4.78, 10.00, 29.0)}
GGUF_NAME = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
             "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
             "mlp.down_proj": "ffn_down"}


def gguf_name(hf_name):
    _, _, i, rest = hf_name.split(".", 3)
    return f"blk.{i}.{GGUF_NAME[rest]}.weight"


class LlamaCpp:
    def __init__(self, root, workdir, log=print):
        self.root, self.work, self.log = root, workdir, log
        os.makedirs(workdir, exist_ok=True)
        sys.path.insert(0, os.path.join(root, "gguf-py"))
        self.threads = str(os.cpu_count() or 4)

    def bin(self, name):
        return os.path.join(self.root, "build", "bin", name)

    def sh(self, *cmd):
        self.log("  $ " + " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
            raise RuntimeError(f"command failed: {cmd[0]}")
        return r

    def path(self, name):
        return os.path.join(self.work, name)

    def tensors(self, gguf_path, names):
        from gguf import GGUFReader
        r = GGUFReader(gguf_path)
        by = {t.name: t for t in r.tensors}
        return {n: by[gguf_name(n)] for n in names}

    def convert(self, snapshot, dense, names):
        bf16 = self.path("model-bf16.gguf")
        if not os.path.exists(bf16):
            self.sh(sys.executable, os.path.join(self.root, "convert_hf_to_gguf.py"), snapshot, "--outtype", "bf16",
                    "--outfile", bf16)
        # round trip: bf16 GGUF -> F32 must equal the HF weights bit for bit (no q/k row permutation for Qwen)
        f32 = self.path("model-bf16-as-f32.gguf")
        self.sh(self.bin("llama-quantize"), "--allow-requantize", bf16, f32, "F32", self.threads)
        mods = dict(dense.named_modules())
        bad = []
        for n, t in self.tensors(f32, names).items():
            W = torch.from_numpy(t.data.copy()).reshape(mods[n].weight.shape)
            if not torch.equal(W, mods[n].weight.detach().float().cpu()):
                bad.append(n)
        os.remove(f32)
        if bad:
            raise RuntimeError(f"GGUF round trip is NOT exact for {len(bad)} linears, e.g. {bad[:3]}")
        self.log("  GGUF round trip exact for all linears")
        return bf16

    def imatrix(self, bf16, calib_text):
        out = self.path("imatrix.gguf")
        if not os.path.exists(out):
            txt = self.path("calib.txt")
            with open(txt, "w") as f:
                f.write(calib_text)
            self.sh(self.bin("llama-imatrix"), "-m", bf16, "-f", txt, "-o", out, "-c", "512", "--chunks", "32",
                    "-t", self.threads)
        return out

    def quantize(self, bf16, imat, fmt, names, shapes):
        q = self.path(f"q-{fmt}.gguf")
        self.sh(self.bin("llama-quantize"), "--imatrix", imat, bf16, q, fmt, self.threads)
        tq = self.tensors(q, names)
        nbytes = sum(int(t.n_bytes) for t in tq.values())
        types = {}
        for t in tq.values():
            types[t.tensor_type.name] = types.get(t.tensor_type.name, 0) + 1
        f32 = self.path(f"q-{fmt}-f32.gguf")
        self.sh(self.bin("llama-quantize"), "--allow-requantize", q, f32, "F32", self.threads)
        W = {n: torch.from_numpy(t.data.copy()).reshape(shapes[n]) for n, t in self.tensors(f32, names).items()}
        os.remove(q)
        os.remove(f32)
        return W, nbytes, types
