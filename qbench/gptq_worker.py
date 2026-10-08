"""Runs the official GPTQModel pipeline in a subprocess and writes the effective (dequantized) weights of the
196 linears. Dequantization is format-agnostic: each quantized module is probed with the identity matrix
(y = I·Wᵀ), i.e. exactly what its own torch kernel computes at inference."""
import argparse
import json
import os
import shutil

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--sym", type=int, required=True)
    ap.add_argument("--desc_act", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", required=True)
    a = ap.parse_args()
    from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
    from safetensors import safe_open

    names = json.load(open(a.names))
    calib = torch.load(a.calib)
    data = [{"input_ids": r.tolist(), "attention_mask": [1] * len(r)} for r in calib]
    qc = QuantizeConfig(bits=a.bits, group_size=128, sym=bool(a.sym), desc_act=bool(a.desc_act))
    model = GPTQModel.load(a.model, qc)
    model.quantize(data, batch_size=1)
    shutil.rmtree(a.tmp, ignore_errors=True)
    model.save(a.tmp)
    del model
    torch.cuda.empty_cache()

    nbytes = 0
    for f in os.listdir(a.tmp):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(a.tmp, f), "pt") as st:
                for k in st.keys():
                    if any(k.startswith(n + ".") for n in names):
                        t = st.get_slice(k)
                        n_el = 1
                        for s in t.get_shape():
                            n_el *= s
                        nbytes += n_el * torch.empty((), dtype=getattr(torch, {"I32": "int32", "F16": "float16",
                                  "BF16": "bfloat16", "F32": "float32", "I16": "int16", "I8": "int8",
                                  "U8": "uint8", "I64": "int64"}[t.get_dtype()])).element_size()

    qm = GPTQModel.load(a.tmp, backend=BACKEND.GPTQ_TORCH, device="cuda:0")
    hf = qm.model
    W = {}
    with torch.no_grad():
        for n in names:
            mod = hf.get_submodule(n)
            n_in = mod.in_features
            dt = getattr(mod, "scales", torch.empty(0, dtype=torch.float16)).dtype
            y = mod(torch.eye(n_in, dtype=dt, device="cuda:0"))
            W[n] = y.float().T.contiguous().cpu()
    torch.save({"weights": W, "bytes": nbytes}, a.out)
    shutil.rmtree(a.tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
