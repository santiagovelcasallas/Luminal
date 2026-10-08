"""Official AutoAWQ 0.2.9 pipeline (run with PYTHONPATH pointing to an isolated site with transformers 4.51.3).
Scales/clipping are searched and applied by AutoAWQ (export_compatible=True), then every linear is quantized with
AutoAWQ's own pseudo_quantize_tensor (the exact values its GEMM kernel would dequantize). The norm rescaling AWQ
introduces is folded back into the linears so that norms stay identical to the dense model (same function)."""
import argparse
import json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    names = json.load(open(a.names))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoAWQForCausalLM.from_pretrained(a.model)
    hf = model.model
    lin = set(names)
    before = {n: p.detach().clone().float().cpu() for n, p in hf.named_parameters()
              if n.rsplit(".", 1)[0] not in lin}
    calib = [r.tolist() for r in torch.load(a.calib)]
    model.quantize(tok, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": a.bits, "version": "GEMM"},
                   calib_data=calib, max_calib_samples=len(calib), max_calib_seq_len=512, export_compatible=True)
    q = model.quantizer
    W = {}
    with torch.no_grad():
        for n in names:
            w = hf.get_submodule(n).weight.data.to("cuda:0").half()
            W[n] = q.pseudo_quantize_tensor(w)[0].float().cpu()
        after = {n: p.detach().float().cpu() for n, p in hf.named_parameters() if n.rsplit(".", 1)[0] not in lin}
        changed = [n for n in before if not torch.equal(before[n], after[n])]
        for n in changed:
            if not n.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")):
                raise RuntimeError(f"AWQ modified an unexpected non-linear parameter: {n}")
            r = after[n] / before[n]
            layer = n.rsplit(".", 2)[0]
            outs = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] if "input_layernorm" in n \
                else ["mlp.gate_proj", "mlp.up_proj"]
            for o in outs:
                W[f"{layer}.{o}"] = W[f"{layer}.{o}"] * r[None, :]
    g = 128
    bpw = a.bits + (16 + a.bits) / g  # GEMM layout: packed codes + fp16 scale + packed zero per group of 128
    torch.save({"weights": W, "bpw": bpw, "folded_norms": changed}, a.out)


if __name__ == "__main__":
    main()
