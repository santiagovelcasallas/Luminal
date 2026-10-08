"""HANDOVER §4.1: H = XᵀX / N per linear, from the ORIGINAL model on the 32 calibration windows."""
import torch

# Linears that read the same activation share one Hessian (q/k/v read input_layernorm, gate/up read
# post_attention_layernorm), so each distinct input is accumulated once.
SHARED = {"self_attn.k_proj": "self_attn.q_proj", "self_attn.v_proj": "self_attn.q_proj", "mlp.up_proj": "mlp.gate_proj"}


@torch.no_grad()
def collect_hessians(model, calib, names, device, bs=4):
    mods = dict(model.named_modules())
    owners = [n for n in names if not any(n.endswith(k) for k in SHARED)]
    acc, hooks = {}, []

    def mk(name):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            h = (x.T @ x).double()
            acc[name] = h if name not in acc else acc[name] + h
        return hook

    for n in owners:
        hooks.append(mods[n].register_forward_hook(mk(n)))
    try:
        for i in range(0, len(calib), bs):
            model(calib[i:i + bs].to(device))
    finally:
        for h in hooks:
            h.remove()
    N = calib.numel()
    H = {n: (acc[n] / N).float().cpu() for n in owners}
    del acc
    for n in names:
        if n not in H:
            pre, suf = n.split(".", 3)[:3], n.split(".", 3)[3]
            H[n] = H[".".join(pre) + "." + SHARED[suf]]
    return H
