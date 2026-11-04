# Luminal Quant — Copyright (C) 2026 Santiago Vélez Casallas. AGPL-3.0-only.
"""Parity of the Rust uniform GPTQ kernel against the reference Python pipeline in qbench.
Same problem, same Hinv; we require identical integer codes on the vast majority of weights and a
dequantized-weight error within the tolerance qbench's own reference test uses."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
luminal_quant = pytest.importorskip("luminal_quant")
from qbench import quant as Q  # noqa: E402


def make_problem(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(4 * cols, cols, generator=g) * (torch.rand(cols, generator=g) * 3 + 0.1)
    H = (X.T @ X / X.shape[0]).double()
    W = (torch.randn(rows, cols, generator=g) * 0.02)
    return W, H


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("group", [128, 64])
def test_rust_matches_python(bits, group):
    rows, cols = 64, 512
    W, H = make_problem(rows, cols, seed=bits * 10 + group)
    Hinv, dead = Q.prep_hinv(H)                      # reference float32 upper Cholesky factor
    assert int(dead.sum()) == 0

    # reference (Python/PyTorch), uniform q path
    opt = Q.Opt("q", bits, "fp16", 0, 0, group)
    finder = Q.FinderQ(opt)
    codes_ref, p0_ref, p1_ref = Q.gptq_quantize(W.double(), H, Hinv, dead, opt, finder)

    # Rust kernel on the identical inputs
    codes_r, p0_r, p1_r = luminal_quant.gptq_uniform(
        W.float().numpy(), Hinv.float().numpy(), bits, group, 128)

    # Functional parity, not bit-exactness: PyTorch's reduction order is not reproducible in a
    # sequential Rust sum, so the 33-candidate argmin occasionally picks an adjacent clip value (a
    # near-tie) and a few scales land one grid step apart. The meaningful gate is the Hessian-weighted
    # dequantization error below; these two just bound how often the tie-break diverges.
    s_close = (np.abs(p1_r - p1_ref.numpy()) < 5e-3).mean()
    assert s_close > 0.97, f"scale agreement {s_close:.4f} (bits={bits}, group={group})"
    agree = (codes_r == codes_ref.numpy()).mean()
    assert agree > 0.97, f"code agreement {agree:.4f} (bits={bits}, group={group})"

    # dequantized effective weights: close in Hessian-weighted error, like qbench's reference test
    def deq(codes, p0, p1):
        c = torch.from_numpy(np.asarray(codes)).float()
        P0 = torch.from_numpy(np.asarray(p0)).float().repeat_interleave(group, 1)
        P1 = torch.from_numpy(np.asarray(p1)).float().repeat_interleave(group, 1)
        return c * P1 + P0

    Wr_ref = deq(codes_ref.numpy(), p0_ref.numpy(), p1_ref.numpy())
    Wr_rust = deq(codes_r, p0_r, p1_r)
    D = (Wr_ref - Wr_rust).double()
    num = float(torch.einsum("ij,jk,ik->", D, H, D))
    den = float(torch.einsum("ij,jk,ik->", Wr_ref.double(), H, Wr_ref.double()))
    # Rust vs PyTorch (different float engines): the two dequantized matrices agree within ~0.5% in
    # Hessian-weighted norm — functionally the same quantizer, not bit-identical.
    assert num / den < 5e-3, f"rel Hessian error {num / den:.2e}"
