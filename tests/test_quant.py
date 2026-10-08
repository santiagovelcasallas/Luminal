"""Checks the qbench implementation against the HANDOVER reference code (copied verbatim below)."""
import math
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qbench import quant as Q
from qbench.hadamard import hadamard, had_rot, rotation_signs

# ------------------------------------------------------------------ HANDOVER reference (verbatim)
CS = np.linspace(1.0, 5.0, 33)


def ref_grid_clip(blk, qmax):
    mu = blk.mean(1, keepdim=True); sd = blk.std(1, keepdim=True) + 1e-12; be = None
    for c in CS:
        l = mu - c * sd; s = 2 * c * sd / qmax; e = ((torch.clamp(torch.round((blk - l) / s), 0, qmax) * s + l - blk) ** 2).sum(1, keepdim=True)
        if be is None: be, lo, hi = e, l, mu + c * sd
        else: m = e < be; be = torch.where(m, e, be); lo = torch.where(m, l, lo); hi = torch.where(m, mu + c * sd, hi)
    s = (hi - lo) / qmax; return lo, torch.where(s == 0, torch.ones_like(s), s)


def ref_prepH(Hraw, W):
    H = Hraw * 2; dead = torch.diag(H) == 0; H[dead, dead] = 1.0; W[:, dead] = 0
    H[range(H.shape[0]), range(H.shape[0])] += 0.01 * torch.diag(H).mean(); return H


def ref_gptq(W, Hraw, bits, group=128, clip=ref_grid_clip):
    W = W.clone(); inn = W.shape[1]; H = ref_prepH(Hraw, W); L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True); qmax = 2 ** bits - 1
    for i in range(inn):
        if i % group == 0: lo, s = clip(W[:, i:i + group], qmax)
        w = W[:, i].clone(); q = (torch.clamp(torch.round((w[:, None] - lo) / s), 0, qmax) * s + lo)[:, 0]; W[:, i] = q
        err = (w - q) / Hinv[i, i]
        if i + 1 < inn: W[:, i + 1:] -= err[:, None] * Hinv[i, i + 1:][None, :]
    return W


def ref_gptq_tern(W, Hraw, group=128):
    W = W.clone(); inn = W.shape[1]; H = ref_prepH(Hraw, W); L = torch.linalg.cholesky(H); Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    for i in range(inn):
        if i % group == 0: J = slice(i, min(i + group, inn)); al, mu = Q.tgrid(W[:, J], Hraw[J, J])
        w = W[:, i].clone(); q = (mu + al * torch.clamp(torch.round((w[:, None] - mu) / al), -1, 1))[:, 0]; W[:, i] = q; err = (w - q) / Hinv[i, i]
        if i + 1 < inn: W[:, i + 1:] -= err[:, None] * Hinv[i, i + 1:][None, :]
    return W


# ------------------------------------------------------------------ helpers
def rand_problem(rows, cols, seed=0, dead=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(4 * cols, cols, generator=g) * (torch.rand(cols, generator=g) * 3 + 0.1)
    if dead:
        X[:, :dead] = 0
    H = X.T @ X / X.shape[0]
    W = torch.randn(rows, cols, generator=g) * 0.02
    return W, H


class NoRound(Q.FinderQ):
    def __call__(self, blk, S):
        return Q.grid_clip(blk, self.qmax)


class NoRoundTern(Q.FinderTern):
    def __call__(self, blk, S):
        return Q.tgrid(blk, S)[::-1]


def run_mine(W, H, opt, finder_cls):
    Hinv, dead = Q.prep_hinv(H)
    finder = finder_cls(opt)
    codes, P0, P1 = Q.gptq_quantize(W.double(), H.double(), Hinv, dead, opt, finder)
    qm = Q.QMat(opt, codes, P0, P1, torch.ones(W.shape[1], dtype=torch.float64))
    return qm.dequant_rot("cpu")


def proxy_loss(W, Wq, H):
    D = (W - Wq).double()
    return float(torch.einsum("ij,jk,ik->", D, H.double(), D))


# ------------------------------------------------------------------ tests
def test_hadamard():
    for n in (1024, 2048, 3072):
        Hn = hadamard(n)
        assert torch.allclose(Hn @ Hn.T, torch.eye(n, dtype=torch.float64), atol=1e-10), n
        V = had_rot(n, rotation_signs(n, 0, 5))
        assert torch.allclose(V @ V.T, torch.eye(n, dtype=torch.float64), atol=1e-10)
    assert not torch.equal(rotation_signs(1024, 0, 1), rotation_signs(1024, 0, 2))
    assert not torch.equal(rotation_signs(1024, 0, 1), rotation_signs(1024, 1, 1))
    print("hadamard ok")


def test_grid_clip():
    g = torch.Generator().manual_seed(1)
    for qmax in (3, 7, 15):
        blk = torch.randn(300, 128, generator=g) * 0.03 + torch.randn(300, 1, generator=g) * 0.01
        lo1, s1 = ref_grid_clip(blk, qmax)
        lo2, s2 = Q.grid_clip(blk, qmax)
        assert torch.allclose(lo1, lo2, rtol=1e-5, atol=1e-7) and torch.allclose(s1, s2, rtol=1e-5, atol=1e-8), qmax
    print("grid_clip ok")


def test_gptq_matches_reference():
    for bits in (2, 3, 4):
        for dead in (0, 5):
            W, H = rand_problem(64, 384, seed=bits + dead, dead=dead)
            ref = ref_gptq(W, H, bits)
            mine = run_mine(W, H, Q.Opt("q", bits), NoRound).float()
            same = (ref - mine).abs() < 1e-5
            lr, lm = proxy_loss(W, ref, H), proxy_loss(W, mine, H)
            assert same.float().mean() > 0.995, (bits, dead, same.float().mean())
            assert abs(lr - lm) / lr < 2e-3, (bits, lr, lm)
    print("gptq ok")


def test_tern_matches_reference():
    W, H = rand_problem(64, 384, seed=7)
    ref = ref_gptq_tern(W, H)
    mine = run_mine(W, H, Q.Opt("tern"), NoRoundTern).float()
    same = (ref - mine).abs() < 1e-5
    lr, lm = proxy_loss(W, ref, H), proxy_loss(W, mine, H)
    assert same.float().mean() > 0.995, same.float().mean()
    assert abs(lr - lm) / lr < 2e-3, (lr, lm)
    print("ternary ok")


def test_group64_and_log_and_rotation():
    W, H = rand_problem(96, 512, seed=3)
    signs = rotation_signs(512, 0, 0)
    Wr, Hr = Q.rotate(W, H, signs)
    V = had_rot(512, signs)
    assert torch.allclose(Wr, W.double() @ V.T) and torch.allclose(Hr, V @ H.double() @ V.T, atol=1e-10)
    Hinv, dead = Q.prep_hinv(Hr)
    losses = {}
    for opt in [Q.Opt("q", 2), Q.Opt("q", 2, "log", 8, 6, 128), Q.Opt("q", 2, "log", 6, 4, 128),
                Q.Opt("q", 2, "log", 8, 6, 64), Q.Opt("q", 4), Q.Opt("q", 4, "log", 8, 6, 128),
                Q.Opt("tern"), Q.Opt("tern", 0, "log", 8, 6, 128)]:
        qm = Q.quantize_matrix(Wr, Hr, Hinv, dead, opt, signs)
        Weff = qm.weff("cpu")
        # the stored codes/params reproduce exactly what GPTQ committed; W_eff = Q·V maps back to the original basis
        loss = proxy_loss(W, Weff, H)
        losses[opt.name] = loss / proxy_loss(W, torch.zeros_like(W), H)
        if opt.kind == "q":
            assert int(qm.codes.min()) >= 0 and int(qm.codes.max()) <= 2 ** opt.bits - 1
        else:
            assert set(qm.codes.unique().tolist()) <= {-1, 0, 1}
        if opt.scale == "log":
            lo_t, step = qm.tensor_params
            p = qm.p1 if opt.kind == "q" else qm.p1.abs()
            idx = (torch.log2(p) - lo_t) / step
            assert (idx - idx.round()).abs().max() < 1e-3, "scales are not on the log grid"
            assert idx.round().min() >= 0 and idx.round().max() <= 2 ** opt.k - 1
        if opt.scale == "fp16":
            assert torch.equal(qm.p1, qm.p1.half().float()) and torch.equal(qm.p0, qm.p0.half().float())
    for k, v in losses.items():
        print(f"  rel. proxy loss {k:28s} {v:.4f}")
    assert losses["q2.log.k8z6g128"] < 1.15 * losses["q2"]
    assert losses["q2.log.k8z6g64"] < losses["q2.log.k8z6g128"]
    assert losses["q4"] < losses["q2"] and losses["tern"] > losses["q2"]
    print("group64/log/rotation ok")


def test_bits():
    r, n = 2048, 1024
    assert Q.opt_bits(Q.Opt("q", 2), r, n) == r * n * 2 + r * 8 * 32 + n
    assert Q.opt_bits(Q.Opt("tern"), r, n) == math.ceil(r * n / 5) * 8 + r * 8 * 32 + n
    assert Q.opt_bits(Q.Opt("q", 3, "log", 8, 6, 64), r, n) == r * n * 3 + r * 16 * 14 + 64 + n
    print("bits ok")


if __name__ == "__main__":
    torch.set_num_threads(4)
    test_hadamard(); test_grid_clip(); test_bits(); test_gptq_matches_reference(); test_tern_matches_reference()
    test_group64_and_log_and_rotation()
    print("ALL OK")
