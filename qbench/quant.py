"""HANDOVER §4.3–4.7 (standardized grid, GPTQ, PT2-style ternary, log scales) and §5 (bit accounting)."""
import math
from dataclasses import dataclass

import numpy as np
import torch

from .hadamard import hadamard

CS = [float(c) for c in np.linspace(1.0, 5.0, 33)]


@dataclass(frozen=True)
class Opt:
    kind: str            # "q" (asymmetric grid) | "tern"
    bits: int = 0        # grid bits (unused for "tern")
    scale: str = "fp16"  # "fp16" | "log"
    k: int = 0           # log2-scale index bits (scale == "log")
    bz: int = 0          # zero-point / mu bits (scale == "log")
    group: int = 128

    @property
    def name(self):
        base = "tern" if self.kind == "tern" else f"q{self.bits}"
        if self.scale == "fp16" and self.group == 128:
            return base
        return f"{base}.{self.scale}.k{self.k}z{self.bz}g{self.group}"


def opt_bits(opt, rows, cols):
    """§5: payload + per-(row, group) metadata + per-tensor log range (2×fp32) + Hadamard signs (cols bits)."""
    ng = cols // opt.group
    payload = rows * cols * opt.bits if opt.kind == "q" else math.ceil(rows * cols / 5) * 8
    meta = rows * ng * (32 if opt.scale == "fp16" else opt.k + opt.bz)
    tensor = 64 if opt.scale == "log" else 0
    return payload + meta + tensor + cols


# ---------------------------------------------------------------- §4.3 standardized grid (vectorized over c)
def grid_clip(blk, qmax):
    mu = blk.mean(1, keepdim=True)
    sd = blk.std(1, keepdim=True) + 1e-12
    c = torch.tensor(CS, dtype=blk.dtype, device=blk.device).view(1, -1, 1)
    l = mu.unsqueeze(1) - c * sd.unsqueeze(1)
    s = 2 * c * sd.unsqueeze(1) / qmax
    x = blk.unsqueeze(1)
    e = ((torch.clamp(torch.round((x - l) / s), 0, qmax) * s + l - x) ** 2).sum(-1)
    cj = c.view(-1)[e.argmin(1)].unsqueeze(1)          # argmin keeps the first minimum, like the spec's strict '<'
    lo, hi = mu - cj * sd, mu + cj * sd
    s = (hi - lo) / qmax
    return lo, torch.where(s == 0, torch.ones_like(s), s)


def grid_err(blk, lo, s, qmax):
    """Squared error per (row, candidate). blk (r,G); lo,s (r,C)."""
    x = blk.unsqueeze(1)
    l, ss = lo.unsqueeze(-1), s.unsqueeze(-1)
    return ((torch.clamp(torch.round((x - l) / ss), 0, qmax) * ss + l - x) ** 2).sum(-1)


# ---------------------------------------------------------------- §4.5 ternary (ITF + AGA), verbatim
def itf(Wb, iters=10):
    mu = Wb.mean(1, keepdim=True)
    Wt = Wb - mu
    D = 0.75 * Wt.abs().mean(1, keepdim=True)
    T = torch.sign(Wt) * (Wt.abs() > D)
    al = (T * Wt).sum(1, keepdim=True) / T.abs().sum(1, keepdim=True).clamp(min=1)
    g = Wb.shape[1]
    Tp = None
    for _ in range(iters):
        a = (T * T).sum(1, keepdim=True); s = T.sum(1, keepdim=True)
        wt = (Wb * T).sum(1, keepdim=True); w1 = Wb.sum(1, keepdim=True)
        den = g * a - s * s; ok = den.abs() > 1e-9; dd = torch.where(ok, den, torch.ones_like(den))
        al = torch.where(ok, (g * wt - s * w1) / dd, al); mu = torch.where(ok, (a * w1 - s * wt) / dd, mu)
        al = torch.where(al.abs() < 1e-12, torch.full_like(al, 1e-6), al)
        T = torch.clamp(torch.round((Wb - mu) / al), -1, 1)
        if Tp is not None and torch.equal(T, Tp):
            break
        Tp = T.clone()
    return al, mu, T


def aga(Wb, T, S):
    one = torch.ones_like(T); TS = T @ S; OS = one @ S; WS = Wb @ S
    tSt = (TS * T).sum(1, keepdim=True); tS1 = (TS * one).sum(1, keepdim=True); oSo = (OS * one).sum(1, keepdim=True)
    wSt = (WS * T).sum(1, keepdim=True); wS1 = (WS * one).sum(1, keepdim=True)
    den = tSt * oSo - tS1 * tS1; ok = den.abs() > 1e-12; dd = torch.where(ok, den, torch.ones_like(den))
    return (wSt * oSo - tS1 * wS1) / dd, (tSt * wS1 - tS1 * wSt) / dd, ok


def tgrid(Wb, S):
    al, mu, T = itf(Wb)
    a2, m2, ok = aga(Wb, T, S)
    al = torch.where(ok, a2, al); mu = torch.where(ok, m2, mu)
    return torch.where(al.abs() < 1e-12, torch.full_like(al, 1e-6), al), mu


def tern_err(blk, mu, al, S):
    """Hessian-weighted error per (row, candidate), the same objective AGA minimizes. mu, al (r,C)."""
    x = blk.unsqueeze(1)
    m, a = mu.unsqueeze(-1), al.unsqueeze(-1)
    R = x - (m + a * torch.clamp(torch.round((x - m) / a), -1, 1))
    return torch.einsum("rcj,jk,rck->rc", R, S, R)


# ---------------------------------------------------------------- grid finders (called at each group start)
NB3 = (-1.0, 0.0, 1.0)


class FinderQ:
    """fp16 (lo, s) per row and group."""

    def __init__(self, opt):
        self.qmax = 2 ** opt.bits - 1

    def __call__(self, blk, S):
        lo, s = grid_clip(blk, self.qmax)
        lo, s = lo.half().float(), s.half().float()
        return lo, torch.where(s == 0, torch.ones_like(s), s)


class FinderQLog:
    """§4.7: s on a per-tensor log2 grid (k bits), lo = -u·ŝ with u on a bz-bit grid over [-0.5, qmax+0.5].
    Only already-quantized (ŝ, lo) pairs are evaluated: {l0-1, l0, l0+1} × {z0-1, z0, z0+1}, min squared error."""

    def __init__(self, opt, lo_t, step):
        self.qmax = 2 ** opt.bits - 1
        self.L, self.Z = 2 ** opt.k - 1, 2 ** opt.bz - 1
        self.r = (self.qmax + 1) / 2 ** opt.bz
        self.lo_t, self.step = lo_t, step

    def decode(self, l, z):
        sh = torch.exp2(self.lo_t + l * self.step)
        return -sh * (-0.5 + (z + 0.5) * self.r), sh

    def __call__(self, blk, S):
        lo_c, s_c = grid_clip(blk, self.qmax)
        nb = torch.tensor(NB3, device=blk.device)
        l0 = torch.round((torch.log2(s_c) - self.lo_t) / self.step)
        ls = (l0 + nb).clamp(0, self.L)                                    # (r,3)
        sh = torch.exp2(self.lo_t + ls * self.step)
        z0 = torch.round((-lo_c / sh + 0.5) / self.r - 0.5)
        zs = (z0.unsqueeze(-1) + nb).clamp(0, self.Z)                       # (r,3,3)
        lo, s = self.decode(ls.unsqueeze(-1).expand_as(zs), zs)
        lo, s = lo.reshape(len(blk), 9), s.reshape(len(blk), 9)
        j = grid_err(blk, lo, s, self.qmax).argmin(1, keepdim=True)
        return lo.gather(1, j), s.gather(1, j)


class FinderTern:
    """fp16 (mu, |alpha|) per row and group (the ternary grid {mu-a, mu, mu+a} is symmetric in the sign of a)."""

    def __init__(self, opt):
        pass

    def __call__(self, blk, S):
        al, mu = tgrid(blk, S)
        al = al.abs().half().float()
        return mu.half().float(), torch.where(al == 0, torch.full_like(al, 1e-6), al)


class FinderTernLog:
    """V3: |alpha| on a per-tensor log2 grid (k bits), mu = alpha·(-1 + 2z/Z) with z on bz bits;
    3×3 quantized candidates, chosen by the AGA (Hessian-weighted) error."""

    def __init__(self, opt, lo_t, step):
        self.L, self.Z = 2 ** opt.k - 1, 2 ** opt.bz - 1
        self.lo_t, self.step = lo_t, step

    def __call__(self, blk, S):
        al_c, mu_c = tgrid(blk, S)
        nb = torch.tensor(NB3, device=blk.device)
        l0 = torch.round((torch.log2(al_c.abs()) - self.lo_t) / self.step)
        ls = (l0 + nb).clamp(0, self.L)
        ah = torch.exp2(self.lo_t + ls * self.step)                         # (r,3)
        z0 = torch.round((mu_c / ah + 1) / 2 * self.Z)
        zs = (z0.unsqueeze(-1) + nb).clamp(0, self.Z)                       # (r,3,3)
        a = ah.unsqueeze(-1).expand_as(zs)
        mu = a * (-1 + 2 * zs / self.Z)
        mu, a = mu.reshape(len(blk), 9), a.reshape(len(blk), 9)
        j = tern_err(blk, mu, a, S).argmin(1, keepdim=True)
        return mu.gather(1, j), a.gather(1, j)


# ---------------------------------------------------------------- per-tensor log ranges (pre-pass on W' before GPTQ)
def log_range(vals, k):
    v = vals.flatten().float()
    lo_t = torch.quantile(v, 0.01).item() - 0.5
    hi_t = v.max().item() + 0.5
    step = (hi_t - lo_t) / (2 ** k - 1)
    return float(np.float32(lo_t)), float(np.float32(step))


def prepass_q(Wr, opt):
    r, n = Wr.shape
    _, s = grid_clip(Wr.reshape(r * (n // opt.group), opt.group), 2 ** opt.bits - 1)
    return log_range(torch.log2(s), opt.k)


def prepass_tern(Wr, Hr, opt):
    G = opt.group
    als = [tgrid(Wr[:, j:j + G], Hr[j:j + G, j:j + G])[0].abs() for j in range(0, Wr.shape[1], G)]
    return log_range(torch.log2(torch.cat(als, 1)), opt.k)


def make_finder(opt, Wr, Hr):
    if opt.kind == "q":
        if opt.scale == "fp16":
            return FinderQ(opt), None
        lt = prepass_q(Wr, opt)
        return FinderQLog(opt, *lt), lt
    if opt.scale == "fp16":
        return FinderTern(opt), None
    lt = prepass_tern(Wr, Hr, opt)
    return FinderTernLog(opt, *lt), lt


# ---------------------------------------------------------------- §4.4 GPTQ (lazy-batch form of the spec's column loop)
def prep_hinv(Hraw):
    """prepH + Cholesky of the inverse, done in float64 for robustness (same math as the spec)."""
    n = Hraw.shape[0]
    H = Hraw.double() * 2
    dead = torch.diag(H) == 0
    d = dead.nonzero().flatten()
    H[d, d] = 1.0
    idx = torch.arange(n, device=H.device)
    H[idx, idx] += 0.01 * torch.diag(H).mean()
    L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    return Hinv.float(), dead


@dataclass
class QMat:
    opt: Opt
    codes: torch.Tensor    # int8 (rows, cols): grid index in [0, qmax] or trit in {-1, 0, 1}
    p0: torch.Tensor       # float32 (rows, groups): lo (grid) | mu (ternary)
    p1: torch.Tensor       # float32 (rows, groups): s (grid)  | alpha (ternary)
    signs: torch.Tensor    # float64 (cols,): Hadamard signs
    tensor_params: tuple = None

    def dequant_rot(self, device):
        G = self.opt.group
        c = self.codes.to(device).float()
        p0 = self.p0.to(device).repeat_interleave(G, 1)
        p1 = self.p1.to(device).repeat_interleave(G, 1)
        return (p0 + p1 * c) if self.opt.kind == "tern" else (c * p1 + p0)

    def weff(self, device, dtype=torch.float32):
        """W_eff = Q · V with V = H_n · diag(s)."""
        Q = self.dequant_rot(device).to(dtype)
        n = Q.shape[1]
        return ((Q @ hadamard_on(n, device, dtype)) * self.signs.to(device, dtype)[None, :]).float()


_HCACHE = {}


def hadamard_on(n, device, dtype=torch.float64):
    key = (n, str(device), dtype)
    if key not in _HCACHE:
        _HCACHE[key] = hadamard(n).to(device, dtype)
    return _HCACHE[key]


def rotate(W, Hraw, signs):
    """W' = W·Vᵀ, H' = V·H·Vᵀ (float64)."""
    dev = W.device
    n = W.shape[1]
    Hn = hadamard_on(n, dev)
    s = signs.to(dev)
    Wr = (W.double() * s[None, :]) @ Hn.T
    Hr = Hn @ (s[:, None] * Hraw.to(dev).double() * s[None, :]) @ Hn.T
    return Wr, Hr


@torch.no_grad()
def gptq_quantize(Wr, Hr, Hinv, dead, opt, finder, blocksize=128):
    W = Wr.float().clone()
    W[:, dead] = 0
    Hrf = Hr.float()
    r, n = W.shape
    G = opt.group
    assert blocksize % G == 0 and n % G == 0
    tern = opt.kind == "tern"
    qmax = 2 ** opt.bits - 1
    codes = torch.empty((r, n), dtype=torch.int8, device=W.device)
    P0 = torch.empty((r, n // G), device=W.device)
    P1 = torch.empty_like(P0)
    for i1 in range(0, n, blocksize):
        i2 = min(i1 + blocksize, n)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % G == 0:
                p0, p1 = finder(W1[:, i:i + G], Hrf[col:col + G, col:col + G])
                P0[:, col // G] = p0[:, 0]
                P1[:, col // G] = p1[:, 0]
            w = W1[:, i]
            if tern:
                t = torch.clamp(torch.round((w[:, None] - p0) / p1), -1, 1)
                q = (p0 + p1 * t)[:, 0]
            else:
                t = torch.clamp(torch.round((w[:, None] - p0) / p1), 0, qmax)
                q = (t * p1 + p0)[:, 0]
            codes[:, col] = t[:, 0].to(torch.int8)
            err = (w - q) / Hinv1[i, i]
            W1[:, i] = q
            if i + 1 < i2 - i1:
                W1[:, i + 1:] -= err[:, None] * Hinv1[i, i + 1:][None, :]
            Err1[:, i] = err
        if i2 < n:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return codes, P0, P1


def quantize_matrix(Wr, Hr, Hinv, dead, opt, signs):
    finder, tp = make_finder(opt, Wr.float(), Hr.float())
    codes, P0, P1 = gptq_quantize(Wr, Hr, Hinv, dead, opt, finder)
    return QMat(opt, codes.cpu(), P0.cpu(), P1.cpu(), signs, tp)
