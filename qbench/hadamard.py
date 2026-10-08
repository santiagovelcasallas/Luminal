"""HANDOVER §4.2: randomized Hadamard rotation (input side, one seed per matrix)."""
import functools
import math

import torch


def sylvester(n):
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def is_prime(q):
    return q > 1 and all(q % d for d in range(2, int(q ** .5) + 1))


def paley1(q):
    qr = {(x * x) % q for x in range(1, q)}
    chi = lambda a: 0 if a % q == 0 else (1 if a % q in qr else -1)
    Q = torch.tensor([[chi(j - i) for j in range(q)] for i in range(q)], dtype=torch.float64)
    S = torch.zeros(q + 1, q + 1, dtype=torch.float64)
    S[0, 1:] = 1
    S[1:, 0] = -1
    S[1:, 1:] = Q
    return S + torch.eye(q + 1, dtype=torch.float64)


@functools.lru_cache(maxsize=None)
def hadamard(n):
    for m in sorted({1, 2} | {q + 1 for q in range(3, n) if is_prime(q) and q % 4 == 3 and n % (q + 1) == 0}):
        k = n // m
        if n % m == 0 and (k & (k - 1)) == 0:
            Hm = torch.ones(1, 1, dtype=torch.float64) if m == 1 else (sylvester(2) if m == 2 else paley1(m - 1))
            return torch.kron(sylvester(k), Hm) / math.sqrt(n)
    raise ValueError(f"no Hadamard construction for n={n}")


def rotation_signs(n, seed, matrix_index):
    g = torch.Generator().manual_seed(1_000_003 * (seed + 1) + matrix_index)
    return torch.randint(0, 2, (n,), generator=g).double() * 2 - 1


def had_rot(n, signs):
    """V = H_n · diag(s) (H_n already normalized by 1/sqrt(n)); V is orthogonal."""
    return hadamard(n) * signs[None, :]
