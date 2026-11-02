# Luminal Quant — Copyright (C) 2026 Santiago Vélez Casallas. AGPL-3.0-only.
"""Parity: the compiled Rust rotation must match a NumPy reference bit-for-bit (same orthonormal map).
Run after `maturin develop` in engine/. This is the gate before trusting the native path in the pipeline."""
import numpy as np
import pytest

luminal_quant = pytest.importorskip("luminal_quant")


def ref_hadamard(n):
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


@pytest.mark.parametrize("n", [64, 256, 1024, 2048])
def test_rotate_matches_numpy(n):
    rng = np.random.default_rng(0)
    W = rng.standard_normal((40, n))
    signs = luminal_quant.rotation_signs(n, seed=0, matrix_index=3)
    assert set(np.unique(signs)) <= {-1.0, 1.0}
    got = luminal_quant.rotate(W, signs)
    want = (W * signs) @ ref_hadamard(n).T
    assert np.allclose(got, want, atol=1e-10), f"max diff {np.abs(got - want).max()}"


def test_orthonormal():
    n = 512
    signs = luminal_quant.rotation_signs(n, 1, 1)
    W = np.random.default_rng(1).standard_normal((8, n))
    out = luminal_quant.rotate(W, signs)
    assert np.allclose((W ** 2).sum(1), (out ** 2).sum(1), atol=1e-9)


def test_non_pow2_raises():
    with pytest.raises(ValueError):
        luminal_quant.rotate(np.zeros((2, 3072)), luminal_quant.rotation_signs(3072, 0, 0))
