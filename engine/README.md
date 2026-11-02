# Luminal Quant

High-ratio post-training quantization for LLMs — Hadamard rotation + standardized grid + mixed
precision with logarithmic scales — down to **2 bits per weight**, at quality that beats the open
state of the art below ~3 bpw on the measured benchmarks.

> **Status: 0.1 skeleton.** The compiled Rust core currently exposes the Hadamard rotation
> primitive, validated bit-for-bit against a NumPy reference. The full pipeline (GPTQ, ternary,
> mixed-precision knapsack) is being ported from the reference Python implementation in
> [`../qbench`](../qbench); CUDA kernels land behind the `cuda` feature.

## Why

Quantization shrinks a model so it runs on smaller hardware. Most open tools degrade sharply below
3 bits. Luminal Quant stays usable at 2–2.3 bits by rotating each weight matrix so its outliers
spread out, then spending bits where the model is most sensitive. It is **free software** (AGPL-3.0):
anyone may use and improve it, and every derivative or network service built on it must stay open too.

## Architecture

```
luminal-quant/
  src/            Rust core — numerics, rayon-parallel, no Python dependency
    hadamard.rs   fast Walsh–Hadamard rotation (O(n log n))
    error.rs      typed errors
    python.rs     PyO3 bridge (feature "python")
  cuda/           CUDA C++ kernels (feature "cuda") — GPU dequant/matmul
  python/         importable `luminal_quant` package (thin wrapper)
  tests/          Python parity tests vs NumPy reference
```

Core in Rust, GPU kernels in CUDA C++, a thin Python surface via PyO3 — the same split `candle`
(Hugging Face) uses.

## Build

```bash
# Rust core tests (no Python needed)
cargo test --lib

# Build and install the Python extension into the current environment
pip install maturin
maturin develop --release --features python
pytest tests/
```

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE). Copyright (C) 2026 Santiago Vélez
Casallas. For commercial terms outside the AGPL, contact the author.
