# Luminal Quant

High-ratio post-training quantization for LLMs — Hadamard rotation + standardized grid + mixed
precision with logarithmic scales — down to **2 bits per weight**, at quality that beats the open
state of the art below ~3 bpw on the measured benchmarks.

> **Status: 0.1.** The compiled Rust core exposes two primitives, each validated against the reference
> Python pipeline in [`../qbench`](../qbench): the **Hadamard rotation** (bit-for-bit vs NumPy) and the
> **uniform-grid GPTQ** quantizer (functional parity — same codes >99%, Hessian-weighted dequant error
> <0.5%). GPTQ runs row-parallel (`rayon`); on CPU it beats the PyTorch path ~2.8× at 1024×1024 and
> ties at 2048×2048 — the trailing error-compensation step is BLAS-bound and needs the CUDA kernel
> (behind the `cuda` feature) to scale. Log-scale and ternary finders, and the mixed-precision
> knapsack, stay in Python for now (already backed by PyTorch C/CUDA; porting them buys little).

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
