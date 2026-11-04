# NEXT STEPS — CUDA kernel for GPTQ error compensation

Status after session: the uniform-grid GPTQ loop is ported to Rust (`src/gptq.rs`), row-parallel via
`rayon`, verified against the Python reference (`../qbench`) — codes match >99%, Hessian-weighted
dequant error <0.5% (12 parity tests in `tests/`). On CPU it beats PyTorch ~2.8× at 1024×1024 but
**ties at 2048×2048**, because the trailing error-compensation step is BLAS-bound. The real speed win
needs a GPU. This file is the plan to pick up in a fresh session (everything is committed).

## Why CPU ties at large sizes

GPTQ per block does two kinds of work:
1. **Per-column, sequential**: quantize column, compute `err`, update the rest of the block. Small,
   dispatch-bound — this is where Python was wasteful and Rust already wins.
2. **Trailing update** `W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]`: a dense matmul, (rows × bw) @ (bw ×
   tail). PyTorch runs it on optimized BLAS; the naive Rust scalar triple-loop does not scale. At
   2048 this term dominates and erases the win.

So the target is: keep the sequential part fused, and run the trailing matmul on the GPU.

## The target kernel

Port `quantize_row` / `gptq_uniform` (src/gptq.rs) to CUDA, one of two designs:

- **Design A (simplest, recommended first):** keep the whole matrix resident in GPU memory. Run the
  per-column sequential inner loop over a block in one kernel (one thread per row — rows are
  independent), then call **cuBLAS `gemm`** for the trailing update `Err1 @ Hinv`. Alternate kernel
  and gemm per block. This mirrors the Rust structure and reuses cuBLAS for the heavy part.
- **Design B (faster, harder):** fuse the block's inner loop and the trailing update into fewer
  kernel launches with shared-memory tiling of `Hinv`. Only attempt after A is correct.

Expected: 5–20× over the PyTorch path for the real matrix sizes (2048–8192), because the sequential
dispatch cost disappears and the matmul stays on cuBLAS.

## Concrete tasks

1. **Build plumbing.** Add a `build.rs` that compiles `cuda/gptq.cu` with `nvcc` when the `cuda`
   feature is on, and links `cudart` + `cublas`. Gate all CUDA code behind `#[cfg(feature = "cuda")]`.
   A machine with CUDA toolkit + an NVIDIA GPU is required to build and test (Kaggle T4, Colab, or an
   EC2 g5/g6 — the same environments already used for the benchmarks).
2. **`cuda/gptq.cu`**: the per-block inner-loop kernel (one thread per row) + cuBLAS gemm calls.
   Inputs: device pointers for `W` (rows×n f32), `Hinv` (n×n f32), outputs `codes` (i8), `p0`/`p1`.
   `grid_clip_row` becomes a `__device__` function; `rint` = `rintf` (round-ties-even, matches torch).
   fp16 scale rounding = `__float2half`/`__half2float`.
3. **Rust FFI** (`src/cuda.rs`): `extern "C"` bindings to the kernel, host-side alloc/copy, and a
   `gptq_uniform_cuda(...)` that mirrors the CPU signature. Expose it in `python.rs` as
   `gptq_uniform_cuda`, and have the Python wrapper pick CUDA when a tensor is on GPU, else CPU.
4. **Parity gate (reuse what exists):** `tests/test_gptq_parity.py` already defines the tolerance
   (codes >99%, Hessian error <0.5%). Run it against the CUDA path on the GPU box. Must pass before
   trusting it.
5. **Benchmark** at 2048/4096/8192 vs the PyTorch `qbench` path; record numbers in the README table.

## After the kernel

- Port `FinderQLog` (log scales) and the ternary finder to `__device__` functions so V1/V2/V3 — not
  just the control path — use the fast kernel. The Python versions stay as the reference oracle.
- Wire `qbench/quant.py` to call the native path when available (a flag: native kernel if importable,
  else the current PyTorch loop), so the whole benchmark pipeline speeds up with zero change in
  results (parity guarantees it).

## Do not port (stays in Python — already PyTorch C/CUDA, porting buys nothing)

Hessian collection, the standardized-grid vectorized math outside the loop, sensitivity (runs the
model), and the knapsack (runs once per budget on tiny lists).
