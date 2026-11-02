# CUDA kernels (feature `cuda`)

GPU kernels for the hot inference path — weight dequantization, the Hadamard rotation applied to
activations, and the quantized matmul. Written in CUDA C++ (the only first-class GPU language), built
behind the `cuda` Cargo feature and called from the Rust core via FFI.

**Not yet implemented in 0.1.** The CPU path in `src/` is the reference; these kernels must reproduce
it bit-for-bit (same parity gate as `tests/test_parity.py`). Placeholder so the layout and the build
feature are fixed from the start.
