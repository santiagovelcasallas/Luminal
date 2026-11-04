// Luminal Quant — high-ratio post-training quantization engine.
// Copyright (C) 2026 Santiago Vélez Casallas
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU Affero General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
// details. You should have received a copy of the GNU Affero General Public
// License along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Core quantization engine. The numerical algorithms live in pure Rust (CPU, `rayon`-parallel);
//! optional CUDA kernels compile behind the `cuda` feature. The Python extension is a thin wrapper.

pub mod error;
pub mod gptq;
pub mod hadamard;

pub use error::{QuantError, Result};

#[cfg(feature = "python")]
mod python;

/// Panic-free check of the single compile-time invariant the engine relies on.
#[cfg(test)]
mod smoke {
    #[test]
    fn f64_is_ieee() {
        assert_eq!(core::mem::size_of::<f64>(), 8);
    }
}
