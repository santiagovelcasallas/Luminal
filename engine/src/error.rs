// Luminal Quant — Copyright (C) 2026 Santiago Vélez Casallas. AGPL-3.0-only.
// See LICENSE at the repository root.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum QuantError {
    #[error("dimension {0} has no supported Hadamard construction (need n = 2^k · m, m ∈ {{1,2}} ∪ Paley orders)")]
    NoHadamard(usize),

    #[error("shape mismatch: expected {expected}, got {got}")]
    Shape { expected: String, got: String },

    #[error("invalid bit width {0} (supported: 2, 3, 4)")]
    BitWidth(u32),
}

pub type Result<T> = std::result::Result<T, QuantError>;
