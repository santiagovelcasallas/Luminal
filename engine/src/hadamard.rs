// Luminal Quant — Copyright (C) 2026 Santiago Vélez Casallas. AGPL-3.0-only.
// See LICENSE at the repository root.

//! Randomized Hadamard rotation, the first hot primitive of the engine.
//!
//! For a power-of-two dimension `n`, the normalized Walsh–Hadamard transform is applied in-place in
//! O(n log n) (the fast transform), never as an n×n matmul. The randomized sign flip `diag(s)` is
//! folded in as a cheap elementwise pass, giving an orthogonal map `V = H_n · diag(s)`.
//!
//! Non-power-of-two dimensions (e.g. 3072 = 1024·3) need a Paley/Kronecker construction; that path
//! is not in this 0.1 skeleton and returns `QuantError::NoHadamard`.

use crate::error::{QuantError, Result};

/// In-place fast Walsh–Hadamard transform on a power-of-two slice, normalized by 1/sqrt(n) so the
/// map is orthonormal. Panics never: callers validate the length via [`is_pow2`].
fn fwht_normalized(data: &mut [f64]) {
    let n = data.len();
    let mut len = 1;
    while len < n {
        let mut i = 0;
        while i < n {
            for j in i..i + len {
                let a = data[j];
                let b = data[j + len];
                data[j] = a + b;
                data[j + len] = a - b;
            }
            i += len << 1;
        }
        len <<= 1;
    }
    let inv = 1.0 / (n as f64).sqrt();
    for x in data.iter_mut() {
        *x *= inv;
    }
}

#[inline]
pub fn is_pow2(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Deterministic ±1 sign vector for one matrix, from a (seed, matrix_index) pair. Must match the
/// reference generator so quantized weights reproduce bit-for-bit across the Python and Rust paths.
pub fn rotation_signs(n: usize, seed: u64, matrix_index: u64) -> Vec<f64> {
    // SplitMix64 — tiny, fast, well-distributed; the stream is fixed by the two inputs.
    let mut state = seed
        .wrapping_mul(1_000_003)
        .wrapping_add(matrix_index)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.push(if z & 1 == 0 { 1.0 } else { -1.0 });
    }
    out
}

/// Apply `x -> H_n · (s ⊙ x)` in place over each contiguous row of length `n`. `signs.len() == n`.
pub fn rotate_rows(rows: &mut [f64], n: usize, signs: &[f64]) -> Result<()> {
    if !is_pow2(n) {
        return Err(QuantError::NoHadamard(n));
    }
    if signs.len() != n {
        return Err(QuantError::Shape {
            expected: format!("signs of length {n}"),
            got: format!("{}", signs.len()),
        });
    }
    if rows.len() % n != 0 {
        return Err(QuantError::Shape {
            expected: format!("row length multiple of {n}"),
            got: format!("{}", rows.len()),
        });
    }
    for row in rows.chunks_mut(n) {
        for (v, s) in row.iter_mut().zip(signs) {
            *v *= s;
        }
        fwht_normalized(row);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_is_orthonormal() {
        // Applying the rotation twice (it is symmetric and involutory up to the sign flip) and
        // checking norm preservation is enough to catch a broken butterfly or normalization.
        let n = 1024;
        let signs = vec![1.0; n];
        let mut x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let norm0: f64 = x.iter().map(|v| v * v).sum();
        rotate_rows(&mut x, n, &signs).unwrap();
        let norm1: f64 = x.iter().map(|v| v * v).sum();
        assert!((norm0 - norm1).abs() < 1e-9, "norm not preserved: {norm0} vs {norm1}");
    }

    #[test]
    fn signs_are_deterministic_and_vary() {
        assert_eq!(rotation_signs(1024, 0, 1), rotation_signs(1024, 0, 1));
        assert_ne!(rotation_signs(1024, 0, 1), rotation_signs(1024, 0, 2));
        assert_ne!(rotation_signs(1024, 0, 1), rotation_signs(1024, 1, 1));
        assert!(rotation_signs(64, 7, 3).iter().all(|&s| s == 1.0 || s == -1.0));
    }

    #[test]
    fn rejects_non_pow2() {
        let mut x = vec![0.0; 3072];
        assert!(matches!(rotate_rows(&mut x, 3072, &vec![1.0; 3072]), Err(QuantError::NoHadamard(3072))));
    }
}
