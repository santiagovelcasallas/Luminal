// Luminal Quant — Copyright (C) 2026 Santiago Vélez Casallas. AGPL-3.0-only.
// See LICENSE at the repository root.

//! GPTQ with the standardized grid, uniform bit width (the control path of the engine).
//!
//! This is the one hot loop that is wasteful in Python: it walks the matrix column by column
//! (1024–3072 Python iterations per matrix), launching several tiny tensor ops each step, so the
//! interpreter's dispatch — not the arithmetic — dominates. Here it is a single fused pass.
//!
//! Numerics mirror the reference (`qbench/quant.py`): weights and Hinv in f32, population std with
//! ddof=1, round-half-to-even, scales rounded to fp16. The log-scale and ternary finders are not
//! here yet; this is the uniform `q{2,3,4}` path.

use half::f16;
use rayon::prelude::*;

/// 33 clip candidates, linspace(1.0, 5.0, 33) — identical to the reference `CS`.
fn clip_candidates() -> [f32; 33] {
    let mut cs = [0.0f32; 33];
    for (i, c) in cs.iter_mut().enumerate() {
        *c = 1.0 + 4.0 * (i as f32) / 32.0;
    }
    cs
}

#[inline]
fn fp16(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

/// Round half to even (torch.round / numpy semantics).
#[inline]
fn rint(x: f32) -> f32 {
    x.round_ties_even()
}

/// grid_clip for one row of a group: returns (lo, s), both already fp16-rounded, s guarded != 0.
/// `blk` is the row's `g` values. Mirrors `FinderQ(grid_clip(...))`.
fn grid_clip_row(blk: &[f32], qmax: f32) -> (f32, f32) {
    let g = blk.len();
    let mu: f32 = blk.iter().sum::<f32>() / g as f32;
    let var: f32 = blk.iter().map(|&x| (x - mu) * (x - mu)).sum::<f32>() / (g as f32 - 1.0); // ddof=1
    let sd = var.sqrt() + 1e-12;

    let mut best_err = f32::INFINITY;
    let mut best_c = clip_candidates()[0];
    for &c in clip_candidates().iter() {
        let lo = mu - c * sd;
        let s = 2.0 * c * sd / qmax;
        let mut err = 0.0f32;
        for &x in blk {
            let q = rint((x - lo) / s).clamp(0.0, qmax) * s + lo;
            err += (q - x) * (q - x);
        }
        if err < best_err {
            // strict '<' keeps the first minimum, like the reference argmin
            best_err = err;
            best_c = c;
        }
    }
    let lo = fp16(mu - best_c * sd);
    let hi = mu + best_c * sd;
    let s = fp16((hi - lo) / qmax);
    (lo, if s == 0.0 { 1.0 } else { s })
}

/// Quantize one rotated weight matrix with uniform `bits`, GPTQ error compensation and the
/// standardized grid. `w` is (rows × n) row-major f32 (consumed as scratch), `hinv` is the upper
/// Cholesky factor of the inverse Hessian, (n × n) row-major f32. `group` divides `n`, and
/// `blocksize` is a multiple of `group`.
///
/// Returns `(codes, p0, p1)`: codes are the integer grid indices in `[0, qmax]` as i8 (rows × n),
/// p0/p1 are lo/s per (row, group) (rows × n/group), row-major. This is the exact triple the
/// reference `gptq_quantize` returns for the `q` path.
pub fn gptq_uniform(
    w: &mut [f32],
    hinv: &[f32],
    rows: usize,
    n: usize,
    bits: u32,
    group: usize,
    blocksize: usize,
) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
    assert_eq!(w.len(), rows * n);
    assert_eq!(hinv.len(), n * n);
    assert!(n % group == 0 && blocksize % group == 0 && n % blocksize == 0);
    assert!(group <= 1024);
    let qmax = ((1u32 << bits) - 1) as f32;
    let ng = n / group;
    let mut codes = vec![0i8; rows * n];
    let mut p0 = vec![0f32; rows * ng];
    let mut p1 = vec![0f32; rows * ng];

    // Each row is quantized independently (GPTQ does not couple rows), so rows run in parallel; Hinv
    // is shared read-only. Within a row the columns are strictly sequential (the error feeds forward).
    codes
        .par_chunks_mut(n)
        .zip(p0.par_chunks_mut(ng))
        .zip(p1.par_chunks_mut(ng))
        .zip(w.par_chunks_mut(n))
        .for_each(|(((codes_row, p0_row), p1_row), w_row)| {
            quantize_row(w_row, hinv, n, qmax, group, blocksize, codes_row, p0_row, p1_row);
        });
    (codes, p0, p1)
}

#[allow(clippy::too_many_arguments)]
fn quantize_row(
    w: &mut [f32],
    hinv: &[f32],
    n: usize,
    qmax: f32,
    group: usize,
    blocksize: usize,
    codes: &mut [i8],
    p0: &mut [f32],
    p1: &mut [f32],
) {
    let mut lo = 0f32;
    let mut s = 1f32;
    let mut err_block = vec![0f32; blocksize];
    let mut i1 = 0;
    while i1 < n {
        let i2 = (i1 + blocksize).min(n);
        let bw = i2 - i1;
        for i in 0..bw {
            let col = i1 + i;
            if col % group == 0 {
                let (l, ss) = grid_clip_row(&w[col..col + group], qmax);
                lo = l;
                s = ss;
                p0[col / group] = l;
                p1[col / group] = ss;
            }
            let wv = w[col];
            let t = rint((wv - lo) / s).clamp(0.0, qmax);
            let q = t * s + lo;
            codes[col] = t as i8;
            let err = (wv - q) / hinv[col * n + col];
            w[col] = q;
            err_block[i] = err;
            for jj in (col + 1)..i2 {
                w[jj] -= err * hinv[col * n + jj];
            }
        }
        if i2 < n {
            for jj in i2..n {
                let mut acc = 0f32;
                for i in 0..bw {
                    acc += err_block[i] * hinv[(i1 + i) * n + jj];
                }
                w[jj] -= acc;
            }
        }
        i1 = i2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_clip_picks_a_candidate() {
        let blk: Vec<f32> = (0..128).map(|i| (i as f32 * 0.01).sin() * 0.03).collect();
        let (lo, s) = grid_clip_row(&blk, 3.0);
        assert!(s > 0.0 && lo.is_finite());
        // lo and s must be exact fp16 values
        assert_eq!(lo, super::fp16(lo));
        assert_eq!(s, super::fp16(s));
    }

    #[test]
    fn codes_in_range_and_shapes() {
        let (rows, n, g, bs, bits) = (8usize, 256usize, 128usize, 128usize, 2u32);
        let mut w: Vec<f32> = (0..rows * n).map(|i| ((i as f32) * 0.001).cos() * 0.02).collect();
        // identity-ish upper triangular Hinv (well-conditioned): 1 on diag, small above
        let mut hinv = vec![0f32; n * n];
        for a in 0..n {
            hinv[a * n + a] = 1.0;
            for b in (a + 1)..n {
                hinv[a * n + b] = 0.001 / ((b - a) as f32);
            }
        }
        let (codes, p0, p1) = gptq_uniform(&mut w, &hinv, rows, n, bits, g, bs);
        let qmax = ((1 << bits) - 1) as i8;
        assert_eq!(codes.len(), rows * n);
        assert_eq!(p0.len(), rows * (n / g));
        assert_eq!(p1.len(), rows * (n / g));
        assert!(codes.iter().all(|&c| c >= 0 && c <= qmax));
        assert!(p1.iter().all(|&s| s > 0.0));
    }
}
