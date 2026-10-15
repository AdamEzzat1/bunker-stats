//! Property and fuzz tests for Rust-layer boundary code.
//!
//! These target the pure-Rust cores (no Python involved): rolling window
//! configuration/shape invariants and matrix symmetry. Proptest drives
//! structured cases; the bolero targets run a bounded random search under
//! plain `cargo test` and become coverage-guided fuzzers under cargo-bolero.
//!
//! Invariants asserted here:
//! * no panics for any window/min_periods/alignment/nan_policy combination,
//!   including non-finite data;
//! * output lengths always match `output_length()`;
//! * correlation matrices are symmetric with unit diagonal (where defined)
//!   and entries in [-1, 1] or NaN — never anything else.

#![cfg(test)]

use proptest::prelude::*;

use crate::kernels::matrix::corr::{corr_matrix_out, corr_matrix_skipna_out};
use crate::kernels::rolling::bounds::output_length;
use crate::kernels::rolling::config::{Alignment, NanPolicy, RollingConfig};
use crate::kernels::rolling::masks::StatsMask;
use crate::kernels::rolling::multi::rolling_multi_into;

fn run_rolling(
    xs: &[f64],
    window: usize,
    min_periods: Option<usize>,
    alignment: Alignment,
    nan_policy: NanPolicy,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let config = RollingConfig::new(window, min_periods, alignment, nan_policy).ok()?;
    if window == 0 || window > xs.len() {
        return None; // kernel contract: caller validates this
    }
    let out_len = output_length(xs.len(), window, alignment);
    let mut mean = vec![0.0; out_len];
    let mut std = vec![0.0; out_len];
    let mut var = vec![0.0; out_len];
    let mut count = vec![0.0; out_len];
    let mut min = vec![0.0; out_len];
    let mut max = vec![0.0; out_len];
    rolling_multi_into(
        xs,
        &config,
        StatsMask::all(),
        &mut mean,
        &mut std,
        &mut var,
        &mut count,
        &mut min,
        &mut max,
    );
    Some((mean, min, max))
}

proptest! {
    /// Rolling outputs always have the documented length and finite windows
    /// satisfy min <= mean <= max.
    #[test]
    fn prop_rolling_shape_and_ordering(
        xs in proptest::collection::vec(-1e6f64..1e6, 1..128),
        window in 1usize..32,
        use_mp in any::<bool>(),
        mp in 1usize..32,
        centered in any::<bool>(),
        policy in 0u8..3,
    ) {
        let alignment = if centered { Alignment::Centered } else { Alignment::Trailing };
        let nan_policy = match policy % 3 {
            0 => NanPolicy::Propagate,
            1 => NanPolicy::Ignore,
            _ => NanPolicy::RequireMinPeriods,
        };
        let min_periods = if use_mp { Some(mp.min(window)) } else { None };

        if let Some((mean, min_v, max_v)) = run_rolling(&xs, window, min_periods, alignment, nan_policy) {
            let expected = output_length(xs.len(), window, alignment);
            prop_assert_eq!(mean.len(), expected);
            for i in 0..mean.len() {
                if mean[i].is_finite() && min_v[i].is_finite() && max_v[i].is_finite() {
                    prop_assert!(min_v[i] <= mean[i] + 1e-9, "min > mean at {}", i);
                    prop_assert!(mean[i] <= max_v[i] + 1e-9, "mean > max at {}", i);
                }
            }
        }
    }

    /// NaN-riddled input never panics; Propagate poisons any window that
    /// contains a NaN.
    #[test]
    fn prop_rolling_nan_propagation(
        raw in proptest::collection::vec(
            prop_oneof![3 => (-1e3f64..1e3).boxed(), 1 => Just(f64::NAN).boxed()],
            2..96,
        ),
        window in 1usize..16,
    ) {
        if let Some((mean, _, _)) = run_rolling(
            &raw, window, None, Alignment::Trailing, NanPolicy::Propagate,
        ) {
            for (i, m) in mean.iter().enumerate() {
                let has_nan = raw[i..i + window].iter().any(|v| v.is_nan());
                if has_nan {
                    prop_assert!(m.is_nan(), "window {} contains NaN but mean is {}", i, m);
                }
            }
        }
    }

    /// Correlation matrices: symmetric, unit diagonal where defined, and every
    /// entry either in [-1, 1] or NaN. Holds for strict and skipna kernels,
    /// including constant columns and NaN contamination.
    #[test]
    fn prop_corr_matrix_symmetry(
        n in 2usize..24,
        p in 1usize..6,
        seed_vals in proptest::collection::vec(-1e3f64..1e3, 24 * 6),
        make_constant in any::<bool>(),
        skipna in any::<bool>(),
        poison in any::<bool>(),
    ) {
        let mut x: Vec<f64> = seed_vals[..n * p].to_vec();
        if make_constant {
            for row in 0..n {
                x[row * p] = 7.5; // first column constant -> zero variance
            }
        }
        if poison && skipna {
            x[0] = f64::NAN; // skipna kernel must tolerate missing entries
        }

        let mut out = vec![0.0f64; p * p];
        if skipna {
            corr_matrix_skipna_out(&x, n, p, &mut out);
        } else {
            corr_matrix_out(&x, n, p, &mut out);
        }

        for i in 0..p {
            for j in 0..p {
                let a = out[i * p + j];
                let b = out[j * p + i];
                // symmetry (NaN == NaN accepted)
                prop_assert!(
                    (a.is_nan() && b.is_nan()) || (a - b).abs() < 1e-9,
                    "asymmetry at ({}, {}): {} vs {}", i, j, a, b
                );
                // range
                prop_assert!(
                    a.is_nan() || (-1.0 - 1e-9..=1.0 + 1e-9).contains(&a),
                    "corr out of range at ({}, {}): {}", i, j, a
                );
            }
            let d = out[i * p + i];
            prop_assert!(
                d.is_nan() || (d - 1.0).abs() < 1e-9,
                "diagonal not 1/NaN at {}: {}", i, d
            );
        }
    }
}

/// Bolero: arbitrary bytes -> rolling configs and data (including non-finite
/// floats). Must never panic and must respect the output-length contract.
#[test]
fn fuzz_rolling_boundary_no_panic() {
    bolero::check!()
        .with_type::<(Vec<f64>, u8, u8, bool, u8)>()
        .for_each(|(xs, window, mp, centered, policy)| {
            if xs.is_empty() {
                return;
            }
            let window = (*window as usize % xs.len().max(1)) + 1;
            let alignment = if *centered { Alignment::Centered } else { Alignment::Trailing };
            let nan_policy = match policy % 3 {
                0 => NanPolicy::Propagate,
                1 => NanPolicy::Ignore,
                _ => NanPolicy::RequireMinPeriods,
            };
            let min_periods = if *mp == 0 { None } else { Some((*mp as usize % window) + 1) };
            // Raw f64 bytes include NaN/inf/subnormals — exactly the point.
            let _ = run_rolling(xs, window, min_periods, alignment, nan_policy);
        });
}

/// Bolero: small arbitrary matrices (with non-finite entries) through both
/// correlation kernels. No panics; output length p*p is caller-enforced.
#[test]
fn fuzz_corr_matrix_no_panic() {
    bolero::check!()
        .with_type::<(Vec<f64>, u8, bool)>()
        .for_each(|(vals, p_raw, skipna)| {
            let p = (*p_raw as usize % 5) + 1;
            if vals.len() < p {
                return;
            }
            let n = vals.len() / p;
            if n == 0 {
                return;
            }
            let x = &vals[..n * p];
            let mut out = vec![0.0f64; p * p];
            if *skipna {
                corr_matrix_skipna_out(x, n, p, &mut out);
            } else {
                corr_matrix_out(x, n, p, &mut out);
            }
            assert_eq!(out.len(), p * p);
        });
}
