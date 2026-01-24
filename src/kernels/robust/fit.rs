/// Robust fit and score kernels - OPTIMIZED
///
/// Optimizations:
/// - Fused median+MAD kernel for default robust_fit (Optimization #3)
/// - Enum-based dispatch (no string parsing in hot path)
/// - Zero-allocation paths for common cases

use super::policy::{LocationPolicy, RobustConfig, ScalePolicy};
use super::extended::{
    iqr_slice, mad_slice, median_slice, qn_scale_slice,
    trimmed_mean_slice, median_mad_fused,
};

/// Compute robust location and scale - OPTIMIZED
///
/// Uses fused median+MAD kernel for default configuration.
/// Returns (location, scale) tuple.
///
/// # Performance
/// - Default (median+MAD): Single allocation, fused computation
/// - Custom configs: Minimal allocations via optimized kernels
#[inline]
pub fn robust_fit_slice(xs: &[f64], cfg: &RobustConfig) -> (f64, f64) {
    if xs.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    // FAST PATH: Fused median+MAD (most common case) - Optimization #3
    if matches!(cfg.location, LocationPolicy::Median) {
        if let ScalePolicy::Mad { consistent } = cfg.scale {
            let (median, mad) = median_mad_fused(xs);
            let scale = if consistent {
                mad * 1.482_602_218_505_602
            } else {
                mad
            };
            return (median, scale);
        }
    }

    // GENERAL PATH: Separate location and scale computation
    let location = compute_location(xs, &cfg.location);
    let scale = compute_scale(xs, &cfg.scale);

    (location, scale)
}

/// Compute robust z-scores - OPTIMIZED
///
/// Uses fused median+MAD for default configuration.
#[inline]
pub fn robust_score_slice(xs: &[f64], cfg: &RobustConfig) -> Vec<f64> {
    if xs.is_empty() {
        return vec![f64::NAN];
    }

    // FAST PATH: Fused computation for median+MAD - Optimization #3
    let (loc, scale) = if matches!(cfg.location, LocationPolicy::Median) {
        if let ScalePolicy::Mad { consistent } = cfg.scale {
            let (median, mad) = median_mad_fused(xs);
            let scale = if consistent {
                mad * 1.482_602_218_505_602
            } else {
                mad
            };
            (median, scale)
        } else {
            // General path
            let loc = compute_location(xs, &cfg.location);
            let scale = compute_scale(xs, &cfg.scale);
            (loc, scale)
        }
    } else {
        // General path
        let loc = compute_location(xs, &cfg.location);
        let scale = compute_scale(xs, &cfg.scale);
        (loc, scale)
    };

    if !scale.is_finite() || scale == 0.0 {
        return vec![f64::NAN; xs.len()];
    }

    xs.iter().map(|&x| (x - loc) / scale).collect()
}

// ============================================================================
// Internal dispatch functions
// ============================================================================

#[inline]
fn compute_location(xs: &[f64], policy: &LocationPolicy) -> f64 {
    match policy {
        LocationPolicy::Median => median_slice(xs),
        
        LocationPolicy::TrimmedMean { trim } => {
            trimmed_mean_slice(xs, *trim)
        }
        
        LocationPolicy::Huber { c, max_iter, tol, scale_policy } => {
            huber_location_with_policy(xs, *c, *max_iter, *tol, scale_policy)
        }
    }
}

#[inline]
fn compute_scale(xs: &[f64], policy: &ScalePolicy) -> f64 {
    match policy {
        ScalePolicy::Mad { consistent } => {
            let mad = mad_slice(xs);
            if *consistent {
                mad * 1.482_602_218_505_602
            } else {
                mad
            }
        }
        
        ScalePolicy::Iqr => iqr_slice(xs),
        
        ScalePolicy::Qn => qn_scale_slice(xs),
    }
}

/// Huber location with configurable scale estimator
#[inline]
fn huber_location_with_policy(
    xs: &[f64],
    k: f64,
    max_iter: usize,
    tol: f64,
    scale_policy: &ScalePolicy,
) -> f64 {
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }

    // Initialize with median
    let mut mu = median_slice(xs);
    
    // Compute scale using specified policy
    let scale = compute_scale(xs, scale_policy);

    if !scale.is_finite() || scale == 0.0 {
        return mu;
    }

    // Iterative refinement
    for _ in 0..max_iter {
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for &x in xs {
            let r = (x - mu) / scale;
            let psi = if r.abs() <= k { r } else { k * r.signum() };

            numerator += psi;
            denominator += if r.abs() <= k { 1.0 } else { k / r.abs() };
        }

        if denominator == 0.0 {
            break;
        }

        let delta = scale * numerator / denominator;
        mu += delta;

        if delta.abs() < tol * scale {
            break;
        }
    }

    mu
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_fit_median_mad_fused() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = RobustConfig {
            location: LocationPolicy::Median,
            scale: ScalePolicy::Mad { consistent: false },
        };
        let (loc, scale) = robust_fit_slice(&data, &cfg);
        assert!((loc - 3.0).abs() < EPS);
        assert!((scale - 1.0).abs() < EPS);
    }

    #[test]
    fn test_fit_median_mad_consistent() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = RobustConfig {
            location: LocationPolicy::Median,
            scale: ScalePolicy::Mad { consistent: true },
        };
        let (loc, scale) = robust_fit_slice(&data, &cfg);
        assert!((loc - 3.0).abs() < EPS);
        assert!((scale - 1.4826).abs() < 0.001);
    }

    #[test]
    fn test_fit_trimmed_iqr() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let cfg = RobustConfig {
            location: LocationPolicy::TrimmedMean { trim: 0.1 },
            scale: ScalePolicy::Iqr,
        };
        let (loc, scale) = robust_fit_slice(&data, &cfg);
        assert!((loc - 5.5).abs() < EPS);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_score_fused() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let cfg = RobustConfig {
            location: LocationPolicy::Median,
            scale: ScalePolicy::Mad { consistent: false },
        };
        let scores = robust_score_slice(&data, &cfg);
        assert_eq!(scores.len(), 5);
        assert!((scores[0] - (-2.0)).abs() < EPS);
        assert!((scores[2] - 0.0).abs() < EPS);
        assert!((scores[4] - 2.0).abs() < EPS);
    }

    #[test]
    fn test_empty_input() {
        let cfg = RobustConfig::default();
        let (loc, scale) = robust_fit_slice(&[], &cfg);
        assert!(loc.is_nan());
        assert!(scale.is_nan());
    }

    #[test]
    fn test_determinism() {
        let data = vec![3.1, 1.4, 5.9, 2.6, 5.3];
        let cfg = RobustConfig {
            location: LocationPolicy::Huber {
                c: 1.345,
                max_iter: 50,
                tol: 1e-6,
                scale_policy: ScalePolicy::Mad { consistent: true },
            },
            scale: ScalePolicy::Qn,
        };

        let (loc1, scale1) = robust_fit_slice(&data, &cfg);
        let (loc2, scale2) = robust_fit_slice(&data, &cfg);

        assert_eq!(loc1, loc2);
        assert_eq!(scale1, scale2);
    }
}