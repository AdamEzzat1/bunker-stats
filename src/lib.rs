// The crate deliberately retains a number of pure-`&[f64]` slice-core kernel
// helpers and work-in-progress time-series functions (e.g. `kernels::tsa::rolling`,
// several `infer::common` reducers) that are part of the internal kernel library
// but are not all wired to a `#[pyfunction]` yet. Allowing dead_code keeps
// `cargo check` a clean release gate without deleting API that callers/tests may
// wire up next. A follow-up pass should prune what is truly obsolete (tracked
// separately; see the review's dead-code findings).
#![allow(dead_code)]

mod kernels;
mod infer;
pub(crate) mod util;

// Property/fuzz tests for boundary code (proptest + bolero); test builds only.
#[cfg(test)]
mod boundary_tests;


use numpy::{
    ndarray::{Array2, ArrayViewD, Axis},
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
};


use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::types::{PyAny, PyTuple};
use pyo3::prelude::*;

use kernels::rolling::engine::rolling_mean_std_vec;
use kernels::rolling::zscore::zscore_from_mean_std;
use kernels::rolling::axis0::{rolling_mean_axis0_vec, rolling_std_axis0_vec, rolling_mean_std_axis0_vec};
use kernels::rolling::covcorr::rolling_cov_vec;

use kernels::rolling::var::vars_from_stds;

// NEW v0.2.9: Fused multi-stat rolling kernels
use kernels::rolling::{
    bounds::output_length,
    config::{Alignment, NanPolicy, RollingConfig},
    masks::StatsMask,
    multi::rolling_multi_into,
    multi_axis0::rolling_multi_axis0_into,
};


use kernels::quantile::percentile::percentile_slice as percentile_slice_k;
use kernels::quantile::iqr::iqr_slice as iqr_slice_k;
use kernels::quantile::winsor::winsorize_vec as winsorize_vec_k;
use crate::kernels::matrix::cov::{
    cov_matrix_out,
    cov_matrix_bias_out,
    cov_matrix_centered_out,
    cov_matrix_skipna_out,
    xtx_matrix_out,
    xxt_matrix_out,
    pairwise_euclidean_cols_out,
    pairwise_cosine_cols_out,
}; 
use kernels::matrix::corr::{
    corr_matrix_out as corr_matrix_out_k,
    corr_matrix_skipna_out as corr_matrix_skipna_out_k,
    corr_distance_out as corr_distance_out_k,
};
use kernels::robust::extended::{
    mad_slice as mad_slice_k,
    trimmed_mean_slice as trimmed_mean_slice_k,
    median_slice as median_slice_k,
    iqr_slice as iqr_robust_slice_k,
    winsorized_mean_slice as winsorized_mean_slice_k,
    trimmed_std_slice as trimmed_std_slice_k,
    mad_std_slice as mad_std_slice_k,
    biweight_midvariance_slice as biweight_midvariance_slice_k,
    qn_scale_slice as qn_scale_slice_k,
    huber_location_slice as huber_location_slice_k,
    median_slice_skipna as median_slice_skipna_k,
    mad_slice_skipna as mad_slice_skipna_k,
    trimmed_mean_slice_skipna as trimmed_mean_slice_skipna_k,
    iqr_slice_skipna as iqr_slice_skipna_k,
};

// resampling
use crate::kernels::resampling::bootstrap::{
    bootstrap_mean, bootstrap_mean_ci, bootstrap_ci, bootstrap_ci_with_draws, bootstrap_corr,
    bootstrap_se, bootstrap_var, bootstrap_t_ci_mean, bootstrap_bca_ci,
    bayesian_bootstrap_ci,
    permutation_corr_test, permutation_mean_diff_test,
    moving_block_bootstrap_mean_ci, circular_block_bootstrap_mean_ci, stationary_bootstrap_mean_ci,
};
use crate::kernels::resampling::jackknife::{ jackknife_mean, jackknife_mean_ci, influence_mean, delete_d_jackknife_mean, jackknife_after_bootstrap_se_mean };

// tsa
use crate::kernels::tsa::stationarity::{ 
    adf_test, kpss_test, pp_test,
    variance_ratio_test, zivot_andrews_test, trend_stationarity_test,
    integration_order_test, seasonal_diff_test, seasonal_unit_root_test
};
use crate::kernels::tsa::diagnostics::{ 
    ljung_box, durbin_watson, bg_test, box_pierce, runs_test, acf_zero_crossing 
};
use crate::kernels::tsa::acf_pacf::{ 
    acf, pacf, pacf_yw, acovf, acf_with_ci, ccf, pacf_innovations, pacf_burg 
};

use crate::kernels::tsa::spectral::{ 
    periodogram, welch_psd, cumulative_periodogram, dominant_frequency,
    spectral_entropy, bartlett_psd, spectral_peaks, spectral_flatness,
    band_power, spectral_centroid, spectral_rolloff
};
use crate::kernels::tsa::rolling_autocorr::{
    rolling_autocorr, rolling_correlation, rolling_autocorr_multi
};
 

// dist
use crate::kernels::dist::normal::{
    norm_pdf, norm_cdf, norm_ppf,
    norm_logpdf, norm_sf, norm_logsf, norm_cumhazard,
};
use crate::kernels::dist::exponential::{
    exp_pdf, exp_cdf, exp_ppf,
    exp_logpdf, exp_sf, exp_logsf, exp_cumhazard,
};
use crate::kernels::dist::uniform::{
    unif_pdf, unif_cdf, unif_ppf,
    unif_logpdf, unif_sf, unif_logsf,
};




// ======================
// Core slice helpers
// ======================

pub(crate) fn mean_slice(xs: &[f64]) -> f64 {

    if xs.is_empty() {
        return f64::NAN;
    }
    let mut sum = 0.0;
    for &x in xs {
        sum += x;
    }
    sum / (xs.len() as f64)
}

// Sample variance (ddof=1)
fn var_slice(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n <= 1 {
        return f64::NAN;
    }
    let m = mean_slice(xs);
    let mut acc = 0.0;
    for &x in xs {
        let d = x - m;
        acc += d * d;
    }
    acc / ((n - 1) as f64)
}

fn std_slice(xs: &[f64]) -> f64 {
    var_slice(xs).sqrt()
}

// NaN-aware helpers (skip NaNs, ddof=1 for var/std when >=2 valid values)

fn mean_slice_skipna(xs: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for &x in xs {
        if x.is_nan() {
            continue;
        }
        sum += x;
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / (count as f64)
    }
}

fn var_slice_skipna(xs: &[f64]) -> f64 {
    let mut values = Vec::with_capacity(xs.len());
    for &x in xs {
        if !x.is_nan() {
            values.push(x);
        }
    }
    let n = values.len();
    if n <= 1 {
        return f64::NAN;
    }
    let m = mean_slice(&values);
    let mut acc = 0.0;
    for &v in &values {
        let d = v - m;
        acc += d * d;
    }
    acc / ((n - 1) as f64)
}

fn std_slice_skipna(xs: &[f64]) -> f64 {
    var_slice_skipna(xs).sqrt()
}


fn mad_slice(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    // NaN in input -> NaN out (numpy semantics); also keeps NaN away from the
    // comparator, which under panic="abort" would abort the interpreter.
    if crate::util::any_nan(xs) {
        return f64::NAN;
    }
    // Copy once, sort once for the median, then reuse the same buffer for deviations.
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));

    let n = v.len();
    let med = if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    };

    // Reuse `v` to hold absolute deviations, then sort to get the MAD median.
    for val in &mut v {
        *val = (*val - med).abs();
    }
    v.sort_by(|a, b| a.total_cmp(b));

    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

// ======================
// Parse helper functions for rolling config
// ======================

/// Parse alignment string to Alignment enum
fn parse_alignment(s: &str) -> PyResult<Alignment> {
    match s.to_lowercase().as_str() {
        "trailing" => Ok(Alignment::Trailing),
        "centered" => Ok(Alignment::Centered),
        _ => Err(PyErr::new::<PyValueError, _>(
            format!("Invalid alignment: '{}'. Must be 'trailing' or 'centered'", s)
        )),
    }
}

/// Parse nan_policy string to NanPolicy enum
fn parse_nan_policy(s: &str) -> PyResult<NanPolicy> {
    match s.to_lowercase().as_str() {
        "propagate" => Ok(NanPolicy::Propagate),
        "ignore" => Ok(NanPolicy::Ignore),
        "require_min_periods" => Ok(NanPolicy::RequireMinPeriods),
        _ => Err(PyErr::new::<PyValueError, _>(
            format!(
                "Invalid nan_policy: '{}'. Must be 'propagate', 'ignore', or 'require_min_periods'",
                s
            )
        )),
    }
}

// ======================
// Basic stats (1-D)
// ======================

#[pyfunction]
fn mean_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(mean_slice(a.as_slice()?))
}

#[pyfunction]
fn mean_skipna_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(mean_slice_skipna(a.as_slice()?))
}

#[pyfunction]
fn var_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(var_slice(a.as_slice()?))
}

#[pyfunction]
fn var_skipna_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(var_slice_skipna(a.as_slice()?))
}

#[pyfunction]
fn std_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(std_slice(a.as_slice()?))
}

#[pyfunction]
fn std_skipna_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(std_slice_skipna(a.as_slice()?))
}

// --- NaN-aware aliases (Python API compatibility) ---
// Python facade expects *_nan_np names in some versions.
#[pyfunction]
fn mean_nan_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    mean_skipna_np(a)
}
#[pyfunction]
fn var_nan_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    var_skipna_np(a)
}
#[pyfunction]
fn std_nan_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    std_skipna_np(a)
}

#[pyfunction]
fn percentile_np(a: PyReadonlyArray1<f64>, q: f64) -> PyResult<f64> {
    if !(0.0..=100.0).contains(&q) {
        return Err(PyValueError::new_err(
            "q must be in the range [0, 100] (numpy.percentile semantics)",
        ));
    }
    Ok(percentile_slice_k(a.as_slice()?, q))
}

#[pyfunction]
fn iqr_np(a: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, f64)> {
    Ok(iqr_slice_k(a.as_slice()?))
}

// Scalar IQR width (kept for convenience; avoids conflicting with tuple iqr_np)
#[pyfunction]
fn iqr_width_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let (q1, q3, iqr) = iqr_slice_k(a.as_slice()?);
    if q1.is_nan() || q3.is_nan() || iqr.is_nan() {
        Ok(f64::NAN)
    } else {
        Ok(iqr)
    }
}

#[pyfunction]
fn mad_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(mad_slice_k(a.as_slice()?))
}

/// Validate a trim/cut proportion: must be finite and in [0, 0.5).
#[inline]
fn check_proportion_to_cut(p: f64) -> PyResult<()> {
    if !p.is_finite() || p < 0.0 || p >= 0.5 {
        return Err(PyValueError::new_err(
            "proportion_to_cut must be in the range [0, 0.5)",
        ));
    }
    Ok(())
}

#[pyfunction]
fn trimmed_mean_np(
    a: PyReadonlyArray1<f64>,
    proportion_to_cut: f64,
) -> PyResult<f64> {
    check_proportion_to_cut(proportion_to_cut)?;
    Ok(trimmed_mean_slice_k(a.as_slice()?, proportion_to_cut))
}

// ======================
// Robust Statistics - Extended Functions
// ======================

#[pyfunction]
fn median_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(median_slice_k(a.as_slice()?))
}

#[pyfunction]
fn iqr_robust_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(iqr_robust_slice_k(a.as_slice()?))
}

#[pyfunction]
fn winsorized_mean_np(
    a: PyReadonlyArray1<f64>,
    lower_percentile: f64,
    upper_percentile: f64,
) -> PyResult<f64> {
    Ok(winsorized_mean_slice_k(a.as_slice()?, lower_percentile, upper_percentile))
}

#[pyfunction]
fn trimmed_std_np(a: PyReadonlyArray1<f64>, proportion_to_cut: f64) -> PyResult<f64> {
    Ok(trimmed_std_slice_k(a.as_slice()?, proportion_to_cut))
}

#[pyfunction]
fn mad_std_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(mad_std_slice_k(a.as_slice()?))
}

#[pyfunction(signature = (a, c=None))]
fn biweight_midvariance_np(a: PyReadonlyArray1<f64>, c: Option<f64>) -> PyResult<f64> {
    let c_val = c.unwrap_or(9.0);  // Default tuning constant
    Ok(biweight_midvariance_slice_k(a.as_slice()?, c_val))
}

#[pyfunction]
fn qn_scale_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(qn_scale_slice_k(a.as_slice()?))
}

#[pyfunction(signature = (a, k=None, max_iter=None))]
fn huber_location_np(
    a: PyReadonlyArray1<f64>,
    k: Option<f64>,
    max_iter: Option<usize>,
) -> PyResult<f64> {
    let k_val = k.unwrap_or(1.345);  // Default for 95% efficiency
    let max_iter_val = max_iter.unwrap_or(30);
    Ok(huber_location_slice_k(a.as_slice()?, k_val, max_iter_val))
}

// NaN-aware robust estimators
#[pyfunction]
fn median_skipna_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(median_slice_skipna_k(a.as_slice()?))
}

#[pyfunction]
fn mad_skipna_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(mad_slice_skipna_k(a.as_slice()?))
}

#[pyfunction]
fn trimmed_mean_skipna_np(a: PyReadonlyArray1<f64>, proportion_to_cut: f64) -> PyResult<f64> {
    check_proportion_to_cut(proportion_to_cut)?;
    Ok(trimmed_mean_slice_skipna_k(a.as_slice()?, proportion_to_cut))
}

#[pyfunction]
fn iqr_skipna_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(iqr_slice_skipna_k(a.as_slice()?))
}


#[pyfunction]
fn zscore_np<'py>(py: Python<'py>, a: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let n = xs.len();
    if n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }
    let m = mean_slice(xs);
    let s = std_slice(xs); // sample std (ddof=1)
    if !s.is_finite() || s == 0.0 {
        return Ok(PyArray1::from_vec_bound(py, vec![f64::NAN; n]));
    }
    let out: Vec<f64> = xs.iter().map(|&x| (x - m) / s).collect();
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
fn zscore_skipna_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let m = mean_slice_skipna(xs);
    let s = std_slice_skipna(xs);
    let out: Vec<f64> = xs
        .iter()
        .map(|&v| {
            if v.is_nan() {
                f64::NAN
            } else if s == 0.0 || s.is_nan() {
                f64::NAN
            } else {
                (v - m) / s
            }
        })
        .collect();
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
fn skew_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = a.as_slice()?;
    if xs.len() < 3 {
        return Ok(f64::NAN);
    }
    // Population skewness g1 = m3 / m2^(3/2) with CONSISTENT central moments
    // (both divided by n). Matches scipy.stats.skew (bias=True). The previous
    // code divided m3 by the SAMPLE std (ddof=1), mixing estimators, so it
    // matched neither scipy convention.
    let m = mean_slice(xs);
    let n = xs.len() as f64;
    let (mut m2, mut m3) = (0.0f64, 0.0f64);
    for &v in xs {
        let d = v - m;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= n;
    m3 /= n;
    if m2 <= 0.0 {
        return Ok(f64::NAN);
    }
    Ok(m3 / m2.powf(1.5))
}

#[pyfunction]
fn kurtosis_np(a: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = a.as_slice()?;
    if xs.len() < 4 {
        return Ok(f64::NAN);
    }
    // Excess (Fisher) kurtosis = m4 / m2^2 - 3 with CONSISTENT central moments
    // (both divided by n). Matches scipy.stats.kurtosis (bias=True). The previous
    // code divided m4 by the SAMPLE variance^2 (ddof=1), mixing estimators.
    let m = mean_slice(xs);
    let n = xs.len() as f64;
    let (mut m2, mut m4) = (0.0f64, 0.0f64);
    for &v in xs {
        let d = v - m;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= n;
    m4 /= n;
    if m2 <= 0.0 {
        return Ok(f64::NAN);
    }
    Ok(m4 / (m2 * m2) - 3.0)
}

// ======================
// Multi-D mean_axis (1D & 2D + skipna)
// ======================

#[pyfunction(signature = (x, axis, skipna=None))]
fn mean_axis_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<f64>,
    axis: isize,
    skipna: Option<bool>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let use_skipna = skipna.unwrap_or(false);
    let a: ArrayViewD<'_, f64> = x.as_array();
    let ndim = a.ndim();

    match ndim {
        1 => {
            if axis != 0 {
                return Err(PyValueError::new_err(
                    "mean_axis_np: for 1D input, axis must be 0",
                ));
            }
            let slice = a
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("mean_axis_np: 1D input must be contiguous"))?;
            let m = if use_skipna {
                mean_slice_skipna(slice)
            } else {
                mean_slice(slice)
            };
            Ok(PyArray1::from_vec_bound(py, vec![m]))
        }
        2 => {
            let axis_u = match axis {
                0 => 0usize,
                1 => 1usize,
                _ => {
                    return Err(PyValueError::new_err(
                        "mean_axis_np: for 2D input, axis must be 0 or 1",
                    ))
                }
            };

            let mut out: Vec<f64> = Vec::new();

            if axis_u == 0 {
                let n_cols = a.len_of(Axis(1));
                for j in 0..n_cols {
                    let col = a.index_axis(Axis(1), j);
                    let v: Vec<f64> = col.iter().copied().collect();
                    out.push(if use_skipna { mean_slice_skipna(&v) } else { mean_slice(&v) });
                }
            } else {
                let n_rows = a.len_of(Axis(0));
                for i in 0..n_rows {
                    let row = a.index_axis(Axis(0), i);
                    let v: Vec<f64> = row.iter().copied().collect();
                    out.push(if use_skipna { mean_slice_skipna(&v) } else { mean_slice(&v) });
                }
            }

            Ok(PyArray1::from_vec_bound(py, out))
        }
        _ => Err(PyValueError::new_err(
            "mean_axis_np currently supports only 1D or 2D arrays",
        )),
    }
}

// ======================
// N-D: mean over last axis (any ndim)
// ======================

#[pyfunction]
fn mean_over_last_axis_dyn_np<'py>(
    py: Python<'py>,
    arr: PyReadonlyArrayDyn<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let view = arr.as_array();
    let ndim = view.ndim();

    if ndim == 0 {
        let v = *view.iter().next().unwrap_or(&f64::NAN);
        return PyArray1::from_vec_bound(py, vec![v]);
    }

    let shape = view.shape();
    let last_dim = shape[ndim - 1];
    let batch_size: usize = shape[..ndim - 1].iter().product();

    let reshaped = view
        .to_owned()
        .into_shape((batch_size, last_dim))
        .expect("reshape failed in mean_over_last_axis_dyn_np");

    let mut out = Vec::with_capacity(batch_size);
    for row in reshaped.axis_iter(Axis(0)) {
        let sum: f64 = row.iter().copied().sum();
        let len = row.len() as f64;
        out.push(if len > 0.0 { sum / len } else { f64::NAN });
    }

    PyArray1::from_vec_bound(py, out)
}

// ======================
// Rolling stats (1-D) — truncated length (n-window+1) fast path
// ======================


/// Shared edge rule for strict rolling kernels (v0.3 contract):
/// window < 1 is a caller error; window > n yields an empty result.
#[inline]
fn check_window_ge1(window: usize) -> PyResult<()> {
    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    Ok(())
}

#[pyfunction]
fn rolling_mean_std_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    check_window_ge1(window)?;
    let xs = a.as_slice()?;
    let (means, stds) = rolling_mean_std_vec(xs, window);
    Ok((PyArray1::from_vec_bound(py, means), PyArray1::from_vec_bound(py, stds)))
}

#[pyfunction]
fn rolling_mean_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    check_window_ge1(window)?;
    let xs = a.as_slice()?;
    let (means, _stds) = rolling_mean_std_vec(xs, window);
    Ok(PyArray1::from_vec_bound(py, means))
}

#[pyfunction]
fn rolling_std_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    check_window_ge1(window)?;
    let xs = a.as_slice()?;
    let (_means, stds) = rolling_mean_std_vec(xs, window);
    Ok(PyArray1::from_vec_bound(py, stds))
}

#[pyfunction]
fn rolling_var_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    check_window_ge1(window)?;
    let xs = a.as_slice()?;

// single rolling pass (engine)
let (_means, stds) = rolling_mean_std_vec(xs, window);

// final linear pass (std -> var)
let vars = vars_from_stds(&stds);

Ok(PyArray1::from_vec_bound(py, vars))

}

#[pyfunction]
fn rolling_zscore_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    check_window_ge1(window)?;
    let xs = a.as_slice()?;

    // single rolling pass
    let (means, stds) = rolling_mean_std_vec(xs, window);

    // final linear pass
    let out = zscore_from_mean_std(xs, &means, &stds);

    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
fn rolling_mean_nan_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    check_window_ge1(window)?;
    let xs = a.as_slice()?;
    let n = xs.len();
    if n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let mut out = vec![f64::NAN; n];
    let mut sum = 0.0f64;
    let mut cnt = 0usize;

    for i in 0..n {
        let x_new = xs[i];
        if !x_new.is_nan() {
            sum += x_new;
            cnt += 1;
        }

        if i >= window {
            let x_old = xs[i - window];
            if !x_old.is_nan() {
                sum -= x_old;
                cnt -= 1;
            }
        }

        if cnt > 0 {
            out[i] = sum / (cnt as f64);
        }
    }

    Ok(PyArray1::from_vec_bound(py, out))
}


#[pyfunction]
fn rolling_std_nan_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    check_window_ge1(window)?;
    let xs = a.as_slice()?;
    let n = xs.len();
    if n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let mut out = vec![f64::NAN; n];
    // Shift by a finite constant near the data magnitude: variance is
    // translation-invariant and the shift prevents catastrophic cancellation
    // in `sumsq - sum^2/c` on large-offset data (see rolling::engine).
    let off = first_finite_offset(xs);
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut cnt = 0usize;

    for i in 0..n {
        let x_new = xs[i];
        if !x_new.is_nan() {
            let xs_ = x_new - off;
            sum += xs_;
            sumsq += xs_ * xs_;
            cnt += 1;
        }

        if i >= window {
            let x_old = xs[i - window];
            if !x_old.is_nan() {
                let xs_ = x_old - off;
                sum -= xs_;
                sumsq -= xs_ * xs_;
                cnt -= 1;
            }
        }

        if cnt >= 2 {
            let c = cnt as f64;
            let var = (sumsq - (sum * sum) / c) / (c - 1.0);
            out[i] = var.max(0.0).sqrt();
        }
    }

    Ok(PyArray1::from_vec_bound(py, out))
}


#[pyfunction]
fn rolling_zscore_nan_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    check_window_ge1(window)?;
    let xs = a.as_slice()?;
    let n = xs.len();
    if n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let mut out = vec![f64::NAN; n];
    // Offset shift for cancellation safety (see rolling_std_nan_np). The
    // z-score is invariant under a common shift of the data and the mean.
    let off = first_finite_offset(xs);
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut cnt = 0usize;

    for i in 0..n {
        let x_new = xs[i];
        if !x_new.is_nan() {
            let xs_ = x_new - off;
            sum += xs_;
            sumsq += xs_ * xs_;
            cnt += 1;
        }

        if i >= window {
            let x_old = xs[i - window];
            if !x_old.is_nan() {
                let xs_ = x_old - off;
                sum -= xs_;
                sumsq -= xs_ * xs_;
                cnt -= 1;
            }
        }

        let x = xs[i];
        if x.is_nan() {
            out[i] = f64::NAN;
            continue;
        }

        if cnt >= 2 {
            let c = cnt as f64;
            let mean_shifted = sum / c;
            let var = (sumsq - (sum * sum) / c) / (c - 1.0);
            let std = var.max(0.0).sqrt();
            out[i] = if std > 0.0 && std.is_finite() {
                ((x - off) - mean_shifted) / std
            } else {
                f64::NAN
            };
        } else {
            out[i] = f64::NAN;
        }
    }

    Ok(PyArray1::from_vec_bound(py, out))
}


#[pyfunction]
fn ewma_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    alpha: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
        return Err(PyValueError::new_err("alpha must satisfy 0 < alpha <= 1"));
    }
    let xs = a.as_slice()?;
    let n = xs.len();
    if n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }
    let mut out = Vec::with_capacity(n);
    let mut prev = xs[0];
    out.push(prev);
    let one_minus = 1.0 - alpha;
    for i in 1..n {
        let val = alpha * xs[i] + one_minus * prev;
        out.push(val);
        prev = val;
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

// ======================
// Rolling axis=0 (2D) — truncated (n-window+1, p) fast path
// ======================


// Cache-friendly axis-0 rolling mean+std (flat buffers; direct indexing; optional Rayon over columns).


#[pyfunction]
fn rolling_mean_std_axis0_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    window: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    check_window_ge1(window)?;
    let a = x.as_array();
    let (n_rows, n_cols) = a.dim();
    if window > n_rows {
        let empty = PyArray2::zeros_bound(py, (0, n_cols), false);
        return Ok((empty.clone(), empty));
    }
    let flat = a.as_slice().ok_or_else(|| PyValueError::new_err("array must be contiguous"))?;
    let out_rows = n_rows - window + 1;

    let (means, stds) = rolling_mean_std_axis0_vec(flat, n_rows, n_cols, window);
    let means2 = Array2::from_shape_vec((out_rows, n_cols), means).unwrap();
    let stds2 = Array2::from_shape_vec((out_rows, n_cols), stds).unwrap();
    Ok((means2.into_pyarray_bound(py), stds2.into_pyarray_bound(py)))
}

#[pyfunction]
fn rolling_mean_axis0_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    check_window_ge1(window)?;
    let a = x.as_array();
    let (n_rows, n_cols) = a.dim();
    if window > n_rows {
        return Ok(PyArray2::zeros_bound(py, (0, n_cols), false));
    }
    let flat = a.as_slice().ok_or_else(|| PyValueError::new_err("array must be contiguous"))?;
    let out_rows = n_rows - window + 1;

    let means = rolling_mean_axis0_vec(flat, n_rows, n_cols, window);
    let out2 = Array2::from_shape_vec((out_rows, n_cols), means).unwrap();
    Ok(out2.into_pyarray_bound(py))
}

#[pyfunction]
fn rolling_std_axis0_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    check_window_ge1(window)?;
    let a = x.as_array();
    let (n_rows, n_cols) = a.dim();
    if window > n_rows {
        return Ok(PyArray2::zeros_bound(py, (0, n_cols), false));
    }
    let flat = a.as_slice().ok_or_else(|| PyValueError::new_err("array must be contiguous"))?;
    let out_rows = n_rows - window + 1;

    let stds = rolling_std_axis0_vec(flat, n_rows, n_cols, window);
    let out2 = Array2::from_shape_vec((out_rows, n_cols), stds).unwrap();
    Ok(out2.into_pyarray_bound(py))
}

// ======================
// Outliers & scaling
// ======================

#[pyfunction(signature = (a, k=1.5))]
fn iqr_outliers_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    k: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let xs = a.as_slice()?;
    let (q1, q3, iqr) = iqr_slice_k(xs);
    if iqr.is_nan() {
        return Ok(PyArray1::from_vec_bound(py, vec![false; xs.len()]));
    }
    let low = q1 - k * iqr;
    let high = q3 + k * iqr;
    let mask: Vec<bool> = xs.iter().map(|&x| x < low || x > high).collect();
    Ok(PyArray1::from_vec_bound(py, mask))
}

#[pyfunction(signature = (a, threshold=3.0))]
fn zscore_outliers_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let xs = a.as_slice()?;
    if xs.is_empty() {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }
    let m = mean_slice(xs);
    let s = std_slice(xs);
    if s == 0.0 || s.is_nan() {
        return Ok(PyArray1::from_vec_bound(py, vec![false; xs.len()]));
    }
    let mask: Vec<bool> = xs.iter().map(|&x| ((x - m) / s).abs() > threshold).collect();
    Ok(PyArray1::from_vec_bound(py, mask))
}

#[pyfunction]
fn minmax_scale_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64, f64)> {
    let xs = a.as_slice()?;
    if xs.is_empty() {
        return Ok((PyArray1::from_vec_bound(py, vec![]), f64::NAN, f64::NAN));
    }
    let mut mn = xs[0];
    let mut mx = xs[0];
    for &x in xs.iter().skip(1) {
        if x < mn {
            mn = x;
        }
        if x > mx {
            mx = x;
        }
    }
    if mx == mn {
        return Ok((PyArray1::from_vec_bound(py, vec![0.0; xs.len()]), mn, mx));
    }
    let scale = mx - mn;
    let scaled: Vec<f64> = xs.iter().map(|&x| (x - mn) / scale).collect();
    Ok((PyArray1::from_vec_bound(py, scaled), mn, mx))
}

#[pyfunction]
fn robust_scale_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    scale_factor: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64, f64)> {
    let xs = a.as_slice()?;
    if xs.is_empty() {
        return Ok((PyArray1::from_vec_bound(py, vec![]), f64::NAN, f64::NAN));
    }
    // NaN in input -> NaN out (numpy semantics), never a crash under panic="abort".
    if crate::util::any_nan(xs) {
        let nans = vec![f64::NAN; xs.len()];
        return Ok((PyArray1::from_vec_bound(py, nans), f64::NAN, f64::NAN));
    }
    let mad = mad_slice(xs);
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let med = if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) };

    let denom = if mad == 0.0 { 1e-12 } else { mad * scale_factor };
    let scaled: Vec<f64> = xs.iter().map(|&x| (x - med) / denom).collect();
    Ok((PyArray1::from_vec_bound(py, scaled), med, mad))
}

// Quantile-based winsorize. v0.3 contract: quantile fractions in [0, 1] only
// (0.05, 0.95). Percent-style arguments (5, 95) are rejected — the previous
// dual-unit auto-detection made 1.0 ambiguous.
#[pyfunction]
fn winsorize_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    lower_q: f64,
    upper_q: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if !(0.0..=1.0).contains(&lower_q) || !(0.0..=1.0).contains(&upper_q) || lower_q >= upper_q {
        return Err(PyValueError::new_err(
            "winsorize quantiles must satisfy 0 <= lower_q < upper_q <= 1",
        ));
    }
    let xs = a.as_slice()?;
    if xs.is_empty() {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    // Delegate to the kernel (single source of truth)
    let out = winsorize_vec_k(xs, lower_q, upper_q);
    Ok(PyArray1::from_vec_bound(py, out))
}
// Clip-based winsorize (explicit bounds)
#[pyfunction]
fn winsorize_clip_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    lower: f64,
    upper: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    if xs.is_empty() {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    // If caller swapped them, fix it deterministically
    let (lo, hi) = if lower <= upper { (lower, upper) } else { (upper, lower) };

    let out: Vec<f64> = xs
        .iter()
        .map(|&v| if v < lo { lo } else if v > hi { hi } else { v })
        .collect();

    Ok(PyArray1::from_vec_bound(py, out))
}



// ======================
// diff / cum / ecdf / bins / sign helpers
// ======================

#[pyfunction]
#[pyo3(signature = (a, periods=1))]
fn diff_np<'py>(py: Python<'py>, a: PyReadonlyArray1<f64>, periods: isize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let n = xs.len();
    if n == 0 || periods == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![0.0; n]));
    }
    let p = periods.abs() as usize;
    if p >= n {
        return Ok(PyArray1::from_vec_bound(py, vec![f64::NAN; n]));
    }

    let mut out = vec![f64::NAN; n];
    if periods > 0 {
        for i in p..n {
            out[i] = xs[i] - xs[i - p];
        }
    } else {
        for i in 0..(n - p) {
            out[i] = xs[i] - xs[i + p];
        }
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
fn pct_change_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    periods: isize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let n = xs.len();
    if n == 0 || periods == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![f64::NAN; n]));
    }

    let p = periods.abs() as usize;
    if p >= n {
        return Ok(PyArray1::from_vec_bound(py, vec![f64::NAN; n]));
    }

    let mut out = vec![f64::NAN; n];
    if periods > 0 {
        for i in p..n {
            let base = xs[i - p];
            out[i] = if base == 0.0 { f64::NAN } else { (xs[i] - base) / base };
        }
    } else {
        for i in 0..(n - p) {
            let base = xs[i + p];
            out[i] = if base == 0.0 { f64::NAN } else { (xs[i] - base) / base };
        }
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
fn cumsum_np<'py>(py: Python<'py>, a: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let mut out = Vec::with_capacity(xs.len());
    let mut s = 0.0;
    for &x in xs {
        s += x;
        out.push(s);
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
fn cummean_np<'py>(py: Python<'py>, a: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = a.as_slice()?;
    let mut out = Vec::with_capacity(xs.len());
    let mut s = 0.0;
    for (i, &x) in xs.iter().enumerate() {
        s += x;
        out.push(s / ((i + 1) as f64));
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
fn ecdf_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let xs = a.as_slice()?;
    if xs.is_empty() {
        return Ok((PyArray1::from_vec_bound(py, vec![]), PyArray1::from_vec_bound(py, vec![])));
    }
    let mut vals = xs.to_vec();
    // total_cmp is a total order over f64 (NaN sorts last), so this never panics.
    vals.sort_by(|a, b| a.total_cmp(b));
    let n = vals.len();
    let cdf: Vec<f64> = (0..n).map(|i| (i + 1) as f64 / (n as f64)).collect();
    Ok((PyArray1::from_vec_bound(py, vals), PyArray1::from_vec_bound(py, cdf)))
}

#[pyfunction]
fn quantile_bins_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    n_bins: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if n_bins == 0 {
        return Err(PyValueError::new_err("n_bins must be >= 1"));
    }
    let xs = a.as_slice()?;
    let n = xs.len();
    if n == 0 {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let mut pairs: Vec<(f64, usize)> = xs.iter().copied().zip(0..n).collect();
    // total_cmp is a total order over f64 (NaN sorts last), so this never panics.
    pairs.sort_by(|(v1, _), (v2, _)| v1.total_cmp(v2));

    let mut bins = vec![-1_i64; n];
    let mut start = 0usize;
    for b in 0..n_bins {
        let end = if b == n_bins - 1 { n } else { ((b + 1) * n) / n_bins };
        for i in start..end {
            let (_, idx) = pairs[i];
            bins[idx] = b as i64;
        }
        start = end;
    }

    Ok(PyArray1::from_vec_bound(py, bins))
}

#[pyfunction]
fn sign_mask_np<'py>(py: Python<'py>, a: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let xs = a.as_slice()?;
    let out: Vec<i8> = xs
        .iter()
        .map(|&x| if x > 0.0 { 1 } else if x < 0.0 { -1 } else { 0 })
        .collect();
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
fn demean_with_signs_np<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i8>>)> {
    let xs = a.as_slice()?;
    let m = mean_slice(xs);
    let mut demeaned = Vec::with_capacity(xs.len());
    let mut signs = Vec::with_capacity(xs.len());
    for &x in xs {
        let d = x - m;
        demeaned.push(d);
        signs.push(if d > 0.0 { 1 } else if d < 0.0 { -1 } else { 0 });
    }
    Ok((PyArray1::from_vec_bound(py, demeaned), PyArray1::from_vec_bound(py, signs)))
}

// ======================
// Covariance / correlation (non-NaN)
// ======================

/// First finite element (or 0.0). Covariance/correlation are invariant under a
/// constant shift of each series, so shifting by a value near the data
/// magnitude keeps the one-pass accumulators from catastrophically cancelling
/// on large-offset data (same pattern as kernels::rolling::covcorr).
#[inline]
fn first_finite_offset(v: &[f64]) -> f64 {
    for &x in v {
        if x.is_finite() {
            return x;
        }
    }
    0.0
}

fn cov_impl(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n <= 1 {
        return f64::NAN;
    }
    let xs = &xs[..n];
    let ys = &ys[..n];
    let (ox, oy) = (first_finite_offset(xs), first_finite_offset(ys));

    // Strict: if any NaN is present in either series, return NaN (matches prior behavior).
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xy = 0.0f64;

    for i in 0..n {
        let x = xs[i];
        let y = ys[i];
        if x.is_nan() || y.is_nan() {
            return f64::NAN;
        }
        let x = x - ox;
        let y = y - oy;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
    }

    let c = n as f64;
    (sum_xy - (sum_x * sum_y) / c) / ((n - 1) as f64)
}

#[pyfunction]
fn cov_np(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err(
            "x and y must have the same length",
        ));
    }
    Ok(cov_impl(xs, ys))
}

#[pyfunction]
fn corr_np(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err(
            "x and y must have the same length",
        ));
    }
    let n = xs.len().min(ys.len());
    if n <= 1 {
        return Ok(f64::NAN);
    }
    let (ox, oy) = (first_finite_offset(xs), first_finite_offset(ys));

    // Strict: if any NaN is present in either series, return NaN (matches prior behavior).
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xx = 0.0f64;
    let mut sum_yy = 0.0f64;
    let mut sum_xy = 0.0f64;

    for i in 0..n {
        let x = xs[i];
        let y = ys[i];
        if x.is_nan() || y.is_nan() {
            return Ok(f64::NAN);
        }
        let x = x - ox;
        let y = y - oy;
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_yy += y * y;
        sum_xy += x * y;
    }

    let c = n as f64;
    let denom = c - 1.0;

    let cov = (sum_xy - (sum_x * sum_y) / c) / denom;
    let varx = (sum_xx - (sum_x * sum_x) / c) / denom;
    let vary = (sum_yy - (sum_y * sum_y) / c) / denom;

    if varx <= 0.0 || vary <= 0.0 || !varx.is_finite() || !vary.is_finite() || !cov.is_finite() {
        return Ok(f64::NAN);
    }
    Ok((cov / (varx * vary).sqrt()).clamp(-1.0, 1.0))
}

// ======================
// Matrix helpers (accept float32/float64; keep kernels on &[f64])
// ======================
#[inline]
fn extract_mat_f64<'py>(x: &Bound<'py, PyAny>) -> PyResult<(Vec<f64>, usize, usize)> {
    // Fast path: float64 (copy from contiguous or materialize non-contiguous)
    if let Ok(x64) = x.extract::<PyReadonlyArray2<f64>>() {
        let arr = x64.as_array();
        let (n_rows, n_cols) = (arr.shape()[0], arr.shape()[1]);

        // C-contiguous input: straight copy. Anything else (Fortran-ordered,
        // sliced/strided views): collect in LOGICAL row-major order via iter().
        // The previous fallback used to_owned().as_slice().expect(..), but
        // ndarray's to_owned() preserves Fortran layout for F-contiguous
        // sources, so as_slice() returned None and the expect() aborted the
        // interpreter (panic="abort") on np.asfortranarray input.
        let xs: Vec<f64> = match arr.as_slice() {
            Some(s) => s.to_vec(),
            None => arr.iter().copied().collect(),
        };

        return Ok((xs, n_rows, n_cols));
    }

    // Accept float32 by upcasting to f64
    if let Ok(x32) = x.extract::<PyReadonlyArray2<f32>>() {
        let arr = x32.as_array();
        let (n_rows, n_cols) = (arr.shape()[0], arr.shape()[1]);

        let mut v = Vec::<f64>::with_capacity(n_rows * n_cols);
        for &val in arr.iter() {
            v.push(val as f64);
        }
        return Ok((v, n_rows, n_cols));
    }

    Err(PyTypeError::new_err(
        "expected a 2D NumPy array of dtype float32 or float64",
    ))
}

#[pyfunction]
pub fn cov_matrix_np<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (xs_vec, n_rows, n_cols) = extract_mat_f64(x)?;
    let xs = xs_vec.as_slice();

    if n_rows < 2 || n_cols == 0 {
        let out2 = numpy::ndarray::Array2::<f64>::from_elem((n_cols, n_cols), f64::NAN);
        return Ok(out2.into_pyarray_bound(py));
    }

    let mut out = vec![0.0f64; n_cols * n_cols];
    cov_matrix_out(xs, n_rows, n_cols, &mut out);

    let out2 = Array2::from_shape_vec((n_cols, n_cols), out)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("cov_matrix_np: from_shape_vec failed"))?;
    Ok(out2.into_pyarray_bound(py))
}

#[pyfunction]
pub fn cov_matrix_bias_np<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (xs_vec, n_rows, n_cols) = extract_mat_f64(x)?;
    let xs = xs_vec.as_slice();

    if n_rows == 0 || n_cols == 0 {
        let out2 = numpy::ndarray::Array2::<f64>::zeros((n_cols, n_cols));
        return Ok(out2.into_pyarray_bound(py));
    }

    let mut out = vec![0.0f64; n_cols * n_cols];
    cov_matrix_bias_out(xs, n_rows, n_cols, &mut out);

    let out2 = Array2::from_shape_vec((n_cols, n_cols), out)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("cov_matrix_bias_np: from_shape_vec failed"))?;
    Ok(out2.into_pyarray_bound(py))
}

#[pyfunction]
pub fn cov_matrix_centered_np<'py>(
    py: Python<'py>,
    x_centered: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (xs_vec, n_rows, n_cols) = extract_mat_f64(x_centered)?;
    let xs = xs_vec.as_slice();

    if n_rows < 2 || n_cols == 0 {
        let out2 = numpy::ndarray::Array2::<f64>::zeros((n_cols, n_cols));
        return Ok(out2.into_pyarray_bound(py));
    }

    let mut out = vec![0.0f64; n_cols * n_cols];
    cov_matrix_centered_out(xs, n_rows, n_cols, &mut out);

    let out2 = Array2::from_shape_vec((n_cols, n_cols), out)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("cov_matrix_centered_np: from_shape_vec failed"))?;
    Ok(out2.into_pyarray_bound(py))
}

#[pyfunction]
pub fn cov_matrix_skipna_np<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (xs_vec, n_rows, n_cols) = extract_mat_f64(x)?;
    let xs = xs_vec.as_slice();

    if n_rows < 2 || n_cols == 0 {
        let out2 = numpy::ndarray::Array2::<f64>::zeros((n_cols, n_cols));
        return Ok(out2.into_pyarray_bound(py));
    }

    let mut out = vec![0.0f64; n_cols * n_cols];
    cov_matrix_skipna_out(xs, n_rows, n_cols, &mut out);

    let out2 = Array2::from_shape_vec((n_cols, n_cols), out)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("cov_matrix_skipna_np: from_shape_vec failed"))?;
    Ok(out2.into_pyarray_bound(py))
}

#[pyfunction]
pub fn xtx_matrix_np<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (xs_vec, n_rows, n_cols) = extract_mat_f64(x)?;
    let xs = xs_vec.as_slice();

    if n_rows == 0 || n_cols == 0 {
        let out2 = numpy::ndarray::Array2::<f64>::zeros((n_cols, n_cols));
        return Ok(out2.into_pyarray_bound(py));
    }

    let mut out = vec![0.0f64; n_cols * n_cols];
    xtx_matrix_out(xs, n_rows, n_cols, &mut out);

    let out2 = Array2::from_shape_vec((n_cols, n_cols), out)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("xtx_matrix_np: from_shape_vec failed"))?;
    Ok(out2.into_pyarray_bound(py))
}

#[pyfunction]
pub fn xxt_matrix_np<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (xs_vec, n_rows, n_cols) = extract_mat_f64(x)?;
    let xs = xs_vec.as_slice();

    if n_rows == 0 || n_cols == 0 {
        let out2 = numpy::ndarray::Array2::<f64>::zeros((n_rows, n_rows));
        return Ok(out2.into_pyarray_bound(py));
    }

    // WARNING: n_rows*n_rows can be huge. We leave sizing responsibility to caller.
    let mut out = vec![0.0f64; n_rows * n_rows];
    xxt_matrix_out(xs, n_rows, n_cols, &mut out);

    let out2 = Array2::from_shape_vec((n_rows, n_rows), out)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("xxt_matrix_np: from_shape_vec failed"))?;
    Ok(out2.into_pyarray_bound(py))
}

#[pyfunction]
pub fn pairwise_euclidean_cols_np<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (xs_vec, n_rows, n_cols) = extract_mat_f64(x)?;
    let xs = xs_vec.as_slice();

    if n_cols == 0 {
        let out2 = numpy::ndarray::Array2::<f64>::zeros((0, 0));
        return Ok(out2.into_pyarray_bound(py));
    }

    let mut out = vec![0.0f64; n_cols * n_cols];
    pairwise_euclidean_cols_out(xs, n_rows, n_cols, &mut out);

    let out2 = Array2::from_shape_vec((n_cols, n_cols), out)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("pairwise_euclidean_cols_np: from_shape_vec failed"))?;
    Ok(out2.into_pyarray_bound(py))
}

#[pyfunction]
pub fn pairwise_cosine_cols_np<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (xs_vec, n_rows, n_cols) = extract_mat_f64(x)?;
    let xs = xs_vec.as_slice();

    if n_cols == 0 {
        let out2 = numpy::ndarray::Array2::<f64>::zeros((0, 0));
        return Ok(out2.into_pyarray_bound(py));
    }

    let mut out = vec![0.0f64; n_cols * n_cols];
    pairwise_cosine_cols_out(xs, n_rows, n_cols, &mut out);

    let out2 = Array2::from_shape_vec((n_cols, n_cols), out)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("pairwise_cosine_cols_np: from_shape_vec failed"))?;
    Ok(out2.into_pyarray_bound(py))
}

#[pyfunction]
pub fn corr_matrix_np<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (xs_vec, n_rows, n_cols) = extract_mat_f64(a)?;
    let xs = xs_vec.as_slice();

    if n_rows < 2 || n_cols == 0 {
        let out2 = Array2::<f64>::zeros((n_cols, n_cols));
        return Ok(out2.into_pyarray_bound(py));
    }

    let mut out = vec![0.0f64; n_cols * n_cols];
    corr_matrix_out_k(xs, n_rows, n_cols, &mut out);

    let out2 = Array2::from_shape_vec((n_cols, n_cols), out)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("corr_matrix_np: from_shape_vec failed"))?;
    Ok(out2.into_pyarray_bound(py))
}

#[pyfunction]
pub fn corr_matrix_skipna_np<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (xs_vec, n_rows, n_cols) = extract_mat_f64(a)?;
    let xs = xs_vec.as_slice();

    if n_rows < 2 || n_cols == 0 {
        let out2 = Array2::<f64>::zeros((n_cols, n_cols));
        return Ok(out2.into_pyarray_bound(py));
    }

    let mut out = vec![0.0f64; n_cols * n_cols];
    corr_matrix_skipna_out_k(xs, n_rows, n_cols, &mut out);

    let out2 = Array2::from_shape_vec((n_cols, n_cols), out)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("corr_matrix_skipna_np: from_shape_vec failed"))?;
    Ok(out2.into_pyarray_bound(py))
}

#[pyfunction]
pub fn corr_distance_np<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (xs_vec, n_rows, n_cols) = extract_mat_f64(a)?;
    let xs = xs_vec.as_slice();

    if n_rows < 2 || n_cols == 0 {
        let out2 = Array2::<f64>::zeros((n_cols, n_cols));
        return Ok(out2.into_pyarray_bound(py));
    }

    let mut out = vec![0.0f64; n_cols * n_cols];
    corr_distance_out_k(xs, n_rows, n_cols, &mut out);

    let out2 = Array2::from_shape_vec((n_cols, n_cols), out)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("corr_distance_np: from_shape_vec failed"))?;
    Ok(out2.into_pyarray_bound(py))
}

#[pyfunction]
pub fn diag_np<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (xs_vec, n_rows, n_cols) = extract_mat_f64(a)?;
    if n_rows != n_cols {
        return Err(PyValueError::new_err("diag_np expects a square 2D array"));
    }
    let xs = xs_vec.as_slice();

    let n = n_rows;
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        out[i] = xs[i * n + i];
    }

    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
pub fn trace_np(a: &Bound<'_, PyAny>) -> PyResult<f64> {
    let (xs_vec, n_rows, n_cols) = extract_mat_f64(a)?;
    if n_rows != n_cols {
        return Err(PyValueError::new_err("trace_np expects a square 2D array"));
    }
    let xs = xs_vec.as_slice();

    let n = n_rows;
    let mut sum = 0.0_f64;
    for i in 0..n {
        sum += xs[i * n + i];
    }
    Ok(sum)
}

#[pyfunction]
pub fn is_symmetric_np(a: &Bound<'_, PyAny>, tol: f64) -> PyResult<bool> {
    let (xs_vec, n_rows, n_cols) = extract_mat_f64(a)?;
    if n_rows != n_cols {
        // By definition a non-square matrix can't be symmetric.
        return Ok(false);
    }
    let xs = xs_vec.as_slice();
    let n = n_rows;

    for i in 0..n {
        for j in (i + 1)..n {
            let a_ij = xs[i * n + j];
            let a_ji = xs[j * n + i];

            if (a_ij - a_ji).abs() > tol {
                return Ok(false);
            }
        }
    }
    Ok(true)
}


#[pyfunction]
fn rolling_cov_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;

    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have the same length"));
    }
    check_window_ge1(window)?;
    if window > xs.len() {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let out = rolling_cov_vec(xs, ys, window);
    Ok(PyArray1::from_vec_bound(py, out))
}


#[pyfunction]
fn rolling_corr_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;

    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y must have the same length"));
    }
    let n = xs.len();
    check_window_ge1(window)?;
    if window > n {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }
    let mut out = Vec::with_capacity(n - window + 1);

    // Offset shift (correlation is translation-invariant) to avoid
    // catastrophic cancellation on large-offset data.
    let (ox, oy) = (first_finite_offset(xs), first_finite_offset(ys));

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut sum_xy = 0.0;

    for i in 0..window {
        let xi = xs[i] - ox;
        let yi = ys[i] - oy;
        sum_x += xi;
        sum_y += yi;
        sum_x2 += xi * xi;
        sum_y2 += yi * yi;
        sum_xy += xi * yi;
    }

    for i in (window - 1)..n {
        if i > window - 1 {
            let xi_new = xs[i] - ox;
            let yi_new = ys[i] - oy;
            let xi_old = xs[i - window] - ox;
            let yi_old = ys[i - window] - oy;

            sum_x += xi_new - xi_old;
            sum_y += yi_new - yi_old;
            sum_x2 += xi_new * xi_new - xi_old * xi_old;
            sum_y2 += yi_new * yi_new - yi_old * yi_old;
            sum_xy += xi_new * yi_new - xi_old * yi_old;
        }

        let w = window as f64;
        let mx = sum_x / w;
        let my = sum_y / w;
        let var_x = (sum_x2 - w * mx * mx) / ((window - 1) as f64);
        let var_y = (sum_y2 - w * my * my) / ((window - 1) as f64);
        let cov = (sum_xy - w * mx * my) / ((window - 1) as f64);

        let denom = (var_x.max(0.0).sqrt()) * (var_y.max(0.0).sqrt());
        out.push(if denom == 0.0 || denom.is_nan() {
            f64::NAN
        } else {
            (cov / denom).clamp(-1.0, 1.0)
        });
    }

    Ok(PyArray1::from_vec_bound(py, out))
}


// ======================
// Welford (single-pass, NaN-skipping)
// ======================

#[pyfunction]
pub fn welford_np(a: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, usize)> {
    let xs = a.as_slice()?;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut n = 0usize;

    for &x in xs {
        if x.is_nan() {
            continue;
        }
        n += 1;
        let delta = x - mean;
        mean += delta / n as f64;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }

    if n == 0 {
        // No valid observations: the mean is undefined, not 0.0.
        return Ok((f64::NAN, f64::NAN, 0));
    }
    if n < 2 {
        return Ok((mean, f64::NAN, n));
    }

    let var = m2 / (n as f64 - 1.0);
    Ok((mean, var, n))
}

// ======================
// NaN-aware covariance/correlation (pairwise deletion, ddof=1)
// ======================

#[pyfunction]
pub fn cov_nan_np(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("length mismatch"));
    }

    // Skipna: drop pairs where either value is NaN.
    let (ox, oy) = (first_finite_offset(xs), first_finite_offset(ys));
    let mut count = 0usize;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xy = 0.0f64;

    for i in 0..xs.len() {
        let xi = xs[i];
        let yi = ys[i];
        if xi.is_nan() || yi.is_nan() {
            continue;
        }
        let xi = xi - ox;
        let yi = yi - oy;
        count += 1;
        sum_x += xi;
        sum_y += yi;
        sum_xy += xi * yi;
    }

    if count < 2 {
        return Ok(f64::NAN);
    }

    let c = count as f64;
    let cov = sum_xy - (sum_x * sum_y) / c;
    Ok(cov / (c - 1.0))
}

#[pyfunction]
pub fn corr_nan_np(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("length mismatch"));
    }

    // Skipna: drop pairs where either value is NaN.
    let (ox, oy) = (first_finite_offset(xs), first_finite_offset(ys));
    let mut count = 0usize;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xx = 0.0f64;
    let mut sum_yy = 0.0f64;
    let mut sum_xy = 0.0f64;

    for i in 0..xs.len() {
        let x = xs[i];
        let y = ys[i];
        if x.is_nan() || y.is_nan() {
            continue;
        }
        let x = x - ox;
        let y = y - oy;
        count += 1;
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_yy += y * y;
        sum_xy += x * y;
    }

    if count < 2 {
        return Ok(f64::NAN);
    }

    let c = count as f64;
    let denom = c - 1.0;

    let cov = (sum_xy - (sum_x * sum_y) / c) / denom;
    let varx = (sum_xx - (sum_x * sum_x) / c) / denom;
    let vary = (sum_yy - (sum_y * sum_y) / c) / denom;

    if varx <= 0.0 || vary <= 0.0 || !varx.is_finite() || !vary.is_finite() || !cov.is_finite() {
        return Ok(f64::NAN);
    }
    Ok((cov / (varx * vary).sqrt()).clamp(-1.0, 1.0))
}
// ======================================================================================
// Rolling skipna pair-state (used for rolling cov/corr/beta/linreg)
// NOTE: This is deliberately pure Rust / slice-based logic. Keep PyO3 wrappers thin.
// ======================================================================================

#[derive(Default, Copy, Clone)]
struct RollingPairState {
    count: usize,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
}

impl RollingPairState {
    #[inline(always)]
    fn add_pair(&mut self, x: f64, y: f64) {
        if x.is_nan() || y.is_nan() {
            return;
        }
        self.count += 1;
        self.sum_x += x;
        self.sum_y += y;
        self.sum_xx += x * x;
        self.sum_yy += y * y;
        self.sum_xy += x * y;
    }

    #[inline(always)]
    fn remove_pair(&mut self, x: f64, y: f64) {
        if x.is_nan() || y.is_nan() {
            return;
        }
        self.count -= 1;
        self.sum_x -= x;
        self.sum_y -= y;
        self.sum_xx -= x * x;
        self.sum_yy -= y * y;
        self.sum_xy -= x * y;
    }

    #[inline(always)]
    fn cov(&self) -> f64 {
        if self.count < 2 {
            return f64::NAN;
        }
        let c = self.count as f64;
        (self.sum_xy - (self.sum_x * self.sum_y) / c) / (c - 1.0)
    }

    #[inline(always)]
    fn var_x(&self) -> f64 {
        if self.count < 2 {
            return f64::NAN;
        }
        let c = self.count as f64;
        (self.sum_xx - (self.sum_x * self.sum_x) / c) / (c - 1.0)
    }

    #[inline(always)]
    fn corr(&self) -> f64 {
        if self.count < 2 {
            return f64::NAN;
        }
        let c = self.count as f64;
        let denom = c - 1.0;

        let cov = (self.sum_xy - (self.sum_x * self.sum_y) / c) / denom;
        let varx = (self.sum_xx - (self.sum_x * self.sum_x) / c) / denom;
        let vary = (self.sum_yy - (self.sum_y * self.sum_y) / c) / denom;

        if varx <= 0.0 || vary <= 0.0 || !varx.is_finite() || !vary.is_finite() || !cov.is_finite() {
            return f64::NAN;
        }
        (cov / (varx * vary).sqrt()).clamp(-1.0, 1.0)
    }
}


#[pyfunction]
pub fn rolling_cov_nan_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    let n = xs.len();

    if ys.len() != n {
        return Err(PyValueError::new_err("length mismatch"));
    }
    check_window_ge1(window)?;
    if window > n {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let out_len = n - window + 1;
    let mut out = vec![f64::NAN; out_len];

    // init window. NaN pairs are dropped by add_pair; requiring a FULL window
    // of valid pairs below matches pandas rolling(...).cov's default
    // min_periods=window (NaN whenever any pair in the window is missing).
    // Values are shifted by a finite offset for cancellation safety
    // (covariance is translation-invariant).
    let (ox, oy) = (first_finite_offset(xs), first_finite_offset(ys));
    let mut st = RollingPairState::default();
    for i in 0..window {
        st.add_pair(xs[i] - ox, ys[i] - oy);
    }
    out[0] = if st.count == window { st.cov() } else { f64::NAN };

    // slide
    for i in window..n {
        st.remove_pair(xs[i - window] - ox, ys[i - window] - oy);
        st.add_pair(xs[i] - ox, ys[i] - oy);
        out[i - window + 1] = if st.count == window { st.cov() } else { f64::NAN };
    }

    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
pub fn rolling_corr_nan_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    let n = xs.len();

    if ys.len() != n {
        return Err(PyValueError::new_err("length mismatch"));
    }
    check_window_ge1(window)?;
    if window > n {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let out_len = n - window + 1;
    let mut out = vec![f64::NAN; out_len];

    // init window. Full-window requirement mirrors pandas rolling(...).corr
    // default min_periods=window (see rolling_cov_nan_np). Values are shifted
    // by a finite offset for cancellation safety.
    let (ox, oy) = (first_finite_offset(xs), first_finite_offset(ys));
    let mut st = RollingPairState::default();
    for i in 0..window {
        st.add_pair(xs[i] - ox, ys[i] - oy);
    }
    out[0] = if st.count == window { st.corr() } else { f64::NAN };

    // slide
    for i in window..n {
        st.remove_pair(xs[i - window] - ox, ys[i - window] - oy);
        st.add_pair(xs[i] - ox, ys[i] - oy);
        out[i - window + 1] = if st.count == window { st.corr() } else { f64::NAN };
    }

    Ok(PyArray1::from_vec_bound(py, out))
}

// --------------------
// Clean exports (preferred) + new rolling linear-model primitives
// These are thin wrappers over the existing implementations to keep Python API stable.
// --------------------

#[pyfunction]
pub fn cov_skipna(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    cov_nan_np(x, y)
}

#[pyfunction]
pub fn corr_skipna(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
    corr_nan_np(x, y)
}

#[pyfunction]
pub fn rolling_cov_skipna<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    rolling_cov_nan_np(py, x, y, window)
}

#[pyfunction]
pub fn rolling_corr_skipna<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    rolling_corr_nan_np(py, x, y, window)
}

#[pyfunction]
pub fn rolling_beta_skipna<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    let n = xs.len();

    if ys.len() != n {
        return Err(PyValueError::new_err("length mismatch"));
    }
    check_window_ge1(window)?;
    if window > n {
        return Ok(PyArray1::from_vec_bound(py, vec![]));
    }

    let out_len = n - window + 1;
    let mut out = vec![f64::NAN; out_len];

    // Slope (cov/var) is translation-invariant, so shifted accumulation is
    // exact and avoids catastrophic cancellation on large-offset data.
    let (ox, oy) = (first_finite_offset(xs), first_finite_offset(ys));
    let mut st = RollingPairState::default();
    for i in 0..window {
        st.add_pair(xs[i] - ox, ys[i] - oy);
    }
    let varx0 = st.var_x();
    out[0] = if varx0 <= 0.0 || !varx0.is_finite() { f64::NAN } else { st.cov() / varx0 };

    for i in window..n {
        st.remove_pair(xs[i - window] - ox, ys[i - window] - oy);
        st.add_pair(xs[i] - ox, ys[i] - oy);

        let varx = st.var_x();
        out[i - window + 1] = if varx <= 0.0 || !varx.is_finite() { f64::NAN } else { st.cov() / varx };
    }

    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
pub fn rolling_linreg_skipna<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;
    let n = xs.len();

    if ys.len() != n {
        return Err(PyValueError::new_err("length mismatch"));
    }
    check_window_ge1(window)?;
    if window > n {
        return Ok((
            PyArray1::from_vec_bound(py, vec![]),
            PyArray1::from_vec_bound(py, vec![]),
        ));
    }

    let out_len = n - window + 1;
    let mut slope = vec![f64::NAN; out_len];
    let mut intercept = vec![f64::NAN; out_len];

    // Shifted accumulation for cancellation safety. The slope is
    // translation-invariant; the intercept of the shifted fit is unshifted via
    // a = a_shifted + oy - b*ox.
    let (ox, oy) = (first_finite_offset(xs), first_finite_offset(ys));
    let mut st = RollingPairState::default();
    for i in 0..window {
        st.add_pair(xs[i] - ox, ys[i] - oy);
    }

    // window 0
    if st.count >= 2 {
        let c = st.count as f64;
        let denom = c - 1.0;
        let sxx = (st.sum_xx - (st.sum_x * st.sum_x) / c) / denom; // var_x
        if sxx > 0.0 && sxx.is_finite() {
            let sxy = (st.sum_xy - (st.sum_x * st.sum_y) / c) / denom; // cov_xy
            let b = sxy / sxx;
            let a = (st.sum_y / c) - b * (st.sum_x / c) + oy - b * ox;
            slope[0] = b;
            intercept[0] = a;
        }
    }

    // slide
    for i in window..n {
        st.remove_pair(xs[i - window] - ox, ys[i - window] - oy);
        st.add_pair(xs[i] - ox, ys[i] - oy);

        let o = i - window + 1;
        if st.count < 2 {
            continue;
        }

        let c = st.count as f64;
        let denom = c - 1.0;
        let sxx = (st.sum_xx - (st.sum_x * st.sum_x) / c) / denom;
        if sxx <= 0.0 || !sxx.is_finite() {
            continue;
        }
        let sxy = (st.sum_xy - (st.sum_x * st.sum_y) / c) / denom;
        let b = sxy / sxx;
        let a = (st.sum_y / c) - b * (st.sum_x / c) + oy - b * ox;

        slope[o] = b;
        intercept[o] = a;
    }

    Ok((
        PyArray1::from_vec_bound(py, slope),
        PyArray1::from_vec_bound(py, intercept),
    ))
}


#[pyfunction]
fn pad_nan_np<'py>(py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(PyArray1::from_vec_bound(py, vec![f64::NAN; n]))
}

// ======================
// Gaussian KDE
// ======================

/// Gaussian kernel density estimate evaluated on a regular grid.
///
/// Bandwidth defaults to Scott's rule, h = n^(-1/5) * std(x, ddof=1), matching
/// scipy.stats.gaussian_kde's default for 1-D data. The grid spans
/// [min - 3h, max + 3h] with `points` samples.
///
/// Python signature:
///     kde_gaussian(x, points=200, bandwidth=None) -> (grid, density)
#[pyfunction]
#[pyo3(signature = (x, points=200, bandwidth=None))]
fn kde_gaussian_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    points: usize,
    bandwidth: Option<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let xs = x.as_slice()?;
    let n = xs.len();

    if points == 0 {
        return Err(PyValueError::new_err("points must be >= 1"));
    }
    if n == 0 {
        return Err(PyValueError::new_err("x must have at least 1 element"));
    }
    if xs.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("x contains NaN or Inf"));
    }
    if let Some(h) = bandwidth {
        if !(h > 0.0) || !h.is_finite() {
            return Err(PyValueError::new_err("bandwidth must be a positive finite number"));
        }
    }

    let h = match bandwidth {
        Some(h) => h,
        None => {
            let s = std_slice(xs);
            if !(s > 0.0) || !s.is_finite() {
                return Err(PyValueError::new_err(
                    "cannot infer a bandwidth from constant or singleton data; pass bandwidth explicitly",
                ));
            }
            (n as f64).powf(-0.2) * s
        }
    };

    let mut mn = xs[0];
    let mut mx = xs[0];
    for &v in &xs[1..] {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    let lo = mn - 3.0 * h;
    let hi = mx + 3.0 * h;

    let grid: Vec<f64> = if points == 1 {
        vec![0.5 * (lo + hi)]
    } else {
        let step = (hi - lo) / ((points - 1) as f64);
        (0..points).map(|i| lo + step * (i as f64)).collect()
    };

    let inv_h = 1.0 / h;
    let norm = 1.0 / ((n as f64) * h * (2.0 * std::f64::consts::PI).sqrt());
    let density: Vec<f64> = grid
        .iter()
        .map(|&g| {
            let mut acc = 0.0f64;
            for &xi in xs {
                let z = (g - xi) * inv_h;
                acc += (-0.5 * z * z).exp();
            }
            acc * norm
        })
        .collect();

    Ok((
        PyArray1::from_vec_bound(py, grid),
        PyArray1::from_vec_bound(py, density),
    ))
}

// ======================
// Effect-size naming compatibility
// ======================

#[pyfunction(signature = (x, y, pooled=None))]
fn hedges_g_2samp_np(
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    pooled: Option<bool>,
) -> PyResult<f64> {
    // Public API: defaults to pooled variance
    let pooled = pooled.unwrap_or(true);
    infer::effect::hedges_g_2samp_np2(x, y, pooled)
}

#[pyfunction]
fn hedges_g_2samp_raw_np(
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    pooled: bool,
) -> PyResult<f64> {
    // Raw API: explicit pooled flag, no defaults
    infer::effect::hedges_g_2samp_np2(x, y, pooled)
}
// ======================
// Module definition
// ======================

// --------------------------------------------------------------------------------------
// Uniform cumulative hazard wrapper
//
// Fixes NaN behavior for x > b:
// - mathematically, SF is 0 for x >= b, so -log(SF) -> +inf
// - but repeated +inf values cause np.diff to produce NaN (inf - inf), and some downstream
//   monotonic checks treat NaN as a failure.
// We therefore:
// - keep +inf exactly at x == b
// - for x > b, clamp to the largest representable float < b (next_down(b)), producing a
//   very large but finite value that stays constant for all x > b.
// This preserves correct tail behavior while keeping the function numerically well-behaved.
//
// NOTE: This wrapper shadows the kernel implementation via the `wrap_pyfunction!` registration.
// --------------------------------------------------------------------------------------

#[inline]
fn next_down(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == f64::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        // negative smallest subnormal
        return f64::from_bits((1u64 << 63) | 1u64);
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        // x < 0.0
        f64::from_bits(bits + 1)
    }
}

#[pyfunction]
pub fn unif_cumhazard(py: Python<'_>, x: PyReadonlyArray1<f64>, a: f64, b: f64) -> PyResult<Py<PyArray1<f64>>> {
    if !a.is_finite() || !b.is_finite() || !(b > a) {
        return Err(PyValueError::new_err("Uniform parameters require finite a,b with b > a"));
    }

    let xs = x.as_slice()?;
    let n = xs.len();
    let mut out = Vec::with_capacity(n);

    let width = b - a;

    // Precompute a "clamped" tail value for x > b (finite, huge).
    let b_prev = next_down(b);
    // If b_prev collapsed below a (pathological), fall back to +inf.
    let tail_val = if b_prev > a {
        let sf_tail = (b - b_prev) / width; // tiny positive
        // sf_tail should be > 0, but be defensive.
        if sf_tail > 0.0 {
            -sf_tail.ln()
        } else {
            f64::INFINITY
        }
    } else {
        f64::INFINITY
    };

    for &xi in xs {
        if xi.is_nan() {
            out.push(f64::NAN);
        } else if xi < a {
            out.push(0.0);
        } else if xi < b {
            let sf = (b - xi) / width;
            if sf <= 0.0 {
                out.push(f64::INFINITY);
            } else {
                out.push(-sf.ln());
            }
        } else if xi == b {
            out.push(f64::INFINITY);
        } else {
            out.push(tail_val);
        }
    }

    Ok(out.into_pyarray_bound(py).unbind())
}

// ============================================================================
// NEW v0.2.9: Fused multi-stat rolling functions
// ============================================================================

/*
/// Parse alignment from string.
fn parse_alignment(s: &str) -> PyResult<Alignment> {
    match s.to_lowercase().as_str() {
        "trailing" => Ok(Alignment::Trailing),
        "centered" => Ok(Alignment::Centered),
        _ => Err(PyErr::new::<PyValueError, _>(
            format!("Invalid alignment: '{}'. Must be 'trailing' or 'centered'", s),
        )),
    }
}

/// Parse NaN policy from string.
fn parse_nan_policy(s: &str) -> PyResult<NanPolicy> {
    match s.to_lowercase().as_str() {
        "propagate" => Ok(NanPolicy::Propagate),
        "ignore" => Ok(NanPolicy::Ignore),
        "require_min_periods" => Ok(NanPolicy::RequireMinPeriods),
        _ => Err(PyErr::new::<PyValueError, _>(
            format!(
                "Invalid nan_policy: '{}'. Must be 'propagate', 'ignore', or 'require_min_periods'",
                s
            ),
        )),
    }
}
*/

///
/// Unified fused rolling statistics function (1D).
///
/// Returns a tuple of requested statistics arrays.
///
/// # Arguments
/// - x: 1D numpy array
/// - window: window size
/// - min_periods: minimum valid observations (None = window)
/// - alignment: "trailing" or "centered"
/// - nan_policy: "propagate", "ignore", or "require_min_periods"
/// - stats: tuple of stat names, e.g., ("mean", "std", "var")
#[pyfunction]
#[pyo3(signature = (x, window, min_periods=None, alignment="trailing", nan_policy="propagate", stats=None))]
fn rolling_multi_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    window: usize,
    min_periods: Option<usize>,
    alignment: &str,
    nan_policy: &str,
    stats: Option<Vec<String>>,
) -> PyResult<Py<PyAny>> {
    let xs = x.as_slice()?;
    let n = xs.len();
    
    // Parse config
    let align = parse_alignment(alignment)?;
    let nan_pol = parse_nan_policy(nan_policy)?;
    let config = RollingConfig::new(window, min_periods, align, nan_pol)
        .map_err(|e| PyErr::new::<PyValueError, _>(e))?;
    
    // Parse stats mask
    let stats = stats.unwrap_or_else(|| vec!["mean".to_string()]);
    let stats_str = stats.join(",");
    let mask = StatsMask::from_str_list(&stats_str)
        .map_err(|e| PyErr::new::<PyValueError, _>(e))?;
    
    let out_len = output_length(n, window, align);
    
    // Allocate outputs - only if out_len > 0
    let mut out_mean = if mask.has_mean() && out_len > 0 { vec![0.0; out_len] } else { Vec::new() };
    let mut out_std = if mask.has_std() && out_len > 0 { vec![0.0; out_len] } else { Vec::new() };
    let mut out_var = if mask.has_var() && out_len > 0 { vec![0.0; out_len] } else { Vec::new() };
    let mut out_count = if mask.has_count() && out_len > 0 { vec![0.0; out_len] } else { Vec::new() };
    let mut out_min = if mask.has_min() && out_len > 0 { vec![0.0; out_len] } else { Vec::new() };
    let mut out_max = if mask.has_max() && out_len > 0 { vec![0.0; out_len] } else { Vec::new() };
    
    // Compute
    rolling_multi_into(
        xs, &config, mask,
        &mut out_mean, &mut out_std, &mut out_var,
        &mut out_count, &mut out_min, &mut out_max,
    );
    
    // Build result tuple in requested order
    let mut results = Vec::new();
    for stat in &stats {
        let arr: Py<PyAny> = match stat.to_lowercase().as_str() {
            "mean" => PyArray1::from_vec_bound(py, out_mean.clone()).into_py(py),
            "std" => PyArray1::from_vec_bound(py, out_std.clone()).into_py(py),
            "var" => PyArray1::from_vec_bound(py, out_var.clone()).into_py(py),
            "count" => PyArray1::from_vec_bound(py, out_count.clone()).into_py(py),
            "min" => PyArray1::from_vec_bound(py, out_min.clone()).into_py(py),
            "max" => PyArray1::from_vec_bound(py, out_max.clone()).into_py(py),
            _ => return Err(PyErr::new::<PyValueError, _>(format!("Unknown stat: {}", stat))),
        };
        results.push(arr);
    }
    
    Ok(PyTuple::new_bound(py, results).into())
}

/// Unified fused rolling statistics function (2D axis=0).
///
/// Computes rolling statistics along axis 0 (column-wise) for 2D arrays.
///
/// # Arguments
/// - x: 2D input array
/// - window: window size
/// - min_periods: minimum valid observations required (defaults to window)
/// - alignment: "trailing" or "centered"
/// - nan_policy: "propagate", "ignore", or "require_min_periods"
/// - stats: tuple of stat names, e.g., ("mean", "std", "var")
#[pyfunction]
#[pyo3(signature = (x, window, min_periods=None, alignment="trailing", nan_policy="propagate", stats=None))]
fn rolling_multi_axis0_np<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    window: usize,
    min_periods: Option<usize>,
    alignment: &str,
    nan_policy: &str,
    stats: Option<Vec<String>>,
) -> PyResult<Py<PyAny>> {
    let arr = x.as_array();
    let shape = arr.shape();
    let n_rows = shape[0];
    let n_cols = shape[1];
    
    let data = arr.as_slice()
        .ok_or_else(|| PyErr::new::<PyValueError, _>("Array must be contiguous"))?;
    
    // Parse config
    let align = parse_alignment(alignment)?;
    let nan_pol = parse_nan_policy(nan_policy)?;
    let config = RollingConfig::new(window, min_periods, align, nan_pol)
        .map_err(|e| PyErr::new::<PyValueError, _>(e))?;
    
    // Parse stats mask
    let stats = stats.unwrap_or_else(|| vec!["mean".to_string()]);
    let stats_str = stats.join(",");
    let mask = StatsMask::from_str_list(&stats_str)
        .map_err(|e| PyErr::new::<PyValueError, _>(e))?;
    
    let out_rows = output_length(n_rows, window, align);
    let out_len = out_rows * n_cols;
    
    // Allocate outputs - only if out_len > 0
    let mut out_mean = if mask.has_mean() && out_len > 0 { vec![0.0; out_len] } else { Vec::new() };
    let mut out_std = if mask.has_std() && out_len > 0 { vec![0.0; out_len] } else { Vec::new() };
    let mut out_var = if mask.has_var() && out_len > 0 { vec![0.0; out_len] } else { Vec::new() };
    let mut out_count = if mask.has_count() && out_len > 0 { vec![0.0; out_len] } else { Vec::new() };
    let mut out_min = if mask.has_min() && out_len > 0 { vec![0.0; out_len] } else { Vec::new() };
    let mut out_max = if mask.has_max() && out_len > 0 { vec![0.0; out_len] } else { Vec::new() };
    
    // Compute
    rolling_multi_axis0_into(
        data, n_rows, n_cols, &config, mask,
        &mut out_mean, &mut out_std, &mut out_var,
        &mut out_count, &mut out_min, &mut out_max,
    );
    
    // Build result tuple
    let mut results = Vec::new();
    for stat in &stats {
        let vec_data = match stat.to_lowercase().as_str() {
            "mean" => out_mean.clone(),
            "std" => out_std.clone(),
            "var" => out_var.clone(),
            "count" => out_count.clone(),
            "min" => out_min.clone(),
            "max" => out_max.clone(),
            _ => return Err(PyErr::new::<PyValueError, _>(format!("Unknown stat: {}", stat))),
        };
        
        let arr_2d = PyArray2::from_vec2_bound(
            py,
            &vec_data.chunks(n_cols).map(|c| c.to_vec()).collect::<Vec<_>>()
        )?;
        results.push(arr_2d.into_py(py));
    }
    
    Ok(PyTuple::new_bound(py, results).into())
}

#[pymodule]
fn bunker_stats_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // basic stats
    m.add_function(wrap_pyfunction!(mean_np, m)?)?;
    m.add_function(wrap_pyfunction!(mean_skipna_np, m)?)?;
    m.add_function(wrap_pyfunction!(mean_nan_np, m)?)?;
    m.add_function(wrap_pyfunction!(var_np, m)?)?;
    m.add_function(wrap_pyfunction!(var_skipna_np, m)?)?;
    m.add_function(wrap_pyfunction!(var_nan_np, m)?)?;
    m.add_function(wrap_pyfunction!(std_np, m)?)?;
    m.add_function(wrap_pyfunction!(std_skipna_np, m)?)?;
    m.add_function(wrap_pyfunction!(std_nan_np, m)?)?;

    m.add_function(wrap_pyfunction!(zscore_np, m)?)?;
    m.add_function(wrap_pyfunction!(zscore_skipna_np, m)?)?;
    m.add_function(wrap_pyfunction!(skew_np, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis_np, m)?)?;

    m.add_function(wrap_pyfunction!(percentile_np, m)?)?;
    m.add_function(wrap_pyfunction!(iqr_np, m)?)?;
    m.add_function(wrap_pyfunction!(iqr_width_np, m)?)?;
    m.add_function(wrap_pyfunction!(mad_np, m)?)?;
    m.add_function(wrap_pyfunction!(trimmed_mean_np, m)?)?;

    // Robust statistics - extended functions
    m.add_function(wrap_pyfunction!(median_np, m)?)?;
    m.add_function(wrap_pyfunction!(iqr_robust_np, m)?)?;
    m.add_function(wrap_pyfunction!(winsorized_mean_np, m)?)?;
    m.add_function(wrap_pyfunction!(trimmed_std_np, m)?)?;
    m.add_function(wrap_pyfunction!(mad_std_np, m)?)?;
    m.add_function(wrap_pyfunction!(biweight_midvariance_np, m)?)?;
    m.add_function(wrap_pyfunction!(qn_scale_np, m)?)?;
    m.add_function(wrap_pyfunction!(huber_location_np, m)?)?;
    
    // Robust statistics - skipna variants
    m.add_function(wrap_pyfunction!(median_skipna_np, m)?)?;
    m.add_function(wrap_pyfunction!(mad_skipna_np, m)?)?;
    m.add_function(wrap_pyfunction!(trimmed_mean_skipna_np, m)?)?;
    m.add_function(wrap_pyfunction!(iqr_skipna_np, m)?)?;

    // ========================================================================
    // NEW: Robust statistics - policy-driven API (v0.2.9)
    // ========================================================================
    m.add_class::<kernels::robust::RobustStats>()?;
    m.add_function(wrap_pyfunction!(kernels::robust::robust_fit, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::robust::robust_score, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::robust::rolling_median, m)?)?;

    // multi-D
    m.add_function(wrap_pyfunction!(mean_axis_np, m)?)?;
    m.add_function(wrap_pyfunction!(mean_over_last_axis_dyn_np, m)?)?;

    // rolling (truncated)
    m.add_function(wrap_pyfunction!(rolling_mean_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_var_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean_std_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_zscore_np, m)?)?;

    // rolling axis0 (truncated)
    m.add_function(wrap_pyfunction!(rolling_mean_std_axis0_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean_axis0_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std_axis0_np, m)?)?;

    // rolling (NaN-aware, full-length)
    m.add_function(wrap_pyfunction!(rolling_mean_nan_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std_nan_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_zscore_nan_np, m)?)?;
    m.add_function(wrap_pyfunction!(ewma_np, m)?)?;

    // NEW v0.2.9: Fused multi-stat rolling functions
    m.add_function(wrap_pyfunction!(rolling_multi_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_multi_axis0_np, m)?)?;

    // outliers / scaling
    m.add_function(wrap_pyfunction!(iqr_outliers_np, m)?)?;
    m.add_function(wrap_pyfunction!(zscore_outliers_np, m)?)?;
    m.add_function(wrap_pyfunction!(minmax_scale_np, m)?)?;
    m.add_function(wrap_pyfunction!(robust_scale_np, m)?)?;
    m.add_function(wrap_pyfunction!(winsorize_np, m)?)?;
    m.add_function(wrap_pyfunction!(winsorize_clip_np, m)?)?;

    // diff / cum / ecdf / bins / signs
    m.add_function(wrap_pyfunction!(diff_np, m)?)?;
    m.add_function(wrap_pyfunction!(pct_change_np, m)?)?;
    m.add_function(wrap_pyfunction!(cumsum_np, m)?)?;
    m.add_function(wrap_pyfunction!(cummean_np, m)?)?;
    m.add_function(wrap_pyfunction!(ecdf_np, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_bins_np, m)?)?;
    m.add_function(wrap_pyfunction!(sign_mask_np, m)?)?;
    m.add_function(wrap_pyfunction!(demean_with_signs_np, m)?)?;

    // covariance / correlation
    m.add_function(wrap_pyfunction!(cov_np, m)?)?;
    m.add_function(wrap_pyfunction!(corr_np, m)?)?;
    m.add_function(wrap_pyfunction!(cov_matrix_np, m)?)?;
    m.add_function(wrap_pyfunction!(cov_matrix_bias_np, m)?)?;
    m.add_function(wrap_pyfunction!(cov_matrix_centered_np, m)?)?;
    m.add_function(wrap_pyfunction!(cov_matrix_skipna_np, m)?)?;
    m.add_function(wrap_pyfunction!(xtx_matrix_np, m)?)?;
    m.add_function(wrap_pyfunction!(xxt_matrix_np, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_euclidean_cols_np, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_cosine_cols_np, m)?)?;
    m.add_function(wrap_pyfunction!(corr_matrix_np, m)?)?;
    m.add_function(wrap_pyfunction!(corr_matrix_skipna_np, m)?)?;
    m.add_function(wrap_pyfunction!(corr_distance_np, m)?)?;
    m.add_function(wrap_pyfunction!(diag_np, m)?)?;
    m.add_function(wrap_pyfunction!(trace_np, m)?)?;
    m.add_function(wrap_pyfunction!(is_symmetric_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_cov_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_corr_np, m)?)?;

    // Welford + NaN-aware covariance/correlation
    m.add_function(wrap_pyfunction!(welford_np, m)?)?;
    m.add_function(wrap_pyfunction!(cov_nan_np, m)?)?;
    m.add_function(wrap_pyfunction!(corr_nan_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_cov_nan_np, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_corr_nan_np, m)?)?;

// preferred clean names + new rolling linear-model primitives
m.add_function(wrap_pyfunction!(cov_skipna, m)?)?;
m.add_function(wrap_pyfunction!(corr_skipna, m)?)?;
m.add_function(wrap_pyfunction!(rolling_cov_skipna, m)?)?;
m.add_function(wrap_pyfunction!(rolling_corr_skipna, m)?)?;
m.add_function(wrap_pyfunction!(rolling_beta_skipna, m)?)?;
m.add_function(wrap_pyfunction!(rolling_linreg_skipna, m)?)?;


    // KDE
    m.add_function(wrap_pyfunction!(kde_gaussian_np, m)?)?;

    // ============================================================================
	// INFERENCE MODULE - OPTIMIZED VERSION
	// ============================================================================

	// Existing tests (with bug fixes)
	m.add_function(wrap_pyfunction!(infer::ttest::t_test_1samp_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::ttest::t_test_2samp_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::chi2::chi2_gof_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::chi2::chi2_independence_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::effect::mean_diff_ci_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::effect::cohens_d_2samp_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::effect::hedges_g_2samp_np2, m)?)?;
	m.add_function(wrap_pyfunction!(infer::mann_whitney::mann_whitney_u_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::ks::ks_1samp_np, m)?)?;

	// NEW: ANOVA
	m.add_function(wrap_pyfunction!(infer::anova::f_test_oneway_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::anova::levene_test_np, m)?)?;

	// NEW: Normality tests
	m.add_function(wrap_pyfunction!(infer::normality::jarque_bera_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::normality::anderson_darling_np, m)?)?;

	// NEW: Correlation tests
	m.add_function(wrap_pyfunction!(infer::correlation::pearson_corr_test_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::correlation::spearman_corr_test_np, m)?)?;

	// NEW: Variance tests
	m.add_function(wrap_pyfunction!(infer::variance_tests::f_test_var_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::variance_tests::bartlett_test_np, m)?)?;

	// v0.3.1: additional inference functions
	m.add_function(wrap_pyfunction!(infer::paired::t_test_paired_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::p_adjust::p_adjust_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::proportion::proportion_ztest_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::proportion::two_proportions_ztest_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::corr_ci::corr_ci_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::var_ci::var_ci_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::odds_ratio::odds_ratio_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::effect_nonparam::rank_biserial_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::effect_nonparam::cliffs_delta_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::effect_nonparam::anova_effect_sizes_np, m)?)?;
	m.add_function(wrap_pyfunction!(infer::effect_nonparam::normality_summary_np, m)?)?;

	    // ----------------------
    // sandboxstats payload
    // ----------------------

    // resampling
    m.add_function(wrap_pyfunction!(bootstrap_mean, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_mean_ci, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_ci, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_ci_with_draws, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_corr, m)?)?;
    m.add_function(wrap_pyfunction!(jackknife_mean, m)?)?;
    m.add_function(wrap_pyfunction!(jackknife_mean_ci, m)?)?;

    // extended resampling
    m.add_function(wrap_pyfunction!(bootstrap_se, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_var, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_t_ci_mean, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_bca_ci, m)?)?;
    m.add_function(wrap_pyfunction!(bayesian_bootstrap_ci, m)?)?;

    // time-series resampling
    m.add_function(wrap_pyfunction!(moving_block_bootstrap_mean_ci, m)?)?;
    m.add_function(wrap_pyfunction!(circular_block_bootstrap_mean_ci, m)?)?;
    m.add_function(wrap_pyfunction!(stationary_bootstrap_mean_ci, m)?)?;

    // permutation tests
    m.add_function(wrap_pyfunction!(permutation_corr_test, m)?)?;
    m.add_function(wrap_pyfunction!(permutation_mean_diff_test, m)?)?;

    // extended jackknife
    m.add_function(wrap_pyfunction!(influence_mean, m)?)?;
    m.add_function(wrap_pyfunction!(delete_d_jackknife_mean, m)?)?;
    m.add_function(wrap_pyfunction!(jackknife_after_bootstrap_se_mean, m)?)?;

    // tsa - stationarity
    m.add_function(wrap_pyfunction!(adf_test, m)?)?;
    m.add_function(wrap_pyfunction!(kpss_test, m)?)?;
    m.add_function(wrap_pyfunction!(pp_test, m)?)?;
	m.add_function(wrap_pyfunction!(variance_ratio_test, m)?)?;
	m.add_function(wrap_pyfunction!(zivot_andrews_test, m)?)?;
	m.add_function(wrap_pyfunction!(trend_stationarity_test, m)?)?;
	m.add_function(wrap_pyfunction!(integration_order_test, m)?)?;
	m.add_function(wrap_pyfunction!(seasonal_diff_test, m)?)?;
	m.add_function(wrap_pyfunction!(seasonal_unit_root_test, m)?)?;

    
    // tsa - diagnostics
    m.add_function(wrap_pyfunction!(ljung_box, m)?)?;
    m.add_function(wrap_pyfunction!(durbin_watson, m)?)?;
    m.add_function(wrap_pyfunction!(bg_test, m)?)?;
    m.add_function(wrap_pyfunction!(box_pierce, m)?)?;
    m.add_function(wrap_pyfunction!(runs_test, m)?)?;
    m.add_function(wrap_pyfunction!(acf_zero_crossing, m)?)?;

    // tsa - acf/pacf
    m.add_function(wrap_pyfunction!(acf, m)?)?;
    m.add_function(wrap_pyfunction!(pacf, m)?)?;
    m.add_function(wrap_pyfunction!(pacf_yw, m)?)?;
    m.add_function(wrap_pyfunction!(acovf, m)?)?;
    m.add_function(wrap_pyfunction!(acf_with_ci, m)?)?;
    m.add_function(wrap_pyfunction!(ccf, m)?)?;
    m.add_function(wrap_pyfunction!(pacf_innovations, m)?)?;
    m.add_function(wrap_pyfunction!(pacf_burg, m)?)?;
    
    // tsa - rolling autocorrelation
    m.add_function(wrap_pyfunction!(rolling_autocorr, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_autocorr_multi, m)?)?;

    // tsa - O(1) rolling helpers (registered under their kernel names; the
    // rolling_mean/std/var/zscore variants in the same module stay unregistered
    // because those names are already taken by the strict kernels above)
    m.add_function(wrap_pyfunction!(kernels::tsa::rolling::rolling_min, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::tsa::rolling::rolling_max, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::tsa::rolling::rolling_range, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::tsa::rolling::rolling_cv, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::tsa::rolling::rolling_count_above, m)?)?;
    m.add_function(wrap_pyfunction!(kernels::tsa::rolling::rolling_pct_above, m)?)?;
    
    // tsa - spectral
    m.add_function(wrap_pyfunction!(periodogram, m)?)?;
    m.add_function(wrap_pyfunction!(welch_psd, m)?)?;
    m.add_function(wrap_pyfunction!(cumulative_periodogram, m)?)?;
    m.add_function(wrap_pyfunction!(dominant_frequency, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(bartlett_psd, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_flatness, m)?)?;
    m.add_function(wrap_pyfunction!(band_power, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_centroid, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_rolloff, m)?)?;

    // dist
    m.add_function(wrap_pyfunction!(norm_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(norm_logpdf, m)?)?;
    m.add_function(wrap_pyfunction!(norm_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(norm_sf, m)?)?;
    m.add_function(wrap_pyfunction!(norm_logsf, m)?)?;
    m.add_function(wrap_pyfunction!(norm_cumhazard, m)?)?;
    m.add_function(wrap_pyfunction!(norm_ppf, m)?)?;

    m.add_function(wrap_pyfunction!(exp_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(exp_logpdf, m)?)?;
    m.add_function(wrap_pyfunction!(exp_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(exp_sf, m)?)?;
    m.add_function(wrap_pyfunction!(exp_logsf, m)?)?;
    m.add_function(wrap_pyfunction!(exp_cumhazard, m)?)?;
    m.add_function(wrap_pyfunction!(exp_ppf, m)?)?;

    m.add_function(wrap_pyfunction!(unif_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(unif_logpdf, m)?)?;
    m.add_function(wrap_pyfunction!(unif_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(unif_sf, m)?)?;
    m.add_function(wrap_pyfunction!(unif_logsf, m)?)?;
    m.add_function(wrap_pyfunction!(unif_cumhazard, m)?)?;
    m.add_function(wrap_pyfunction!(unif_ppf, m)?)?;


    // padding
    m.add_function(wrap_pyfunction!(pad_nan_np, m)?)?;

    Ok(())
}