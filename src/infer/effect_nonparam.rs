// ========================================================================
// Non-parametric effect sizes:
//   - rank_biserial (from Mann-Whitney U)
//   - cliff's delta
//   - eta_squared / omega_squared (from one-way ANOVA SS)
// ========================================================================
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::common::{mean, reject_nonfinite, rankdata_average as common_rankdata, sum_of_squares_kahan};

/// Rank-biserial correlation from Mann-Whitney U.
///
/// r = 1 - 2U / (n1 * n2)
///
/// where U is the Mann-Whitney U statistic (minimum of U1, U2).
/// This is equivalent to the matched-pairs rank-biserial correlation.
///
/// Returns dict: {rank_biserial, u_statistic}
#[pyfunction]
pub fn rank_biserial_np(
    py: Python<'_>,
    x: numpy::PyReadonlyArray1<f64>,
    y: numpy::PyReadonlyArray1<f64>,
) -> PyResult<Py<PyDict>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;

    if xs.is_empty() || ys.is_empty() {
        return Err(PyValueError::new_err("x and y must be non-empty"));
    }
    reject_nonfinite(xs, "x")?;
    reject_nonfinite(ys, "y")?;

    let n1 = xs.len();
    let n2 = ys.len();
    let n1f = n1 as f64;
    let n2f = n2 as f64;

    // Compute ranks of combined data
    let mut combined: Vec<f64> = Vec::with_capacity(n1 + n2);
    combined.extend_from_slice(xs);
    combined.extend_from_slice(ys);

    let ranks = common_rankdata(&combined);

    // Sum of ranks for group x
    let r1: f64 = ranks[..n1].iter().sum();
    let u1 = r1 - n1f * (n1f + 1.0) / 2.0;

    // Rank-biserial: r = 1 - 2*U / (n1*n2)
    // Use U = min(U1, U2) for the standard formula,
    // but for rank-biserial we use U1 directly for directionality.
    let rb = 1.0 - 2.0 * u1 / (n1f * n2f);

    let d = PyDict::new_bound(py);
    d.set_item("rank_biserial", rb)?;
    d.set_item("u_statistic", u1)?;
    Ok(d.unbind())
}

/// Cliff's delta: non-parametric effect size.
///
/// delta = (#{x_i > y_j} - #{x_i < y_j}) / (n1 * n2)
///
/// Range: [-1, 1]. Magnitude thresholds: |d| < 0.147 negligible,
/// < 0.33 small, < 0.474 medium, >= 0.474 large (Romano 2006).
///
/// Returns dict: {delta, dominance_greater, dominance_less, magnitude}
#[pyfunction]
pub fn cliffs_delta_np(
    py: Python<'_>,
    x: numpy::PyReadonlyArray1<f64>,
    y: numpy::PyReadonlyArray1<f64>,
) -> PyResult<Py<PyDict>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;

    if xs.is_empty() || ys.is_empty() {
        return Err(PyValueError::new_err("x and y must be non-empty"));
    }
    reject_nonfinite(xs, "x")?;
    reject_nonfinite(ys, "y")?;

    let n1 = xs.len();
    let n2 = ys.len();
    let total = (n1 * n2) as f64;

    // Count dominance pairs
    let mut n_greater = 0u64;
    let mut n_less = 0u64;

    for &xi in xs {
        for &yj in ys {
            if xi > yj {
                n_greater += 1;
            } else if xi < yj {
                n_less += 1;
            }
        }
    }

    let delta = (n_greater as f64 - n_less as f64) / total;

    let magnitude = if delta.abs() < 0.147 {
        "negligible"
    } else if delta.abs() < 0.33 {
        "small"
    } else if delta.abs() < 0.474 {
        "medium"
    } else {
        "large"
    };

    let d = PyDict::new_bound(py);
    d.set_item("delta", delta)?;
    d.set_item("dominance_greater", n_greater)?;
    d.set_item("dominance_less", n_less)?;
    d.set_item("magnitude", magnitude)?;
    Ok(d.unbind())
}

/// Eta-squared and omega-squared from one-way ANOVA components.
///
/// eta2 = SS_between / SS_total
/// omega2 = (SS_between - df_between * MS_within) / (SS_total + MS_within)
///
/// Omega-squared is less biased than eta-squared for small samples.
///
/// Returns dict: {eta_squared, omega_squared, ss_between, ss_within, ss_total}
#[pyfunction]
pub fn anova_effect_sizes_np(
    py: Python<'_>,
    groups: Vec<numpy::PyReadonlyArray1<f64>>,
) -> PyResult<Py<PyDict>> {
    if groups.len() < 2 {
        return Err(PyValueError::new_err("requires at least 2 groups"));
    }

    let mut group_data: Vec<Vec<f64>> = Vec::with_capacity(groups.len());
    let mut n_total = 0usize;

    for g in groups {
        let xs = g.as_slice()?;
        if xs.len() < 2 {
            return Err(PyValueError::new_err("each group must have at least 2 observations"));
        }
        reject_nonfinite(xs, "group")?;
        n_total += xs.len();
        group_data.push(xs.to_vec());
    }

    let k = group_data.len();

    // Grand mean
    let mut grand_sum = 0.0;
    for group in &group_data {
        grand_sum += group.iter().sum::<f64>();
    }
    let grand_mean = grand_sum / (n_total as f64);

    // Group means
    let group_means: Vec<f64> = group_data.iter().map(|g| mean(g)).collect();

    // SS_between
    let mut ss_between = 0.0;
    for (i, group) in group_data.iter().enumerate() {
        let n = group.len() as f64;
        let diff = group_means[i] - grand_mean;
        ss_between += n * diff * diff;
    }

    // SS_within with Kahan summation
    let mut ss_within = 0.0;
    for (i, group) in group_data.iter().enumerate() {
        ss_within += sum_of_squares_kahan(group, group_means[i]);
    }

    let ss_total = ss_between + ss_within;
    let df_between = (k - 1) as f64;
    let df_within = (n_total - k) as f64;

    // Eta-squared
    let eta2 = if ss_total > 0.0 { ss_between / ss_total } else { 0.0 };

    // Omega-squared (less biased)
    let ms_within = if df_within > 0.0 { ss_within / df_within } else { 0.0 };
    let omega2 = if ss_total + ms_within > 0.0 {
        (ss_between - df_between * ms_within) / (ss_total + ms_within)
    } else {
        0.0
    };
    let omega2 = omega2.max(0.0); // Can be slightly negative with small effects

    let d = PyDict::new_bound(py);
    d.set_item("eta_squared", eta2)?;
    d.set_item("omega_squared", omega2)?;
    d.set_item("ss_between", ss_between)?;
    d.set_item("ss_within", ss_within)?;
    d.set_item("ss_total", ss_total)?;
    Ok(d.unbind())
}

/// Normality summary: runs Jarque-Bera and Anderson-Darling in a single call.
///
/// Computes skewness, kurtosis, JB statistic/p-value, and AD statistic
/// with one pass over the data for moments.
///
/// Returns dict: {skewness, kurtosis, excess_kurtosis,
///                jb_statistic, jb_pvalue, ad_statistic,
///                n}
#[pyfunction]
pub fn normality_summary_np(
    py: Python<'_>,
    x: numpy::PyReadonlyArray1<f64>,
) -> PyResult<Py<PyDict>> {
    let xs = x.as_slice()?;
    let n = xs.len();

    if n < 4 {
        return Err(PyValueError::new_err("normality_summary requires n >= 4"));
    }
    reject_nonfinite(xs, "x")?;

    let m = mean(xs);

    // Single pass for m2, m3, m4
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;

    for &v in xs {
        let d = v - m;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }

    let nf = n as f64;
    m2 /= nf;
    m3 /= nf;
    m4 /= nf;

    if m2 <= 0.0 {
        return Err(PyValueError::new_err("variance is zero"));
    }

    let skewness = m3 / m2.powf(1.5);
    let kurtosis = m4 / (m2 * m2);
    let excess_kurtosis = kurtosis - 3.0;

    // Jarque-Bera
    let jb = (nf / 6.0) * (skewness * skewness + excess_kurtosis * excess_kurtosis / 4.0);

    use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};
    let chi2_dist = ChiSquared::new(2.0).unwrap();
    let jb_p = (1.0 - chi2_dist.cdf(jb)).clamp(0.0, 1.0);

    // Anderson-Darling
    let std_dev = (m2 * nf / (nf - 1.0)).sqrt(); // sample std (ddof=1)
    let mut z: Vec<f64> = xs.iter().map(|&xi| (xi - m) / std_dev).collect();
    z.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let norm = Normal::new(0.0, 1.0).unwrap();
    let mut a2 = 0.0;

    for i in 0..n {
        let z_i = z[i];
        let z_ni = z[n - 1 - i];
        let if64 = (i + 1) as f64;

        let cdf_i = norm.cdf(z_i).max(1e-300).min(1.0 - 1e-300);
        let sf_ni = (1.0 - norm.cdf(z_ni)).max(1e-300).min(1.0 - 1e-300);

        a2 += (2.0 * if64 - 1.0) * (cdf_i.ln() + sf_ni.ln());
    }
    a2 = -nf - a2 / nf;

    // Stephens correction
    let ad_star = a2 * (1.0 + 4.0 / nf - 25.0 / (nf * nf));

    let d = PyDict::new_bound(py);
    d.set_item("skewness", skewness)?;
    d.set_item("kurtosis", kurtosis)?;
    d.set_item("excess_kurtosis", excess_kurtosis)?;
    d.set_item("jb_statistic", jb)?;
    d.set_item("jb_pvalue", jb_p)?;
    d.set_item("ad_statistic", ad_star)?;
    d.set_item("n", n)?;
    Ok(d.unbind())
}
