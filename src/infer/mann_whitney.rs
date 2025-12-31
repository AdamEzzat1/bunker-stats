use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use statrs::distribution::{ContinuousCDF, Normal};

use super::common::{reject_nonfinite, Alternative};

#[derive(Clone, Copy)]
struct Obs {
    v: f64,
    is_x: bool,
}

/// Compute average ranks for tied groups.
/// Returns (sum_ranks_x, tie_term_sum, n_total).
fn rankdata_average(obs: &mut [Obs]) -> (f64, f64, usize) {
    obs.sort_by(|a, b| a.v.partial_cmp(&b.v).unwrap());

    let n = obs.len();
    let mut rank = 1usize;

    let mut sum_ranks_x = 0.0;
    let mut tie_term_sum = 0.0; // Σ(t^3 - t)

    let mut i = 0usize;
    while i < n {
        let v0 = obs[i].v;
        let mut j = i + 1;
        while j < n && obs[j].v == v0 {
            j += 1;
        }

        // tie group is [i, j)
        let t = j - i;
        let r_start = rank as f64;
        let r_end = (rank + t - 1) as f64;
        let r_avg = 0.5 * (r_start + r_end);

        if t > 1 {
            let tf = t as f64;
            tie_term_sum += tf * tf * tf - tf;
        }

        for k in i..j {
            if obs[k].is_x {
                sum_ranks_x += r_avg;
            }
        }

        rank += t;
        i = j;
    }

    (sum_ranks_x, tie_term_sum, n)
}

/// Asymptotic Mann–Whitney U (SciPy-like), with tie correction and continuity correction.
/// Returns (statistic, pvalue).
fn mann_whitney_asymptotic(x: &[f64], y: &[f64], alt: Alternative) -> (f64, f64) {
    let n1 = x.len();
    let n2 = y.len();
    let n = n1 + n2;

    let mut obs: Vec<Obs> = Vec::with_capacity(n);
    obs.extend(x.iter().map(|&v| Obs { v, is_x: true }));
    obs.extend(y.iter().map(|&v| Obs { v, is_x: false }));

    let (r1, tie_sum, n_total) = rankdata_average(&mut obs);

    let n1f = n1 as f64;
    let n2f = n2 as f64;
    let nf = n_total as f64;

    // U1 for sample x
    let u1 = r1 - n1f * (n1f + 1.0) / 2.0;
    let u2 = n1f * n2f - u1;

    let mean_u = n1f * n2f / 2.0;

    // tie-corrected variance (SciPy style)
    // var = n1*n2/12 * (N+1 - tie_sum/(N*(N-1)))
    let denom = nf * (nf - 1.0);
    let tie_c = if denom > 0.0 { tie_sum / denom } else { 0.0 };
    let var_u = n1f * n2f / 12.0 * (nf + 1.0 - tie_c);
    let sd_u = var_u.sqrt();

    // z from u1 (SciPy uses u for x), continuity correction:
    let diff = u1 - mean_u;
    let cc = match alt {
        Alternative::TwoSided => {
            if diff > 0.0 {
                0.5
            } else if diff < 0.0 {
                -0.5
            } else {
                0.0
            }
        }
        Alternative::Greater => 0.5,
        Alternative::Less => -0.5,
    };

    let z = if sd_u > 0.0 { (diff - cc) / sd_u } else { 0.0 };

    let norm = Normal::new(0.0, 1.0).unwrap();
    let p = match alt {
        Alternative::TwoSided => 2.0 * norm.sf(z.abs()),
        Alternative::Greater => norm.sf(z),
        Alternative::Less => norm.cdf(z),
    }
    .clamp(0.0, 1.0);

    // statistic returned: SciPy returns min(U1, U2) for two-sided, else U1
    let stat = match alt {
        Alternative::TwoSided => u1.min(u2),
        _ => u1,
    };

    (stat, p)
}

#[pyfunction]
pub fn mann_whitney_u_np(
    py: Python<'_>,
    x: numpy::PyReadonlyArray1<f64>,
    y: numpy::PyReadonlyArray1<f64>,
    alternative: &str,
) -> PyResult<Py<PyDict>> {
    let xs = x.as_slice()?;
    let ys = y.as_slice()?;

    if xs.is_empty() || ys.is_empty() {
        return Err(PyValueError::new_err("x and y must be non-empty"));
    }
    reject_nonfinite(xs, "x")?;
    reject_nonfinite(ys, "y")?;

    let alt = Alternative::from_str(alternative)?;

    let (stat, p) = mann_whitney_asymptotic(xs, ys, alt);

    let dct = PyDict::new_bound(py);
    dct.set_item("statistic", stat)?;
    dct.set_item("pvalue", p)?;
    Ok(dct.unbind())
}
