use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Alternative hypothesis specification.
/// Matches SciPy semantics.
#[derive(Clone, Copy, Debug)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

impl Alternative {
    /// Parse alternative string from Python API.
    pub fn from_str(s: &str) -> PyResult<Self> {
        match s {
            "two-sided" | "two_sided" => Ok(Self::TwoSided),
            "less" => Ok(Self::Less),
            "greater" => Ok(Self::Greater),
            _ => Err(PyValueError::new_err(
                "alternative must be one of: 'two-sided', 'less', 'greater'",
            )),
        }
    }
}

/// Reject NaN / Inf inputs (v0.3 contract).
#[inline]
pub fn reject_nonfinite(xs: &[f64], name: &str) -> PyResult<()> {
    if xs.iter().any(|v| !v.is_finite()) {
        Err(PyValueError::new_err(format!(
            "{name} contains NaN or Inf; bunker-stats v0.3 nan_policy is 'reject'"
        )))
    } else {
        Ok(())
    }
}

/// Compute the mean of a slice (no NaN handling).
#[inline]
pub fn mean(xs: &[f64]) -> f64 {
    let mut s = 0.0;
    for &v in xs {
        s += v;
    }
    s / (xs.len() as f64)
}

/// Compute sample variance (ddof = 1).
///
/// Assumes:
/// - xs.len() >= 2
/// - mean already computed
#[inline]
pub fn var_sample(xs: &[f64], mean: f64) -> f64 {
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }

    let mut ss = 0.0;
    for &v in xs {
        let d = v - mean;
        ss += d * d;
    }

    ss / ((n - 1) as f64)
}
