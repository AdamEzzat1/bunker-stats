/// PyO3 bindings for robust statistics - OPTIMIZED
///
/// Optimization #4: Store enums directly instead of strings
/// - Parse/validate once in __new__
/// - Zero string operations in fit() and score()
/// - Direct enum dispatch for performance

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use super::fit::{robust_fit_slice, robust_score_slice};
use super::policy::{LocationPolicy, RobustConfig, ScalePolicy};

/// Robust statistics estimator with configurable policies - OPTIMIZED
///
/// Parameters
/// ----------
/// location : str, default="median"
///     Location estimator: "median", "trimmed_mean", or "huber"
/// scale : str, default="mad"
///     Scale estimator: "mad", "iqr", or "qn"
/// c : float, default=1.345
///     Tuning constant for Huber estimator
/// trim : float, default=0.1
///     Trim proportion for trimmed_mean
/// max_iter : int, default=50
///     Maximum iterations for Huber estimator
/// tol : float, default=1e-6
///     Convergence tolerance for Huber estimator
/// mad_consistent : bool, default=True
///     Apply normal-consistency constant to MAD
///
/// Notes
/// -----
/// PERFORMANCE: Configuration is parsed once at construction and stored
/// as efficient enums. No string operations occur in fit() or score().
#[pyclass(module = "bunker_stats")]
#[derive(Clone)]
pub struct RobustStats {
    // OPTIMIZED: Store pre-built config (enums, not strings) - Optimization #4
    config: RobustConfig,
    
    // Keep string copies only for __repr__
    location_name: String,
    scale_name: String,
    c: f64,
    trim: f64,
    mad_consistent: bool,
}

#[pymethods]
impl RobustStats {
    #[new]
    #[pyo3(signature = (
        location="median",
        scale="mad",
        c=1.345,
        trim=0.1,
        max_iter=50,
        tol=1e-6,
        mad_consistent=true
    ))]
    fn new(
        location: &str,
        scale: &str,
        c: f64,
        trim: f64,
        max_iter: usize,
        tol: f64,
        mad_consistent: bool,
    ) -> PyResult<Self> {
        // Parse and validate ONCE at construction (Optimization #4)
        let location_policy = parse_location_policy(
            location, scale, c, trim, max_iter, tol, mad_consistent
        )?;
        let scale_policy = parse_scale_policy(scale, mad_consistent)?;

        let config = RobustConfig {
            location: location_policy,
            scale: scale_policy,
        };

        Ok(Self {
            config,
            location_name: location.to_string(),
            scale_name: scale.to_string(),
            c,
            trim,
            mad_consistent,
        })
    }

    /// Fit robust location and scale - OPTIMIZED
    ///
    /// Zero string operations - uses pre-built enum config.
    fn fit<'py>(
        &self,
        _py: Python<'py>,
        x: PyReadonlyArray1<f64>,
    ) -> PyResult<(f64, f64)> {
        let data = x.as_slice()?;
        // FAST: No parsing, direct enum dispatch (Optimization #4)
        Ok(robust_fit_slice(data, &self.config))
    }

    /// Compute robust z-scores - OPTIMIZED
    ///
    /// Zero string operations - uses pre-built enum config.
    fn score<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let data = x.as_slice()?;
        // FAST: No parsing, direct enum dispatch (Optimization #4)
        let scores = robust_score_slice(data, &self.config);
        Ok(PyArray1::from_vec_bound(py, scores))
    }

    fn __repr__(&self) -> String {
        format!(
            "RobustStats(location='{}', scale='{}', c={:.3}, trim={:.3}, mad_consistent={})",
            self.location_name, self.scale_name, self.c, self.trim, self.mad_consistent
        )
    }
}

// ============================================================================
// PARSING HELPERS (called only at construction, not in hot path)
// ============================================================================

fn parse_location_policy(
    location: &str,
    scale: &str,
    c: f64,
    trim: f64,
    max_iter: usize,
    tol: f64,
    mad_consistent: bool,
) -> PyResult<LocationPolicy> {
    match location {
        "median" => Ok(LocationPolicy::Median),
        
        "trimmed_mean" => Ok(LocationPolicy::TrimmedMean { trim }),
        
        "huber" => {
            let scale_policy = parse_scale_policy(scale, mad_consistent)?;
            Ok(LocationPolicy::Huber {
                c,
                max_iter,
                tol,
                scale_policy,
            })
        }
        
        _ => Err(PyValueError::new_err(
            format!("Invalid location: {}. Must be 'median', 'trimmed_mean', or 'huber'", location)
        )),
    }
}

fn parse_scale_policy(scale: &str, mad_consistent: bool) -> PyResult<ScalePolicy> {
    match scale {
        "mad" => Ok(ScalePolicy::Mad { consistent: mad_consistent }),
        "iqr" => Ok(ScalePolicy::Iqr),
        "qn" => Ok(ScalePolicy::Qn),
        _ => Err(PyValueError::new_err(
            format!("Invalid scale: {}. Must be 'mad', 'iqr', or 'qn'", scale)
        )),
    }
}

// ============================================================================
// MODULE-LEVEL FUNCTIONS (optimized with enum dispatch)
// ============================================================================

/// Compute robust location and scale - OPTIMIZED
#[pyfunction]
#[pyo3(signature = (x, location="median", scale="mad", c=1.345, trim=0.1, max_iter=50, tol=1e-6, mad_consistent=true))]
pub fn robust_fit(
    x: PyReadonlyArray1<f64>,
    location: &str,
    scale: &str,
    c: f64,
    trim: f64,
    max_iter: usize,
    tol: f64,
    mad_consistent: bool,
) -> PyResult<(f64, f64)> {
    let data = x.as_slice()?;
    
    // Parse once, then dispatch
    let location_policy = parse_location_policy(
        location, scale, c, trim, max_iter, tol, mad_consistent
    )?;
    let scale_policy = parse_scale_policy(scale, mad_consistent)?;
    
    let config = RobustConfig {
        location: location_policy,
        scale: scale_policy,
    };
    
    Ok(robust_fit_slice(data, &config))
}

/// Compute robust z-scores - OPTIMIZED
#[pyfunction]
#[pyo3(signature = (x, location="median", scale="mad", c=1.345, trim=0.1, max_iter=50, tol=1e-6, mad_consistent=true))]
pub fn robust_score<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    location: &str,
    scale: &str,
    c: f64,
    trim: f64,
    max_iter: usize,
    tol: f64,
    mad_consistent: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = x.as_slice()?;
    
    // Parse once, then dispatch
    let location_policy = parse_location_policy(
        location, scale, c, trim, max_iter, tol, mad_consistent
    )?;
    let scale_policy = parse_scale_policy(scale, mad_consistent)?;
    
    let config = RobustConfig {
        location: location_policy,
        scale: scale_policy,
    };
    
    let scores = robust_score_slice(data, &config);
    Ok(PyArray1::from_vec_bound(py, scores))
}

/// Rolling median with adaptive algorithm - OPTIMIZED
#[pyfunction]
pub fn rolling_median<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = x.as_slice()?;
    let result = super::rolling::rolling_median(data, window);
    Ok(PyArray1::from_vec_bound(py, result))
}