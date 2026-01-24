/// Policy-driven robust statistics configuration
///
/// Provides composable location and scale estimators with
/// deterministic behavior and minimal allocations.

use std::fmt;

/// Location estimator policy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LocationPolicy {
    /// Median (50th percentile)
    Median,
    
    /// Trimmed mean: remove proportion from each tail
    /// trim must be in [0.0, 0.5)
    TrimmedMean { trim: f64 },
    
    /// Huber M-estimator of location
    /// - c: tuning constant (1.345 for 95% efficiency at normal)
    /// - max_iter: maximum iterations
    /// - tol: convergence tolerance
    /// - scale_policy: scale estimator used internally
    Huber {
        c: f64,
        max_iter: usize,
        tol: f64,
        scale_policy: ScalePolicy,
    },
}

/// Scale estimator policy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalePolicy {
    /// Median Absolute Deviation
    /// consistent=true applies normal-consistency constant (1.4826)
    Mad { consistent: bool },
    
    /// Interquartile Range (Q3 - Q1)
    Iqr,
    
    /// Rousseeuw-Croux Qn estimator
    /// (robust, doesn't require location)
    Qn,
}

/// Complete robust statistics configuration
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobustConfig {
    pub location: LocationPolicy,
    pub scale: ScalePolicy,
}

impl Default for RobustConfig {
    fn default() -> Self {
        Self {
            location: LocationPolicy::Median,
            scale: ScalePolicy::Mad { consistent: true },
        }
    }
}

impl fmt::Display for LocationPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Median => write!(f, "median"),
            Self::TrimmedMean { trim } => write!(f, "trimmed_mean(trim={:.3})", trim),
            Self::Huber { c, max_iter, tol, scale_policy } => {
                write!(f, "huber(c={:.3}, max_iter={}, tol={:.1e}, scale={:?})", 
                       c, max_iter, tol, scale_policy)
            }
        }
    }
}

impl fmt::Display for ScalePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mad { consistent } => {
                write!(f, "mad(consistent={})", consistent)
            }
            Self::Iqr => write!(f, "iqr"),
            Self::Qn => write!(f, "qn"),
        }
    }
}
