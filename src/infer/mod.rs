//! Pillar C (v0.3): Inference Core
//!
//! Contract (LOCKED for v0.3):
//! - float64 only (Rust `f64`, NumPy `float64`)
//! - deterministic outputs (no randomness, no resampling)
//! - NaN / Inf policy: REJECT (raise error)
//! - Must match SciPy within documented tolerances
//!
//! Explicit non-goals (v0.3):
//! - no regression zoo (not statsmodels)
//! - no Bayesian / MCMC
//! - no bootstrap / jackknife in core
//!
//! Design philosophy:
//! Keep this module small, boring, and correct.
//! Performance comes later (Pillar D).

pub mod common;
pub mod ttest;
pub mod chi2;
pub mod effect;

// Wired but intentionally incomplete for v0.3.
// These are enabled later without breaking API.
pub mod mann_whitney;
pub mod ks;
