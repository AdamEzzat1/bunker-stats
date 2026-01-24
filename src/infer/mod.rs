// ========================================================================
// OPTIMIZED INFERENCE MODULE
// ========================================================================
// 
// Improvements:
// 1. Bug fixes for numerical stability
// 2. New cheap functions leveraging existing architecture
// 3. Performance optimizations (Kahan summation, reduced allocations)
// 4. Better error handling with context
// 5. Shared utilities to reduce code duplication

pub mod common;
pub mod ttest;
pub mod chi2;
pub mod effect;
pub mod mann_whitney;
pub mod ks;
pub mod anova;          // NEW
pub mod normality;      // NEW
pub mod correlation;    // NEW
pub mod variance_tests; // NEW

// Re-export for convenience
pub use common::{Alternative, reject_nonfinite, mean, var_sample, rankdata_average};
