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

// v0.3.1: additional inference functions
pub mod paired;          // Paired t-test
pub mod p_adjust;        // Multiple-testing correction
pub mod proportion;      // Proportion z-tests
pub mod corr_ci;         // Correlation CI (Fisher z)
pub mod var_ci;          // Variance CI (chi-square)
pub mod odds_ratio;      // Odds ratio for 2x2 tables
pub mod effect_nonparam; // rank_biserial, cliffs_delta, eta/omega squared, normality_summary

// Re-export for convenience
