/// Robust statistics module - OPTIMIZED
///
/// High-performance robust estimators with:
/// - O(n) median/MAD via select_nth_unstable (not O(n log n) sort)
/// - Fused median+MAD kernel for common case
/// - Workspace API for zero-allocation paths
/// - Enum-based dispatch (no string parsing in hot paths)
/// - Hybrid rolling algorithms based on window size
///
/// Performance improvements over v0.2.9:
/// - 2-5x faster median/MAD at large n
/// - Zero string overhead in RobustStats class
/// - Adaptive rolling median (cache-friendly for small windows)
/// - Fused pipelines eliminate redundant allocations

pub mod policy;
pub mod extended;
pub mod fit;
pub mod rolling;
pub mod pyrobust;

// Re-export PyO3 bindings for lib.rs
pub use pyrobust::{RobustStats, robust_fit, robust_score, rolling_median};