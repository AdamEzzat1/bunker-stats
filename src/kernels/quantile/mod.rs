// src/kernels/quantile/mod.rs
//
// Quantile/selection kernels (exact implementations will be extracted here).
// Keep this module compile-safe while you migrate functions out of lib.rs.

pub(crate) mod select;
pub(crate) mod percentile;
pub(crate) mod iqr;
pub(crate) mod winsor;
