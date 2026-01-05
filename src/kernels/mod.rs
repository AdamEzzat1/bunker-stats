// src/kernels/mod.rs
//
// Top-level kernel modules.
// Add more here only when the directories exist.

pub(crate) mod rolling;
pub(crate) mod quantile;
pub(crate) mod matrix;
pub(crate) mod robust;
pub mod resampling;
pub mod tsa;
pub mod dist;

// src/kernels/rolling/mod.r
