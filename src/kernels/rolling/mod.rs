// src/kernels/rolling/mod.rs

pub(crate) mod engine;
pub(crate) mod mean;
pub(crate) mod std;
pub(crate) mod zscore;
pub(crate) mod var;
pub(crate) mod axis0;
pub(crate) mod covcorr;

// NEW v0.2.9: Multi-stat rolling modules
pub mod bounds;
pub mod config;
pub mod masks;
pub mod multi;
pub mod multi_axis0;