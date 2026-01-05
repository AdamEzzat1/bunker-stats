// src/kernels/tsa/mod.rs

pub(super) fn demean(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    x.iter().map(|v| v - mean).collect()
}

pub(super) fn variance(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let mut ss = 0.0;
    for &v in x {
        let d = v - mean;
        ss += d * d;
    }
    ss / ((n - 1) as f64)
}

pub mod stationarity;
pub mod diagnostics;
pub mod acf_pacf;
pub mod spectral;
pub mod rolling;
pub mod rolling_autocorr;

