use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use statrs::distribution::{ContinuousCDF, Normal};

use super::common::{mean, reject_nonfinite};

/// Jarque-Bera normality test.
/// 
/// Tests whether sample skewness and kurtosis match a normal distribution.
#[pyfunction]
pub fn jarque_bera_np(
    py: Python<'_>,
    x: numpy::PyReadonlyArray1<f64>,
) -> PyResult<Py<PyDict>> {
    let xs = x.as_slice()?;
    
    let n = xs.len();
    if n < 4 {
        return Err(PyValueError::new_err("jarque_bera requires n >= 4"));
    }
    
    reject_nonfinite(xs, "x")?;
    
    let m = mean(xs);
    
    // Compute central moments
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    
    for &v in xs {
        let d = v - m;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    
    let nf = n as f64;
    m2 /= nf;
    m3 /= nf;
    m4 /= nf;
    
    if m2 <= 0.0 {
        return Err(PyValueError::new_err("variance is zero"));
    }
    
    let skewness = m3 / m2.powf(1.5);
    let kurtosis = m4 / (m2 * m2);
    let excess_kurtosis = kurtosis - 3.0;
    
    // JB statistic
    let jb = (nf / 6.0) * (skewness * skewness + excess_kurtosis * excess_kurtosis / 4.0);
    
    // Chi-square with df=2
    use statrs::distribution::ChiSquared;
    let dist = ChiSquared::new(2.0).unwrap();
    let p = (1.0 - dist.cdf(jb)).clamp(0.0, 1.0);
    
    let d = PyDict::new_bound(py);
    d.set_item("statistic", jb)?;
    d.set_item("pvalue", p)?;
    d.set_item("skewness", skewness)?;
    d.set_item("kurtosis", kurtosis)?;
    Ok(d.unbind())
}

/// Anderson-Darling normality test.
///
/// Computes the Anderson-Darling statistic for normality.
#[pyfunction]
pub fn anderson_darling_np(
    py: Python<'_>,
    x: numpy::PyReadonlyArray1<f64>,
) -> PyResult<Py<PyDict>> {
    let xs = x.as_slice()?;
    
    let n = xs.len();
    if n < 2 {
        return Err(PyValueError::new_err("anderson_darling requires n >= 2"));
    }
    
    reject_nonfinite(xs, "x")?;
    
    // Standardize
    let m = mean(xs);
    let mut v = 0.0;
    for &x in xs {
        let d = x - m;
        v += d * d;
    }
    v /= (n - 1) as f64;
    
    if v <= 0.0 {
        return Err(PyValueError::new_err("variance is zero"));
    }
    
    let std = v.sqrt();
    
    let mut z: Vec<f64> = xs.iter().map(|&x| (x - m) / std).collect();
    z.sort_by(|a, b| a.total_cmp(b));
    
    let norm = Normal::new(0.0, 1.0).unwrap();
    
    let mut a2 = 0.0;
    let nf = n as f64;
    
    for i in 0..n {
        let z_i = z[i];
        let z_ni = z[n - 1 - i];
        
        let if64 = (i + 1) as f64;
        
        let cdf_i = norm.cdf(z_i).max(1e-300).min(1.0 - 1e-300);
        let sf_ni = (1.0 - norm.cdf(z_ni)).max(1e-300).min(1.0 - 1e-300);
        
        a2 += (2.0 * if64 - 1.0) * (cdf_i.ln() + sf_ni.ln());
    }
    
    a2 = -nf - a2 / nf;
    
    // Correction for finite sample (Stephens 1974)
    let a2_star = a2 * (1.0 + 4.0 / nf - 25.0 / (nf * nf));
    
    // Critical values are fixed for normality test
    // Just return statistic for now
    let d = PyDict::new_bound(py);
    d.set_item("statistic", a2_star)?;
    // p-value approximation would require interpolation from critical value tables
    Ok(d.unbind())
}
