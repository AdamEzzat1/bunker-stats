use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Alternative hypothesis specification.
/// Matches SciPy semantics.
#[derive(Clone, Copy, Debug)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

impl Alternative {
    /// Parse alternative string from Python API.
    pub fn from_str(s: &str) -> PyResult<Self> {
        match s {
            "two-sided" | "two_sided" => Ok(Self::TwoSided),
            "less" => Ok(Self::Less),
            "greater" => Ok(Self::Greater),
            _ => Err(PyValueError::new_err(
                "alternative must be one of: 'two-sided', 'less', 'greater'",
            )),
        }
    }
}

/// Reject NaN / Inf inputs (v0.3 contract).
#[inline]
pub fn reject_nonfinite(xs: &[f64], name: &str) -> PyResult<()> {
    if xs.iter().any(|v| !v.is_finite()) {
        Err(PyValueError::new_err(format!(
            "{name} contains NaN or Inf; bunker-stats v0.3 nan_policy is 'reject'"
        )))
    } else {
        Ok(())
    }
}

/// Validate sample size for statistical test.
#[inline]
pub fn validate_sample_size(n: usize, min_n: usize, test_name: &str) -> PyResult<()> {
    if n < min_n {
        return Err(PyValueError::new_err(format!(
            "{} requires n >= {}, got n = {}",
            test_name, min_n, n
        )));
    }
    Ok(())
}

/// Compute the mean of a slice (no NaN handling).
#[inline]
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    
    let mut s = 0.0;
    for &v in xs {
        s += v;
    }
    s / (xs.len() as f64)
}

/// Compute the mean with Kahan summation for better numerical stability.
#[inline]
pub fn mean_kahan(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    
    let mut sum = 0.0;
    let mut c = 0.0; // Compensation for lost low-order bits
    
    for &v in xs {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    sum / (xs.len() as f64)
}

/// Compute sample variance (ddof = 1) with Kahan summation.
///
/// Assumes:
/// - xs.len() >= 2
/// - mean already computed
#[inline]
pub fn var_sample(xs: &[f64], mean: f64) -> f64 {
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }

    let mut ss = 0.0;
    let mut c = 0.0; // Kahan compensation
    
    for &v in xs {
        let d = v - mean;
        let term = d * d;
        
        let y = term - c;
        let t = ss + y;
        c = (t - ss) - y;
        ss = t;
    }

    ss / ((n - 1) as f64)
}

/// Compute sample standard deviation (ddof = 1).
#[inline]
pub fn std_sample(xs: &[f64], mean: f64) -> f64 {
    var_sample(xs, mean).sqrt()
}

/// Compute average ranks with tie handling.
///
/// Returns a vector of ranks (1-indexed, averaged for ties).
/// This is the standard "average" method for ranking.
pub fn rankdata_average(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    
    let mut indexed: Vec<(usize, f64)> = data.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    
    // Sort by value
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    
    while i < n {
        let v = indexed[i].1;
        let mut j = i + 1;
        
        // Find end of tied group
        while j < n && indexed[j].1 == v {
            j += 1;
        }
        
        // Tied group is [i, j)
        // Average rank formula: (first_rank + last_rank) / 2
        // where first_rank = i+1, last_rank = j
        let avg_rank = ((i + 1) + j) as f64 / 2.0;
        
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        
        i = j;
    }
    
    ranks
}

/// Compute median of a slice (will sort a copy).
pub fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Compute sum of squared deviations from mean.
#[inline]
pub fn sum_of_squares(xs: &[f64], mean: f64) -> f64 {
    let mut ss = 0.0;
    for &v in xs {
        let d = v - mean;
        ss += d * d;
    }
    ss
}

/// Compute sum of squared deviations from mean with Kahan summation.
#[inline]
pub fn sum_of_squares_kahan(xs: &[f64], mean: f64) -> f64 {
    let mut ss = 0.0;
    let mut c = 0.0;
    
    for &v in xs {
        let d = v - mean;
        let term = d * d;
        
        let y = term - c;
        let t = ss + y;
        c = (t - ss) - y;
        ss = t;
    }
    
    ss
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mean() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&xs), 3.0);
    }
    
    #[test]
    fn test_var_sample() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = mean(&xs);
        let v = var_sample(&xs, m);
        // Sample variance with ddof=1 should be 2.5
        assert!((v - 2.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_rankdata_average() {
        let xs = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
        let ranks = rankdata_average(&xs);
        
        // Expected ranks:
        // 1.0 -> rank 1
        // 2.0, 2.0 -> ranks 2,3 -> avg 2.5
        // 3.0, 3.0, 3.0 -> ranks 4,5,6 -> avg 5.0
        // 4.0 -> rank 7
        let expected = vec![1.0, 2.5, 2.5, 5.0, 5.0, 5.0, 7.0];
        
        for i in 0..xs.len() {
            assert!((ranks[i] - expected[i]).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_median() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(median(&xs), 3.0);
        
        let ys = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(median(&ys), 2.5);
    }
}
