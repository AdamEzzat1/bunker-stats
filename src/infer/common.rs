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

/// Upper tail `P(T > t)` of a Student-t with `df` degrees of freedom, `t >= 0`.
///
/// Evaluated as the lower tail at `-t` (the distribution is symmetric), never
/// as `1 - cdf(t)`: that subtraction cancels to exactly 0.0 once the tail
/// drops below ~1e-16.
///
/// The incomplete beta in statrs also returns exactly 0 once
/// `x = df / (df + t^2)` is below machine epsilon, which for small `df`
/// happens while the true tail is still far from underflow (df = 1, t = 7e7:
/// 4.5e-9). There the tail is taken from the leading terms of the series
/// `I_x(a, 1/2) = x^a / (a B(a, 1/2)) * (1 + a x / (2 (a + 1)) + O(x^2))`,
/// `a = df / 2`, evaluated in log space.
fn t_upper_tail(t: f64, df: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, StudentsT};
    use statrs::function::gamma::ln_gamma;

    if t.is_infinite() {
        return 0.0;
    }
    let x = df / (df + t * t);
    if x < 1e-10 {
        let a = 0.5 * df;
        let ln_beta = ln_gamma(a) + ln_gamma(0.5) - ln_gamma(a + 0.5);
        let ln_lead = a * x.ln() - a.ln() - ln_beta;
        return 0.5 * ln_lead.exp() * (1.0 + a * x / (2.0 * (a + 1.0)));
    }
    StudentsT::new(0.0, 1.0, df).unwrap().cdf(-t)
}

/// p-value of a Student-t statistic with `df` degrees of freedom.
///
/// Handles every statistic a caller can produce: NaN gives NaN, and +/-Inf
/// (zero standard error) gives 0 or 1 according to the alternative -- an
/// infinite statistic in the direction *opposite* to a one-sided alternative
/// is no evidence at all, so its p-value is 1, not 0.
pub fn t_pvalue(t: f64, df: f64, alt: Alternative) -> f64 {
    if t.is_nan() || df.is_nan() {
        return f64::NAN;
    }
    let tail = t_upper_tail(t.abs(), df);
    let p = match alt {
        Alternative::TwoSided => 2.0 * tail,
        Alternative::Less => {
            if t < 0.0 { tail } else { 1.0 - tail }
        }
        Alternative::Greater => {
            if t > 0.0 { tail } else { 1.0 - tail }
        }
    };
    p.clamp(0.0, 1.0)
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
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    
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
    sorted.sort_by(|a, b| a.total_cmp(b));
    
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

#[cfg(test)]
mod t_pvalue_tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    /// `scipy.stats.t.sf(t, df)`. The rows with t >= 7e7 at small df are the
    /// ones the statrs incomplete beta returns exactly 0 for.
    const SF: &[(f64, f64, f64)] = &[
        (1e3, 1.0, 0.00031830978008055887),
        (1e6, 1.0, 3.1830988618368455e-07),
        (7e7, 1.0, 4.547284088339866e-09),
        (1e8, 1.0, 3.1830988618379066e-09),
        (1e10, 1.0, 3.1830988618379065e-11),
        (1e3, 2.0, 4.9999925000125e-07),
        (1e6, 2.0, 4.999999999992501e-13),
        (7e7, 2.0, 1.0204081632653057e-16),
        (1e8, 2.0, 4.999999999999999e-17),
        (1e10, 2.0, 5e-21),
        (1e3, 3.0, 1.1026538212882999e-09),
        (1e6, 3.0, 1.1026577908396162e-18),
        (7e7, 3.0, 3.214745745899666e-24),
        (1e10, 3.0, 1.1026577908435765e-30),
        (1e3, 5.0, 9.490065565989865e-15),
        (1e6, 5.0, 9.490167245460652e-30),
        (7e7, 5.0, 5.6465563429298997e-39),
        (1e10, 5.0, 9.490167245562447e-50),
        (1e3, 10.0, 1.2304123550866324e-26),
        (1e6, 10.0, 1.2304687499436141e-56),
        (1e8, 10.0, 1.2304687500000114e-76),
        (1e10, 10.0, 1.230468750000001e-96),
        (1e3, 30.0, 1.0360017415558858e-69),
        (1e6, 30.0, 1.0364534648044268e-159),
        (1e8, 30.0, 1.0364534652562334e-219),
        (1e10, 30.0, 1.0364534652561783e-279),
    ];

    #[test]
    fn far_tail_matches_scipy() {
        for &(t, df, sf) in SF {
            let greater = t_pvalue(t, df, Alternative::Greater);
            assert!(rel(greater, sf) < 1e-9, "greater t={t} df={df}: {greater} vs {sf}");
            let less = t_pvalue(-t, df, Alternative::Less);
            assert!(rel(less, sf) < 1e-9, "less t={t} df={df}: {less} vs {sf}");
            for sign in [1.0, -1.0] {
                let two = t_pvalue(sign * t, df, Alternative::TwoSided);
                assert!(rel(two, 2.0 * sf) < 1e-9, "two-sided t={t} df={df}: {two}");
            }
            // The opposite one-sided tail is the complement.
            assert!((t_pvalue(t, df, Alternative::Less) - (1.0 - sf)).abs() < 1e-15);
        }
    }

    /// The series branch and the statrs branch must agree where they meet
    /// (x = df / (df + t^2) = 1e-10): evaluate both formulas at the same t.
    #[test]
    fn branches_agree_at_the_switch() {
        use statrs::distribution::{ContinuousCDF, StudentsT};
        for df in [1.0f64, 2.0, 5.0, 30.0] {
            // just inside the series branch
            let t = (df * (1.0 / 0.99e-10 - 1.0)).sqrt();
            let series = t_upper_tail(t, df);
            let direct = StudentsT::new(0.0, 1.0, df).unwrap().cdf(-t);
            assert!(series > 0.0 && direct > 0.0, "df={df}");
            assert!(rel(series, direct) < 1e-9, "df={df}: {series} vs {direct}");
        }
    }

    #[test]
    fn ordinary_values() {
        // scipy.stats.t.sf(2.0, 10)
        let sf = 0.036694017385370196;
        assert!(rel(t_pvalue(2.0, 10.0, Alternative::Greater), sf) < 1e-12);
        assert!(rel(t_pvalue(2.0, 10.0, Alternative::TwoSided), 2.0 * sf) < 1e-12);
        assert!(rel(t_pvalue(-2.0, 10.0, Alternative::Less), sf) < 1e-12);
        assert!(rel(t_pvalue(2.0, 10.0, Alternative::Less), 1.0 - sf) < 1e-12);
        assert_eq!(t_pvalue(0.0, 7.0, Alternative::TwoSided), 1.0);
        assert_eq!(t_pvalue(0.0, 7.0, Alternative::Less), 0.5);
        assert_eq!(t_pvalue(0.0, 7.0, Alternative::Greater), 0.5);
    }

    #[test]
    fn infinite_and_nan_statistics() {
        let inf = f64::INFINITY;
        assert_eq!(t_pvalue(inf, 3.0, Alternative::TwoSided), 0.0);
        assert_eq!(t_pvalue(-inf, 3.0, Alternative::TwoSided), 0.0);
        assert_eq!(t_pvalue(inf, 3.0, Alternative::Greater), 0.0);
        assert_eq!(t_pvalue(inf, 3.0, Alternative::Less), 1.0);
        assert_eq!(t_pvalue(-inf, 3.0, Alternative::Less), 0.0);
        assert_eq!(t_pvalue(-inf, 3.0, Alternative::Greater), 1.0);
        for alt in [Alternative::TwoSided, Alternative::Less, Alternative::Greater] {
            assert!(t_pvalue(f64::NAN, 3.0, alt).is_nan());
        }
        // t^2 overflows: the tail underflows to 0 instead of producing NaN.
        assert_eq!(t_pvalue(1e200, 4.0, Alternative::Greater), 0.0);
    }
}
