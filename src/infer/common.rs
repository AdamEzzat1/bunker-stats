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

/// Upper tail `P(X >= stat)` of a chi-square with `df` degrees of freedom.
///
/// Evaluated as the regularized upper incomplete gamma function, never as
/// `1 - cdf(stat)`: that subtraction is exactly 0.0 once the tail drops below
/// ~1e-16, so every strongly significant test reported p = 0.
///
/// NaN in gives NaN out; a statistic at or below zero (rounding in a
/// statistic that is zero in exact arithmetic) has p = 1.
pub fn chi2_sf(stat: f64, df: f64) -> f64 {
    use statrs::function::gamma::gamma_ur;

    if stat.is_nan() || !(df > 0.0) || df.is_infinite() {
        return f64::NAN;
    }
    if stat <= 0.0 {
        return 1.0;
    }
    if stat.is_infinite() {
        return 0.0;
    }
    gamma_ur(0.5 * df, 0.5 * stat).clamp(0.0, 1.0)
}

/// Upper tail `P(F >= f)` of an F distribution with `(df1, df2)` degrees of
/// freedom: `I_x(df2/2, df1/2)` with `x = df2 / (df2 + df1 f)`.
///
/// `x` is formed directly. Writing it as `1 - df1 f / (df1 f + df2)` (as
/// statrs' own `sf` does) rounds to 0 for a large statistic, and `1 - cdf(f)`
/// cannot go below ~1e-16 at all.
pub fn f_sf(f: f64, df1: f64, df2: f64) -> f64 {
    use statrs::function::beta::beta_reg;

    if f.is_nan() || !(df1 > 0.0) || !(df2 > 0.0) || df1.is_infinite() || df2.is_infinite() {
        return f64::NAN;
    }
    if f <= 0.0 {
        return 1.0;
    }
    let denom = df2 + df1 * f;
    if !denom.is_finite() {
        return 0.0;
    }
    let x = df2 / denom;
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    beta_reg(0.5 * df2, 0.5 * df1, x).clamp(0.0, 1.0)
}

/// Two-sided p-value of a standard-normal statistic, `2 P(Z >= |z|)`, from
/// `erfc` rather than `2 (1 - cdf(|z|))`, which is 0 for |z| > ~8.3.
///
/// `libm::erfc`, as in `kernels::dist::normal`: the statrs one is only good
/// to about 1e-11 relative (0.04999999999775 at z = 1.96).
pub fn norm_two_sided_pvalue(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    libm::erfc(z.abs() / std::f64::consts::SQRT_2).clamp(0.0, 1.0)
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

#[cfg(test)]
mod tail_tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    // References: mpmath at 60 digits (regularized incomplete gamma / beta,
    // erfc), independent of statrs; each was also checked against scipy.
    const CHI2: &[(f64, f64, f64)] = &[
        (0.5, 1.0, 0.47950012218695346),
        (3.0, 2.0, 0.22313016014842983),
        (10.0, 5.0, 0.075235246146512179),
        (40.0, 2.0, 2.0611536224385578e-9),
        (100.0, 1.0, 1.5239706048321052e-23),
        (200.0, 10.0, 1.6139305336977305e-37),
        (800.0, 2.0, 1.9151695967140057e-174),
        (1000.0, 50.0, 7.188706215514835e-177),
        (1400.0, 3.0, 2.9456193610163087e-303),
        (30.0, 40.0, 0.87521878496747518),
        (0.001, 7.0, 0.9999999999997598),
        (500.0, 400.0, 0.00048221275959343374),
        (2000.0, 1000.0, 4.1436785914549917e-69),
    ];

    const F: &[(f64, f64, f64, f64)] = &[
        (1.0, 3.0, 10.0, 0.43233720302169707),
        (4.0, 2.0, 30.0, 0.028844623295696826),
        (50.0, 2.0, 117.0, 2.0229189466309584e-16),
        (500.0, 2.0, 117.0, 4.761702792428355e-58),
        (5000.0, 2.0, 117.0, 4.9356634371962726e-114),
        (1000000.0, 1.0, 5.0, 1.8980131131979715e-14),
        (1000000000000.0, 3.0, 4.0, 3.333333333322963e-24),
        (30.0, 59.0, 199.0, 1.1401187408079199e-72),
        (0.2, 5.0, 5.0, 0.94903026058507082),
        (3.0, 100.0, 1.0, 0.43499952209983999),
        (1.2, 59.0, 2.0, 0.56035168591399539),
        (0.001, 4.0, 9.0, 0.99999756025749564),
        (2.5, 400.0, 500.0, 2.3734602832561327e-22),
    ];

    const NORM2: &[(f64, f64)] = &[
        (0.0, 1.0),
        (0.5, 0.61707507745197379),
        (1.959963984540054, 0.050000000000000022),
        (5.0, 5.7330314375838782e-7),
        (8.0, 1.2441921148543568e-15),
        (8.3, 1.0411139489780508e-16),
        (10.0, 1.5239706048321052e-23),
        (20.0, 5.5072482372124674e-89),
        (30.0, 9.8134278542963741e-198),
        (37.0, 1.1451142445049154e-299),
        (-12.5, 7.4651285977554268e-36),
    ];

    #[test]
    fn chi2_sf_matches_reference_into_the_far_tail() {
        for &(x, df, sf) in CHI2 {
            let p = chi2_sf(x, df);
            assert!(rel(p, sf) < 1e-10, "chi2_sf({x}, {df}) = {p}, want {sf}");
        }
    }

    #[test]
    fn f_sf_matches_reference_into_the_far_tail() {
        for &(f, d1, d2, sf) in F {
            let p = f_sf(f, d1, d2);
            assert!(rel(p, sf) < 1e-10, "f_sf({f}, {d1}, {d2}) = {p}, want {sf}");
        }
    }

    #[test]
    fn norm_two_sided_matches_reference_into_the_far_tail() {
        for &(z, want) in NORM2 {
            let p = norm_two_sided_pvalue(z);
            assert!(rel(p, want) < 1e-12, "z = {z}: {p}, want {want}");
        }
    }

    #[test]
    fn tails_handle_every_statistic_a_caller_can_produce() {
        assert_eq!(chi2_sf(0.0, 3.0), 1.0);
        assert_eq!(chi2_sf(-1e-12, 3.0), 1.0);
        assert_eq!(chi2_sf(f64::INFINITY, 3.0), 0.0);
        assert!(chi2_sf(f64::NAN, 3.0).is_nan());
        assert!(chi2_sf(1.0, 0.0).is_nan());
        assert!(chi2_sf(1.0, f64::NAN).is_nan());
        // true value 3e-2651: underflows, and must do so to 0, not NaN
        assert_eq!(f_sf(1e8, 2.0, 1000.0), 0.0);
        assert_eq!(chi2_sf(1e6, 4.0), 0.0);

        assert_eq!(f_sf(0.0, 2.0, 9.0), 1.0);
        assert_eq!(f_sf(f64::INFINITY, 2.0, 9.0), 0.0);
        assert_eq!(f_sf(f64::MAX, 2.0, 9.0), 0.0);
        assert!(f_sf(f64::NAN, 2.0, 9.0).is_nan());
        assert!(f_sf(1.0, 0.0, 9.0).is_nan());

        assert_eq!(norm_two_sided_pvalue(f64::INFINITY), 0.0);
        assert_eq!(norm_two_sided_pvalue(f64::NEG_INFINITY), 0.0);
        assert!(norm_two_sided_pvalue(f64::NAN).is_nan());
    }

    /// The lower tail of F(d1, d2) at f is the upper tail of F(d2, d1) at 1/f,
    /// so the two must sum to one wherever neither is negligible.
    #[test]
    fn f_tails_are_complementary() {
        for &(f, d1, d2) in &[(0.3, 4.0, 7.0), (1.0, 59.0, 2.0), (2.2, 100.0, 1.0), (5.0, 3.0, 3.0)] {
            let total = f_sf(f, d1, d2) + f_sf(1.0 / f, d2, d1);
            assert!((total - 1.0).abs() < 1e-12, "f={f} ({d1}, {d2}): {total}");
        }
    }
}
