use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use statrs::distribution::{ContinuousCDF, Exp, Normal, Uniform};

use super::common::{reject_nonfinite, Alternative};

enum NamedDist {
    Norm(Normal),
    Uniform(Uniform),
    Expon { loc: f64, exp: Exp },
}

impl NamedDist {
    fn cdf(&self, x: f64) -> f64 {
        match self {
            NamedDist::Norm(d) => d.cdf(x),
            NamedDist::Uniform(d) => d.cdf(x),
            NamedDist::Expon { loc, exp } => {
                let z = x - *loc;
                if z <= 0.0 {
                    0.0
                } else {
                    exp.cdf(z)
                }
            }
        }
    }
}

fn parse_named_dist(name: &str, params: &[f64]) -> PyResult<NamedDist> {
    match name {
        "norm" | "normal" => {
            if params.len() != 2 {
                return Err(PyValueError::new_err("cdf='norm' expects params=[loc, scale]"));
            }
            let loc = params[0];
            let scale = params[1];
            if !loc.is_finite() || !scale.is_finite() {
                return Err(PyValueError::new_err("params must be finite"));
            }
            if scale <= 0.0 {
                return Err(PyValueError::new_err("scale must be > 0"));
            }
            Ok(NamedDist::Norm(
                Normal::new(loc, scale)
                    .map_err(|_| PyValueError::new_err("invalid parameters for normal"))?,
            ))
        }
        "uniform" => {
            if params.len() != 2 {
                return Err(PyValueError::new_err(
                    "cdf='uniform' expects params=[loc, scale]",
                ));
            }
            let loc = params[0];
            let scale = params[1];
            if !loc.is_finite() || !scale.is_finite() {
                return Err(PyValueError::new_err("params must be finite"));
            }
            if scale <= 0.0 {
                return Err(PyValueError::new_err("scale must be > 0"));
            }
            Ok(NamedDist::Uniform(
                Uniform::new(loc, loc + scale)
                    .map_err(|_| PyValueError::new_err("invalid parameters for uniform"))?,
            ))
        }
        "expon" | "exponential" => {
            if params.len() != 2 {
                return Err(PyValueError::new_err("cdf='expon' expects params=[loc, scale]"));
            }
            let loc = params[0];
            let scale = params[1];
            if !loc.is_finite() || !scale.is_finite() {
                return Err(PyValueError::new_err("params must be finite"));
            }
            if scale <= 0.0 {
                return Err(PyValueError::new_err("scale must be > 0"));
            }
            let rate = 1.0 / scale;
            Ok(NamedDist::Expon {
                loc,
                exp: Exp::new(rate)
                    .map_err(|_| PyValueError::new_err("invalid parameters for exponential"))?,
            })
        }
        _ => Err(PyValueError::new_err("cdf must be one of: 'norm', 'uniform', 'expon'")),
    }
}

/// Asymptotic two-sided Kolmogorov distribution survival function with Stephens/Massey correction.
fn ks_pvalue_asymp_two_sided_stephens(n: usize, d: f64) -> f64 {
    if !d.is_finite() || n == 0 {
        return f64::NAN;
    }
    if d <= 0.0 {
        return 1.0;
    }
    if d >= 1.0 {
        return 0.0;
    }

    let nf = n as f64;
    let sqrtn = nf.sqrt();
    // Stephens correction improves agreement for finite n.
    let x = (sqrtn + 0.12 + 0.11 / sqrtn) * d;
    let x2 = x * x;

    // Q_KS(x) = 2 * sum_{k=1}^∞ (-1)^{k-1} exp(-2 k^2 x^2)
    let mut sum = 0.0;
    let mut c = 0.0; // Kahan compensation

    for k in 1..=100000 {
        let term = (-2.0 * (k as f64).powi(2) * x2).exp();
        let signed = if k % 2 == 1 { term } else { -term };

        let y = signed - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;

        if term < 1e-20 {
            break;
        }
    }

    (2.0 * sum).clamp(0.0, 1.0)
}

/// Below this two-sided p-value the tail formula `2 P(D+ >= d)` is used.
///
/// `P(D >= d) = 2 P(D+ >= d) - P(D+ >= d and D- >= d)`. The joint term is zero
/// for `d >= 1/2` and otherwise of the order of the square of the one-sided
/// tail, so at p = 0.01 dropping it changes the result by about 1e-7 relative
/// (measured against an exact evaluation; see the tests).
const KS_TAIL_SWITCH: f64 = 0.01;

/// Two-sided KS survival function (p-value for D_n >= d).
///
/// * small p-values (and every `d >= 1/2`): twice the exact one-sided tail.
///   That is an O(n) sum of positive terms, so it stays accurate down to the
///   smallest representable p-value; the matrix method below can only form
///   `1 - cdf`, which cannot go under ~1e-16, and costs O((n d)^3 log n).
/// * otherwise, `n <= 10_000`: Durbin's exact matrix formula.
/// * otherwise: the Stephens-corrected asymptotic series (within about 0.5%
///   of the exact value for p >= 0.01 at these sizes).
fn ks_pvalue_two_sided_finite(n: usize, d: f64) -> f64 {
    if !d.is_finite() || n == 0 {
        return f64::NAN;
    }
    if d <= 0.0 {
        return 1.0;
    }
    if d >= 1.0 {
        return 0.0;
    }

    // D_n >= 1/(2n) always (the empirical CDF jumps by 1/n), so P(D_n >= d) = 1.
    if d <= 0.5 / (n as f64) {
        return 1.0;
    }

    let twice_one_sided = 2.0 * ks_pvalue_one_sided_exact(n, d);
    if d >= 0.5 || twice_one_sided <= KS_TAIL_SWITCH {
        return twice_one_sided.clamp(0.0, 1.0);
    }

    // Moderate p-values only from here on, so n d^2 is bounded and the matrix
    // below has at most a few hundred rows.
    if n > 10_000 {
        return ks_pvalue_asymp_two_sided_stephens(n, d);
    }
    ks_pvalue_two_sided_durbin(n, d)
}

/// Durbin's matrix formula for the two-sided KS distribution, as implemented
/// by Marsaglia, Tsang & Wang (2003). Returns the survival function.
fn ks_pvalue_two_sided_durbin(n: usize, d: f64) -> f64 {
    use statrs::function::gamma::ln_gamma;

    let nf = n as f64;

    // k = floor(n*d) + 1; m = 2k - 1; h = k - n*d
    let k = (nf * d).floor() as usize + 1;
    let m = 2 * k - 1;
    let h = (k as f64) - nf * d;

    // inv_fact[t] = 1/t! for t=0..=m
    let mut inv_fact = vec![0.0f64; m + 1];
    inv_fact[0] = 1.0;
    for t in 1..=m {
        inv_fact[t] = inv_fact[t - 1] / (t as f64);
    }

    // H matrix (row-major)
    let mut hmat = vec![0.0f64; m * m];
    let idx = |i: usize, j: usize, m: usize| -> usize { i * m + j };

    for i in 0..m {
        for j in 0..m {
            // t = i - j + 1
            if i + 1 >= j {
                let t = i + 1 - j;
                hmat[idx(i, j, m)] = inv_fact[t];
            }
        }
    }

    // First column adjustment: H[i,0] -= h^{i+1} / (i+1)!
    let mut hp = h;
    for i in 0..m {
        hmat[idx(i, 0, m)] -= hp * inv_fact[i + 1];
        hp *= h;
    }

    // Last row adjustment: H[m-1,j] -= h^{m-j} / (m-j)!
    for j in 0..m {
        let t = m - j;
        hmat[idx(m - 1, j, m)] -= h.powi(t as i32) * inv_fact[t];
    }

    // Corner correction: H[m-1,0] += max(0, 2h-1)^m / m!
    let two_h_minus_one = 2.0 * h - 1.0;
    if two_h_minus_one > 0.0 {
        hmat[idx(m - 1, 0, m)] += two_h_minus_one.powi(m as i32) * inv_fact[m];
    }

    fn mat_mul_scaled(a: &[f64], b: &[f64], m: usize) -> (Vec<f64>, f64) {
        // Kahan-compensated accumulation per output cell to reduce cancellation error.
        let mut c = vec![0.0f64; m * m];
        let mut comp = vec![0.0f64; m * m];

        for i in 0..m {
            for k in 0..m {
                let aik = a[i * m + k];
                if aik == 0.0 {
                    continue;
                }
                let bk = &b[k * m..k * m + m];
                let row_base = i * m;
                for j in 0..m {
                    let idx = row_base + j;
                    let prod = aik * bk[j];

                    // Kahan summation: c[idx] += prod
                    let y = prod - comp[idx];
                    let t = c[idx] + y;
                    comp[idx] = (t - c[idx]) - y;
                    c[idx] = t;
                }
            }
        }

        // scale down if needed to avoid overflow/underflow during powering
        let mut max_abs = 0.0f64;
        for &v in &c {
            let av = v.abs();
            if av > max_abs {
                max_abs = av;
            }
        }
        
        if max_abs == 0.0 {
            return (c, 0.0);
        }
        
        // More aggressive scaling thresholds to prevent overflow
        let (target_min, target_max) = (1e-100, 1e100);
        if max_abs > target_max {
            let s = max_abs / target_max;
            for v in &mut c {
                *v /= s;
            }
            return (c, s.ln());
        } else if max_abs < target_min && max_abs > 0.0 {
            let s = max_abs / target_min;
            for v in &mut c {
                *v /= s;
            }
            return (c, s.ln());
        }
        
        (c, 0.0)
    }

    fn mat_pow_scaled(mut base: Vec<f64>, mut exp: usize, m: usize) -> (Vec<f64>, f64) {
        // identity
        let mut res = vec![0.0f64; m * m];
        for i in 0..m {
            res[i * m + i] = 1.0;
        }
        // `res` and `base` are each stored divided by their own scale:
        //   true_res  = res  * exp(res_log)
        //   true_base = base * exp(base_log)
        // A rescaled `base` is squared and reused, so its log-scale DOUBLES on
        // every squaring and is ADDED to the result's on every multiply.
        // Keeping one running sum instead dropped those factors, and every
        // p-value was wrong as soon as a single rescale fired (n >~ 550).
        let mut res_log = 0.0f64;
        let mut base_log = 0.0f64;

        while exp > 0 {
            if exp & 1 == 1 {
                let (tmp, l) = mat_mul_scaled(&res, &base, m);
                res = tmp;
                res_log += base_log + l;
            }
            exp >>= 1;
            if exp > 0 {
                let (tmp, l) = mat_mul_scaled(&base, &base, m);
                base = tmp;
                base_log = 2.0 * base_log + l;
            }
        }
        (res, res_log)
    }

    let (q, log_scale) = mat_pow_scaled(hmat, n, m);

    // Durbin/Marsaglia: CDF = (H^n)[k-1,k-1] * n! / n^n
    let elem = q[(k - 1) * m + (k - 1)];
    if elem.is_nan() {
        return f64::NAN;
    }
    let ln_cdf = elem.abs().ln() + log_scale + ln_gamma(nf + 1.0) - nf * nf.ln();
    if elem <= 0.0 {
        // The CDF is a probability. Exactly zero, or negative by rounding when
        // the true value is negligible, means p = 1. A non-positive element of
        // any real size is a broken computation and must not be reported as
        // "no evidence" -- which is what the old `elem <= 0 => 1.0` did.
        return if ln_cdf < -23.0 { 1.0 } else { f64::NAN };
    }
    let cdf = ln_cdf.exp().clamp(0.0, 1.0);
    (1.0 - cdf).clamp(0.0, 1.0)
}

/// Exact one-sided KS survival function (Birnbaum-Tingey / Smirnov):
///
///   P(D_n^+ >= d) = d * sum_{j=0}^{floor(n(1-d))} C(n,j) (j/n + d)^(j-1) (1 - d - j/n)^(n-j)
///
/// This is what SciPy's `ksone.sf` evaluates for the one-sample one-sided
/// statistics. Binomial coefficients and powers are combined in log space for
/// stability at large n.
fn ks_pvalue_one_sided_exact(n: usize, d: f64) -> f64 {
    use statrs::function::gamma::ln_gamma;

    if !d.is_finite() || n == 0 {
        return f64::NAN;
    }
    if d <= 0.0 {
        return 1.0;
    }
    if d >= 1.0 {
        return 0.0;
    }

    let nf = n as f64;
    let ln_n_fact = ln_gamma(nf + 1.0);
    let j_max = (nf * (1.0 - d)).floor() as usize;

    let mut sum = 0.0f64;
    let mut c = 0.0f64; // Kahan compensation
    for j in 0..=j_max {
        let jf = j as f64;
        let a = jf / nf + d; // > 0
        let b = 1.0 - d - jf / nf; // >= 0
        let ln_binom = ln_n_fact - ln_gamma(jf + 1.0) - ln_gamma(nf - jf + 1.0);

        let term = if b == 0.0 {
            // b^(n-j): zero unless n == j (then b^0 == 1)
            if j == n {
                (ln_binom + (jf - 1.0) * a.ln()).exp()
            } else {
                0.0
            }
        } else {
            (ln_binom + (jf - 1.0) * a.ln() + (nf - jf) * b.ln()).exp()
        };

        let y = term - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    (d * sum).clamp(0.0, 1.0)
}

#[pyfunction]
#[pyo3(signature = (x, cdf, params, alternative="two-sided"))]
pub fn ks_1samp_np(
    py: Python<'_>,
    x: numpy::PyReadonlyArray1<f64>,
    cdf: &str,
    params: Vec<f64>,
    alternative: &str,
) -> PyResult<Py<PyDict>> {
    let xs = x.as_slice()?;
    let (stat, p) = py.allow_threads(|| -> PyResult<_> {
        if xs.is_empty() {
            return Err(PyValueError::new_err("x must have at least 1 element"));
        }
        reject_nonfinite(xs, "x")?;
        let alt = Alternative::from_str(alternative)?;

        let dist = parse_named_dist(cdf, &params)?;

        let mut v = xs.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));

        let n = v.len();
        let nf = n as f64;

        let mut d_plus = 0.0;
        let mut d_minus = 0.0;

        for (i0, &xi) in v.iter().enumerate() {
            let i = (i0 + 1) as f64;
            let f = dist.cdf(xi);

            let fn_i = i / nf;
            let fn_im1 = (i - 1.0) / nf;

            let dp = fn_i - f;
            let dm = f - fn_im1;

            if dp > d_plus {
                d_plus = dp;
            }
            if dm > d_minus {
                d_minus = dm;
            }
        }

        let (stat, p) = match alt {
            Alternative::TwoSided => {
                let d = d_plus.max(d_minus);
                (d, ks_pvalue_two_sided_finite(n, d))
            }
            // SciPy: "greater" uses D+; "less" uses D-
            Alternative::Greater => (d_plus, ks_pvalue_one_sided_exact(n, d_plus)),
            Alternative::Less => (d_minus, ks_pvalue_one_sided_exact(n, d_minus)),
        };
        Ok((stat, p))
    })?;

    let dct = PyDict::new_bound(py);
    dct.set_item("statistic", stat)?;
    dct.set_item("pvalue", p)?;
    Ok(dct.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    /// Exact two-sided survival function: Durbin's matrix formula evaluated
    /// with 60-digit arithmetic (mpmath), independent of this code and of
    /// scipy (whose kstwo is itself only good to ~1e-7 for n >= 400). All of
    /// these are above the tail switch, so they exercise the matrix path;
    /// n >= 550 is where the matrix power has to rescale, and every one of
    /// those used to come back as ~1.0.
    const TWO_SIDED: &[(usize, f64, f64)] = &[
        (5, 0.3, 0.66400000000000004),
        (20, 0.2, 0.35272017362341598),
        (100, 0.1, 0.25269275700639007),
        (400, 0.05, 0.26121260292565059),
        (500, 0.04, 0.39025101632640605),
        (550, 0.04, 0.33349655643955383),
        (600, 0.04, 0.2846796163446239),
        (1000, 0.03, 0.32269024641329994),
        (1000, 0.045, 0.033783028682297522),
        (3000, 0.02, 0.1788981376320468),
    ];

    /// Same reference, below the tail switch (p <= 0.01, d < 1/2): the result
    /// is 2 P(D+ >= d), which overestimates by the joint-tail term.
    const TAIL: &[(usize, f64, f64, f64)] = &[
        (200, 0.25, 1.6840667400683395e-11, 1e-9),
        (100, 0.2, 0.00055519273280280968, 1e-9),
        (60, 0.21, 0.0084503171147745367, 1e-7),
        (30, 0.3, 0.0068522433629125412, 1e-7),
    ];

    #[test]
    fn two_sided_moderate_p_matches_exact_reference() {
        for &(n, d, sf) in TWO_SIDED {
            let p = ks_pvalue_two_sided_finite(n, d);
            assert!(p > KS_TAIL_SWITCH, "n={n} d={d} should take the matrix path");
            assert!(rel(p, sf) < 1e-9, "n={n} d={d}: {p} vs {sf}");
        }
    }

    #[test]
    fn two_sided_tail_matches_exact_reference() {
        for &(n, d, sf, tol) in TAIL {
            let p = ks_pvalue_two_sided_finite(n, d);
            assert!(rel(p, sf) < tol, "n={n} d={d}: {p} vs {sf}");
            assert!(p >= sf * (1.0 - 1e-12), "n={n}: the tail formula never underestimates");
        }
    }

    /// For d >= 1/2 the two tails cannot both be exceeded, so
    /// P(D >= d) = 2 P(D+ >= d) exactly; and P(D+ >= d) has the closed form
    /// (1 - d)^n when d >= 1 - 1/n (only the j = 0 term survives).
    #[test]
    fn two_sided_closed_form_for_large_d() {
        for &(n, d) in &[(2usize, 0.6f64), (3, 0.7), (5, 0.85), (10, 0.95), (50, 0.99)] {
            let exact = 2.0 * (1.0 - d).powi(n as i32);
            let p = ks_pvalue_two_sided_finite(n, d);
            assert!(rel(p, exact) < 1e-12, "n={n} d={d}: {p} vs {exact}");
        }
    }

    /// The tail formula must not be cut off at ~1e-16 the way `1 - cdf` is.
    #[test]
    fn two_sided_far_tail_is_not_zero() {
        // 2 * (1 - d)^n with d >= 1 - 1/n: 2e-200, far below what 1 - cdf can hold
        let p = ks_pvalue_two_sided_finite(100, 0.99);
        let exact = 2.0 * 0.01f64.powi(100);
        assert!(exact > 1e-201 && exact < 1e-199 && rel(p, exact) < 1e-9, "{p} vs {exact}");
    }

    /// The two evaluation paths must agree where they meet.
    #[test]
    fn tail_and_matrix_agree_at_the_switch() {
        for &n in &[30usize, 100, 400, 1000] {
            // bisect for the d at which twice the one-sided tail equals the switch
            let (mut lo, mut hi) = (0.0f64, 0.5f64);
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if 2.0 * ks_pvalue_one_sided_exact(n, mid) > KS_TAIL_SWITCH {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let tail = 2.0 * ks_pvalue_one_sided_exact(n, hi);
            let matrix = ks_pvalue_two_sided_durbin(n, hi);
            assert!(rel(tail, matrix) < 1e-6, "n={n} d={hi}: tail {tail} vs matrix {matrix}");
            assert!(tail >= matrix, "n={n}: dropping the joint term can only overestimate");
        }
    }

    /// D_n cannot be smaller than 1/(2n), so any d at or below that has p = 1.
    #[test]
    fn two_sided_is_one_below_the_smallest_possible_statistic() {
        for &n in &[1usize, 2, 7, 100, 5000] {
            let floor = 0.5 / n as f64;
            assert_eq!(ks_pvalue_two_sided_finite(n, floor), 1.0, "n={n}");
            assert_eq!(ks_pvalue_two_sided_finite(n, floor * 0.3), 1.0, "n={n}");
            let just_above = ks_pvalue_two_sided_finite(n, floor * 1.5);
            assert!(just_above.is_finite() && just_above <= 1.0 && just_above >= 0.5 - 1e-12, "n={n}: {just_above}");
        }
    }

    #[test]
    fn two_sided_is_monotone_in_d() {
        for &n in &[7usize, 60, 550, 2000] {
            let mut prev = 1.0f64;
            for step in 1..200 {
                let d = step as f64 / 200.0;
                let p = ks_pvalue_two_sided_finite(n, d);
                assert!(p.is_finite() && (0.0..=1.0).contains(&p), "n={n} d={d}: {p}");
                assert!(p <= prev * (1.0 + 1e-6), "n={n} d={d}: {p} > {prev}");
                prev = p;
            }
        }
    }
}
