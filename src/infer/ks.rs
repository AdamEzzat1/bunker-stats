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

/// Finite-n two-sided KS survival function (p-value for D_n >= d).
///
/// Based on Durbin's matrix formula as implemented by Marsaglia, Tsang & Wang (2003).
/// This is the core of SciPy's `kstwo.sf` for one-sample two-sided KS.
fn ks_pvalue_two_sided_finite(n: usize, d: f64) -> f64 {
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

    // Practical guard: exact method becomes expensive when m ~ O(n).
    // SciPy switches methods internally too; for very large n, the Stephens
    // approximation is usually sufficient for statistical testing.
    if n > 10_000 {
        return ks_pvalue_asymp_two_sided_stephens(n, d);
    }

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
    // keep values roughly in [1e-150, 1e150]
    let target = 1e150;
    if max_abs > target {
        let s = max_abs / target;
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
        let mut log_scale = 0.0f64;

        while exp > 0 {
            if exp & 1 == 1 {
                let (tmp, l) = mat_mul_scaled(&res, &base, m);
                res = tmp;
                log_scale += l;
            }
            exp >>= 1;
            if exp > 0 {
                let (tmp, l) = mat_mul_scaled(&base, &base, m);
                base = tmp;
                log_scale += l;
            }
        }
        (res, log_scale)
    }

    let (q, log_scale) = mat_pow_scaled(hmat, n, m);

    // Durbin/Marsaglia: CDF = (H^n)[k-1,k-1] * n! / n^n
    let elem = q[(k - 1) * m + (k - 1)];
    if elem <= 0.0 {
        return 1.0;
    }
    let ln_cdf = elem.ln() + log_scale + ln_gamma(nf + 1.0) - nf * nf.ln();
    let cdf = ln_cdf.exp().clamp(0.0, 1.0);
    (1.0 - cdf).clamp(0.0, 1.0)
}

/// One-sided asymptotic approximation: P(D_n >= d) ≈ exp(-2 n d^2)
fn ks_pvalue_asymp_one_sided(n: usize, d: f64) -> f64 {
    if !d.is_finite() {
        return f64::NAN;
    }
    if d <= 0.0 {
        return 1.0;
    }
    let nf = n as f64;
    (-2.0 * nf * d * d).exp().clamp(0.0, 1.0)
}

#[pyfunction]
pub fn ks_1samp_np(
    py: Python<'_>,
    x: numpy::PyReadonlyArray1<f64>,
    cdf: &str,
    params: Vec<f64>,
    alternative: &str,
) -> PyResult<Py<PyDict>> {
    let xs = x.as_slice()?;
    if xs.is_empty() {
        return Err(PyValueError::new_err("x must have at least 1 element"));
    }
    reject_nonfinite(xs, "x")?;
    let alt = Alternative::from_str(alternative)?;

    let dist = parse_named_dist(cdf, &params)?;

    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

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
        Alternative::Greater => (d_plus, ks_pvalue_asymp_one_sided(n, d_plus)),
        Alternative::Less => (d_minus, ks_pvalue_asymp_one_sided(n, d_minus)),
    };

    let dct = PyDict::new_bound(py);
    dct.set_item("statistic", stat)?;
    dct.set_item("pvalue", p)?;
    Ok(dct.unbind())
}
