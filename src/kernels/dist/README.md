# `dist` — Probability Distribution Kernels

Rust implementations (exposed to Python via PyO3) of density, distribution,
survival, hazard, and quantile functions for three continuous distributions:
**Normal**, **Exponential**, and **Uniform**. Each distribution exposes the
same seven function kinds, for a total of **21 functions**.

| Source file | Distribution | Functions |
|---|---|---|
| `normal.rs` | Normal(μ, σ) | `norm_pdf`, `norm_logpdf`, `norm_cdf`, `norm_sf`, `norm_logsf`, `norm_cumhazard`, `norm_ppf` |
| `exponential.rs` | Exponential(λ) | `exp_pdf`, `exp_logpdf`, `exp_cdf`, `exp_sf`, `exp_logsf`, `exp_cumhazard`, `exp_ppf` |
| `uniform.rs` | Uniform(a, b) | `unif_pdf`, `unif_logpdf`, `unif_cdf`, `unif_sf`, `unif_logsf`, `unif_cumhazard`*, `unif_ppf` |

\* The registered `unif_cumhazard` is a wrapper defined in `src/lib.rs` that
supersedes the version in `uniform.rs`; see
[Uniform cumulative hazard tail clamp](#uniform-cumulative-hazard-tail-clamp).

All 21 functions are registered in `src/lib.rs` and re-exported unchanged by
the `bunker_stats` Python facade (`bunker_stats/__init__.py`), so
`from bunker_stats import norm_cdf` and `from bunker_stats_rs import norm_cdf`
resolve to the same native function.


## Using the Python facade

New code should reach these kernels through the `bunker_stats` facade, which
exposes clean names with keyword arguments. The raw `bunker_stats_rs` names
documented below remain available and stable; the facade adds ergonomics on
top of the same kernels:

```python
import bunker_stats as bs

bs.norm_ppf(q)     # facade re-exports the distribution family unchanged
bs.exp_pdf(x, 2.0)  # rate-parameterized
```

Where a statistic has strict and skip-NaN variants, the facade exposes ONE
name with a `skipna=` keyword; `skipna=True` dispatches to the skip-NaN kernel
documented below (the twin kernels stay separate in Rust, so there is no
branch inside the hot loop).

## Input and output conventions

Every function maps a 1-D `float64` NumPy array to a new 1-D `float64` NumPy
array of the same length. There are no separate scalar or `_np` variants —
the array form is the only interface.

Accepted input (verified against the built extension):

- **1-D, C-contiguous, `float64` only.** Python scalars, integer arrays,
  2-D arrays, and non-contiguous views (e.g. `a[::2]`) all raise `TypeError`.
  Convert first: `np.ascontiguousarray(x, dtype=np.float64)`.
- Distribution parameters (`mu`, `sigma`, `lam`, `a`, `b`) are Python floats
  with defaults; they are validated on every call and raise `ValueError`
  when out of range (see [Parameter validation](#parameter-validation)).
- Functions are pure (no shared state) and release no Python objects other
  than the output array; they are safe to call from multiple threads.

## Parameterization conventions

These were confirmed empirically against SciPy, not just from the source.

### Normal — location/scale (μ, σ)

`mu` is the mean, `sigma` the **standard deviation** (not variance).
Defaults `mu=0.0`, `sigma=1.0`. Equivalent to `scipy.stats.norm(loc=mu, scale=sigma)`.

### Exponential — **rate** λ (not scale)

`lam` is the rate parameter: pdf is `λ·exp(−λx)` for `x ≥ 0`, mean is `1/λ`.
Default `lam=1.0`.

Verified: `exp_pdf([1.0], lam=2.0)` returns `0.2706705664732254 = 2e⁻²`,
which equals `scipy.stats.expon(scale=0.5).pdf(1.0)`. So the SciPy
equivalence is

```python
exp_*(x, lam)  ==  scipy.stats.expon(scale=1.0/lam).*(x)
```

SciPy's `expon` takes a *scale* (= 1/rate); passing `lam` where SciPy expects
`scale` silently computes the wrong distribution. This is the single most
common misuse of this module — double-check call sites.

### Uniform — endpoints (a, b)

`a` is the lower endpoint, `b` the upper, requiring `b > a`. Defaults
`a=0.0`, `b=1.0`. Note the difference from SciPy, whose `uniform(loc, scale)`
uses `[loc, loc+scale]`: the equivalence is
`unif_*(x, a, b) == scipy.stats.uniform(loc=a, scale=b-a).*(x)`.

## Function kinds explained

Each distribution provides the same seven kinds. `F` is the CDF, `f` the
density, `S = 1 − F` the survival function.

| Kind | Definition | Why it matters |
|---|---|---|
| `pdf(x)` | density `f(x)` | Likelihood factors, kernel weights, plotting. |
| `logpdf(x)` | `ln f(x)` | Log-likelihood sums without underflow; `pdf` underflows to 0 far in the tails, `logpdf` stays finite and exact. |
| `cdf(x)` | `P(X ≤ x)` | Probabilities, probability-integral transforms, goodness-of-fit statistics. |
| `sf(x)` | `P(X > x) = 1 − F(x)` | Upper-tail p-values **without cancellation**: computed directly, not as `1 − cdf`, so it stays accurate where `1 − cdf(x)` would round to 0 (see [Tail accuracy](#tail-accuracy-sf-vs-1--cdf)). |
| `logsf(x)` | `ln S(x)` | Extreme-tail log p-values; finite long after `sf` itself underflows the smallest subnormal. |
| `cumhazard(x)` | `H(x) = −ln S(x)` | Survival/reliability modeling: cumulative hazard is the integral of the hazard rate, and `H` is what proportional-hazards and counting-process methods accumulate. For the Exponential it is exactly `λx`. |
| `ppf(q)` | quantile: `x` with `F(x) = q` | Value-at-Risk and other quantile risk measures, inverse-transform random sampling, confidence-bound construction. |

Identities that hold across the module (verified): `cdf + sf = 1`,
`cumhazard = −logsf`, `logpdf = ln(pdf)` where `pdf > 0`, and the round trip
`ppf(cdf(x)) ≈ x` (max error `4.4e−16` for Normal over `[−3, 3]`;
`≈1e−12` absolute for Exponential out to `x = 20`, where `cdf` saturates
toward 1 and the inversion loses resolution).

## Normal(μ, σ)

| Function | Signature | Returns | Notes |
|---|---|---|---|
| `norm_pdf` | `norm_pdf(x, mu=0.0, sigma=1.0)` | `f(x)` | Direct evaluation of the Gaussian density. |
| `norm_logpdf` | `norm_logpdf(x, mu=0.0, sigma=1.0)` | `ln f(x)` | Closed form `−½ln(2π) − ln σ − z²/2`; never under/overflows for finite `x`. |
| `norm_cdf` | `norm_cdf(x, mu=0.0, sigma=1.0)` | `Φ((x−μ)/σ)` | `0.5·erfc(−z/√2)` via `libm::erfc`, matching SciPy's `ndtr` formulation. |
| `norm_sf` | `norm_sf(x, mu=0.0, sigma=1.0)` | `1 − Φ(z)` | `0.5·erfc(z/√2)` — direct, cancellation-free upper tail. Matches `scipy.stats.norm.sf(10.0)` to ~4 ulps; stays nonzero to `z ≈ 38` (`sf(38) ≈ 2.9e−316`, subnormal). |
| `norm_logsf` | `norm_logsf(x, mu=0.0, sigma=1.0)` | `ln S(x)` | Log of the stable SF. Exact until `sf` underflows: finite at `z = 38` (`≈ −726.56`), `−inf` from `z ≈ 39`. Not an asymptotic-expansion `logsf`; for `z > 38` use an external asymptotic if you need finite values. |
| `norm_cumhazard` | `norm_cumhazard(x, mu=0.0, sigma=1.0)` | `−ln S(x)` | `+inf` where `sf` underflows to 0 (same threshold as `logsf`). |
| `norm_ppf` | `norm_ppf(q, mu=0.0, sigma=1.0)` | quantile | Validates `q ∈ [0,1]` up front (`ValueError` otherwise); `q=0 → −inf`, `q=1 → +inf`, `NaN → NaN`. Interior values delegate to `statrs` `inverse_cdf` after the explicit boundary handling. |

All seven raise `ValueError("sigma must be positive")` when `sigma ≤ 0`.
`mu` and `sigma` are not checked for NaN/inf; non-finite parameters produce
NaN/degenerate outputs rather than errors.

## Exponential(λ)

Support is `[0, ∞)`; all `x`-domain functions treat `x < 0` by returning the
mathematically correct constant (0, 1, or `−inf`), not an error.

| Function | Signature | Returns | Notes |
|---|---|---|---|
| `exp_pdf` | `exp_pdf(x, lam=1.0)` | `λe^{−λx}` for `x ≥ 0`, else `0` | |
| `exp_logpdf` | `exp_logpdf(x, lam=1.0)` | `ln λ − λx` for `x ≥ 0`, else `−inf` | |
| `exp_cdf` | `exp_cdf(x, lam=1.0)` | `1 − e^{−λx}` for `x ≥ 0`, else `0` | Saturates to exactly 1.0 once `λx ≳ 37` (double rounding). |
| `exp_sf` | `exp_sf(x, lam=1.0)` | `e^{−λx}` for `x ≥ 0`, else `1` | Direct exponential — accurate deep into the tail. |
| `exp_logsf` | `exp_logsf(x, lam=1.0)` | `−λx` for `x ≥ 0`, else `0` | Exact for all finite `x`; never underflows. |
| `exp_cumhazard` | `exp_cumhazard(x, lam=1.0)` | `λx` for `x ≥ 0`, else `0` | The Exponential's constant hazard makes this exactly linear. |
| `exp_ppf` | `exp_ppf(q, lam=1.0)` | `−ln(1−q)/λ` | Computed with `ln_1p` for stability near `q = 0`: `exp_ppf(1e−300)` returns `1e−300` exactly. `q=0 → 0`, `q=1 → +inf`, `NaN → NaN`; out-of-range `q` raises `ValueError`. |

All seven raise `ValueError("lam must be positive")` when `lam ≤ 0`.

## Uniform(a, b)

Support is `[a, b]`. Outside the support the functions return the correct
constants rather than errors.

| Function | Signature | Returns | Notes |
|---|---|---|---|
| `unif_pdf` | `unif_pdf(x, a=0.0, b=1.0)` | `1/(b−a)` on `[a, b]`, else `0` | Endpoints included in the support. **NaN caveat:** `unif_pdf(NaN)` returns `1/(b−a)`, not NaN — see [NaN propagation](#nan-propagation). |
| `unif_logpdf` | `unif_logpdf(x, a=0.0, b=1.0)` | `−ln(b−a)` on `[a, b]`, else `−inf` | Explicit NaN handling (NaN → NaN). |
| `unif_cdf` | `unif_cdf(x, a=0.0, b=1.0)` | `0`, `(x−a)/(b−a)`, or `1` | Piecewise; exactly 0 at `x ≤ a` and 1 at `x ≥ b`. |
| `unif_sf` | `unif_sf(x, a=0.0, b=1.0)` | `1`, `(b−x)/(b−a)`, or `0` | Exact complement of `unif_cdf`. |
| `unif_logsf` | `unif_logsf(x, a=0.0, b=1.0)` | `0`, `ln((b−x)/(b−a))`, or `−inf` | Piecewise logic guarantees no `ln(negative)`; requires **finite** `a`, `b` (stricter validation than pdf/cdf/sf, see below). |
| `unif_cumhazard` | `unif_cumhazard(x, a=0.0, b=1.0)` | `0`, `−ln((b−x)/(b−a))`, `+inf`, or clamped tail | Registered implementation lives in `src/lib.rs`, not `uniform.rs`. Requires finite `a`, `b`. See tail-clamp note below. |
| `unif_ppf` | `unif_ppf(q, a=0.0, b=1.0)` | `a + q·(b−a)` | `q=0 → a`, `q=1 → b` (finite, unlike Normal/Exponential); `NaN → NaN`; out-of-range `q` raises `ValueError`. |

All seven raise `ValueError` when `b ≤ a`. Additionally, `unif_logsf` and
`unif_cumhazard` raise `ValueError("a and b must be finite")` for non-finite
endpoints, while the other five accept them without error (verified:
`unif_pdf(x, 0.0, inf)` does not raise; with `1/(b−a) = 0` it degenerates to
an all-zero density). Do not rely on infinite endpoints anywhere in this
family.

### Uniform cumulative hazard tail clamp

The exported `unif_cumhazard` is a wrapper in `src/lib.rs` (the
`uniform.rs` version is shadowed at registration). Verified behavior for
`a=0, b=1` at `x = [−1, 0, 0.5, 1, 1.5, 2]`:

```
[0.0, 0.0, 0.6931, +inf, 36.7368, 36.7368]
```

- `x < a` → `0`; `a ≤ x < b` → `−ln((b−x)/(b−a))`.
- `x == b` → `+inf` exactly.
- `x > b` → a **large finite constant**: the hazard evaluated at the largest
  representable double below `b` (`−ln((b − next_down(b))/(b−a))`,
  ≈ `36.74` for unit width) instead of `+inf`.

The clamp keeps downstream arithmetic on the tail finite, at the cost of a
non-monotonic step at the boundary (`H(b) = +inf` but `H(b+ε)` finite). If
strict `H = +inf` beyond the support matters to a caller, treat `x ≥ b`
explicitly before calling.

## Edge cases and domain behavior

### PPF at and outside the boundaries

Verified for all three distributions:

- `q` **outside** `[0, 1]` (including `−0.1` and `1.5`): the whole call
  raises `ValueError("q values must be in [0, 1]")` — validation is
  performed over the full array before any output is produced, so one bad
  element fails the entire call. NaN elements are exempt from this check.
- `q = 0` → lower support end: `−inf` (Normal), `0.0` (Exponential),
  `a` (Uniform).
- `q = 1` → upper support end: `+inf` (Normal and Exponential), `b` (Uniform).
- `q = NaN` → NaN in that position (no error).

### NaN propagation

NaN inputs produce NaN outputs for every function **except `unif_pdf`**,
where a NaN element falls through the support test (`v < a || v > b` is
false for NaN) and yields `1/(b−a)`. Verified:
`unif_pdf([nan], 0.0, 1.0) == [1.0]`. Pre-filter NaNs if this matters.
The Normal and Exponential pdf/cdf/sf paths propagate NaN either through an
explicit check or arithmetically; the log/hazard/ppf paths all check
explicitly.

### Tail accuracy: `sf` vs `1 − cdf`

`1 − norm_cdf(x)` rounds to exactly 0 once the true tail probability drops
below double resolution near 1 (around `z ≈ 9`). `norm_sf` computes the
tail directly via `erfc` and remains accurate to `z ≈ 38`
(`sf(38) ≈ 2.9e−316`). Verified at `z = 10`: `norm_sf` agrees with
`scipy.stats.norm.sf` to ~`3e−38` absolute (a few ulps), while `1 − cdf`
returns 0. For p-values beyond that, use `norm_logsf`, which is finite
through `z = 38` and `−inf` after. The Exponential equivalents (`exp_sf`,
`exp_logsf`) are exact closed forms and have no comparable breakdown;
`exp_logsf = −λx` never underflows.

### Parameter validation

| Check | Functions | Error |
|---|---|---|
| `sigma > 0` | all `norm_*` | `ValueError: sigma must be positive` |
| `lam > 0` | all `exp_*` | `ValueError: lam must be positive` |
| `b > a` | all `unif_*` | `ValueError: b must be greater than a` |
| `a, b` finite | `unif_logsf`, `unif_cumhazard` only | `ValueError: a and b must be finite` |

Parameters are scalars, checked before any element is processed; parameter
errors never produce partial output.

## Implementation notes

- **Normal CDF/SF/logSF** use `libm::erfc` rather than the `statrs`
  distribution object; the `erfc` formulation matches SciPy's `ndtr` and was
  adopted for its tail precision (`statrs`' normal CDF is several digits
  worse in the far tail — see comments at the top of `normal.rs`).
- **Normal PPF** uses `statrs::distribution::Normal::inverse_cdf` for
  interior `q`, with NaN and the `q ∈ {0, 1}` boundaries short-circuited
  before the call to avoid `statrs` edge-case behavior.
- **Exponential PPF** uses `ln_1p` so that quantiles near `q = 0` keep full
  relative precision.
- **Uniform log-SF / cumulative hazard** are piecewise so `ln` is only ever
  applied to a strictly positive ratio.
- Outputs are freshly allocated `Vec<f64>` converted to NumPy arrays; inputs
  are never modified.
