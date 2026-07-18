# Time-Series Analysis (TSA) Module

Rust kernels (PyO3/NumPy bindings) for time-series statistics: unit-root and
stationarity testing, serial-correlation diagnostics, autocorrelation and
partial-autocorrelation estimation, spectral analysis, and rolling-window
correlation measures.

All functions accept 1-D `numpy.ndarray` of `float64` and are exposed at the
top level of the package:

```python
import numpy as np
import bunker_stats as bs          # Python facade (re-exports everything below)
# or: import bunker_stats_rs as b  # compiled extension module directly

stat, pvalue = bs.adf_test(x)
```

The only function not re-exported by the facade is `kpss_test_debug`, which is
available on the compiled module `bunker_stats_rs` only.

Source layout:

| File | Contents |
|---|---|
| `stationarity.rs` | ADF, KPSS, Phillips-Perron, variance ratio, Zivot-Andrews, integration-order and seasonal helpers |
| `diagnostics.rs` | Ljung-Box, Box-Pierce, Breusch-Godfrey, Durbin-Watson, runs test, ACF zero crossing |
| `acf_pacf.rs` | ACF, ACovF, ACF confidence bands, CCF, four PACF algorithms |
| `spectral.rs` | Periodogram, Welch/Bartlett PSD, spectral descriptors, band power |
| `rolling_autocorr.rs` | Rolling autocorrelation and rolling correlation |
| `rolling.rs` | Work-in-progress rolling statistics — **not registered in the Python API** |

---


## Using the Python facade

New code should reach these kernels through the `bunker_stats` facade, which
exposes clean names with keyword arguments. The raw `bunker_stats_rs` names
documented below remain available and stable; the facade adds ergonomics on
top of the same kernels:

```python
import bunker_stats as bs

bs.adf_test(x, regression="ct")
bs.kpss_test(x)
bs.rolling_autocorr_multi(x, lags=[1, 2, 5], window=60)
```

Where a statistic has strict and skip-NaN variants, the facade exposes ONE
name with a `skipna=` keyword; `skipna=True` dispatches to the skip-NaN kernel
documented below (the twin kernels stay separate in Rust, so there is no
branch inside the hot loop).

## Function summary

### Stationarity and unit-root tests

| Function | Signature | Returns |
|---|---|---|
| `adf_test` | `(x, regression="c", max_lag=None)` | `(statistic, pvalue)` |
| `kpss_test` | `(x, regression="c", max_lag=None)` | `(statistic, pvalue)` |
| `pp_test` | `(x, regression="c")` | `(statistic, pvalue)` |
| `variance_ratio_test` | `(x, lags=2)` | `(vr, z_score, pvalue)` |
| `zivot_andrews_test` | `(x, max_lag=None)` | `(statistic, breakpoint_index, pvalue)` |
| `trend_stationarity_test` | `(x)` | `(statistic, pvalue, is_stationary)` |
| `integration_order_test` | `(x)` | `(is_i0, is_i1, adf_level, adf_diff1)` |
| `seasonal_diff_test` | `(x, period=12)` | `(statistic, pvalue, is_stationary)` |
| `seasonal_unit_root_test` | `(x, period=12)` | `list[(lag, statistic, pvalue)]` |
| `kpss_test_debug` | `(x, regression="c", max_lag=None)` | `(statistic, pvalue)` + stderr trace |

### Serial-correlation diagnostics

| Function | Signature | Returns |
|---|---|---|
| `ljung_box` | `(x, lags=20)` | `(statistic, pvalue)` |
| `box_pierce` | `(x, lags=20)` | `(statistic, pvalue)` |
| `bg_test` | `(resid, max_lag=5)` | `(statistic, pvalue)` |
| `durbin_watson` | `(x)` | `float` in `[0, 4]` |
| `runs_test` | `(x)` | `(n_runs, z_score, pvalue)` |
| `acf_zero_crossing` | `(x, max_lag=100)` | `int` or `None` |

### Autocorrelation / partial autocorrelation

| Function | Signature | Returns |
|---|---|---|
| `acf` | `(x, nlags=40)` | `ndarray`, length `nlags+1` |
| `acovf` | `(x, nlags=40)` | `ndarray`, length `nlags+1` |
| `acf_with_ci` | `(x, nlags=40, alpha=0.05)` | `(acf, lower, upper)` arrays |
| `ccf` | `(x, y, nlags=40)` | `ndarray`, length `2*nlags+1` |
| `pacf` | `(x, nlags=40)` | `ndarray`, length `nlags+1` |
| `pacf_yw` | `(x, nlags=40)` | `ndarray`, length `nlags+1` |
| `pacf_innovations` | `(x, nlags=40)` | `ndarray`, length `nlags+1` |
| `pacf_burg` | `(x, nlags=40)` | `ndarray`, length `nlags+1` |

### Spectral analysis

| Function | Signature | Returns |
|---|---|---|
| `periodogram` | `(x)` | `(freqs, power)` arrays |
| `welch_psd` | `(x, nperseg=256, noverlap=None)` | `(freqs, psd)` arrays |
| `bartlett_psd` | `(x, nperseg=256)` | `(freqs, psd)` arrays |
| `cumulative_periodogram` | `(x)` | `(freqs, cumulative_power)` arrays |
| `dominant_frequency` | `(x)` | `float` |
| `spectral_entropy` | `(x)` | `float` (nats) |
| `spectral_peaks` | `(x, n_peaks=5, min_height=0.0)` | `(freqs, powers)` arrays |
| `spectral_flatness` | `(x)` | `float` in `(0, 1]` |
| `spectral_centroid` | `(x)` | `float` |
| `spectral_rolloff` | `(x, percentile=0.85)` | `float` |
| `band_power` | `(x, freq_low=0.0, freq_high=0.5)` | `float` |

### Rolling-window measures

| Function | Signature | Returns |
|---|---|---|
| `rolling_autocorr` | `(x, lag=1, window=50)` | `ndarray`, length `n-window+1` |
| `rolling_correlation` | `(x, y, window=50)` | `ndarray`, length `n-window+1` |
| `rolling_autocorr_multi` | `(x, lags, window=50)` | `ndarray`, shape `(n-window+1, len(lags))` |

---

## Stationarity and unit-root tests

Unit-root testing is the standard first step before fitting ARMA-family
models, running regressions on time series (to avoid spurious regression), or
choosing a differencing order. ADF/PP take a unit root as the null; KPSS takes
stationarity as the null, so the two families complement each other.

### `adf_test(x, regression="c", max_lag=None) -> (statistic, pvalue)`

Augmented Dickey-Fuller test. Fits by OLS

```
Δy_t = [deterministic] + β·y_{t-1} + Σ_{i=1}^{p} γ_i·Δy_{t-i} + ε_t
```

and reports the t-ratio on `β`. Null hypothesis: unit root (non-stationary).
Small p-value ⇒ reject the unit root.

- `regression`: `"c"` constant (default), `"ct"` constant + linear trend,
  `"n"`/`"nc"` no deterministic terms. Any other value raises `ValueError`.
- `max_lag`: number of augmenting lagged differences `p`. `None` or `0` gives
  the plain Dickey-Fuller regression. Note that the reference implementation in
  statsmodels selects the lag order by AIC by default; to reproduce a
  `bunker-stats` result there, pass `maxlag=p, autolag=None`.

P-values come from the MacKinnon (1994) regression-surface approximation, the
same method used by statsmodels' `mackinnonp`, including its tail behavior:
statistics outside the tabulated range return exactly `0.0` (far left tail) or
`1.0` (far right tail).

Verified: for fixed lag order, the statistic agrees with
`statsmodels.tsa.stattools.adfuller(x, maxlag=p, regression=r, autolag=None)`
to ~1e-13 across `"c"`, `"ct"`, and `"n"`, and p-values agree including the
extreme tails.

```python
stat, p = bs.adf_test(prices, regression="ct", max_lag=4)
if p < 0.05:
    print("unit root rejected: series looks trend-stationary")
```

Edge cases: returns `(nan, nan)` when the sample is too short for the
requested lag order or the design matrix is rank-deficient.

### `kpss_test(x, regression="c", max_lag=None) -> (statistic, pvalue)`

Kwiatkowski-Phillips-Schmidt-Shin test. Null hypothesis: the series is
(level- or trend-) stationary — the opposite orientation to ADF. The statistic
is `Σ S_t² / (n² λ̂²)` where `S_t` is the cumulative sum of regression
residuals and `λ̂²` is a Newey-West long-run variance with Bartlett weights.

- `regression`: `"c"` level stationarity (residuals from demeaning) or `"ct"`
  trend stationarity (residuals from a linear-trend fit). Other values raise
  `ValueError`.
- `max_lag`: HAC bandwidth. When `None` (default), the bandwidth is chosen by
  the automatic data-dependent rule of Hobijn, Franses & Ooms — identical to
  statsmodels `kpss(..., nlags="auto")`, which is the statsmodels default.

Verified: the statistic matches `statsmodels.tsa.stattools.kpss` to machine
precision for both `"c"` and `"ct"` with automatic bandwidth.

P-values are interpolated from the Kwiatkowski et al. (1992) critical-value
table and therefore **clamped to the interval [0.01, 0.10]**: `0.10` means
"p ≥ 0.10" and `0.01` means "p ≤ 0.01" (statsmodels reports the same bounds
with a warning).

Edge cases: `n < 3`, a singular trend fit, or a non-positive long-run variance
return `(nan, nan)`.

### `pp_test(x, regression="c") -> (statistic, pvalue)`

Phillips-Perron test. Runs the plain Dickey-Fuller regression (no augmenting
lags) and then corrects the t-ratio non-parametrically for serial correlation
using a Newey-West (Bartlett-kernel) long-run variance with the Schwert
`12·(n/100)^{1/4}` bandwidth:

```
Z_t = sqrt(γ0/λ²)·t  −  (λ² − γ0)·T·SE(β) / (2·λ·s)
```

(Hamilton 1994, Prop. 17.6). The corrected statistic `Z_t` is evaluated
against the Dickey-Fuller distribution via the same MacKinnon p-value surface
as `adf_test`. Null hypothesis: unit root. Accepts `"c"`, `"ct"`, `"n"`.

Use when serial correlation in `ε_t` is expected but you prefer a
HAC correction over choosing an ADF lag order.

Edge cases: `(nan, nan)` on short samples, rank-deficient designs, or
non-positive variance estimates.

### `variance_ratio_test(x, lags=2) -> (vr, z_score, pvalue)`

Lo-MacKinlay variance-ratio test of the random-walk hypothesis.

**Input convention: `x` is the increment (return) series, not the level
series.** For a price series pass `np.diff(prices)` (or log returns).

The q-period variance uses **overlapping** q-sums with the bias-adjusted
denominator `m = q(n−q+1)(1−q/n)` (Lo & MacKinlay 1988), consistent with the
homoscedastic asymptotic variance `θ = 2(2q−1)(q−1)/(3qn)` used for the
z-score. The p-value is two-sided Normal.

Interpretation under the random-walk null `VR ≈ 1`:

- `VR < 1` — negative serial correlation in increments (mean reversion),
- `VR > 1` — positive serial correlation (momentum / trending).

```python
vr, z, p = bs.variance_ratio_test(np.diff(np.log(prices)), lags=2)
```

Edge cases: requires `lags >= 2` and `n >= lags + 2`; otherwise returns
`(nan, nan, nan)`. A zero-variance increment series also returns NaNs.

### `zivot_andrews_test(x, max_lag=None) -> (statistic, breakpoint_index, pvalue)`

Zivot-Andrews unit-root test allowing a single endogenous structural break,
Model C (simultaneous shift in level and trend):

```
Δy_t = μ + β·t + θ·DU_t(τ) + γ·DT_t(τ) + α·y_{t-1} + Σ φ_i·Δy_{t-i} + ε_t
```

Every candidate breakpoint `τ` in the central 85% of the sample (15% trimming
from each end) is tried; the reported statistic is the minimum (most negative)
t-ratio on `α` and `breakpoint_index` is the `τ` that attains it. Null
hypothesis: unit root with no break.

- `max_lag`: augmenting lag order; default is `min(floor(sqrt(n)), 12)`
  (at least 1).
- The p-value is a coarse lookup against Model C critical values
  (−5.57 / −5.08 / −4.82 at 1% / 5% / 10%) and takes only the discrete values
  `{0.01, 0.05, 0.10, 0.15}`, where `0.15` means "p > 0.10".

Edge cases: returns `(nan, 0, nan)` for `n < 20` or when the sample cannot
support the trimming plus lag order; singular candidate regressions are
skipped.

### `trend_stationarity_test(x) -> (statistic, pvalue, is_stationary)`

Convenience wrapper: `kpss_test(x, "ct")` with the boolean decision
`is_stationary = pvalue > 0.05` (failing to reject the KPSS null of trend
stationarity). Inherits the KPSS p-value clamp to [0.01, 0.10].

### `integration_order_test(x) -> (is_i0, is_i1, adf_level, adf_diff1)`

Quick I(0)/I(1) classification via two ADF tests (`regression="c"`, no
augmenting lags):

- `is_i0 = True` if the ADF p-value on the levels is < 0.05,
- `is_i1 = True` if the levels test fails but the ADF p-value on the first
  difference is < 0.05,
- `adf_level`, `adf_diff1` are the two ADF statistics.

Edge cases: `n < 3` returns `(False, False, nan, nan)`.

### `seasonal_diff_test(x, period=12) -> (statistic, pvalue, is_stationary)`

Applies the seasonal difference `y_t − y_{t−period}` and runs
`adf_test(·, "c")` on the result. `is_stationary = pvalue < 0.05` (unit root
rejected after seasonal differencing). Returns `(nan, nan, False)` if
`n <= period`.

### `seasonal_unit_root_test(x, period=12) -> list[(lag, statistic, pvalue)]`

Screens for unit roots at the regular and seasonal frequencies: an ADF test on
the levels (entry `lag=1`), plus `seasonal_diff_test` at `period` and at
`2*period` when the sample allows. Returns an empty list if `n < 2*period`.

### `kpss_test_debug(x, regression="c", max_lag=None)`

Diagnostic variant of `kpss_test` that prints every intermediate quantity
(residuals, cumulative sums, per-lag autocovariances, bandwidth, statistic) to
stderr. Exposed on `bunker_stats_rs` only, not via the `bunker_stats` facade.

Note: with `max_lag=None` the debug variant selects its bandwidth with the
Schwert rule rather than the automatic Hobijn rule used by `kpss_test`, so the
default-argument statistics of the two functions can differ. Pass an explicit
`max_lag` to make them identical.

---

## Serial-correlation diagnostics

Standard checks on regression/ARMA residuals: if residuals are serially
correlated, coefficient standard errors are wrong and the model is
mis-specified. These tests are also useful directly on returns to detect
predictability.

### `ljung_box(x, lags=20) -> (statistic, pvalue)`

Ljung-Box portmanteau test. Null hypothesis: no autocorrelation up to `lags`.

```
Q = n(n+2) Σ_{k=1}^{L} r_k² / (n−k)   ~  χ²(L)
```

Uses the biased (denominator-`n`) autocorrelation estimator, matching
statsmodels. `lags` is capped at `n−1`.

Verified: agrees with `statsmodels.stats.diagnostic.acorr_ljungbox` to
~1e-14 in both statistic and p-value.

Edge cases: empty input, `lags == 0`, or a zero-variance series (including
any NaN in the input) return `(nan, nan)`.

### `box_pierce(x, lags=20) -> (statistic, pvalue)`

Box-Pierce statistic `Q = n Σ r_k²`, the simpler predecessor of Ljung-Box
(no small-sample `(n−k)` weighting). Same χ²(L) reference distribution, same
edge-case behavior. Prefer `ljung_box` for small samples.

### `bg_test(resid, max_lag=5) -> (statistic, pvalue)`

Breusch-Godfrey LM test for serial correlation of order up to `max_lag` in a
residual series. Runs the auxiliary regression of `e_t` on an intercept and
`e_{t-1}, …, e_{t-max_lag}` over the full sample with pre-sample lags
zero-padded (the same convention as statsmodels'
`acorr_breusch_godfrey`), and reports

```
LM = n · R²   ~  χ²(max_lag)
```

Null hypothesis: no serial correlation. Unlike the portmanteau tests, the LM
test remains valid when the residuals come from a model with lagged dependent
variables.

Note: this function takes the residual vector directly and regresses on
lagged residuals only; it does not include the original model's regressors in
the auxiliary regression, which the full Breusch-Godfrey procedure would.

Edge cases: `max_lag == 0` or `n <= max_lag + 1` returns `(nan, nan)`, as
does a singular auxiliary regression.

### `durbin_watson(x) -> float`

Durbin-Watson statistic `Σ (x_t − x_{t-1})² / Σ x_t²` on a residual series.
Range `[0, 4]`; `≈ 2` no first-order autocorrelation, `→ 0` positive,
`→ 4` negative. The denominator is the raw (not demeaned) sum of squares, so
pass residuals (mean ≈ 0), not an arbitrary series.

Edge cases: `n < 2` or an all-zero series returns `nan`.

### `runs_test(x) -> (n_runs, z_score, pvalue)`

Wald-Wolfowitz runs test for randomness about the median. Counts runs of
values above/below the sample median, compares against the expected run count
under independence with a ±0.5 continuity correction, and reports a two-sided
Normal p-value. Too few runs ⇒ positive dependence (clustering); too many ⇒
negative dependence (oscillation).

Values exactly equal to the median are treated as ties and excluded from run
transitions. NaN comparisons behave like ties, so **NaN inputs are skipped
rather than causing a crash** (median extraction uses a total ordering).

Edge cases: `n < 2`, or all observations on one side of the median, return
NaN statistic/p-value.

### `acf_zero_crossing(x, max_lag=100) -> int | None`

First lag `k` at which the autocorrelation function crosses from positive to
non-positive. A quick decorrelation-length summary (e.g., for choosing block
lengths in block bootstraps). Returns `None` if no crossing occurs within
`max_lag` (capped at `n−1`), or if the series has zero variance.

---

## Autocorrelation / partial autocorrelation

The ACF/PACF pair is the classical tool for ARMA order identification: an
AR(p) process shows a PACF cutoff after lag p; an MA(q) process shows an ACF
cutoff after lag q.

All estimators use the biased (denominator-`n`) sample moments, matching
statsmodels defaults. `nlags` is always capped at `n−1`. Empty input yields
empty arrays. A constant (zero-variance) series yields an all-ones ACF by
convention.

### `acf(x, nlags=40) -> ndarray`

Sample autocorrelation at lags `0..nlags` (index 0 is always 1.0).

Verified: matches `statsmodels.tsa.stattools.acf(x, nlags, fft=False)` to
machine precision.

### `acovf(x, nlags=40) -> ndarray`

Sample autocovariance at lags `0..nlags` (unnormalized ACF; index 0 is the
biased sample variance).

Verified: matches `statsmodels.tsa.stattools.acovf(x, nlag=nlags,
adjusted=False)` to machine precision.

### `acf_with_ci(x, nlags=40, alpha=0.05) -> (acf, lower, upper)`

ACF with Bartlett-formula confidence bands: the standard error at lag `k` is
`sqrt((1 + 2 Σ_{j<k} r_j²)/n)`. The Normal critical value is taken from a
fixed table — 2.576 for `alpha <= 0.01`, 1.96 for `alpha <= 0.05`, 1.645
otherwise — not computed from `alpha` exactly. Bands at lag 0 are pinned to
`[1, 1]`.

### `ccf(x, y, nlags=40) -> ndarray`

Cross-correlation between two equal-length series, normalized by the product
of the population standard deviations. Output has length `2*nlags + 1`;
index `i` corresponds to lag `ℓ = i − nlags` and holds
`Corr(x_t, y_{t−ℓ})`:

- a peak at **negative** `ℓ` means `x` leads `y` (`y` echoes `x` after `|ℓ|`
  steps),
- a peak at **positive** `ℓ` means `y` leads `x`.

```python
c = bs.ccf(x, y, nlags=5)          # length 11, center index 5 is lag 0
lead = np.argmax(c) - 5            # negative => x leads y
```

Edge cases: empty input or a length mismatch returns an empty array; zero
variance in either series returns an all-NaN array.

### `pacf(x, nlags=40) -> ndarray`

Partial autocorrelation via the Levinson-Durbin recursion on the sample ACF
(O(k²) by exploiting the Toeplitz structure, versus O(k³) for a generic
solve). This is the default and fastest PACF.

Verified: matches `statsmodels.tsa.stattools.pacf(x, method="ywm")` to
machine precision.

If the innovation variance is exhausted mid-recursion (numerically singular
problems, e.g. near-perfectly predictable series), remaining lags are NaN.

### `pacf_yw(x, nlags=40) -> ndarray`

Same Yule-Walker quantity computed by explicitly solving the k×k Toeplitz
system at each lag. Slower than `pacf`; retained for cross-checking. Matches
`pacf` and statsmodels `"ywm"` to machine precision; singular systems yield
NaN at the affected lag.

### `pacf_innovations(x, nlags=40) -> ndarray`

PACF via the innovations algorithm — a numerically stable alternative for
ill-conditioned autocovariance sequences. Remaining lags are NaN after a
non-positive prediction-error variance.

### `pacf_burg(x, nlags=40) -> ndarray`

Reflection coefficients from Burg's maximum-entropy method on the demeaned
series; well suited to short series. The sign convention matches the other
PACF variants here. Note this is a distinct estimator: values differ from the
Yule-Walker-based PACFs at higher lags, and it is **not** numerically
identical to statsmodels' `pacf(method="burg")` (a different variant of the
recursion); agreement is close at low lags but only approximate beyond them.

---

## Spectral analysis

Frequency-domain analysis: locate periodicities/cycles, quantify how "tonal"
versus noise-like a series is, and measure power in frequency bands. All
functions assume unit sampling frequency; frequencies are in **cycles per
sample** on `[0, 0.5]` (a period-`T` cycle appears at frequency `1/T`).

### `periodogram(x) -> (freqs, power)`

One-sided raw periodogram. Uses an FFT for `n >= 64` and a direct DFT below
that (identical results). No detrending or windowing is applied.

Verified: matches `scipy.signal.periodogram(x, scaling="spectrum",
detrend=False, window="boxcar")` to ~1e-17, including frequency grid and
one-sided doubling rules (interior bins doubled; Nyquist handling depends on
parity of `n`). Because the input is not demeaned, a nonzero mean shows up as
power at frequency 0.

Output length is `n//2 + 1`; empty input gives empty arrays.

### `welch_psd(x, nperseg=256, noverlap=None) -> (freqs, psd)`

Welch's averaged-periodogram PSD estimate: the series is split into segments
of length `nperseg` with overlap `noverlap` (default `nperseg // 2`), each
segment is demeaned, Hann-windowed, transformed, and the per-segment spectra
are averaged. Averaging trades frequency resolution for a large variance
reduction relative to the raw periodogram.

Scaling note: the relative spectral shape matches `scipy.signal.welch`, but
the absolute normalization differs by a constant factor (this implementation
applies no one-sided interior doubling and includes an extra `1/nperseg`
factor). Use it for peak location, shape, and ratios rather than absolute
PSD levels, or calibrate the constant against a known reference.

Fallback: if `n < nperseg`, or the overlap leaves a zero step size, the
function silently returns the raw periodogram of the full series (with the
periodogram's scipy-matching scaling).

### `bartlett_psd(x, nperseg=256) -> (freqs, psd)`

Bartlett's method: identical to `welch_psd` with zero overlap
(non-overlapping segments). Same scaling caveats.

### `cumulative_periodogram(x) -> (freqs, cumulative_power)`

Cumulative sum of periodogram power, normalized to end at 1.0. The basis of
Kolmogorov-Smirnov-type white-noise tests: for white noise the curve is close
to the diagonal; sharp jumps reveal concentrated periodic components.

### `dominant_frequency(x) -> float`

Frequency of the largest periodogram bin, excluding the DC (frequency-0)
component. Returns `nan` when fewer than two bins exist (`n < 2`).

```python
f = bs.dominant_frequency(signal)   # period estimate: 1/f samples
```

### `spectral_entropy(x) -> float`

Shannon entropy (natural log, nats) of the periodogram normalized to a
probability distribution. Low values ⇒ power concentrated at few frequencies
(strong periodicity); high values ⇒ power spread out (noise-like). The
maximum possible value is `ln(n//2 + 1)`. Returns `nan` for empty or
zero-power input.

### `spectral_peaks(x, n_peaks=5, min_height=0.0) -> (freqs, powers)`

Top `n_peaks` local maxima of the periodogram (strictly greater than both
neighbors, at least `min_height`), sorted by descending power. The DC bin and
the final bin are never candidates. Returns fewer than `n_peaks` entries (or
empty arrays) when fewer qualifying peaks exist.

### `spectral_flatness(x) -> float`

Wiener entropy: geometric mean / arithmetic mean of periodogram power.
Near 1 for white noise, near 0 for tonal signals.

Caveat: bins with power ≤ 1e-10 are excluded before averaging. For a
noiseless pure sinusoid nearly all off-peak bins fall below this threshold,
so the ratio is computed over the peak alone and returns ≈ 1 rather than ≈ 0.
With any realistic noise floor the measure behaves conventionally (e.g. a
sine in noise scores far below pure noise). Returns `nan` when no bin
survives the threshold.

### `spectral_centroid(x) -> float`

Power-weighted mean frequency (center of mass of the spectrum). Values near
0 indicate low-frequency-dominated series; white noise centers near 0.25.
Returns `nan` for empty or zero-power input.

### `spectral_rolloff(x, percentile=0.85) -> float`

Smallest frequency below which `percentile` of the total power is contained.
Returns the highest frequency if the threshold is never reached, and `nan`
for empty or zero-power input.

### `band_power(x, freq_low=0.0, freq_high=0.5) -> float`

Sum of periodogram power over bins with `freq_low <= f <= freq_high`
(inclusive on both ends; the DC bin is included when `freq_low == 0.0`).
Since the periodogram uses "spectrum" scaling, band powers over disjoint
bands sum to the total power. Returns `nan` for empty input.

---

## Rolling-window measures

Windowed second-moment statistics for detecting time variation in dependence
structure (regime changes, evolving momentum/mean-reversion).

Unlike most of the module, these functions **raise `ValueError`** on invalid
parameters rather than returning NaN: `window` must satisfy
`1 <= window <= n`, every lag must be `< window`, and `rolling_correlation`
requires equal-length inputs.

### `rolling_autocorr(x, lag=1, window=50) -> ndarray`

Lag-`lag` autocorrelation computed within each sliding window. Output length
is `n - window + 1`; element `i` covers `x[i : i+window]`. Windows with zero
variance produce NaN.

### `rolling_correlation(x, y, window=50) -> ndarray`

Pearson correlation between `x` and `y` within each sliding window. Output
length `n - window + 1`; zero-variance windows produce NaN.

### `rolling_autocorr_multi(x, lags, window=50) -> ndarray`

Returns a 2-D array of shape `(n - window + 1, len(lags))` whose column `j`
equals `rolling_autocorr(x, lags[j], window)`. `lags` has no default and must
be provided.

Equivalent to (but cheaper than, since the window mean/variance are computed
once per window rather than once per lag):

```python
out = np.column_stack([bs.rolling_autocorr(x, k, window) for k in lags])
```

Historical note: before 0.2.9 the multi-lag output buffer was written
column-major but reshaped row-major, scrambling the result whenever more than
one lag was requested. This is fixed and covered by a regression test
(`test_rolling_autocorr_multi_row_major`).

---

## Statistical conventions and reference-implementation parity

- **ACF/ACovF/PACF**: biased (denominator-`n`) moment estimators, matching
  statsmodels defaults. `acf`, `acovf`, `pacf`, and `pacf_yw` agree with
  statsmodels to machine precision; `pacf_burg` is a distinct Burg variant
  with only approximate agreement.
- **ADF**: MacKinnon (1994) regression-surface p-values with exact 0/1
  outside the tabulated tau range; statistic matches
  `adfuller(..., autolag=None)` to ~1e-13 for all three deterministic
  specifications. No automatic lag selection — `max_lag` is taken literally
  (`None` ⇒ 0 lags).
- **KPSS**: automatic bandwidth reproduces statsmodels `nlags="auto"`;
  statistic matches to machine precision; p-values are table-interpolated and
  clamped to [0.01, 0.10].
- **Phillips-Perron**: `Z_t` correction per Hamilton (1994) Prop. 17.6 with
  Newey-West/Bartlett long-run variance (Schwert bandwidth), evaluated
  against the Dickey-Fuller distribution.
- **Ljung-Box**: matches `acorr_ljungbox` to ~1e-14.
- **Breusch-Godfrey**: full-sample, zero-padded auxiliary regression;
  `LM = n·R²`, χ²(`max_lag`).
- **Variance ratio**: Lo-MacKinlay overlapping estimator, bias-adjusted
  denominator, homoscedastic z; input is the increment series.
- **Periodogram**: exactly `scipy.signal.periodogram(scaling="spectrum",
  detrend=False, window="boxcar")`. Welch/Bartlett PSDs match scipy in shape
  but not in absolute normalization (constant factor).
- **Frequencies** are cycles per sample (`fs = 1`), range `[0, 0.5]`.

References: Dickey & Fuller (1979); MacKinnon (1994); Kwiatkowski,
Phillips, Schmidt & Shin (1992); Hobijn, Franses & Ooms (2004); Phillips &
Perron (1988); Hamilton (1994); Lo & MacKinlay (1988); Zivot & Andrews
(1992); Ljung & Box (1978); Box & Pierce (1970); Breusch (1978) / Godfrey
(1978); Durbin & Watson (1950); Wald & Wolfowitz (1940); Bartlett (1946,
1950); Durbin (1960); Burg (1975); Welch (1967).

---

## Edge-case behavior

| Condition | Behavior |
|---|---|
| Empty input | ACF/PACF/CCF/spectral array outputs: empty arrays. Scalar tests: NaN results. |
| Sample too short for the test (e.g. `n < 3` KPSS, `n < 20` Zivot-Andrews, `n <= period` seasonal tests) | NaN statistic and p-value (empty list for `seasonal_unit_root_test`); no exception. |
| Constant / zero-variance series | `acf` returns all ones; `ljung_box`/`box_pierce`/`durbin_watson` return NaN; `ccf` returns all-NaN; `acf_zero_crossing` returns `None`; zero-variance rolling windows yield NaN entries. |
| NaN in input | Propagates to NaN outputs for moment-based statistics (no crash). `runs_test` treats NaNs as median ties and skips them. Sorting-based paths use total ordering and cannot abort. |
| `nlags` / `lags` / `max_lag` larger than the sample supports | Silently capped at `n − 1` (ACF/PACF/diagnostics/KPSS bandwidth). ADF/Zivot-Andrews return NaN if the lag order leaves too few rows. |
| Invalid `regression` string | `ValueError` (`adf_test`/`pp_test`: `"c"`, `"ct"`, `"n"`/`"nc"`; `kpss_test`: `"c"`, `"ct"`). |
| Invalid rolling parameters (`window` out of range, `lag >= window`, length mismatch) | `ValueError`. |
| `variance_ratio_test` with `lags < 2` | `(nan, nan, nan)`. |
| Rank-deficient / singular regressions | NaN results (ADF, PP, BG); singular breakpoints skipped (Zivot-Andrews); NaN at the affected lag (`pacf_yw`). |
| KPSS p-value outside table | Clamped: `0.10` ⇒ p ≥ 0.10, `0.01` ⇒ p ≤ 0.01. |
| ADF statistic outside MacKinnon range | Exact `0.0` / `1.0`, matching statsmodels. |
| `welch_psd` with `n < nperseg` or zero step | Falls back to the full-series periodogram. |
| `rolling_autocorr_multi` with multiple lags | Each column `j` equals `rolling_autocorr(x, lags[j], window)`; zero-variance windows produce NaN. |
