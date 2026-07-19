# Resampling Kernels

Rust kernels (`src/kernels/resampling/`) exposed to Python through PyO3 as flat
functions on the `bunker_stats_rs` extension module. This document covers the
19 registered resampling functions: the i.i.d. bootstrap family, block
bootstraps for time series, permutation tests, and jackknife estimators.

Sources:

- `bootstrap.rs` — bootstrap CIs (percentile, BCa, bootstrap-t, Bayesian),
  bootstrap SE/variance, block bootstraps, permutation tests.
- `jackknife.rs` — jackknife mean, delete-d jackknife, influence values,
  jackknife-after-bootstrap.


## Using the Python facade

New code should reach these kernels through the `bunker_stats` facade, which
exposes clean names with keyword arguments. The raw `bunker_stats_rs` names
documented below remain available and stable; the facade adds ergonomics on
top of the same kernels:

```python
import bunker_stats as bs

bs.bootstrap_mean_ci(x, 1000, random_state=42)
bs.jackknife_mean(x)
```

Where a statistic has strict and skip-NaN variants, the facade exposes ONE
name with a `skipna=` keyword; `skipna=True` dispatches to the skip-NaN kernel
documented below (the twin kernels stay separate in Rust, so there is no
branch inside the hot loop).

## Overview

The **bootstrap** approximates the sampling distribution of a statistic by
recomputing it on many datasets drawn with replacement from the observed
sample (or, for time series, from blocks of it), and reads standard errors and
confidence intervals off that empirical distribution. The **jackknife**
instead recomputes the statistic on leave-one-out (or leave-d-out) subsamples,
giving fast, deterministic bias and standard-error estimates and the influence
values that feed the BCa interval's acceleration constant. **Permutation
tests** address hypothesis testing rather than estimation: they build the null
distribution of a test statistic by randomly relabeling the data
(shuffling one variable, or pooling and resplitting two groups) and report the
fraction of relabelings at least as extreme as the observed statistic.

All array inputs are 1-D contiguous `float64` NumPy arrays. NaN values are not
filtered by these kernels; NaN in, NaN (or NaN-contaminated resamples) out.

## Function summary

| Function | Purpose | Returns |
|---|---|---|
| `bootstrap_mean(x, n_resamples, random_state=None)` | Mean of bootstrap means | `float` |
| `bootstrap_mean_ci(x, n_resamples, conf=0.95, random_state=None)` | Percentile CI for the mean | `(point, lo, hi)` |
| `bootstrap_ci(x, stat="mean", n_resamples=1000, conf=0.95, random_state=None)` | Percentile CI for mean/median/std | `(point, lo, hi)` |
| `bootstrap_corr(x, y, n_resamples, conf=0.95, random_state=None)` | Percentile CI for Pearson correlation | `(point, lo, hi)` |
| `bootstrap_se(x, stat="mean", n_resamples=1000, random_state=None)` | Bootstrap standard error | `float` |
| `bootstrap_var(x, stat="mean", n_resamples=1000, random_state=None)` | Bootstrap variance (= SE squared) | `float` |
| `bootstrap_t_ci_mean(x, n_resamples=1000, conf=0.95, random_state=None)` | Studentized (bootstrap-t) CI for the mean | `(point, lo, hi)` |
| `bootstrap_bca_ci(x, stat="mean", n_resamples=1000, conf=0.95, random_state=None)` | BCa CI for mean/median/std | `(point, lo, hi)` |
| `bayesian_bootstrap_ci(x, stat="mean", n_resamples=1000, conf=0.95, random_state=None)` | Rubin's Bayesian bootstrap CI | `(point, lo, hi)` |
| `moving_block_bootstrap_mean_ci(x, block_len, n_resamples=1000, conf=0.95, random_state=None)` | Moving-block bootstrap CI for the mean | `(point, lo, hi)` |
| `circular_block_bootstrap_mean_ci(x, block_len, n_resamples=1000, conf=0.95, random_state=None)` | Circular-block bootstrap CI for the mean | `(point, lo, hi)` |
| `stationary_bootstrap_mean_ci(x, block_len, n_resamples=1000, conf=0.95, random_state=None)` | Politis–Romano stationary bootstrap CI | `(point, lo, hi)` |
| `permutation_corr_test(x, y, n_permutations=1000, alternative="two-sided", random_state=None)` | Permutation test for Pearson correlation | `(observed, p_value)` |
| `permutation_mean_diff_test(x, y, n_permutations=1000, alternative="two-sided", random_state=None)` | Permutation test for mean(x) − mean(y) | `(observed, p_value)` |
| `jackknife_mean(x)` | Jackknife estimate, bias, SE of the mean | `(estimate, bias, se)` |
| `jackknife_mean_ci(x, conf=0.95)` | Jackknife mean with percentile CI over LOO estimates | `(estimate, lo, hi)` |
| `influence_mean(x)` | Leave-one-out influence values for the mean | `ndarray, shape (n,)` |
| `delete_d_jackknife_mean(x, d)` | Delete-d (block) jackknife for the mean | `(estimate, bias, se)` |
| `jackknife_after_bootstrap_se_mean(x, n_resamples=200, random_state=None)` | Jackknife-after-bootstrap SE of the bootstrap SE | `float` |

Where a `stat` parameter exists, the accepted values are `"mean"`,
`"median"`, and `"std"`. `bootstrap_ci`, `bootstrap_bca_ci`, and
`bayesian_bootstrap_ci` raise `ValueError` for anything else;
`bootstrap_se` and `bootstrap_var` instead return `NaN` silently (verified).

---

## Bootstrap family (i.i.d.)

### `bootstrap_mean`

```python
bootstrap_mean(x, n_resamples, random_state=None) -> float
```

Draws `n_resamples` samples of size `n` with replacement, computes each
resample mean, and returns the average of those means. Mostly a smoke-test /
building-block routine; for inference use `bootstrap_mean_ci` or
`bootstrap_se` instead.

- **Returns** a scalar; `NaN` if `len(x) == 0` or `n_resamples == 0`.
- Each resample uses an independent RNG derived from
  `mix_seed(random_state or 0, resample_index)`, so individual resample means
  are exactly reproducible. The per-resample means are collected in index
  order and averaged with a serial sum, so the result is bit-identical across
  thread counts — see [Determinism & seeding](#determinism--seeding).

### `bootstrap_mean_ci`

```python
bootstrap_mean_ci(x, n_resamples, conf=0.95, random_state=None) -> (point, lo, hi)
```

Percentile bootstrap CI for the mean. The point estimate is the mean of the
bootstrap means (not the sample mean); `lo`/`hi` are the `alpha/2` and
`1 - alpha/2` percentiles of the sorted bootstrap distribution using
floor-index selection.

**Why it matters.** The percentile interval is the simplest, most robust
default: no normality assumption, no analytic SE. It is first-order accurate;
when the statistic's distribution is skewed or biased, prefer
`bootstrap_bca_ci`, and for heavier tails consider `bootstrap_t_ci_mean`.

### `bootstrap_ci`

```python
bootstrap_ci(x, stat="mean", n_resamples=1000, conf=0.95, random_state=None) -> (point, lo, hi)
```

Generic percentile CI for `stat` in `{"mean", "median", "std"}`. The `"std"`
statistic is the population-style (ddof=0) standard deviation of each
resample. The point estimate is the mean of the bootstrap statistics. The
median path reuses a per-thread scratch buffer, so it costs one sort per
resample without per-resample allocation.

- Unsupported `stat` raises
  `ValueError: Unsupported stat. Use 'mean', 'median', or 'std'.`

### `bootstrap_corr`

```python
bootstrap_corr(x, y, n_resamples, conf=0.95, random_state=None) -> (point, lo, hi)
```

Paired bootstrap for the Pearson correlation: each resample draws index pairs
`(x[i], y[i])` jointly, preserving the dependence structure. Degenerate
resamples (zero variance in either coordinate) produce NaN and are dropped
before the percentile step; if every resample is degenerate the result is a
NaN tuple.

- Returns a NaN tuple when `len(x) == 0`, `n_resamples == 0`, or
  `len(y) != len(x)` (length mismatch does not raise).

### `bootstrap_se` / `bootstrap_var`

```python
bootstrap_se(x, stat="mean", n_resamples=1000, random_state=None) -> float
bootstrap_var(x, stat="mean", n_resamples=1000, random_state=None) -> float
```

`bootstrap_se` returns the standard deviation (ddof=1 across resamples) of the
resampled statistic — the Monte Carlo estimate of the statistic's standard
error. `bootstrap_var` is exactly `bootstrap_se(...) ** 2` (it delegates to
the same code path, so the two are always consistent for the same seed).

Unlike the CI functions, these two do **not** raise on an unsupported `stat`;
they return `NaN`.

**Why it matters.** The bootstrap SE is the workhorse for statistics with no
convenient analytic SE (medians, trimmed means, ratios). For the mean it
should closely track `std(x, ddof=1) / sqrt(n)`, which makes it a useful
sanity check.

### `bootstrap_t_ci_mean`

```python
bootstrap_t_ci_mean(x, n_resamples=1000, conf=0.95, random_state=None) -> (point, lo, hi)
```

Studentized (bootstrap-t) interval for the mean. Each resample produces a
pivot `t_b = (mean_b - mean_hat) / se_b`, where `se_b` is the resample's own
standard error; the interval is `mean_hat - t_(1-alpha/2) * se_hat` to
`mean_hat - t_(alpha/2) * se_hat`. The point estimate here is the **sample
mean**, not the mean of bootstrap means. Resamples with zero within-resample
variance are dropped; if none survive, `lo`/`hi` are NaN.

**Why it matters.** Studentizing makes the interval second-order accurate and
notably better than the percentile interval when the data have heavy tails or
substantial skew — the pivot's distribution stabilizes faster than the raw
statistic's. The cost is sensitivity to tiny `se_b` in small samples, which
can produce wide intervals.

- Requires `n >= 2`; returns a NaN tuple for `n <= 1` or `n_resamples == 0`.

### `bootstrap_bca_ci`

```python
bootstrap_bca_ci(x, stat="mean", n_resamples=1000, conf=0.95, random_state=None) -> (point, lo, hi)
```

Bias-corrected and accelerated (BCa) interval for `stat` in
`{"mean", "median", "std"}`. Implementation details:

- Bias correction `z0` from the fraction of bootstrap replicates strictly
  below the observed statistic (clamped away from 0/1 before the inverse
  normal CDF).
- Acceleration `a` from leave-one-out jackknife influence values, computed in
  O(n) for mean and std and via a single sort for the median.
- Adjusted percentiles are read from the sorted bootstrap distribution with
  linear interpolation; the normal CDF/quantile use Acklam-style
  approximations (adequate for CI work).
- The point estimate is the **observed sample statistic** (for `"std"`, the
  ddof=1 sample standard deviation; bootstrap replicates for `"std"` are also
  ddof=1).

**Why it matters.** BCa corrects the two failure modes of the percentile
interval: median bias (the bootstrap distribution not being centered on the
estimate) and non-constant variance of the statistic as the parameter varies
(skew). It is second-order accurate and is the recommended default interval
for skewed statistics such as `"std"` or the median at moderate `n`. It costs
one extra jackknife pass; for very heavy-tailed data on the mean,
`bootstrap_t_ci_mean` is a strong alternative.

- Requires `n >= 3`; returns a NaN tuple for `n <= 2` or `n_resamples == 0`.

### `bayesian_bootstrap_ci`

```python
bayesian_bootstrap_ci(x, stat="mean", n_resamples=1000, conf=0.95, random_state=None) -> (point, lo, hi)
```

Rubin's Bayesian bootstrap: instead of resampling observations, each replicate
draws Dirichlet(1, …, 1) weights (via normalized Exp(1) draws) and computes
the weighted statistic — weighted mean, weighted population-style std, or
weighted median (first value whose cumulative weight reaches 0.5). The
interval is the equal-tailed credible interval from the replicate
distribution; the point estimate is the plain sample statistic.

**Why it matters.** The Bayesian bootstrap produces a smooth posterior over
the statistic without the discreteness artifacts of with-replacement
resampling (no observation is ever entirely absent from a replicate, only
down-weighted). Replicate distributions are typically slightly smoother and
slightly narrower than the classical bootstrap in small samples.

- Returns a NaN tuple when `n == 0`, `n_resamples == 0`, `conf` outside
  `(0, 1)`, or (`stat="std"`) `n < 2`.

---

## Block bootstraps (time series)

All three functions target the mean of a 1-D series whose observations are
autocorrelated. Naive i.i.d. resampling destroys serial dependence and
typically **understates** the variance of the sample mean for positively
autocorrelated data, producing overconfident (too-narrow) intervals. Block
methods resample contiguous runs so short-range dependence survives inside
each block. Choose `block_len` large enough to capture the dependence length
(a common heuristic is on the order of `n^(1/3)`).

All three return `(point, lo, hi)` where `point` is the **sample mean** of
`x`, and a NaN tuple when `n == 0`, `n_resamples == 0`, or `block_len == 0`.

### `moving_block_bootstrap_mean_ci`

```python
moving_block_bootstrap_mean_ci(x, block_len, n_resamples=1000, conf=0.95, random_state=None)
```

Draws blocks starting at uniform random positions; a block that would run past
the end of the series is truncated at the boundary (this implementation does
not wrap). Blocks are concatenated until at least `n` observations are
collected; the replicate statistic is the mean over the collected values.

### `circular_block_bootstrap_mean_ci`

```python
circular_block_bootstrap_mean_ci(x, block_len, n_resamples=1000, conf=0.95, random_state=None)
```

Same scheme but the series is treated as a circle: blocks wrap around from the
end to the beginning, so every observation has equal inclusion probability.
This removes the edge-effect bias of the moving-block scheme, where
observations near the boundaries are under-sampled.

### `stationary_bootstrap_mean_ci`

```python
stationary_bootstrap_mean_ci(x, block_len, n_resamples=1000, conf=0.95, random_state=None)
```

Politis–Romano stationary bootstrap: block lengths are geometric with mean
`block_len` (restart probability `p = 1/block_len` at each step, with circular
wrapping), so replicates form a stationary series. Less sensitive to the exact
choice of `block_len` than fixed-block schemes; a good default when the
dependence length is uncertain.

**Note on seeding.** These three functions use a single sequential RNG stream
(not per-resample mixing), seeded with `random_state.unwrap_or(0)`:
`random_state=None` equals `random_state=0` and repeated calls are
bit-identical. See [Determinism & seeding](#determinism--seeding).

---

## Permutation tests

Both tests return `(observed_statistic, p_value)` with add-one smoothing:
`p = (n_extreme + 1) / (n_permutations + 1)`, which avoids reporting p = 0 and
is the standard finite-sample-valid estimate. `alternative` is one of
`"two-sided"`, `"greater"`, `"less"`.

> **Caveat:** an unrecognized `alternative` string is **not** rejected — it
> counts zero permutations as extreme and silently returns
> `p = 1 / (n_permutations + 1)`, which looks highly significant. Spell the
> alternative correctly.

### `permutation_corr_test`

```python
permutation_corr_test(x, y, n_permutations=1000, alternative="two-sided", random_state=None) -> (obs, p)
```

Tests the null of zero association by Fisher–Yates shuffling `y` against fixed
`x` and recomputing the Pearson correlation each time. Degenerate permutations
(zero variance) are counted as non-extreme. If the observed correlation is
itself undefined (constant `x` or `y`), returns `(NaN, NaN)`; likewise for
empty input, `n_permutations == 0`, or length mismatch.

### `permutation_mean_diff_test`

```python
permutation_mean_diff_test(x, y, n_permutations=1000, alternative="two-sided", random_state=None) -> (obs, p)
```

Two-sample test of `mean(x) - mean(y)`: pools both samples, shuffles the pool,
resplits into groups of the original sizes, and recomputes the difference.
`x` and `y` may have different lengths. `"greater"` tests whether `mean(x)`
exceeds `mean(y)`.

**Why it matters.** Permutation tests give exact finite-sample error control
under exchangeability with no distributional assumptions — a robust
alternative to the t-test / correlation t-approximation for small samples,
skewed data, or outliers.

---

## Jackknife family

The jackknife is deterministic (no RNG) except for
`jackknife_after_bootstrap_se_mean`, which embeds a bootstrap.

### `jackknife_mean`

```python
jackknife_mean(x) -> (estimate, bias, se)
```

Leave-one-out jackknife for the mean: returns the bias-corrected jackknife
estimate `n * theta_full - (n-1) * mean(theta_loo)`, the estimated bias
(`estimate - sample_mean`; identically ~0 for the mean, since the mean is
unbiased), and the jackknife SE `sqrt((n-1)/n * sum((theta_i - theta_bar)^2))`.

- Requires `n >= 2`; returns a NaN tuple otherwise.

### `jackknife_mean_ci`

```python
jackknife_mean_ci(x, conf=0.95) -> (estimate, lo, hi)
```

Jackknife estimate plus a percentile interval taken **over the n leave-one-out
means**. Note this is a spread of LOO estimates, which is much narrower than a
sampling-distribution CI (each LOO mean differs from the full mean by
O(1/n)) — treat it as a leave-one-out sensitivity band, not as a substitute
for a bootstrap CI of the mean.

### `influence_mean`

```python
influence_mean(x) -> ndarray  # shape (n,), float64
```

Leave-one-out influence values `infl[i] = (n-1) * (theta_full - theta_loo_i)`,
which for the mean simplifies to `x[i] - mean(x)`; the values sum to ~0.

**Why it matters.** Influence values identify which observations drive an
estimate (outlier/leverage diagnostics) and are the ingredient for the BCa
acceleration constant. For `n <= 1` returns an **empty array** rather than
NaN.

### `delete_d_jackknife_mean`

```python
delete_d_jackknife_mean(x, d) -> (estimate, bias, se)
```

Delete-d jackknife using contiguous, non-overlapping blocks of size `d`
(`ceil(n/d)` blocks; the last block may be short). Returns the bias-corrected
estimate, bias, and an SE computed across block-deleted estimates.

**Why it matters.** The delete-1 jackknife is inconsistent for non-smooth
statistics (e.g. the median); deleting groups restores consistency. Contiguous
blocks also make it a simple dependence-aware diagnostic for serial data.

- Requires `n >= 3`, `d >= 1`, and `d <= n - 2`; otherwise returns a NaN
  tuple.

### `jackknife_after_bootstrap_se_mean`

```python
jackknife_after_bootstrap_se_mean(x, n_resamples=200, random_state=None) -> float
```

Jackknife-after-bootstrap (JAB) diagnostic: for each i, recomputes the
bootstrap SE of the mean on the sample with observation i removed, then
jackknifes those n SE values. The result estimates the sampling variability of
the bootstrap SE itself.

**Why it matters.** The bootstrap SE is itself a random estimate; JAB answers
"how much would my bootstrap SE move if the data changed slightly?" A large
JAB value relative to the SE signals that `n` or `n_resamples` is too small,
or that a few observations dominate the SE. It costs roughly `n` full
bootstraps — keep `n_resamples` modest (100–500).

- Requires `n >= 3`; returns NaN for `n <= 2` or `n_resamples == 0`.
- Follows the shared seeding policy: `random_state=None` equals
  `random_state=0` (bit-identical results).

---

## Determinism & seeding

All randomized kernels use the PCG64 generator (`rand_pcg::Pcg64`) and one
seeding policy: **`random_state=None` means seed 0.** For every resampler in
this module, `random_state=None` and `random_state=0` return bit-identical
results, and repeated calls (seeded or not) are bit-identical.

**Per-resample seed mixing (i.i.d. bootstrap family, permutation tests,
JAB).** Each resample b gets its own generator seeded with
`mix_seed(base_seed, b) = base_seed * 0x9E3779B97F4A7C15 + b` (wrapping
arithmetic; golden-ratio multiply for avalanche). Consequences:

- Resample b's draw sequence depends only on `(base_seed, b)` — never on
  thread scheduling or on other resamples. Results are identical whether the
  crate is built with or without the `parallel` (rayon) feature, and across
  thread counts.
- This covers `bootstrap_mean`, `bootstrap_mean_ci`, `bootstrap_ci`,
  `bootstrap_corr`, `bootstrap_se`, `bootstrap_var`, `bootstrap_t_ci_mean`,
  `bootstrap_bca_ci`, `bayesian_bootstrap_ci`, `permutation_corr_test`,
  `permutation_mean_diff_test`, and `jackknife_after_bootstrap_se_mean`.

**Block bootstraps** (`moving_block_bootstrap_mean_ci`,
`circular_block_bootstrap_mean_ci`, `stationary_bootstrap_mean_ci`) use a
single sequential PCG64 stream seeded with `random_state.unwrap_or(0)` — the
same None-means-0 policy. (Before 0.3.0 they drew entropy when unseeded;
unseeded calls are now reproducible.)

**Parallel aggregation.** With the `parallel` cargo feature, resamples are
evaluated with rayon. Every function collects per-resample statistics into a
vector (rayon's indexed `collect` preserves order) and reduces sequentially —
including `bootstrap_mean`, which as of 0.3.0 performs its final average as a
serial sum — or reduces with an exact integer sum (permutation tests). Output
therefore does not depend on thread count or work-stealing schedule.

Without the `parallel` feature, all iteration is sequential
(`IntoParIterCompat` maps `into_par_iter()` to `into_iter()`).

## Edge cases

Invalid sizes return NaN sentinels rather than raising; only an unsupported
`stat` string raises `ValueError`.

| Condition | Functions | Result |
|---|---|---|
| `len(x) == 0` or `n_resamples == 0` | all bootstrap CI functions, `bootstrap_se`, `bootstrap_var`, block bootstraps | `NaN` scalar or `(NaN, NaN, NaN)` |
| `n_permutations == 0` or empty input | permutation tests | `(NaN, NaN)` |
| `len(y) != len(x)` | `bootstrap_corr`, `permutation_corr_test` | NaN tuple (no exception) |
| `n <= 1` | `bootstrap_t_ci_mean`, `jackknife_mean`, `jackknife_mean_ci` | `(NaN, NaN, NaN)` |
| `n <= 1` | `influence_mean` | empty array (length 0) |
| `n <= 2` | `bootstrap_bca_ci`, `delete_d_jackknife_mean`, `jackknife_after_bootstrap_se_mean` | NaN tuple / NaN |
| `d == 0` or `d >= n - 1` | `delete_d_jackknife_mean` | `(NaN, NaN, NaN)` |
| `block_len == 0` | block bootstraps | `(NaN, NaN, NaN)` |
| `conf` outside `(0, 1)` | `bayesian_bootstrap_ci` | NaN tuple (other CI functions do not validate `conf`) |
| constant `x` or `y` | `permutation_corr_test` | `(NaN, NaN)` |
| all resamples degenerate | `bootstrap_corr` | NaN tuple; `bootstrap_t_ci_mean`, `bayesian_bootstrap_ci` → `(point, NaN, NaN)` |
| `stat` not in `{"mean","median","std"}` | `bootstrap_ci`, `bootstrap_bca_ci`, `bayesian_bootstrap_ci` | raises `ValueError` |
| `stat` not in `{"mean","median","std"}` | `bootstrap_se`, `bootstrap_var` | returns `NaN` (no exception) |
| unknown `alternative` | permutation tests | **not rejected**; returns `p = 1/(n_permutations+1)` |
| NaN in input | all | NaN propagates into resample statistics (no filtering) |
