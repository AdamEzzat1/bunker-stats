# Inference Module Reference

Statistical hypothesis tests and effect-size estimators for the `bunker-stats` extension.
All functions are implemented in Rust (PyO3 + `statrs`) and exposed to Python through the
compiled `bunker_stats_rs` module. The pure-Python facade package `bunker_stats` re-exports
each function under a clean alias without the `_np` suffix (e.g. `bunker_stats.t_test_1samp`
wraps `t_test_1samp_np`); the `*_np` names remain available as deprecated aliases at the
facade level.

Source layout (`src/infer/`):

| File | Contents |
|---|---|
| `ttest.rs` | `t_test_1samp_np`, `t_test_2samp_np` |
| `chi2.rs` | `chi2_gof_np`, `chi2_independence_np` |
| `anova.rs` | `f_test_oneway_np`, `levene_test_np` |
| `ks.rs` | `ks_1samp_np` |
| `mann_whitney.rs` | `mann_whitney_u_np` |
| `normality.rs` | `jarque_bera_np`, `anderson_darling_np` |
| `correlation.rs` | `pearson_corr_test_np`, `spearman_corr_test_np` |
| `variance_tests.rs` | `f_test_var_np`, `bartlett_test_np` |
| `effect.rs` | `cohens_d_2samp_np`, `hedges_g_2samp_np2`, `mean_diff_ci_np` |
| `common.rs` | Shared helpers: `Alternative` parsing, non-finite rejection, Kahan-compensated mean/variance, average ranking, median |


## Using the Python facade

New code should reach these kernels through the `bunker_stats` facade, which
exposes clean names with keyword arguments. The raw `bunker_stats_rs` names
documented below remain available and stable; the facade adds ergonomics on
top of the same kernels:

```python
import bunker_stats as bs

bs.t_test_2samp(x, y)                      # defaults: pooled, two-sided
bs.t_test_2samp(x, y, equal_var=False)     # Welch's t
bs.cohens_d_2samp(x, y)                    # pooled=True default
```

Where a statistic has strict and skip-NaN variants, the facade exposes ONE
name with a `skipna=` keyword; `skipna=True` dispatches to the skip-NaN kernel
documented below (the twin kernels stay separate in Rust, so there is no
branch inside the hot loop).

## API contract

- **Input type**: every array argument must be a C-contiguous 1-D (or 2-D for
  `chi2_independence_np`) `numpy.ndarray` of `float64`. Integer arrays raise `TypeError`
  (`'ndarray' object cannot be converted`); non-contiguous views (e.g. `a[::2]`) raise
  `TypeError: The given array is not contiguous`. Cast with `np.ascontiguousarray(a, dtype=np.float64)`
  when in doubt.
- **NaN policy**: `reject`. Any NaN or Inf in an input raises
  `ValueError: <name> contains NaN or Inf; bunker-stats v0.3 nan_policy is 'reject'`.
  There is no `nan_policy` parameter; drop or impute missing values before calling.
- **`alternative` parameter**: where present, accepts `"two-sided"` (also spelled
  `"two_sided"`), `"less"`, or `"greater"`, with SciPy semantics. Anything else raises
  `ValueError`.
- **Return format**: tests return a plain `dict` (exact keys listed per function below).
  Effect sizes return a `float`; `mean_diff_ci_np` returns a `(lower, upper)` tuple.
- **Errors**: all input-validation failures raise `ValueError` (or `TypeError` for
  dtype/contiguity issues). No function silently returns NaN for invalid input; degenerate
  but well-defined cases (e.g. zero variance in a t-test) return `0.0` or `±inf` as
  documented in the edge-case table.

## Function summary

| Family | Function | Returns | Min n |
|---|---|---|---|
| t-tests | `t_test_1samp_np(x, popmean, alternative="two-sided")` | `{'statistic','pvalue','df','mean'}` | n ≥ 2 |
| t-tests | `t_test_2samp_np(x, y, equal_var, alternative="two-sided")` | `{'statistic','pvalue','df','mean_x','mean_y','equal_var'}` | n ≥ 2 each |
| Chi-square | `chi2_gof_np(observed, expected=None, *, sum_check=True)` | `{'statistic','pvalue','df'}` | k ≥ 2 categories |
| Chi-square | `chi2_independence_np(table)` | `{'statistic','pvalue','df'}` | ≥ 2×2 table |
| ANOVA | `f_test_oneway_np(groups)` | `{'statistic','pvalue','df_between','df_within'}` | ≥ 2 groups, n ≥ 2 each |
| ANOVA | `levene_test_np(groups)` | `{'statistic','pvalue','df_between','df_within'}` | ≥ 2 groups, non-empty |
| Variance | `bartlett_test_np(groups)` | `{'statistic','pvalue','df'}` | ≥ 2 groups, n ≥ 2 each |
| Variance | `f_test_var_np(x, y)` | `{'statistic','pvalue','df1','df2'}` | n ≥ 2 each |
| KS | `ks_1samp_np(x, cdf, params, alternative="two-sided")` | `{'statistic','pvalue'}` | n ≥ 1 |
| Rank | `mann_whitney_u_np(x, y, alternative="two-sided")` | `{'statistic','pvalue'}` | non-empty each |
| Normality | `jarque_bera_np(x)` | `{'statistic','pvalue','skewness','kurtosis'}` | n ≥ 4 |
| Normality | `anderson_darling_np(x)` | `{'statistic'}` | n ≥ 2 |
| Correlation | `pearson_corr_test_np(x, y)` | `{'correlation','statistic','pvalue','df'}` | n ≥ 3 |
| Correlation | `spearman_corr_test_np(x, y)` | `{'correlation','statistic','pvalue','df'}` | n ≥ 3 |
| Effect size | `cohens_d_2samp_np(x, y, pooled)` | `float` | n ≥ 2 each |
| Effect size | `hedges_g_2samp_np2(x, y, pooled)` | `float` | n ≥ 2 each |
| Effect size | `mean_diff_ci_np(x, y=None, alpha=0.05, equal_var=True)` | `(lower, upper)` tuple | n ≥ 2 each |

Multi-group tests (`f_test_oneway_np`, `levene_test_np`, `bartlett_test_np`) take a single
Python list of 1-D arrays, not variadic arguments: `f_test_oneway_np([g1, g2, g3])`.

---

## t-tests

### `t_test_1samp_np(x, popmean, alternative="two-sided")`

One-sample Student t-test of H₀: mean(x) = `popmean`.

```python
import numpy as np
import bunker_stats_rs as bs

x = np.random.default_rng(0).normal(0.1, 1.0, 50)
res = bs.t_test_1samp_np(x, 0.0, "two-sided")
# {'statistic': ..., 'pvalue': ..., 'df': 49.0, 'mean': ...}
```

**Parameters**
- `x` (1-D float64 array): sample, n ≥ 2.
- `popmean` (float): hypothesized population mean.
- `alternative` (str): `"two-sided"` (default), `"less"`, `"greater"`.

**Returns** `{'statistic': t, 'pvalue': p, 'df': n-1, 'mean': sample_mean}`.

**Why it matters**: the standard location test when the population variance is unknown.
Variance is computed with Kahan-compensated summation (ddof = 1).

**Assumptions**: observations approximately normal, or n large enough for the CLT.
For heavy-tailed or ordinal data, prefer a rank-based procedure or use
`permutation_mean_diff_test` from the resampling module.

**Edge cases**: constant `x` equal to `popmean` gives `t = 0.0, p = 1.0`; constant `x`
different from `popmean` gives `t = ±inf, p = 0.0` (p is 0.0 regardless of `alternative`
in this degenerate branch). p-values match SciPy `ttest_1samp` to ≈1e-13 for all three
alternatives.

### `t_test_2samp_np(x, y, equal_var, alternative="two-sided")`

Two-sample t-test of H₀: mean(x) = mean(y). **`equal_var` is a required positional
argument** (no default): `True` runs the pooled-variance Student test with
df = n₁+n₂−2; `False` runs **Welch's t-test** with Welch–Satterthwaite df. Both match
`scipy.stats.ttest_ind` (statistic, p-value, and df) to ≈1e-12.

```python
res = bs.t_test_2samp_np(x, y, False)          # Welch
res = bs.t_test_2samp_np(x, y, True, "less")   # pooled, one-sided
```

**Returns** `{'statistic': t, 'pvalue': p, 'df': df, 'mean_x': m1, 'mean_y': m2, 'equal_var': bool}`.

**Why it matters**: the workhorse two-group mean comparison. Use `equal_var=False`
(Welch) by default unless there is a strong prior reason to assume equal variances —
Welch is safe under heteroscedasticity and loses almost no power when variances are
equal.

**Assumptions**: normality within each group (pooled additionally assumes equal
variances). Under non-normality, switch to `mann_whitney_u_np` or a permutation test.

**Implementation notes**: the Welch df is guarded against degenerate variance inputs —
if both variances are zero the pooled df is used, and the computed df is clamped to
`[1, n1+n2-2]` (a no-op for regular data since the Welch df always lies in
`[min(n1,n2)-1, n1+n2-2]`). Both samples constant and equal (|Δ| < 1e-15) gives
`t = 0, p = 1`; constant and different gives `t = ±inf, p → {0, 1}` per the alternative.

---

## Chi-square tests

### `chi2_gof_np(observed, expected=None, *, sum_check=True)`

Chi-square goodness-of-fit test. With `expected=None`, tests against a uniform
distribution over the k categories. `sum_check` is **keyword-only**.

```python
obs = np.array([16., 18., 16., 14., 12., 12.])
exp = np.array([16., 16., 16., 16., 16.,  8.])
bs.chi2_gof_np(obs, exp)          # matches scipy.stats.chisquare(obs, exp)
bs.chi2_gof_np(obs)               # uniform expected
```

**Parameters**
- `observed` (1-D float64 array): non-negative counts, k ≥ 2 categories, sum > 0.
- `expected` (1-D float64 array, optional): strictly positive, same length as `observed`.
- `sum_check` (bool, keyword-only, default `True`): require `sum(observed)` and
  `sum(expected)` to agree to relative tolerance `sqrt(eps)` (≈1.49e-8), mirroring the
  SciPy consistency check. Set `False` to skip (e.g. when `expected` are rates rather
  than matched counts).

**Returns** `{'statistic': chi2, 'pvalue': p, 'df': k-1}`. There is no `ddof` parameter;
if parameters were estimated from the data, adjust the reference df yourself.

**Why it matters**: the standard test for "do these counts match this distribution".
Statistic accumulation uses Kahan summation; parity with `scipy.stats.chisquare` is
≈1e-15 on both statistic and p-value.

**Assumptions**: counts are independent and expected cell counts are not tiny (the usual
rule of thumb is every expected count ≥ 5); for sparse tables use an exact test.

### `chi2_independence_np(table)`

Chi-square test of independence on an r×c contingency table.

```python
table = np.array([[10., 20.], [20., 10.]])
bs.chi2_independence_np(table)
# {'statistic': 6.6667, 'pvalue': 0.00982, 'df': 1.0}
```

**Parameters**: `table` — 2-D C-contiguous float64 array, at least 2×2, non-negative
entries, positive total. Cells whose expected count is 0 (an all-zero row/column) are
skipped in the statistic.

**Returns** `{'statistic': chi2, 'pvalue': p, 'df': (r-1)(c-1)}`.

**Important convention — no Yates correction**: for 2×2 tables this function applies
**no continuity correction**. It reproduces
`scipy.stats.chi2_contingency(table, correction=False)`, whereas SciPy's *default* is
`correction=True` for 2×2 tables. Expect a larger statistic and smaller p-value than
SciPy's default on 2×2 inputs (e.g. the table above: 6.667/0.0098 here vs 5.4/0.0201
with Yates). For r×c tables larger than 2×2 the correction never applies and results
agree with SciPy's default.

**Assumptions**: independent observations, adequate expected cell counts. For small 2×2
tables where the correction (or exactness) matters, use Fisher's exact test instead.

---

## ANOVA and variance tests

### `f_test_oneway_np(groups)`

One-way ANOVA F-test of H₀: all group means equal. Takes a list of 1-D arrays.

```python
bs.f_test_oneway_np([g1, g2, g3])
# {'statistic': F, 'pvalue': p, 'df_between': k-1, 'df_within': N-k}
```

**Requirements**: ≥ 2 groups, each with ≥ 2 observations. Verified to match
`scipy.stats.f_oneway` to ≈1e-14 on statistic and p-value.

**Why it matters**: the standard omnibus test for comparing more than two group means
without inflating the type-I rate via pairwise t-tests.

**Assumptions**: normal residuals and equal group variances. Check the variance
assumption with `levene_test_np`; when it fails, note that this module does not provide
a Welch ANOVA — fall back to `scipy.stats.f_oneway`-alternatives or pairwise Welch
t-tests with a multiplicity correction.

**Edge cases**: all groups constant and identical → `F = 0, p = 1`; within-variance zero
but means differ → `F = inf, p = 0`.

### `levene_test_np(groups)`

Levene's test of H₀: all group variances equal — **median-centered variant**
(Brown–Forsythe). Each observation is transformed to |x − median(group)| and a one-way
ANOVA is run on the transformed values. This is exactly
`scipy.stats.levene(*groups, center='median')`, which is also SciPy's default center;
parity verified to ≈1e-14.

```python
bs.levene_test_np([g1, g2, g3])
# {'statistic': W, 'pvalue': p, 'df_between': k-1, 'df_within': N-k}
```

**Requirements**: ≥ 2 groups; each group need only be **non-empty** (n = 1 groups are
accepted, unlike `bartlett_test_np`).

**Why it matters**: the robust pre-check for the equal-variance assumption of pooled
t-tests and ANOVA. Median centering makes it far less sensitive to non-normality than
Bartlett's test.

**Edge cases**: all groups constant → `W = 0, p = 1` (the zero-within-variance branch
returns 0 rather than inf).

### `bartlett_test_np(groups)`

Bartlett's test of H₀: equal variances across k groups. Matches `scipy.stats.bartlett`
to ≈1e-14.

```python
bs.bartlett_test_np([g1, g2, g3])
# {'statistic': chi2, 'pvalue': p, 'df': k-1}
```

**Requirements**: ≥ 2 groups, each with n ≥ 2 and strictly positive sample variance
(a constant group raises `ValueError: all group variances must be positive`).

**Why it matters**: more powerful than Levene when the data really are normal.

**Assumptions**: strongly normality-dependent — under heavy tails it rejects far too
often. Prefer `levene_test_np` unless normality is well established.

### `f_test_var_np(x, y)`

F-test of H₀: var(x) = var(y).

```python
bs.f_test_var_np(x, y)
# {'statistic': F, 'pvalue': p, 'df1': ..., 'df2': ...}
```

**Convention — symmetric upper-tail form**: the statistic is always
`larger_variance / smaller_variance` (F ≥ 1), with `df1` the df of the larger-variance
sample and `df2` the smaller's. The two-sided p-value is `2 · P(F_{df1,df2} ≥ F)`.
Consequently `f_test_var_np(x, y) == f_test_var_np(y, x)` exactly (verified), and the
`df1`/`df2` values follow the variance ordering, not the argument order. SciPy has no
direct equivalent (`scipy.stats.f` gives the distribution only); if comparing to a
textbook F = s₁²/s₂² form, account for this ordering convention.

**Requirements**: n ≥ 2 in both samples; both variances strictly positive (constant
input raises `ValueError`).

**Assumptions**: extremely normality-sensitive — this test is unreliable under any
non-normality. Prefer `levene_test_np` for real data.

---

## Kolmogorov–Smirnov

### `ks_1samp_np(x, cdf, params, alternative="two-sided")`

One-sample KS test of H₀: x is drawn from the named distribution.

```python
bs.ks_1samp_np(x, "norm", [0.0, 1.0])            # N(loc=0, scale=1)
bs.ks_1samp_np(u, "uniform", [2.0, 3.0])         # Uniform on [2, 5]
bs.ks_1samp_np(e, "expon", [1.0, 2.0])           # shifted exponential
```

**Parameters**
- `x` (1-D float64 array): n ≥ 1.
- `cdf` (str): `"norm"`/`"normal"`, `"uniform"`, or `"expon"`/`"exponential"` — nothing
  else is supported.
- `params` (list of 2 floats): `[loc, scale]` with SciPy parameterization
  (uniform covers `[loc, loc+scale]`; expon has rate `1/scale`); `scale > 0` required.
- `alternative` (str): `"two-sided"`, `"less"` (uses D⁻), `"greater"` (uses D⁺) —
  SciPy's one-sided conventions.

**Returns** `{'statistic': D, 'pvalue': p}`.

**Why it matters**: distribution-free test of a fully specified null distribution.

**p-value methods**:
- Two-sided, n ≤ 10 000: exact finite-n survival function via the Durbin matrix method
  (Marsaglia–Tsang–Wang), the same core as SciPy's `kstwo.sf`. Verified against
  `scipy.stats.ks_1samp` to ≈1e-15 for `norm`, `uniform`, and `expon`.
- Two-sided, n > 10 000: asymptotic Kolmogorov series with the Stephens finite-n
  correction factor `(sqrt(n) + 0.12 + 0.11/sqrt(n))`.
- **One-sided (`less`/`greater`): asymptotic approximation `p ≈ exp(-2 n D²)` only.**
  This does *not* match SciPy's exact one-sided p-values (e.g. n = 30: 0.479 here vs
  0.4465 from SciPy); the approximation overstates p slightly (conservative). The
  statistic itself matches exactly.

**Caveat**: like all KS tests against a named distribution, parameters must be specified
a priori. If `loc`/`scale` are estimated from the same data, the p-value is invalid
(use Lilliefors-type corrections or `anderson_darling_np` for normality).

---

## Rank-based

### `mann_whitney_u_np(x, y, alternative="two-sided")`

Mann–Whitney U test (Wilcoxon rank-sum) of H₀: the two distributions are equal.
Asymptotic normal approximation with **average ranks for ties, tie-corrected variance,
and a continuity correction** — equivalent to
`scipy.stats.mannwhitneyu(x, y, method="asymptotic")` (which uses `use_continuity=True`
by default). Two-sided and one-sided p-values verified to ≈1e-10 against SciPy's
asymptotic method.

```python
bs.mann_whitney_u_np(x, y)             # {'statistic': U, 'pvalue': p}
bs.mann_whitney_u_np(x, y, "greater")
```

**Statistic convention — differs from current SciPy for two-sided**: for one-sided
alternatives the returned statistic is U₁ (the U of the first argument `x`), matching
SciPy. For `alternative="two-sided"` the returned statistic is **min(U₁, U₂)** (the
classical tabulated U), whereas SciPy ≥ 0.17 always returns U₁. Verified: with samples
where U₁ = 507, U₂ = 243, this function returns 243 two-sided (SciPy: 507) and 507 for
`greater` (SciPy: 507). p-values are unaffected. Recover the counterpart via
`U1 + U2 = n1 * n2`.

**Why it matters**: the default two-group comparison when normality cannot be assumed;
tests stochastic dominance rather than means.

**Method note**: p-values are always asymptotic; there is no exact small-sample method
(SciPy's default switches to the exact method for small tie-free samples, so small-n
tie-free results differ from SciPy's *default* but match `method="asymptotic"`).
The approximation is standardly considered adequate for n₁, n₂ ≳ 8.

**Edge cases**: empty input raises; all values tied across both samples gives
`sd = 0 → z = 0, p = 1`.

---

## Normality tests

### `jarque_bera_np(x)`

Jarque–Bera test of H₀: skewness = 0 and excess kurtosis = 0 (normality), with the
statistic referred to χ²(2). Matches `scipy.stats.jarque_bera` to ≈1e-14. Requires
n ≥ 4 and non-zero variance.

```python
bs.jarque_bera_np(x)
# {'statistic': JB, 'pvalue': p, 'skewness': g1, 'kurtosis': b2}
```

**Return-key note**: `'skewness'` is the biased moment estimator m₃/m₂^1.5;
`'kurtosis'` is the **raw (Pearson) kurtosis** m₄/m₂² — i.e. ≈ 3.0 for normal data —
*not* the excess kurtosis that `scipy.stats.kurtosis` returns (subtract 3 to compare).

**Why it matters**: cheap asymptotic normality screen; standard in econometrics.
The χ² approximation is poor below n ≈ 50 (over-rejects); prefer `anderson_darling_np`
or a Shapiro-type test for small samples.

### `anderson_darling_np(x)`

Anderson–Darling test statistic for normality, with mean and variance estimated from
the sample.

```python
bs.anderson_darling_np(x)
# {'statistic': A2_star}
```

**Returns `{'statistic'}` only — no p-value.** Compare the statistic against fixed
critical values (case 3, both parameters estimated; e.g. 0.787 at 5 % for large n, or
the significance tables SciPy ships).

**Convention — corrected statistic**: the returned value is the small-sample-corrected
statistic `A²* = A² · (1 + 4/n − 25/n²)` (Stephens 1974). `scipy.stats.anderson`
returns the **uncorrected** A² (its critical-value table absorbs the correction
differently). Verified: `scipy_A2 * (1 + 4/n - 25/n**2)` equals this function's output
to ≈1e-10. CDF/SF evaluations are clamped to [1e-300, 1−1e-300] to keep the logs finite
for extreme observations.

**Requirements**: n ≥ 2, non-zero variance.

**Why it matters**: more powerful than KS in the tails, and valid when the normal
parameters are estimated from the data (which invalidates a naive `ks_1samp_np` against
`"norm"`).

---

## Correlation tests

### `pearson_corr_test_np(x, y)` / `spearman_corr_test_np(x, y)`

Test of H₀: zero (linear / monotonic) association, two-sided only. Both use the exact
t-transform `t = r · sqrt(df / (1 − r²))` with df = n − 2. Spearman ranks both inputs
with average ranks for ties and then applies the Pearson machinery to the ranks —
equivalent to `scipy.stats.spearmanr` (tie handling verified). Parity with
`scipy.stats.pearsonr`/`spearmanr` p-values ≈1e-13.

```python
bs.pearson_corr_test_np(x, y)
# {'correlation': r, 'statistic': t, 'pvalue': p, 'df': n-2.0}
```

**Requirements**: equal lengths, n ≥ 3, non-zero variance in both inputs (for Spearman:
non-zero variance *of the ranks*, so any constant input fails).

**Why it matters**: Pearson for linear association under approximate bivariate
normality; Spearman as the robust alternative under monotone-but-nonlinear
relationships, outliers, or ordinal data. For small samples or exact inference under
exotic nulls, use `permutation_corr_test` from the resampling module.

**Edge cases**: |r| = 1 exactly returns `statistic = ±inf, pvalue = 0.0`. There is no
`alternative` parameter (two-sided only).

---

## Effect sizes and confidence intervals

### `cohens_d_2samp_np(x, y, pooled)`

Cohen's d = (mean(x) − mean(y)) / s. **`pooled` is a required positional argument.**
With `pooled=True`, s is the pooled standard deviation
`sqrt(((n1-1)s1² + (n2-1)s2²) / (n1+n2-2))`; with `pooled=False`, s is
`sqrt((s1² + s2²)/2)` (the unweighted RMS of the two SDs, ignoring sample sizes).
Returns a `float`. Zero denominator (both samples constant) returns `0.0` regardless of
the mean difference.

### `hedges_g_2samp_np2(x, y, pooled)`

Small-sample bias-corrected standardized mean difference:

```
g = d · J,   J = 1 − 3 / (4·(n1 + n2 − 2) − 1)
```

verified to reproduce `cohens_d_2samp_np(x, y, pooled) * J` exactly. Note the `_np2`
suffix — this is the only Hedges g symbol exported by the compiled module (the facade
exposes it as `bunker_stats.hedges_g_2samp`).

### `mean_diff_ci_np(x, y=None, alpha=0.05, equal_var=True)`

Analytic t-based confidence interval, returned as a `(lower, upper)` tuple of floats.

- `y=None`: CI for mean(x), df = n − 1.
- `y` given: CI for mean(x) − mean(y); `equal_var=True` uses the pooled SE with
  df = n₁+n₂−2, `equal_var=False` uses the Welch SE and Welch–Satterthwaite df (same
  guarded computation as `t_test_2samp_np`).

Coverage level is `1 − alpha` (default 95 %). Verified against the confidence intervals
of `scipy.stats.ttest_1samp` / `ttest_ind` (pooled and Welch) to ≈1e-12. No bootstrap —
purely analytic (see the resampling module for bootstrap CIs).

**Why effect sizes matter**: p-values conflate magnitude with sample size; report d or g
(with `mean_diff_ci_np` for the raw scale) alongside any t-test. Use Hedges g rather
than d whenever n₁ + n₂ ≲ 40.

---

## Conventions and SciPy-parity notes

All parity statements below were checked against SciPy 1.15 with float64 inputs.

| Topic | This module | SciPy comparison |
|---|---|---|
| Welch t-test | `t_test_2samp_np(..., equal_var=False)` is Welch's t | Matches `ttest_ind(equal_var=False)` statistic/p/df to ≈1e-12 |
| 2×2 chi-square | **No Yates continuity correction** | Matches `chi2_contingency(..., correction=False)`; differs from SciPy default `correction=True` on 2×2 tables |
| Levene center | Median-centered (Brown–Forsythe) | Identical to `levene(center='median')`, SciPy's default |
| Anderson–Darling | Returns corrected `A²* = A²(1 + 4/n − 25/n²)`; no p-value | `scipy.stats.anderson` returns raw A²; multiply by the factor to compare |
| Mann–Whitney statistic | `min(U1, U2)` for two-sided; `U1` for one-sided | SciPy always returns U₁; p-values match `method="asymptotic"` |
| Mann–Whitney p-value | Always asymptotic normal + tie & continuity corrections | SciPy default auto-switches to exact for small tie-free samples |
| KS two-sided p | Exact Durbin–Marsaglia for n ≤ 10 000, Stephens-corrected asymptotic above | Matches `ks_1samp` to ≈1e-15 in the exact regime |
| KS one-sided p | Asymptotic `exp(-2nD²)` only | SciPy uses exact one-sided SF; expect visible differences (≈0.03 absolute at n = 30) |
| `f_test_var_np` | F = larger/smaller variance (symmetric in arguments), p = 2 × upper tail | No SciPy equivalent function |
| Hedges g | `g = d × (1 − 3/(4(n₁+n₂−2) − 1))` | — |
| Jarque–Bera | Matches `jarque_bera` to ≈1e-14; returns raw kurtosis (normal ≈ 3), not excess | `scipy.stats.kurtosis` default is excess |
| ANOVA / Bartlett / correlations / chi2_gof / mean_diff_ci | Direct parity | ≈1e-13 or better |

Not provided by this module: paired t-test, Wilcoxon signed-rank, Kruskal–Wallis,
Welch ANOVA, Fisher exact, Shapiro–Wilk, two-sample KS, Kendall's tau, and p-values for
Anderson–Darling. Permutation-based mean-difference and correlation tests live outside
`src/infer/` (`permutation_mean_diff_test`, `permutation_corr_test`).

## Edge-case behavior

| Input condition | Behavior |
|---|---|
| NaN or Inf anywhere in an input array | `ValueError` (nan_policy `'reject'`) |
| Integer dtype or non-contiguous array | `TypeError` |
| `t_test_1samp_np`: n < 2 | `ValueError: t_test_1samp requires n >= 2` |
| `t_test_1samp_np`: constant x, mean == popmean | `t = 0.0, p = 1.0` |
| `t_test_1samp_np`: constant x, mean != popmean | `t = ±inf, p = 0.0` (regardless of alternative) |
| `t_test_2samp_np`: both samples constant, equal means | `t = 0.0, p = 1.0` |
| `t_test_2samp_np`: both samples constant, unequal means | `t = ±inf`, p from the t direction and alternative |
| `chi2_gof_np`: k < 2, negative counts, zero total, non-positive expected | `ValueError` |
| `chi2_gof_np`: sums disagree with `sum_check=True` | `ValueError` (relative tolerance ≈1.49e-8) |
| `chi2_independence_np`: table smaller than 2×2 / negative / zero total | `ValueError` |
| `f_test_oneway_np`: < 2 groups or any group n < 2 | `ValueError` |
| `f_test_oneway_np`: zero within-variance, equal means / unequal means | `F = 0, p = 1` / `F = inf, p = 0` |
| `levene_test_np`: < 2 groups or empty group | `ValueError` (n = 1 groups allowed) |
| `levene_test_np`: all groups constant | `W = 0, p = 1` |
| `bartlett_test_np`: any group constant or n < 2 | `ValueError` |
| `f_test_var_np`: either sample constant or n < 2 | `ValueError` |
| `ks_1samp_np`: empty x, unknown `cdf` name, `scale <= 0`, params length ≠ 2 | `ValueError` |
| `mann_whitney_u_np`: empty sample | `ValueError` |
| `mann_whitney_u_np`: all observations tied | `p = 1.0` |
| `jarque_bera_np`: n < 4 / zero variance | `ValueError` |
| `anderson_darling_np`: n < 2 / zero variance | `ValueError` |
| Correlation tests: length mismatch, n < 3, constant input | `ValueError` |
| Correlation tests: perfect correlation | `statistic = ±inf, p = 0.0` |
| `cohens_d_2samp_np` / `hedges_g_2samp_np2`: n < 2 | `ValueError` |
| `cohens_d_2samp_np`: zero pooled SD | returns `0.0` |
| `mean_diff_ci_np`: n < 2 in x or y | `ValueError` |

## Testing

Inference behavior is covered by the Python test suites under `tests/`
(e.g. `pytest tests/ -k "infer or ttest or chi2 or mann or ks or anova"`), plus Rust unit
tests in `src/infer/common.rs` (`cargo test`).
