# bunker-stats

**Production-grade statistical computing library combining Rust performance with Python ergonomics**

Version: 0.3.1  
Status: Released  
License: See LICENSE file

---

## Overview

`bunker-stats` is a high-performance statistical computing library that delivers production-grade functionality through Rust backend kernels with Python bindings via PyO3. The library emphasizes **deterministic results**, **numerical stability**, and **minimal allocations** while maintaining an intuitive, Pythonic API.

### Core Principles

🎯 **Deterministic** - Bit-exact given the same seed: every randomized routine is reproducible, and `random_state=None` behaves as seed 0  
⚡ **High-Performance** - 2-5× faster than SciPy/pandas/numpy on its hot paths (rolling windows, robust estimators, bootstrap); see [Performance](#performance-highlights) for measured, per-function numbers  
🔢 **Numerically Stable** - Kahan summation, Welford's algorithm, careful conditioning  
🧪 **Thoroughly Tested** - 53 Rust unit/property tests + 600+ Python tests (numpy/scipy/pandas/statsmodels parity, edge cases, and never-panic properties); ~99% of public functions have a direct reference test  
🔒 **Type-Safe** - Rust implementation with full input validation  
📦 **Zero Dependencies** - Core functionality requires only NumPy

---

## Release Notes — Unreleased

Notebook layer hardening (`bunker_stats.notebook`), plus three kernel fixes
found while reviewing it.

- **`rich=True` on every styler and transform.** Stylers return a
  `StyleResult` that unpacks as `styler, frame, metadata`; `scale_columns`,
  `winsorize_columns` and `robust_scale_column` return a `TransformResult`
  (`frame, metadata`). Metadata carries sample sizes, fitted values, fences,
  counts and the colors used. The default return types are unchanged.
- **New helpers:** `missingness_style` (where the gaps are),
  `scale_compare_style` (classical vs robust scores side by side, to expose
  masking) and `available_themes`.
- **Themes.** `theme="default" | "colorblind"` or a `role -> color` mapping on
  every styler. Styled cells now set a text color as well as a background, so
  they stay legible in dark-mode notebooks.

Behavior changes (each replaces a wrong or late-failing result):

- Duplicate column labels raise `ValueError`. They used to be pooled into one
  sample — `robust_summary` reported `n=6` for a 3-row frame.
- Complex columns raise `TypeError` instead of dropping the imaginary part
  behind a NumPy warning; `columns=None` no longer selects timedelta columns
  (which then crashed).
- Stylers raise `ValueError` for a non-unique index at the call, instead of a
  bare `KeyError` when the notebook cell renders.
- `corr_heatmap` paints undefined correlations with the theme's `missing`
  color. pandas rendered them solid black, the visually strongest cell.
- `threshold`, `k` and `z_threshold` must be finite and `> 0` everywhere
  (`NaN` and, in `outlier_style`, negative values used to be accepted).
- `style_significance` rejects p-values outside `[0, 1]`.
- The `outlier_report(rich=True)` zero-spread warning said "nothing can be
  flagged"; when the fences collapse, every value off that point *is* flagged.

Kernel fixes (Rust):

- **`qn_scale` no longer aborts the process on large input.** It materialised
  all n(n−1)/2 pairwise differences — 160 GB at n = 200,000 — and the failed
  allocation killed the interpreter with no Python exception, taking
  `robust_summary` and `describe_fast` with it. It now runs in O(n log n) time
  and O(n) memory, including for input containing ±inf.
- **`qn_scale` is now Rousseeuw & Croux's Qn.** It selected the pairwise
  difference of rank ⌊C(n,2)/4⌋ (the first quartile) and multiplied by a
  rounded 2.2219. The definition is the C(⌊n/2⌋+1, 2)-th smallest difference
  times 1/(√2·Φ⁻¹(5/8)) = 2.21914…; the two ranks coincide only for odd
  n ≤ 9. **Results change, mostly for small samples**: for normal data the
  old value was typically 0.68× the definition at n = 6, 0.82× at n = 10,
  0.87× at n = 20, 0.97× at n = 100 and 0.998× at n = 1000 (and always 0.37×
  at n = 2). It now equals `statsmodels.robust.scale.qn_scale` to 12
  digits. This reaches `robust_summary`, `describe_fast` and
  `robust_fit(scale="qn")`. No finite-sample correction is applied (as in
  statsmodels; R's `robustbase::Qn` applies one by default).
- **`corr_matrix(skipna=True)` is pairwise-complete.** It divided the
  pairwise covariance by standard deviations taken over each column's own
  rows, which is not the correlation of any one set of rows when columns are
  missing in different places (−0.696 where `DataFrame.corr()` gives −0.671
  on an 8×3 example) and could leave [−1, 1]. Each pair that involves a
  column with gaps is now computed on its own rows. **Results change** for
  data with missing values; gap-free data is unchanged.
- **Chi-square p-values between 4.9e-324 and 2.2e-308 are reported instead
  of 0.** The chi-square tail gave exactly 0 below the smallest *normal*
  float although smaller values are representable (the t and F tails already
  reached them). It is now evaluated in log space there and checked against
  60-digit references, for `chi2_gof`, `chi2_independence`, `bartlett_test`,
  `jarque_bera` and the time-series tests. A p-value of exactly 0.0 means
  the true value is below 4.9e-324.
- **Permutation tests can be interrupted.** `permutation_corr_test` and
  `permutation_mean_diff_test` do not store anything per permutation, so a
  huge `n_permutations` neither fails nor aborts: it runs. They now work in
  batches and check for Ctrl-C between them. Results are unchanged.
- **`Rolling(..., full_length=True)`** returns one value per observation for a
  trailing window, as pandas does: the first `window - 1` rows use partial
  windows and honour `min_periods`. The default output (`n - window + 1`
  values, one per full window) is unchanged. `rolling_report` uses it.
- **t-based p-values are accurate in the far tail.** Upper tails were
  computed as `1 - cdf(t)`, which cancels below about 1e-16, so
  `pearson_corr_test`, `spearman_corr_test`, `t_test_1samp`, `t_test_2samp`
  and `t_test_paired` reported `pvalue == 0.0` for strong effects (and a
  two-sided t-test gave different answers when the samples were swapped).
  They now agree with scipy to 9 significant digits down to about 1e-280,
  including at 1–2 degrees of freedom. A p-value of exactly 0.0 now means the
  true value is below the smallest representable number.
- **One-sided t-tests with zero variance point the right way.** Constant data
  gives an infinite statistic; `t_test_1samp` and `t_test_paired` returned
  p = 0 for it under every alternative. An infinite statistic *against* a
  one-sided alternative now gives p = 1, as in scipy, and a NaN statistic
  gives a NaN p-value instead of 0.
- **`ks_1samp` two-sided p-values were wrong for n from about 550 to
  10,000.** They came back as ~1.0 whatever the data (1,000 points from
  N(0.15, 1) tested against N(0, 1): p = 1.0; scipy 0.012). The exact matrix
  method lost track of its scale factors once it had to rescale, and a
  non-positive result was then reported as "no evidence". The bookkeeping is
  fixed and checked against a 60-digit evaluation of the same formula. Small
  p-values (and every statistic ≥ 0.5) now use twice the exact one-sided tail,
  which is accurate far below 1e-16 and takes O(n) time: a case that took
  minutes (n = 4,000, D ≈ 0.5) is now immediate. For n > 10,000 with
  p > 0.01 the Stephens series is still used (within about 0.5%).
- **`anderson_darling` no longer depends on the sign of an outlier.** The
  upper tail was `1 - cdf(z)`, exactly 0 beyond z ≈ 8.3 and then clamped, so a
  sample with one value at +12σ scored 5.22 and its mirror image 3.96. It now
  matches scipy's A² times the documented small-sample factor either way. For
  n ≤ 3 that factor is negative and the statistic was reported negative;
  `anderson_darling` now requires n ≥ 4 and `normality_summary` /
  `normality_report` show NaN.
- **Rolling windows:** a centered window longer than the data returned all
  zeros (now truncated windows, as pandas with `min_periods`); rolling
  min/max were inexact for data with a large offset (`[1e16, 1, 3, …]` gave a
  minimum of 0; now exact); the std/var of a one-point trailing window was
  0.0 (now NaN, as in pandas).
- **`pearson_corr_test` is scale-invariant.** Data scaled by 1e-100 gave
  r = inf, and by 1e100 gave r = 0.
- **Bad arguments raise `ValueError` instead of killing the process:** a
  frame with no columns in `Rolling(..., axis=0)`; `f_test_var` on data whose
  variance overflows; `mean_diff_ci` with `alpha` outside (0, 1); NaN as the
  `confidence` of `odds_ratio`, `var_ci` or `corr_ci` (`var_ci` silently
  returned a zero-width interval).
- **F, chi-square and normal p-values are accurate in the far tail.** These
  were still `1 - cdf(stat)`, which is exactly 0.0 below about 1e-16, so
  `f_test_oneway`, `levene_test`, `bartlett_test`, `f_test_var`, `chi2_gof`,
  `chi2_independence`, `jarque_bera`, `normality_summary`, `ljung_box`,
  `box_pierce`, `bg_test`, `runs_test` and `variance_ratio_test` reported
  `pvalue == 0.0` for any strong effect. They now use the upper incomplete
  gamma / beta functions and `erfc` directly and agree with scipy down to
  about 1e-300. Ordinary p-values are unchanged (to ~1e-11).
- **`f_test_var` doubled the wrong tail when the sample sizes differ a lot.**
  It put the larger variance on top and doubled the upper tail, but that is
  not always the smaller tail: with 101 and 2 observations and a variance
  ratio of 1.5 it reported p = 1.0 where 2·min(lower, upper) is 0.83. It now
  takes the smaller of the two tails. Results for equal sample sizes are
  unchanged.
- **Absurd counts and lags raise or return NaN instead of killing the
  process.** `n_resamples=10**12` asked for an 8 TB result buffer, and the
  failed allocation aborted the interpreter; every resampler (and `pad_nan`)
  now raises `MemoryError`. A lag near 2**64 in `adf_test`,
  `zivot_andrews_test`, `variance_ratio_test` or `bg_test` wrapped the
  "enough rows?" arithmetic and then indexed out of bounds or never returned;
  a lag at or beyond the series length now returns NaN, like any other lag
  that is too large. `quantile_bins` no longer loops over the bins, so an
  absurd `n_bins` returns immediately.
- **t-tests on constant samples.** The mean of n copies of a value is not
  always that value in floating point, which left a variance of ~1e-34 and a
  t statistic made of rounding error: two samples holding the same constant
  with different sizes could come out "significantly different". Constant
  samples are now detected and handled exactly, in `t_test_1samp`,
  `t_test_2samp` and `t_test_paired`. `t_test_2samp` also compared the means
  of two zero-variance samples with an absolute 1e-15 tolerance, so constants
  of 1e-20 and 2e-20 were reported as equal (t = 0, p = 1); they are now
  compared exactly (t = -inf, p = 0).
- **`mann_whitney_u`, `proportion_ztest` and `two_proportions_ztest`** take
  their normal tail from `erfc` and agree with scipy to 1e-13 (it was ~1e-11).
- **Rolling `count` is the number of valid values in the window, always.**
  Under `nan_policy="propagate"` it was NaN for a window holding a NaN, and a
  valid count under every other policy and alignment. It is now the same
  number everywhere and never NaN: it is what explains a NaN in the other
  statistics. **Behaviour change** for `propagate` windows that contain NaN.
- **`jackknife_mean` returns the mean.** It formed `n·θ − (n−1)·mean(loo)`,
  which subtracts two numbers the size of the data: for
  `[0, 0, 0, 999999, -999945, 0]` it returned 9.000000000014559 and reported
  the difference as a bias. For the mean the jackknife has a closed form
  (estimate = mean, bias = 0, SE = s/√n), which is what it now computes.
- **Centered rolling windows with an even `window` covered `window + 1`
  points.** `Rolling(x, window=4, alignment="centered")` averaged five values.
  The window is now `[i - w//2, i + (w-1)//2]`, as in pandas
  `rolling(w, center=True)`. Odd windows are unchanged.

Further notebook-layer fixes:

- Reports accept **MultiIndex columns** (every one raised `ValueError`). The
  report's rows are indexed by the same levels, as are both axes of the
  `correlation_report` matrix. Derived columns put their suffix on the last
  level -- `("g1", "a")` gives `("g1", "a_zscore")` -- in the transforms, the
  stylers and `rolling_report`; they used to come out as the string
  `"('g1', 'a')_zscore"`.
- **`normality_report` decides by Anderson-Darling, not Jarque-Bera**, and
  gains `ad_pvalue` and `n_missing` columns. Verdicts change where the two
  tests disagree: 30 exponential draws that Jarque-Bera passed (p = 0.28) are
  now rejected (p = 0.03). Fewer than 8 finite values gives "inconclusive
  (n < 8)" instead of a verdict from five points, and a constant column says
  so instead of "sample too small". The `conclusion` strings name the test.
- **Every helper takes `rich=True`**: `describe_fast`, `normality_report`,
  `missingness_report` and `rolling_report` now return a `DescribeReport`,
  `NormalityReport`, `MissingnessReport` and `RollingReport`. Reports unpack
  as `data, meta`, and reports and style results answer to each other's
  attribute names.
- **`to_dict()` is strict JSON, and `to_json()` is new.** It was documented
  as JSON-safe but passed NaN through (invalid JSON), and raised on MultiIndex
  labels, Timestamps and numpy scalar arguments. NaN and ±inf are now `None`;
  code that read NaN out of `to_dict()` should read `.data` / `.metadata`.
- **Warnings that were only on the rich result are also raised as
  `UserWarning` on the plain path** when they change what the numbers mean: a
  derived column replacing an existing one, robust scaling with a zero MAD
  (the output is ~1e12, not a score), fewer than 100 bootstrap resamples, a
  bootstrap interval from fewer than 30 values (it under-covers).
  New: integer columns beyond 2**53 warn, because float64 collapses nearby
  values (`2**53 + [1, 3, 5, 2, 4]` has std 1.58; the computed value is 2.24).
- New notes on rich results: a z-score threshold the sample size can never
  reach (`outlier_report` / `outlier_style`, `method="zscore"`); pairs with
  fewer than 3 overlapping rows, constant columns and small overlaps in
  `correlation_report`, which used to return no warnings at all; and the
  number of uncorrected tests when `pvalues=True`.
- `style_significance(adjusted=True)` declares the p-values already corrected
  and drops the "no correction" warning, which used to fire on a column that
  had just come out of `p_adjust`.
- `missingness_report` and `missingness_style` share one rule, so they agree:
  infinities in an `object` column of numbers are counted and painted, and a
  complex column no longer loses its imaginary part (with a `ComplexWarning`).
- `CorrelationReport.style()` on the long form colors only the `correlation`
  column; the [-1, 1] gradient used to be applied to `n`, `statistic` and
  `pvalue` too.
- Validation: non-numeric `alpha`, `conf`, `lower_q` / `upper_q` raise a
  `TypeError` naming the argument; `bootstrap_ci_report` rejects
  `n_resamples < 1`; `style_effect_size` rejects thresholds that are not
  finite and > 0 (negative ones silently put every row in "large").
- **`bunker_stats.pandas.corr_df` / `cov_df` default to `skipna=True`.** One
  NaN used to make the whole matrix NaN. They are now pairwise-complete, like
  `DataFrame.corr()` / `.cov()` and `correlation_report`, and ±inf is skipped
  along with NaN. `skipna=False` keeps the strict behaviour.
- **One name per option.** `suffix=` works on every helper that adds a column;
  `add_suffix=` (on `zscore_style`, `demean_style`, `robust_scale_column`) is
  kept as an alias. Outlier style metadata carries `lower_bound` /
  `upper_bound` as well as `lower_fence` / `upper_fence`, matching
  `outlier_report`. `scale_compare_style` takes its pair of suffixes as
  `suffix=` too (`suffixes=` still works).
- The Anderson-Darling p-value no longer flattens out at 5e-31: beyond the
  published fit it is continued log-linearly, so stronger departures still
  get smaller p-values. Treat values that small as an order of magnitude.
- `describe_fast` is no longer described as faster than `df.describe()`. With
  `robust=False` it is (about 2x at 100,000 rows); the default adds six robust
  statistics and is about 10x slower there, mostly `qn_scale`.
- **`normality_report` `kurtosis` is now excess kurtosis** (0 for a normal
  distribution), matching `robust_summary`. It was Pearson kurtosis, so the
  two reports differed by 3 under the same column name. Values in this column
  drop by 3; the Jarque-Bera statistic and p-value are unchanged.
- **`bootstrap_ci_report` `estimate` is the statistic of the sample.** It was
  the mean of the bootstrap draws. The interval is unchanged. For
  `stat="std"` the estimate is the ddof=0 standard deviation, because that is
  what the resampling kernel computes for each draw; mixing in the ddof=1
  value would put the estimate outside its own interval at small n. The
  small-sample warning now fires below n = 30 and says why: percentile
  intervals are too narrow at small n.
- **`rolling_report(alignment="centered")` returns NaN where a full window
  does not fit**, unless `min_periods` is lowered, so a `..._roll5_mean`
  column never holds 3-point means. `alignment` is now documented and
  validated.
- **`rolling_report` honours `min_periods` on the leading rows of a trailing
  window.** The first `window - 1` rows were NaN whatever `min_periods` said;
  they now hold partial-window values once enough observations are in, as
  pandas `rolling(window, min_periods=m)` does, and match it on every row.
  The default (`min_periods = window`) is unchanged. `bs.Rolling` has the
  same behaviour behind `full_length=True`.
- **Stylers warn when pandas will truncate the table.** Past
  `styler.render.max_elements` (262,144 cells by default) pandas renders only
  the leading rows, so highlights further down were silently missing while
  the metadata still counted them. Every styler, `corr_heatmap` included, now
  raises a `UserWarning` attributed to the calling line, and the rich result
  carries the same note. `styler.render.max_rows` / `max_columns` and the
  report objects' `.style()` are not covered.

## Release Notes — 0.3.1

The first release whose PyPI artifacts actually ship the documented Python
API. Every 0.2.x artifact — wheel and sdist alike — installed only the raw
`bunker_stats_rs` bindings; `import bunker_stats` failed for installed users.
0.3.1 wheels and sdist carry the full package, and CI now install-tests both
paths on every build so that gap cannot reopen.

Highlights on top of the 0.3.0 base (which was never published):

- **Notebook layer** (`bunker_stats.notebook`, `pip install "bunker-stats-rs[notebook]"`):
  reports, stylers, batch transforms, and report objects with Plotly figures.
- **Rich results everywhere**: `rich=True` on the hypothesis tests and reports
  returns tuple-unpackable result objects with `.to_dict()`, `.info()`,
  `.conclusion(alpha)`, and misuse warnings (small n, NaN, approximate p).
- **Bootstrap draws**: `BootstrapConfig(return_draws=True)` retains the
  resampling distribution with a CI identical to the tuple path (one shared
  kernel and RNG stream), enabling `plot_distribution()`.
- **`bg_test(resid, max_lag, exog=None)`**: the full Breusch-Godfrey
  construction with the original model's regressors.
- **Portable wheels for Linux (glibc + musl), Windows, macOS (Intel + ARM),
  CPython 3.9–3.13**, built and smoke-gated in CI; local `target-cpu=native`
  tuning is explicitly stripped from release builds.

---

## Release Notes — 0.3.0 consistency changes

A correctness-and-consistency pass. Every change below is pinned by a
regression test (see `tests/test_hardening_v030.py`).

### Numeric bug fixes (behavior changes)

- **Catastrophic cancellation eliminated in one-pass covariance/correlation
  kernels.** `cov`/`corr`, `cov_skipna`/`corr_skipna`,
  `cov_matrix_skipna`/`corr_matrix_skipna`, the NaN-aware rolling
  `std`/`zscore` paths, and the strict `rolling_cov`/`rolling_corr` kernels
  now shift data by the first finite value (per series / per column) before
  accumulating. Second moments are translation-invariant, so results are
  mathematically identical but no longer lose all precision on large-offset
  data (e.g. series centered near 1e8 or 1e12). `rolling_beta_skipna` and
  `rolling_linreg_skipna` use the same shift (the linreg intercept is
  un-shifted exactly). Verified against two-pass numpy references at offsets
  1e8 and 1e12 — note that pandas' own one-pass rolling kernels fail this test.
- **Correlation outputs are clamped to [-1, 1]** across scalar, matrix and
  rolling correlation kernels; round-off can no longer produce |r| slightly
  above 1.
- **`welch_psd` / `bartlett_psd` now produce true density-scaled one-sided
  spectra**: scale is `1/(fs·Σw²)` with interior bins doubled (the Nyquist bin
  is not doubled for even `nperseg`). Matches `scipy.signal.welch` to ~1e-15
  relative. Previously the output was neither 'density' nor 'spectrum' scaled
  and interior bins were not doubled.
- **`ks_1samp` one-sided p-values are now exact** (Birnbaum–Tingey / Smirnov
  formula, evaluated in log space), matching
  `scipy.stats.ks_1samp(alternative="greater"/"less")`. Previously a rough
  `exp(-2nD²)` asymptotic was used.
- **`exp_cdf` uses `-expm1(-λx)`**, keeping full precision for tiny arguments
  (`exp_cdf(1e-18) == 1e-18` instead of `0.0`).

### Previously unregistered functions now exported

`rolling_min`, `rolling_max`, `rolling_range`, `rolling_cv`,
`rolling_count_above`, `rolling_pct_above` (O(1) sliding kernels) and
`kde_gaussian` (Gaussian KDE with Scott's-rule bandwidth, parity with
`scipy.stats.gaussian_kde`) are now registered in the extension module. Every
name in the facade's `__all__` now resolves to a callable.

### API consistency changes

One shared edge/validation contract across the library. Each item below is a
deliberate behavior change:

- **Rolling edge rule** (strict, truncated-output kernels): `window < 1` now
  raises `ValueError` everywhere (several kernels silently returned an empty
  array); `window > len(x)` now returns an **empty array** everywhere
  (`rolling_cov`, `rolling_corr`, `rolling_autocorr`, `rolling_correlation`,
  `rolling_autocorr_multi`, `rolling_min/max/range/cv/count_above/pct_above`
  previously raised). Exception: `rolling_median` intentionally keeps its
  pandas-like full-length output (length n, NaN head).
- **Facade rolling wrappers validate `window` as an integer >= 1**, so a
  negative window raises `ValueError` instead of `OverflowError`.
- **`winsorize` accepts quantile fractions in [0, 1] only** with
  `0 <= lower_q < upper_q <= 1`; the old dual-unit auto-detection (which made
  `1.0` ambiguous and silently clamped out-of-range arguments) now raises
  `ValueError`. **`winsorized_mean` (facade) uses the same fraction units**,
  validating and converting to the kernel's percentile units; the raw
  `winsorized_mean_np` binding keeps percentile units for backwards
  compatibility.
- **`percentile` raises `ValueError` for q outside [0, 100]** (previously
  silently clamped).
- **`trimmed_mean` / `trimmed_mean_skipna` raise `ValueError` for
  `proportion_to_cut` outside [0, 0.5)** (previously returned NaN).
- **`ewma` requires `0 < alpha <= 1`**, raising `ValueError` otherwise.
- **`quantile_bins` requires `n_bins >= 1`**; the facade also rejects negative
  `n_bins` with `ValueError` (previously `OverflowError` or empty output).
- **Outlier detectors have keyword defaults**: `iqr_outliers(x, k=1.5)`,
  `zscore_outliers(x, threshold=3.0)`.
- **`alternative` is validated** against `{"two-sided", "less", "greater"}` in
  the t-tests, `mann_whitney_u`, and the facade `t_test_2samp` wrapper
  (`ValueError` otherwise).
- **`cov` / `corr` raise `ValueError` on length mismatch** (previously the
  longer input was silently truncated).
- **`acf`/`pacf` (all methods) on constant input return 1.0 at lag 0 and NaN
  at lags >= 1** (statsmodels semantics; previously reported spurious
  autocorrelation). `pacf_innovations` reports NaN instead of impossible
  coefficients with |value| > 1.
- **`welford([])` returns `(nan, nan, 0)`** — the mean of no observations is
  NaN, not 0.0.

### Determinism

- **`random_state=None` now means seed 0 for every randomized routine.** The
  three block bootstraps (`moving_block_bootstrap_mean_ci`,
  `circular_block_bootstrap_mean_ci`, `stationary_bootstrap_mean_ci`)
  previously drew OS entropy when unseeded, and
  `jackknife_after_bootstrap_se_mean` used a different fixed default seed.
  All 15 resamplers now return bit-identical results for `None` vs `0` and
  across repeated calls.
- **`bootstrap_mean` is bit-identical across thread counts**: per-resample
  means are collected in index order and averaged with a serial sum, removing
  the last parallel floating-point reduction whose rounding could depend on
  rayon's work-stealing schedule.

---

## Release Notes — 0.2.9 Hardening

A full-codebase review and hardening pass. Every fix below is pinned by a
regression test; the suite now stands at **458 Python tests + 53 Rust tests
passing, 0 compiler warnings**.

### Crash fixes (highest impact)

- **NaN inputs no longer kill the Python process.** The release build uses
  `panic = "abort"`, so any Rust panic terminated the whole interpreter rather
  than raising an exception. ~24 sort/select code paths (`median`, `mad`,
  `iqr`, `percentile`, `robust_scale`, `ecdf`, `quantile_bins`, `qn_scale`,
  `runs_test`, and others) panicked on NaN input. All now use total-order
  comparison and propagate NaN (`NaN in → NaN out`) or raise a normal
  `ValueError`. *Impact: a single NaN in production data could previously
  crash the entire process, losing all in-flight work.*
- **`zivot_andrews_test` crashed on every call** via an out-of-bounds buffer
  write (the design-matrix column count was off by one). *Impact: the function
  was unusable; the crash mode was a hard interpreter abort.*

### Statistical correctness

- **Zivot-Andrews regression was numerically scrambled** — the design matrix
  was built row-major but read column-major, so breakpoint estimates and
  t-statistics were noise. It now recovers the true structural break
  (verified against `statsmodels`).
- **Rolling variance/std/cov/corr suffered catastrophic cancellation** on
  large-offset data (e.g. values near 1e8): results silently degraded to
  wrong zeros. All rolling second-moment kernels now use translation-offset
  accumulators. *Impact: correct results on price-level-style series, not
  just mean-zero data.*
- **KPSS** now uses the same automatic bandwidth as `statsmodels`
  (`nlags="auto"`); statistics match statsmodels to machine precision.
- **ADF / Phillips-Perron p-values** now come from the MacKinnon (1994)
  regression surface (statsmodels-identical, including extreme tails). ADF
  honors its `regression` and `max_lag` arguments (previously ignored); PP
  applies the HAC long-run-variance correction and the correct
  Dickey-Fuller reference distribution (previously Normal).
- **Breusch-Godfrey** statistic corrected to the standard `T·R²` form.
- **Variance-ratio test** rewritten as the proper Lo-MacKinlay overlapping
  estimator; a random walk now yields VR ≈ 1 as expected.
- **Biweight midvariance** exponents corrected to the standard
  (Beers-Flynn-Gebhardt / astropy) definition.
- **Skewness/kurtosis** now use scipy-consistent population moments
  (previously mixed sample/population estimators matching no standard
  convention).
- **NaN-aware rolling cov/corr** now match pandas `min_periods=window`
  semantics.
- **`rolling_autocorr_multi`** wrote its output buffer column-major but
  reshaped it row-major, scrambling the matrix whenever more than one lag was
  requested (single-lag calls were coincidentally correct). Each column `j`
  now equals `rolling_autocorr(x, lags[j], window)` exactly.

### Test coverage & tooling

- Function-level test coverage rose from **51% to 99%** of the 188 public
  functions: 67 new parity tests validate outputs directly against
  numpy / scipy / pandas / statsmodels references.
- New test layers: Rust unit + property tests (including
  never-panic-on-any-input properties), a binding-surface test asserting
  every registered export is callable, and a subprocess-isolated NaN-crash
  regression test.
- The `parallel` feature is now genuinely optional
  (`--no-default-features` builds and passes tests with a serial fallback),
  and the crate builds an `rlib` so Rust integration tests and fuzzing can
  link against it.

### Modern facade API (v0.3 layer)

- **One name per statistic; NaN handling is a keyword.** `bs.mean(x, skipna=True)`
  dispatches to the skip-NaN Rust kernel; the strict and skip-NaN kernels stay
  separate underneath, so there is no branch in the hot loop. Applies to
  mean/std/var/median/mad/iqr/zscore/trimmed_mean, cov/corr (scalar, matrix,
  and rolling), and the rolling reducers.
- **Keyword defaults everywhere.** `bs.t_test_2samp(x, y)` works bare
  (pooled, two-sided); options like `equal_var`, `pooled` are keyword-only.
- **Unit-explicit argument names.** `bs.winsorize(x, lower_q=0.05, upper_q=0.95)`
  takes quantiles in [0, 1]; `bs.percentile(x, q=95)` keeps numpy's [0, 100]
  convention — the name tells you the unit.
- **Optional pandas layer.** `bunker_stats.pandas` provides `cov_df`/`corr_df`
  (labeled DataFrame results) and Styler helpers (`corr_heatmap`,
  `zscore_style`, ...). The core package remains numpy-only.
- All raw `*_np` / `*_skipna` names remain available; nothing breaks.

---

## Quick Start

### Installation

```bash
pip install bunker-stats-rs

# ...plus the optional pandas/Jupyter reporting layer (see "Notebook UX" below)
pip install "bunker-stats-rs[notebook]"
```

The core package depends only on **numpy**. `pandas`, `jinja2`, `matplotlib`
and `plotly` arrive with the `[notebook]` extra and are needed solely by
`bunker_stats.notebook`.

### Basic Usage

```python
import bunker_stats as bs
import numpy as np

# Robust statistics - resistant to outliers
data = np.array([1, 2, 3, 4, 5, 100])  # outlier: 100
location, scale = bs.robust_fit(data)   # (3.5, 2.22) vs mean/std (19.17, 38.4)

# Rolling window operations - 2-5× faster than pandas
signal = np.random.randn(10000)
smoothed = bs.rolling_median(signal, window=10)

# NaN handling is a keyword, not a separate function
noisy = signal.copy(); noisy[::97] = np.nan
m = bs.mean(noisy, skipna=True)                  # numpy.nanmean semantics
r = bs.rolling_mean(noisy, window=20, skipna=True)

# Statistical inference - keyword defaults just work
x = np.random.randn(30)
y = np.random.randn(25) + 0.5
result = bs.t_test_2samp(x, y)                   # pooled, two-sided defaults
welch = bs.t_test_2samp(x, y, equal_var=False)  # Welch's t-test

# Quantile arguments carry their unit in the name
clipped = bs.winsorize(x, lower_q=0.05, upper_q=0.95)

# Matrix operations - fast covariance/correlation
X = np.random.randn(1000, 10)
cov = bs.cov_matrix(X)
corr = bs.corr_matrix(X, skipna=True)            # pairwise-complete

# Optional notebook layer (pip install "bunker-stats-rs[notebook]")
# from bunker_stats.notebook import robust_summary, outlier_style
# robust_summary(df)          # DataFrame: median, MAD, IQR, Qn, skew, kurtosis...
# outlier_style(df)           # Styler: highlights outliers across all columns

# Bootstrap confidence intervals (deterministic given random_state)
estimate, lower, upper = bs.bootstrap_mean_ci(data, n_resamples=10000, conf=0.95, random_state=0)
```

---

## Module Documentation

Each module has comprehensive documentation with detailed API references, usage examples, performance benchmarks, and edge case behavior specifications.

### 1. **Robust Statistics** ✅ Production-Ready

**Status:** 73/73 tests passing  
**Performance:** 2-5× faster than SciPy/pandas on order-statistic estimators (median/MAD/Qn)  
**Documentation:** See [src/kernels/robust/README.md](./src/kernels/robust/README.md)

Outlier-resistant statistical estimators including:
- Location estimators (median, trimmed mean, Huber location)
- Scale estimators (MAD, IQR, Qn, Sn)
- Robust fitting (`robust_fit`, `robust_score`)
- Rolling robust statistics
- Skip-NaN variants for all functions

**Key Features:**
- Policy-driven `RobustStats` class with composable configuration
- Fused median+MAD kernel (40% faster joint computation)
- O(n) selection vs O(n log n) sorting (2-5× speedup)
- Perfect SciPy parity with deterministic results

---

### 2. **Inference** ✅ Production-Ready

**Status:** 15/15 tests passing  
**Performance:** 1.2-1.5× faster than SciPy  
**Documentation:** See [src/infer/INFERENCE_README.md](./src/infer/INFERENCE_README.md)

Comprehensive statistical hypothesis testing suite:
- **Chi-square tests:** Goodness-of-fit, independence
- **T-tests:** One-sample, two-sample (pooled/Welch)
- **Non-parametric:** Mann-Whitney U, Kolmogorov-Smirnov
- **Correlation:** Pearson, Spearman with significance tests
- **ANOVA:** F-test, Levene's test, Bartlett's test
- **Normality:** Jarque-Bera, Anderson-Darling
- **Effect sizes:** Cohen's d, Hedges' g

**Key Features:**
- Numerical stability with extreme values (χ² > 1000, n > 5000)
- Exact finite-n algorithms (Durbin-Marsaglia for KS test)
- Welch-Satterthwaite with zero-variance edge case handling
- 100% SciPy parity (rtol ≤ 1e-10)

---

### 3. **Matrix Operations** ✅ Production-Ready

**Status:** 83/83 tests passing  
**Performance:** correctness-first; dense cov/corr are ~0.6× numpy (no BLAS backend) — see Performance Highlights  
**Documentation:** See [src/kernels/matrix/README.md](./src/kernels/matrix/README.md)

High-performance matrix computations for statistical analysis:
- **Covariance matrices:** Sample, population, centered, pairwise-complete
- **Correlation matrices:** Pearson correlation, correlation distance
- **Gram matrices:** X^T X and X X^T for regression/kernel methods
- **Pairwise distances:** Euclidean, cosine
- **Utilities:** Diagonal extraction, trace, symmetry checking

**Key Features:**
- Guaranteed symmetry and positive semi-definiteness
- Optional Rayon parallelism for large matrices
- Comprehensive NaN handling with skip-NaN variants
- Perfect NumPy/SciPy parity with mathematical guarantees verified

---

### 4. **Rolling Windows** ✅ Production-Ready

**Status:** 53/53 tests passing  
**Performance:** 2-5× faster than pandas (rolling std/mean); rolling_median is O(n) selection-based  
**Documentation:** See [src/kernels/rolling/ROLLING_README.md](./src/kernels/rolling/ROLLING_README.md)

Flexible rolling window statistics with policy-driven configuration:
- **Statistics:** Mean, std, variance, min, max, count
- **Alignment:** Trailing (classic) or centered (full-length output; windows match pandas `center=True`, but edge rows use partial windows unless `min_periods` is set)
- **NaN handling:** Propagate, ignore, or minimum periods
- **Multi-stat kernels:** Compute 2-6 statistics in single pass
- **2D support:** Column-wise operations on matrices

**Key Features:**
- `Rolling` class with composable `RollingConfig` policies
- Fused kernels for efficient multi-metric computation
- Kahan summation for numerical stability
- Automatic edge truncation for centered windows
- 100% backward compatibility with legacy functions

---

### 5. **Resampling** ✅ Production-Ready

**Status:** Deterministic given `random_state`; CI ordering and determinism tested  
**Performance:** ~5× faster than a vectorized-numpy bootstrap loop  
**Documentation:** See [src/kernels/resampling/README.md](./src/kernels/resampling/README.md)

Lightning-fast resampling methods with ergonomic interfaces:
- **Bootstrap:** Percentile, BCa, bootstrap-t, and Bayesian CIs; standard error and variance
- **Block bootstraps:** Moving, circular, and stationary variants for autocorrelated series
- **Permutation tests:** Mean-difference and correlation
- **Jackknife:** Leave-one-out, delete-d, jackknife-after-bootstrap, influence values

**Key Features:**
- `BootstrapConfig` class with comprehensive validation
- Flexible NaN handling (propagate or omit)
- Deterministic random seeding for reproducibility
- Zero performance overhead from config layer
- Actionable error messages

---

### 6. **Time Series Analysis** ✅ Production-Ready

**Status:** Core statistics validated against statsmodels (ADF, KPSS, Ljung-Box, ACF/PACF to machine precision)  
**Documentation:** See [src/kernels/tsa/README.md](./src/kernels/tsa/README.md)

Comprehensive temporal data analysis tools:
- **Correlation:** ACF, PACF (Levinson-Durbin, Yule-Walker, Innovations, Burg)
- **Spectral analysis:** Periodogram, Welch PSD, spectral density
- **Diagnostic tests:** Ljung-Box, Box-Pierce, Breusch-Godfrey, Durbin-Watson, runs test
- **Stationarity:** ADF (MacKinnon p-values), KPSS (automatic bandwidth), Phillips-Perron, Zivot-Andrews structural break, variance ratio
- **Rolling operations:** Rolling autocorrelation (single and multi-lag)

---

### 7. **Distributions** ✅ Production-Ready

**Status:** Round-trip (`ppf(cdf(x)) ≈ x`) and reference-parity tested  
**Documentation:** See [src/kernels/dist/README.md](./src/kernels/dist/README.md)

Vectorized distribution functions for Normal, Exponential (rate-parameterized), and Uniform:
- **Densities:** `pdf`, `logpdf`
- **Probabilities:** `cdf`, `sf`, `logsf` — dedicated survival functions keep far-tail
  p-values accurate where `1 - cdf` would cancel to zero
- **Quantiles:** `ppf` with strict domain checks
- **Reliability:** cumulative hazard functions for survival analysis

---

### 8. **Quantiles & Order Statistics** ✅ Production-Ready

**Status:** NumPy-parity tested, NaN-safe  
**Documentation:** See [src/kernels/quantile/README.md](./src/kernels/quantile/README.md)

Order-statistic utilities built on O(n) selection:
- **Percentiles & IQR:** `percentile`, IQR family including skip-NaN variants
- **Winsorization:** Percentile-based and explicit-bound clipping
- **Outlier masks:** IQR fences and z-score thresholds
- **Empirical distributions:** ECDF and quantile binning

---

---

## Rich Results (opt-in result objects)

Hypothesis tests return a plain `dict` by default. Pass **`rich=True`** to get a
result object instead — tuple-unpackable, indexable, and carrying a bit of
extra metadata (sample sizes, the chosen `alternative`, and cheap effect
sizes / CIs computed from the same kernels). This mirrors the result-object
style already used by the time-series tests.

```python
import bunker_stats as bs

# Default: unchanged dict return
bs.t_test_2samp(x, y)
# -> {'statistic': ..., 'pvalue': ..., 'df': ..., 'mean_x': ..., ...}

# Opt in to a rich result
res = bs.t_test_2samp(x, y, equal_var=False, rich=True)

stat, pval = res            # tuple unpacking (statistic, pvalue)
res[0], res[1]              # indexing
res.pvalue, res.df          # named access
res.effect_size            # Cohen's d, computed from the same inputs
res.n1, res.n2, res.equal_var
res.is_significant(0.05)   # -> bool
res.to_dict()              # JSON-friendly dict of populated fields
print(res.conclusion(alpha=0.01))
print(res.info())          # multi-line human summary
```

```text
>>> print(res.info())
t-Test
============================================================
Statistic:          -2.145318
P-value:            0.033987
Test:               two-sample
Alternative:        two-sided
df:                 118
Mean x / y:         0.0512 / 0.4231
Equal variance:     False
Cohen's d:          -0.394
n1 / n2:            60 / 70

Conclusion: Reject H0 (the two means are equal) at alpha=0.05 (p=0.034)
```

### Which tests support `rich=True`

| Function(s) | Result type | Notable extra fields |
|-------------|-------------|----------------------|
| `t_test_1samp`, `t_test_2samp`, `t_test_paired` | `TTestResult` | `effect_size` (Cohen's d, 2-sample), `n1`/`n2`, `equal_var` |
| `chi2_gof`, `chi2_independence` | `ChiSquareResult` | `dof`, `observed`, `test_type` |
| `mann_whitney_u` | `MannWhitneyResult` | `rank_biserial`, `n1`/`n2` |
| `ks_1samp` | `KSResult` | `distribution`, `n` |
| `f_test_oneway` | `ANOVAResult` | `eta_squared`, `omega_squared`, `n_groups`/`n_total` |
| `pearson_corr_test`, `spearman_corr_test` | `CorrelationTestResult` | `r`, `ci_low`/`ci_high` (Pearson) |
| `jarque_bera`, `anderson_darling` | `NormalityResult` | `skewness`, `kurtosis`, `critical_value_5pct` |

The result classes are exported from both the root package and
`bunker_stats.infer` (for `isinstance` checks and type hints):

```python
from bunker_stats.infer import TTestResult, NormalityResult
isinstance(bs.t_test_2samp(x, y, rich=True), TTestResult)   # True
```

### Beyond inference

The same `rich=True` opt-in extends to other modules. Matrix and outlier
results are **array-like** — `np.asarray(result)` returns the payload with no
copy, so they drop straight into NumPy code — while still carrying metadata and
conversions.

| Function / method | Result type | Highlights |
|-------------------|-------------|------------|
| `corr_matrix(X, rich=True, columns=...)` | `CorrelationMatrixResult` | array-like; `.to_frame()`, `.style_heatmap()`, `n_obs` |
| `cov_matrix(X, rich=True)` | `CovarianceMatrixResult` | array-like; `ddof`, `.to_frame()` |
| `robust_fit(x, rich=True)` | `RobustFitResult` | unpacks `location, scale`; `.zscores(x)` |
| `iqr_outliers(x, rich=True)`, `zscore_outliers(x, rich=True)` | `OutlierResult` | array-like mask; `.indices()`, bounds, counts |
| `Rolling(x, w).result(*stats)` | `RollingResult` | dict-like by stat; `.to_frame()`, `.plot()` |
| `bootstrap(x, rich=True)` | `BootstrapResult` | unpacks `estimate, ci_lower, ci_upper` |
| `permutation_test(x, y, rich=True)` | `PermutationTestResult` | unpacks `statistic, pvalue`; `.conclusion()` |

```python
# Array-like matrix result
C = bs.corr_matrix(X, rich=True, columns=["a", "b", "c"])
np.asarray(C)          # the raw matrix, no copy
C.to_frame()           # labeled DataFrame
C.style_heatmap()      # Styler (needs matplotlib)

# Robust fit unpacks like a tuple, but carries more
loc, scale = bs.robust_fit(x, rich=True)

# Outlier mask behaves like the boolean array it wraps
out = bs.iqr_outliers(x, rich=True)
x[out]                 # boolean-index with the result directly
out.indices()          # positions flagged, plus out.lower_bound / out.upper_bound

# Rolling result is dict-like over the statistics
r = bs.Rolling(x, window=20).result("mean", "std")
r["mean"]; r.to_frame(); print(r.info())
```

These result classes are exported from the root package and their submodules
(`bunker_stats.matrix`, `bunker_stats.robust`, `bunker_stats.rolling`,
`bunker_stats.resampling`).

### Protocol shared by every rich result

| Member | Behaviour |
|--------|-----------|
| `for v in res` / `a, b = res` | Yields the primary fields — `(statistic, pvalue)`, except `CorrelationTestResult` yields `(r, pvalue)`. |
| `res[i]` | Integer/slice indexing over those same primary fields. |
| `res.to_dict(array=False)` | All *populated* fields (None omitted). NumPy scalars become Python scalars; arrays become lists unless `array=True`. |
| `res.info()` | Multi-line summary: statistic, p-value, test-specific detail, and a conclusion line. ASCII-only, so it prints on any console. |
| `res.conclusion(alpha=0.05)` | One-line verdict at the given significance level. |
| `res.is_significant(alpha=0.05)` | `bool`; `False` when the p-value is missing/NaN. |

Two behaviours worth noting:

- **Backward compatibility is absolute.** The `rich` keyword is the *only* thing
  the facade intercepts; every other argument is forwarded untouched, so the
  default (`rich=False`) return is byte-for-byte what it always was — including
  the deprecated `*_np` aliases.
- **Anderson-Darling has no p-value.** `NormalityResult.pvalue` is `None` for
  `anderson_darling`; its `conclusion()` / `is_normal()` fall back to the 5%
  critical value (A\* > 0.787 rejects normality).

### Misuse-prevention warnings

Rich results carry a `warnings` list flagging conditions that make the numbers
easy to over-trust — small samples, non-computable p-values, approximate
p-values, zero-variance/all-NaN columns, low bootstrap resample counts.
Warnings appear in `.info()` output and in `.to_dict()["warnings"]`; a clean
result has an empty list:

```python
r = bs.t_test_2samp(x[:5], y[:6], rich=True)
r.warnings
# ['small sample (min n = 5): asymptotic p-values are unreliable']
```

## Notebook UX (optional pandas layer)

`bunker_stats.notebook` is the ergonomic bridge between the Rust kernels and
pandas/Jupyter. Every helper is a thin, validated wrapper — the numbers always
come from the same kernels the core API uses.

```bash
pip install "bunker-stats-rs[notebook]"
```

`import bunker_stats` stays **numpy-only**: pandas is never imported at package
import time, and even `import bunker_stats.notebook` succeeds without pandas
installed. The import happens lazily on the first call, and a missing install
raises a clear `ImportError` naming the command above.

### Naming convention

| Suffix        | Returns                                                  |
|---------------|----------------------------------------------------------|
| `*_report`    | `pandas.DataFrame` of results (usually one row per column) |
| `*_style` / `style_*` | `pandas.io.formats.style.Styler`                 |
| `*_columns`   | `pandas.DataFrame` copy with added/transformed columns    |

Every helper also accepts `rich=True` to get the same output wrapped with
its context. See [Report objects](#report-objects-and-interactive-plotly-figures)
and [Style results](#style-results-richtrue-on-stylers-and-transforms).

**Copy semantics.** No helper ever modifies the DataFrame it is given, and
there is no `inplace` option. Helpers that add columns (`*_columns`,
`demean_style`, `zscore_style`, `scale_compare_style`) return or style a copy;
`replace=True` overwrites the source column *in that copy*. Helpers that only
color existing cells style your frame directly without copying it.

### Quick tour

```python
import bunker_stats as bs
from bunker_stats.notebook import (
    robust_summary, describe_fast, outlier_report, outlier_style,
    normality_report, correlation_report, corr_heatmap, missingness_report,
    scale_columns, winsorize_columns, rolling_report, bootstrap_ci_report,
    style_significance, style_effect_size,
    missingness_style, scale_compare_style,
)

# --- Profiling ---------------------------------------------------------
robust_summary(df)                 # count, mean, std, median, MAD, IQR, Qn,
                                   # trimmed mean, skew, kurtosis
describe_fast(df)                  # a richer df.describe().T (robust=False
                                   # for the plain columns, faster than pandas)
missingness_report(df)             # NaN vs +/-inf vs finite, per column
missingness_style(df)              # the same, painted onto the table

# --- Outliers ----------------------------------------------------------
outlier_report(df, method="iqr", k=1.5)             # counts, %, bounds
outlier_report(df, method="robust_zscore")          # median/MAD fences
outlier_style(df)                                   # highlight across columns
scale_compare_style(df, "price")                    # z-score vs robust score

# --- Distribution & association ----------------------------------------
normality_report(df)                                # Anderson-Darling + J-B
correlation_report(df, method="spearman")           # square matrix
correlation_report(df, pvalues=True)                # long form with p-values
corr_heatmap(df)                                    # gradient Styler

# --- Transforms --------------------------------------------------------
scale_columns(df, method="robust")                  # adds *_robust columns
scale_columns(df, method="zscore", replace=True)    # overwrite in place
winsorize_columns(df, lower_q=0.05, upper_q=0.95)   # adds *_winsor columns

# --- Time series & uncertainty -----------------------------------------
rolling_report(df, "price", window=20, stats=("mean", "std"))
bootstrap_ci_report(df, stat="median", n_resamples=5000, random_state=0)

# --- Styling result tables ---------------------------------------------
style_significance(results, pvalue_column="pvalue", alpha=0.05)
style_effect_size(results, "cohens_d", thresholds=(0.2, 0.5, 0.8))
```

Helpers are also reachable lazily off the package (`bs.notebook.robust_summary(df)`)
and re-exported from `bunker_stats.pandas` alongside `cov_df` / `corr_df`.

### Report objects and interactive Plotly figures

Every report also takes **`rich=True`**, returning a report object instead
of a bare DataFrame — same numbers (`report.data`), plus `meta` (method,
parameters, NaN policy, seeds), misuse-prevention `warnings`, a stable
`to_dict()` schema (`{"title", "data", "meta", "warnings"}`), `to_json()`,
`.style()` and `.info()`. Three of them add interactive **Plotly** figures:

```python
from bunker_stats import notebook as nb

rep = nb.outlier_report(df, rich=True)
rep.plot_counts()          # plotly Figure: outlier counts, hover shows % + method

corr = nb.correlation_report(df, rich=True)
corr.plot_heatmap()        # plotly Figure: hover shows pair + r; NaN cells = gaps

ci = nb.bootstrap_ci_report(df, n_resamples=5000, random_state=0, rich=True)
ci.plot_intervals()        # plotly Figure: estimates with CI error bars

# Result objects grew Plotly methods too:
bs.Rolling(x, window=20).result("mean", "std").plot()   # one trace per stat
za.plot_breakpoint_scan_plotly()                        # Zivot-Andrews scan

# Bootstrap draws are retained on request — same kernel and RNG stream, so
# the estimate/CI match the plain tuple path exactly:
from bunker_stats.resampling import BootstrapConfig
res = BootstrapConfig(n_resamples=5000, random_state=0, return_draws=True).run(x)
res.plot_distribution()      # histogram of draws + estimate/CI lines
```

Figure methods return `plotly.graph_objects.Figure` — they never call
`.show()`, and every figure serializes via `fig.to_json()` / `fig.to_html()`.
Plotly ships with the `notebook` extra; calling a plot method without it raises
`Install with pip install bunker-stats-rs[notebook] to use Plotly
visualizations.` Everything else on a report object works without Plotly.

### Style results (`rich=True` on stylers and transforms)

A highlighted table shows *which* cells stand out but not *why*. With
`rich=True` every styler returns the Styler together with the numbers behind
it, so the highlighting can be described in a report or reproduced later:

```python
from bunker_stats import notebook as nb

result = nb.zscore_style(df, "price", threshold=3, rich=True)
result                      # displays the styled table in Jupyter
styler, frame, meta = result   # frame = copy of df + "price_zscore"

meta["mean"], meta["std"], meta["n"], meta["n_missing"]
meta["n_high"], meta["n_low"], meta["n_flagged"]
meta["expected_pct_if_normal"]   # 0.27 for threshold=3: a calibration point

print(result.info())        # the same, as aligned text, plus any warnings
result.to_dict()            # {"title", "metadata", "warnings"}, strict JSON
result.to_json()            # the same as a string (NaN and inf become null)
result.warnings             # e.g. "with n = 5 no z-score can reach 3 ..."
```

| Helper | `rich=True` returns | Metadata highlights |
|--------|--------------------|---------------------|
| `demean_style` | `DemeanStyleResult` | `mean`, `n_above`, `n_below`, `n_at_mean` |
| `zscore_style` | `ZScoreStyleResult` | `mean`, `std`, `threshold`, `n_high`, `n_low`, `n_flagged` |
| `iqr_outlier_style` | `IQROutlierStyleResult` | `q1`, `q3`, `iqr`, `k`, `lower_fence`, `upper_fence`, `n_low`, `n_high` |
| `outlier_style` | `OutlierStyleResult` | the same per column, under `per_column` |
| `scale_compare_style` | `ScaleCompareStyleResult` | both fits, `n_flagged_both`, `n_robust_only`, `n_classical_only` |
| `corr_heatmap` | `CorrelationHeatmapResult` | `strongest_pairs`, `pairwise_n_min/max`, `constant_columns` |
| `missingness_style` | `MissingnessStyleResult` | `n_missing_cells`, `n_infinite_cells`, `n_complete_rows`, `per_column` |
| `style_significance` | `SignificanceStyleResult` | `n_tests`, `n_per_tier`, `expected_false_positives_if_all_null` |
| `style_effect_size` | `EffectSizeStyleResult` | `thresholds`, `n_per_bucket` |
| `scale_columns`, `winsorize_columns` | `TransformResult` | fitted parameters / clipping bounds per column |
| `robust_scale_column` | `RobustScaleResult` | `median`, `mad`, `scale`, `zero_mad` |

Style results unpack as `styler, frame, metadata`; transform results as
`frame, metadata` (their `.styler` is built on demand and is the only part
that needs jinja2); reports unpack as `data, meta`. All three answer to both
spellings of their parts (`.frame` / `.data`, `.metadata` / `.meta`). Every
result records `frame_is_copy` and how missing values were treated;
single-column results carry `n` and `n_missing` at the top level and
multi-column ones under `per_column`. Without `rich=True` the helpers return
exactly what they always did.

`to_dict()` and `to_json()` produce strict JSON: NaN and ±inf become `null`,
numpy scalars become Python numbers, dates become ISO strings, and tuple
(MultiIndex) labels become string keys. Read `.metadata` / `.data` when you
need the original values.

**Warnings.** Things that make a result misleading are listed in
`result.warnings`. A few mean the numbers are not what they appear to be, and
those are also raised as a `UserWarning` when you are *not* using `rich=True`:
integer values beyond 2**53 (float64 cannot hold them exactly), a derived
column replacing one that already existed, robust scaling of a column whose
MAD is 0, fewer than 100 bootstrap resamples, and a table too large for pandas
to render in full.

**Themes.** Every styler takes `theme=`: `"default"`, `"colorblind"` (blue /
orange instead of green / red), or a mapping that overrides single roles —
`theme={"outlier": "#ffd54f"}`. `nb.available_themes()` lists the roles. The
older per-color keywords (`outlier_color=`, `high_color=`, ...) still work and
take precedence. Each styled cell sets a text color along with its background,
so tables stay readable in dark-mode notebooks.

**What the colors do not mean.** These helpers mark values that are unusual
under a stated rule; they do not identify errors, and the metadata is written
to keep that visible:

- **z-scores mask.** The mean and standard deviation are inflated by the very
  values they are meant to catch, and in a sample of size *n* no z-score can
  exceed (*n* − 1)/√*n*. In `[1, 2, 3, 4, 100]` the value 100 has z = 1.79 and
  is not flagged at 2. `scale_compare_style` puts the robust score (65 here)
  next to it and counts the disagreements in `n_robust_only`.
- **`expected_pct_if_normal`** is how much of an exactly normal sample the
  rule would flag — 0.7% for 1.5 × IQR fences, 4.6% for |z| ≥ 2. Finding that
  share of "outliers" is what the rule does to clean data.
- **`style_significance` applies no multiple-comparison correction.** It
  shades the p-values as given and reports how many would fall below `alpha`
  by chance. Adjust first, then style the adjusted column. `bs.p_adjust`
  rejects NaN, and `correlation_report(pvalues=True)` emits NaN for undefined
  pairs, so adjust the finite p-values only:

  ```python
  p = df["pvalue"].to_numpy(dtype=float)
  ok = np.isfinite(p)
  df["p_adj"] = np.nan
  df.loc[ok, "p_adj"] = bs.p_adjust(p[ok], "holm")
  nb.style_significance(df, pvalue_column="p_adj")
  ```

  The result's multiple-comparison warning still appears for an adjusted
  column: the styler cannot tell that the values were already corrected.
- **Robust scaling with MAD = 0** (more than half the values identical) is not
  a usable score: the kernel divides by a ~1e-12 floor. The result carries
  `zero_mad=True` and a warning; `scale_compare_style` leaves the column NaN.
- **`corr_heatmap` colors encode the size of r, not its significance**, and
  with missing data each cell can come from a different set of rows
  (`pairwise_n_min` / `pairwise_n_max`).

### NaN and infinity policy

The Rust kernels are *strict*: one NaN anywhere poisons the whole result. That
is right for a numerics library and wrong for exploratory analysis, so this
layer makes the choice explicit rather than inheriting it:

| Helper family | Behaviour with NaN / ±inf |
|---------------|---------------------------|
| `*_report`, `robust_summary`, `describe_fast` | Dropped **per column** before the kernel runs; the count appears in `n_missing`. A column with no finite values yields an all-NaN row, not an exception. (`rolling_report` keeps one row per input row instead.) |
| `correlation_report`, `corr_heatmap` | Dropped **pairwise** — each pair uses rows where both columns are finite. `n` per pair is reported when `pvalues=True`. |
| `*_columns` | The transform is fit on finite values only, then results are scattered back into their original positions. Non-finite in ⇒ NaN out; **row alignment is always preserved**. |
| `*_style`, `style_*` | Never raise on NaN; non-finite cells receive no background color (except in `missingness_style`, whose job is to show them). |
| `missingness_report` | The exception — it *counts* rather than drops, and separates NaN from ±inf. |

Two behaviours worth calling out explicitly:

- **`normality_report` decides by Anderson-Darling.** The kernel returns the
  A\* statistic; this layer adds `ad_pvalue` (the D'Agostino-Stephens
  approximation, checked against statsmodels) and bases `normal` /
  `conclusion` on it. Jarque-Bera is reported alongside but only sees skewness
  and kurtosis, and its chi-square approximation is poor below a few hundred
  observations. There is no verdict (`normal` is `None`) for fewer than 8
  finite values or a constant column, and the conclusion says which. With
  thousands of rows either test rejects for departures too small to matter;
  `rich=True` warns about that.
- **`rolling_report(..., nan_policy="ignore")` does nothing on its own.** With
  the default `min_periods == window`, a window containing a NaN still fails
  the count check. To actually skip NaNs, pass `min_periods` below `window`:
  `rolling_report(df, "x", 5, min_periods=3, nan_policy="ignore")`.

### Backwards compatibility

The original five helpers (`demean_style`, `zscore_style`, `iqr_outlier_style`,
`corr_heatmap`, `robust_scale_column`) keep working from
`bunker_stats.pandas_helpers`, which is now a shim over the notebook layer.
They gained dtype validation and NaN safety; `iqr_outlier_style` is now a thin
wrapper over the multi-column `outlier_style`.

## Performance Highlights

Measured with `timeit` (best of 5) on Windows 11, Python 3.10, numpy 2.2 /
pandas 2.3 / scipy 1.15 / statsmodels 0.14. `ratio = reference_time /
bunker_time`, so **> 1× is faster, < 1× is slower**. Numbers are machine- and
size-dependent; reproduce with `benchmarks/`.

| Operation | Size | vs reference | Ratio |
|-----------|------|--------------|-------|
| `rolling_std` (w=50) | n=1e4 | pandas | **5.2×** |
| `rolling_std` (w=50) | n=1e6 | pandas | **2.6×** |
| `rolling_mean` (w=50) | n=1e4 / 1e6 | pandas | **3.3× / 2.4×** |
| `mad` | n=1e6 | numpy | **4.0×** |
| `median` | n=1e4 / 1e6 | numpy | **3.6× / 2.1×** |
| `bootstrap_mean_ci` (1000 res.) | n=1e4 | numpy (vectorized) | **4.9×** |
| `norm_cdf` | n=1e6 | scipy | **2.5×** |
| `kpss_test` | n=5000 | statsmodels | 1.2× |
| `percentile` | n=1e6 | numpy | ~1.1× (parity) |
| `cov_matrix` / `corr_matrix` | 1000×50 | numpy | **0.6×** (slower — see below) |
| `adf_test` (matched lag count) | n=5000 | statsmodels | **0.15×** (slower — see below) |

**Where it wins:** streaming/rolling statistics, order-statistic estimators
(median/MAD/Qn), and bootstrap confidence intervals — the O(n) single-pass and
selection-based kernels are 2–5× faster than the pandas/numpy equivalents,
with the largest margins at small-to-medium `n`.

**Where it does not (honest caveats):**
- **Dense linear algebra** (`cov_matrix`, `corr_matrix`) is ~0.6× numpy. numpy
  dispatches these to a tuned multithreaded BLAS (OpenBLAS/MKL); the portable
  pure-Rust kernels here do not link a BLAS, so they trail on large dense
  products. If matrix throughput is your bottleneck, compute these with numpy.
- **`adf_test` with an explicit lag count** is slower than statsmodels, whose
  OLS path is LAPACK-backed. (The headline "hundreds of ×" figures that
  appeared in earlier drafts compared against statsmodels' *default* automatic
  lag search, which does substantially more work — an apples-to-oranges
  comparison that has been removed.)

---

## Design Philosophy

### 1. **Determinism First**
Results are bit-exact given the same seed. Deterministic kernels produce identical results across runs and thread counts; every randomized routine takes a `random_state`, and leaving it as `None` uses seed 0 rather than entropy, so even "unseeded" calls are reproducible.

### 2. **Edge Cases Matter**
Production data has empty arrays, NaN values, zero variance, and extreme values. All functions handle these gracefully with clear, documented behavior.

### 3. **Performance Without Compromise**
Optimizations never sacrifice correctness or numerical stability. All performance claims are verified against reference implementations.

### 4. **Ergonomic Configuration**
Policy-driven design with composable configuration objects. Sensible defaults, actionable error messages, zero performance overhead.

### 5. **Comprehensive Testing**
Every edge case, every numerical corner, every performance regression is covered by tests. Test failures are treated as bugs, not warnings.

---

## API Compatibility

### NumPy/SciPy Parity
- `cov_matrix` matches `np.cov(X.T, ddof=1)`
- `corr_matrix` matches `np.corrcoef(X.T)`
- Inference functions match SciPy results to machine precision (rtol ≤ 1e-10)
- MAD with `consistent=True` matches SciPy's consistency factor (1.4826)

### Backward Compatibility
- All legacy flat functions preserved
- Config classes add features without breaking existing code
- Deprecation warnings for upcoming changes
- Semantic versioning for API changes

---

## Testing

Run the comprehensive test suite:

```bash
# All tests
pytest tests/ -v

# Specific modules
pytest tests/test_robust_stats.py -v       # Robust statistics (73 tests)
pytest tests/test_inference*.py -v         # Inference (15 tests)
pytest tests/test_matrix.py -v             # Matrix ops (83 tests)
pytest tests/test_rolling*.py -v           # Rolling windows (53 tests)
pytest tests/test_resampling.py -v         # Resampling (25 tests)
pytest tests/test_tsa*.py -v               # Time series (45/47 tests)

# With coverage
pytest tests/ --cov=bunker_stats --cov-report=html
```

**Total Test Coverage:** 294+ tests across all modules

---

## Building from Source

### Requirements
- Python ≥ 3.8
- Rust ≥ 1.70
- NumPy ≥ 1.20

### Build Commands

```bash
# Development build
maturin develop

# Optimized release build
maturin develop --release

# With parallel features (Rayon)
maturin develop --release --features parallel

# Build distributable wheel
maturin build --release
```

---

## Roadmap

### v0.3.1 (Current)
✅ NaN inputs no longer abort the interpreter; panic boundaries hardened  
✅ Numeric correctness pass (cancellation-stable covariance/correlation, exact KS p-values, corrected PSD scaling, MacKinnon ADF/PP p-values)  
✅ TSA core validated against statsmodels (ADF, KPSS, Ljung-Box, ACF/PACF)  
✅ API consistency contract (uniform window/NaN/unit rules; see the 0.3.0 release notes)  
✅ Deterministic resampling (`random_state=None` == seed 0 everywhere)  
✅ ~99% of public functions covered by direct reference tests

### v0.4.0 (Planned)
- **Multivariate robust stats:** MCD, OGK covariance
- **Robust regression:** Huber, Theil-Sen, RANSAC
- **Weighted statistics:** weighted median, MAD, robust_fit
- **BLAS-backed matrix kernels** (current pure-Rust cov/corr are ~0.6× numpy; linking a BLAS would close the gap)
- Bayesian inference module
- Model selection criteria (AIC, BIC)
- Cross-validation utilities
- Spectral density estimation enhancements

---

## Contributing

We welcome contributions! Key areas:

- **New estimators** - Additional robust/Bayesian methods
- **Performance** - SIMD, GPU acceleration
- **Documentation** - Examples, tutorials, benchmarks
- **Testing** - Edge cases, stress tests
- **Bug fixes** - Numerical issues, edge case handling

See CONTRIBUTING.md for guidelines.

---

## Citation

If using in academic work:

```bibtex
@software{bunker_stats,
  title = {bunker-stats: Production-grade statistical computing in Rust and Python},
  author = {[Author Name]},
  year = {2026},
  version = {0.2.9},
  url = {https://github.com/[repo]/bunker-stats}
}
```

---

## License

See LICENSE file in repository root.

---

## Support

- **Documentation:** See module-specific READMEs (listed above)
- **Bug Reports:** Open an issue on GitHub
- **Questions:** GitHub Discussions
- **Performance Issues:** Include benchmarks and system info

---

## Acknowledgments

Built with:
- **Rust** - High-performance kernels
- **PyO3** - Python bindings
- **Rayon** - Optional parallelism
- **statrs** - Statistical distributions

Validated against:
- **NumPy** - Matrix operations
- **SciPy** - Statistical tests and distributions
- **statsmodels** - Time series analysis
- **pandas** - Rolling window operations

---

**bunker-stats: Because real-world data demands production-grade statistics** 🚀
