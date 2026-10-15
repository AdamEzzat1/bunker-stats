"""``rich=True`` facades for the inference tests.

Each public test keeps its current dict return by default. When called with
``rich=True`` it instead returns the matching object from :mod:`.types`, built
from the raw dict plus a little input-derived metadata (sample sizes, the
chosen ``alternative``, and cheap effect sizes / CIs computed from the same
kernels).

The wrapping is done by :func:`bunker_stats._result.rich_facade`, which
forwards every argument to the raw function untouched — so the default path is
exactly backward compatible. The builders here are the only inference-specific
piece; the root package imports :data:`RICH_INFERENCE` and rebinds the names.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .._result import rich_facade
from .types import (
    ANOVAResult,
    ChiSquareResult,
    CorrelationTestResult,
    KSResult,
    MannWhitneyResult,
    NormalityResult,
    TTestResult,
)

# Anderson-Darling 5% critical value for testing normality (Stephens 1974,
# case 3: mean and variance estimated). Used to reach a verdict since the
# kernel returns no p-value.
_AD_CRIT_5PCT = 0.787


def _size(a: Any) -> int | None:
    try:
        return int(np.size(a))
    except Exception:
        return None


def _get_alt(args: tuple, kwargs: dict, pos: int) -> str:
    """Resolve ``alternative`` from kwargs, then a positional slot, else default."""
    if "alternative" in kwargs:
        return kwargs["alternative"]
    if len(args) > pos:
        return args[pos]
    return "two-sided"


def _safe(fn, *a, **k):
    """Call an optional helper kernel, degrading to None on any failure.

    Effect sizes and CIs are enrichments; if one raises, the rich object should
    still be returned with that field left as None rather than propagating an
    error the plain call would not have raised.
    """
    try:
        return fn(*a, **k)
    except Exception:
        return None


# ----------------------------------------------------------------------
# Builders: (raw_dict, args, kwargs) -> rich result
# ----------------------------------------------------------------------

def _build_t_1samp(d, args, kwargs) -> TTestResult:
    r = TTestResult(
        statistic=d["statistic"], pvalue=d["pvalue"], df=d["df"],
        alternative=_get_alt(args, kwargs, pos=2), kind="one-sample",
        mean=d.get("mean"), n1=_size(args[0]) if args else None,
    )
    r._h0 = "H0 (mean equals popmean)"
    return r


def _build_t_paired(d, args, kwargs) -> TTestResult:
    r = TTestResult(
        statistic=d["statistic"], pvalue=d["pvalue"], df=d["df"],
        alternative=_get_alt(args, kwargs, pos=2), kind="paired",
        mean=d.get("mean"), n1=_size(args[0]) if args else None,
    )
    r._h0 = "H0 (mean difference is zero)"
    return r


def _build_t_2samp(d, args, kwargs) -> TTestResult:
    import bunker_stats as bs

    x = args[0] if len(args) > 0 else None
    y = args[1] if len(args) > 1 else None
    effect = _safe(bs.cohens_d_2samp, x, y) if x is not None and y is not None else None
    r = TTestResult(
        statistic=d["statistic"], pvalue=d["pvalue"], df=d["df"],
        alternative=_get_alt(args, kwargs, pos=3), kind="two-sample",
        mean_x=d.get("mean_x"), mean_y=d.get("mean_y"),
        equal_var=d.get("equal_var"),
        effect_size=None if effect is None else float(effect),
        n1=_size(x), n2=_size(y),
    )
    r._h0 = "H0 (the two means are equal)"
    return r


def _build_chi2_gof(d, args, kwargs) -> ChiSquareResult:
    observed = np.asarray(args[0], dtype=float) if args else None
    r = ChiSquareResult(
        statistic=d["statistic"], pvalue=d["pvalue"], dof=d["df"],
        test_type="goodness_of_fit", observed=observed,
    )
    r._h0 = "H0 (observed counts match expected)"
    return r


def _build_chi2_independence(d, args, kwargs) -> ChiSquareResult:
    observed = np.asarray(args[0], dtype=float) if args else None
    r = ChiSquareResult(
        statistic=d["statistic"], pvalue=d["pvalue"], dof=d["df"],
        test_type="independence", observed=observed,
    )
    r._h0 = "H0 (the variables are independent)"
    return r


def _build_mann_whitney(d, args, kwargs) -> MannWhitneyResult:
    import bunker_stats as bs

    x = args[0] if len(args) > 0 else None
    y = args[1] if len(args) > 1 else None
    rb = _safe(bs.rank_biserial, x, y) if x is not None and y is not None else None
    return MannWhitneyResult(
        statistic=d["statistic"], pvalue=d["pvalue"],
        alternative=_get_alt(args, kwargs, pos=2),
        rank_biserial=None if rb is None else float(rb["rank_biserial"]),
        n1=_size(x), n2=_size(y),
    )


def _build_ks(d, args, kwargs) -> KSResult:
    distribution = args[1] if len(args) > 1 else kwargs.get("cdf")
    return KSResult(
        statistic=d["statistic"], pvalue=d["pvalue"],
        alternative=_get_alt(args, kwargs, pos=3),
        distribution=distribution, n=_size(args[0]) if args else None,
    )


def _build_anova(d, args, kwargs) -> ANOVAResult:
    import bunker_stats as bs

    groups = args[0] if args else None
    eff = _safe(bs.anova_effect_sizes, groups) if groups is not None else None
    n_groups = len(groups) if groups is not None else None
    n_total = sum(_size(g) or 0 for g in groups) if groups is not None else None
    return ANOVAResult(
        statistic=d["statistic"], pvalue=d["pvalue"],
        df_between=d["df_between"], df_within=d["df_within"],
        eta_squared=None if eff is None else float(eff["eta_squared"]),
        omega_squared=None if eff is None else float(eff["omega_squared"]),
        n_groups=n_groups, n_total=n_total,
    )


def _make_corr_builder(method: str):
    def _build(d, args, kwargs) -> CorrelationTestResult:
        import bunker_stats as bs

        df = d["df"]
        n = int(df) + 2 if df is not None else None
        ci_low = ci_high = None
        if method == "pearson" and len(args) >= 2:
            ci = _safe(bs.corr_ci, args[0], args[1], method="pearson")
            if ci is not None:
                ci_low, ci_high = float(ci["ci_lower"]), float(ci["ci_upper"])
        return CorrelationTestResult(
            r=d["correlation"], statistic=d["statistic"], pvalue=d["pvalue"],
            df=df, method=method, n=n, ci_low=ci_low, ci_high=ci_high,
        )

    return _build


def _build_jarque_bera(d, args, kwargs) -> NormalityResult:
    return NormalityResult(
        statistic=d["statistic"], pvalue=d.get("pvalue"), method="jarque_bera",
        n=_size(args[0]) if args else None,
        skewness=d.get("skewness"), kurtosis=d.get("kurtosis"),
    )


def _build_anderson_darling(d, args, kwargs) -> NormalityResult:
    return NormalityResult(
        statistic=d["statistic"], pvalue=d.get("pvalue"),
        method="anderson_darling", n=_size(args[0]) if args else None,
        critical_value_5pct=_AD_CRIT_5PCT,
    )


# ----------------------------------------------------------------------
# Misuse-prevention warnings, attached to every rich result post-build.
# ----------------------------------------------------------------------

def _annotate_warnings(r):
    """Attach data-quality warnings a careful statistician would flag.

    Central so every ``rich=True`` test gets the same standards: small
    samples, non-computable p-values, and approximate-p caveats. Warnings
    surface in ``.info()`` and ``.to_dict()['warnings']``.
    """
    import math

    ns = [
        v
        for v in (
            getattr(r, "n1", None),
            getattr(r, "n2", None),
            getattr(r, "n", None),
        )
        if isinstance(v, int)
    ]
    if ns and min(ns) < 8:
        r.add_warning(
            f"small sample (min n = {min(ns)}): asymptotic p-values are unreliable"
        )

    p = getattr(r, "pvalue", None)
    if isinstance(p, float) and math.isnan(p):
        r.add_warning("p-value could not be computed for this input")

    if isinstance(r, NormalityResult):
        if r.method == "jarque_bera" and isinstance(r.n, int) and r.n < 20:
            r.add_warning(
                "Jarque-Bera is asymptotic; results are unreliable below n ~ 20"
            )
        if r.method == "anderson_darling":
            r.add_warning(
                "no exact p-value; verdict compares A* to the 5% critical "
                "value (approximate)"
            )

    if isinstance(r, MannWhitneyResult):
        if ns and min(ns) < 10:
            r.add_warning(
                "normal-approximation p-value; exact tables are recommended "
                "for n < 10"
            )
    return r


def _with_warnings(builder):
    def wrapped(d, args, kwargs):
        return _annotate_warnings(builder(d, args, kwargs))

    return wrapped


# ----------------------------------------------------------------------
# name -> builder. The root package wraps each raw kernel with these.
# ----------------------------------------------------------------------
_BUILDERS = {
    "t_test_1samp": _build_t_1samp,
    "t_test_2samp": _build_t_2samp,
    "t_test_paired": _build_t_paired,
    "chi2_gof": _build_chi2_gof,
    "chi2_independence": _build_chi2_independence,
    "mann_whitney_u": _build_mann_whitney,
    "ks_1samp": _build_ks,
    "f_test_oneway": _build_anova,
    "pearson_corr_test": _make_corr_builder("pearson"),
    "spearman_corr_test": _make_corr_builder("spearman"),
    "jarque_bera": _build_jarque_bera,
    "anderson_darling": _build_anderson_darling,
}


def wrap_inference(raw_funcs: dict):
    """Return ``{name: rich-wrapped fn}`` for every name we have a builder for.

    ``raw_funcs`` maps name -> the raw (dict-returning) callable currently bound
    in the root package. Names without a builder are skipped, so passing the
    whole namespace is safe.
    """
    wrapped = {}
    for name, builder in _BUILDERS.items():
        raw = raw_funcs.get(name)
        if raw is None:
            continue
        wrapped[name] = rich_facade(raw, _with_warnings(builder), name=name)
    return wrapped
