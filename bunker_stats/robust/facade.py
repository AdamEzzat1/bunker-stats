"""``rich=True`` facades for robust fitting and outlier detection."""
from __future__ import annotations

import numpy as np

from .._result import rich_facade
from .types import OutlierResult, RobustFitResult


def _finite_counts(x):
    try:
        arr = np.asarray(x, dtype=float)
    except Exception:
        return None, None
    finite = int(np.isfinite(arr).sum())
    return finite, int(arr.size) - finite


def _build_robust_fit(result, args, kwargs):
    # robust_fit(x, location="median", scale="mad", ...) -> (location, scale)
    location, scale = result
    n, n_missing = _finite_counts(args[0]) if args else (None, None)
    method_location = kwargs.get("location", args[1] if len(args) > 1 else "median")
    method_scale = kwargs.get("scale", args[2] if len(args) > 2 else "mad")
    return RobustFitResult(
        location=float(location), scale=float(scale),
        method_location=method_location, method_scale=method_scale,
        n=n, n_missing=n_missing,
    )


def _outlier_bounds(method, x, param):
    """Recompute the fence bounds the mask kernel used (for reporting)."""
    import bunker_stats as bs

    arr = np.asarray(x, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (np.nan, np.nan)
    if method == "iqr":
        q1 = bs.percentile(finite, 25.0)
        q3 = bs.percentile(finite, 75.0)
        width = q3 - q1
        return (q1 - param * width, q3 + param * width)
    # zscore
    center = bs.mean(finite)
    spread = bs.std(finite)
    return (center - param * spread, center + param * spread)


def _make_outlier_builder(method: str, default_param: float, param_pos: int):
    def _build(mask, args, kwargs):
        m = np.asarray(mask)
        param = kwargs.get(
            "k" if method == "iqr" else "threshold",
            args[param_pos] if len(args) > param_pos else default_param,
        )
        n = int(m.size)
        n_out = int(m.sum())
        lower = upper = None
        if args:
            try:
                lower, upper = _outlier_bounds(method, args[0], float(param))
            except Exception:
                lower = upper = None
        return OutlierResult(
            mask=m, method=method, threshold=float(param),
            lower_bound=lower, upper_bound=upper,
            n=n, n_outliers=n_out,
            proportion_outliers=(n_out / n) if n else None,
        )

    return _build


def wrap_robust(raw_funcs: dict):
    """Return rich-wrapped ``robust_fit`` / ``iqr_outliers`` / ``zscore_outliers``."""
    wrapped = {}
    if raw_funcs.get("robust_fit") is not None:
        wrapped["robust_fit"] = rich_facade(
            raw_funcs["robust_fit"], _build_robust_fit, name="robust_fit",
        )
    if raw_funcs.get("iqr_outliers") is not None:
        wrapped["iqr_outliers"] = rich_facade(
            raw_funcs["iqr_outliers"],
            _make_outlier_builder("iqr", 1.5, param_pos=1),
            name="iqr_outliers",
        )
    if raw_funcs.get("zscore_outliers") is not None:
        wrapped["zscore_outliers"] = rich_facade(
            raw_funcs["zscore_outliers"],
            _make_outlier_builder("zscore", 3.0, param_pos=1),
            name="zscore_outliers",
        )
    return wrapped
