"""Rich result objects for the robust-statistics module.

* :class:`RobustFitResult` -- the (location, scale) pair from ``robust_fit``,
  plus the estimators used and sample counts. Unpacks as ``location, scale``.
* :class:`OutlierResult` -- the boolean mask from ``iqr_outliers`` /
  ``zscore_outliers``, behaving like the mask array while carrying the method,
  bounds and counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .._result import ArrayResult, RichResult

__all__ = ["RobustFitResult", "OutlierResult"]


@dataclass
class RobustFitResult(RichResult):
    """Location/scale fit from :func:`bunker_stats.robust_fit`.

    Unpacks as ``location, scale``. ``method_location`` / ``method_scale`` record
    which estimators were used (e.g. ``median`` / ``mad``, ``huber`` / ``mad``).
    """

    location: float
    scale: float
    method_location: str = "median"
    method_scale: str = "mad"
    n: Optional[int] = None
    n_missing: Optional[int] = None

    _fields = ("location", "scale")
    _title = "Robust Fit"

    def zscores(self, x) -> np.ndarray:
        """Robust z-scores of ``x`` under this fit: ``(x - location) / scale``."""
        arr = np.asarray(x, dtype=float)
        if not np.isfinite(self.scale) or self.scale == 0.0:
            return np.full(arr.shape, np.nan)
        return (arr - self.location) / self.scale

    def _info_rows(self) -> list[tuple[str, Any]]:
        rows = [
            ("Location:", self.location),
            ("Scale:", self.scale),
            ("Location est.:", self.method_location),
            ("Scale est.:", self.method_scale),
        ]
        if self.n is not None:
            rows.append(("n (finite):", self.n))
        if self.n_missing:
            rows.append(("n missing:", self.n_missing))
        return rows


@dataclass
class OutlierResult(ArrayResult):
    """Boolean outlier mask from an IQR- or z-score-based detector.

    Behaves like the mask array (``np.asarray(result)``, ``result[i]``,
    iteration), and carries the method, fence bounds and counts.
    """

    mask: np.ndarray
    method: str = "iqr"
    threshold: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    n: Optional[int] = None
    n_outliers: Optional[int] = None
    proportion_outliers: Optional[float] = None

    _array_field = "mask"
    _title = "Outlier Detection"

    def indices(self) -> np.ndarray:
        """Positions flagged as outliers."""
        return np.flatnonzero(np.asarray(self.mask))

    def _info_rows(self) -> list[tuple[str, Any]]:
        rows: list[tuple[str, Any]] = [("Method:", self.method)]
        if self.threshold is not None:
            label = "k:" if self.method == "iqr" else "threshold:"
            rows.append((label, self.threshold))
        if self.lower_bound is not None:
            rows.append(("Bounds:", f"[{self.lower_bound:.6g}, {self.upper_bound:.6g}]"))
        if self.n_outliers is not None:
            pct = "" if self.proportion_outliers is None else f" ({100*self.proportion_outliers:.1f}%)"
            rows.append(("Outliers:", f"{self.n_outliers} / {self.n}{pct}"))
        return rows
