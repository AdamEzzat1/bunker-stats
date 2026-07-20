"""Robust submodule: rich result types for robust fitting and outliers.

>>> import bunker_stats as bs
>>> fit = bs.robust_fit(x, rich=True)     # doctest: +SKIP
>>> loc, scale = fit
>>> out = bs.iqr_outliers(x, rich=True)   # doctest: +SKIP
>>> out.indices(); out.n_outliers
"""
from __future__ import annotations

from .types import OutlierResult, RobustFitResult

__all__ = ["RobustFitResult", "OutlierResult"]
