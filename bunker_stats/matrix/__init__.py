"""Matrix submodule: rich result types for correlation / covariance matrices.

>>> import bunker_stats as bs
>>> C = bs.corr_matrix(X, rich=True, columns=["a", "b", "c"])   # doctest: +SKIP
>>> C.to_frame()          # labeled DataFrame
>>> C.style_heatmap()     # Styler (needs matplotlib)
>>> import numpy as np; np.asarray(C)   # the raw matrix, no copy
"""
from __future__ import annotations

from .types import CorrelationMatrixResult, CovarianceMatrixResult

__all__ = ["CorrelationMatrixResult", "CovarianceMatrixResult"]
