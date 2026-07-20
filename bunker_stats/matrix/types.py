"""Rich result objects for the matrix / correlation module.

``corr_matrix`` and ``cov_matrix`` return a bare ``ndarray`` by default. With
``rich=True`` they return one of these array-like results, which behave like the
underlying matrix (``np.asarray(result)`` is a no-copy view, ``result[i, j]``
indexes it) while also carrying column labels and observation counts and knowing
how to turn themselves into a labeled DataFrame or a heatmap Styler.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from .._result import ArrayResult

__all__ = ["CorrelationMatrixResult", "CovarianceMatrixResult"]


def _default_columns(n: int, columns: Optional[List[Any]]) -> List[Any]:
    if columns is None:
        return list(range(n))
    cols = list(columns)
    if len(cols) != n:
        raise ValueError(
            f"columns has length {len(cols)} but the matrix is {n}x{n}"
        )
    return cols


class _MatrixResult(ArrayResult):
    """Shared behaviour for the square-matrix results."""

    _array_field = "matrix"

    # Subclasses are dataclasses with at least: matrix, columns, n_obs.
    matrix: np.ndarray
    columns: Optional[List[Any]]
    n_obs: Optional[int]

    def _labels(self) -> List[Any]:
        return _default_columns(self._payload().shape[0], self.columns)

    def to_frame(self):
        """Return the matrix as a labeled ``pandas.DataFrame``."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'pandas is required for .to_frame(); install with '
                'pip install "bunker-stats-rs[notebook]"'
            ) from exc
        labels = self._labels()
        return pd.DataFrame(np.asarray(self.matrix), index=labels, columns=labels)

    def _info_rows(self) -> list[tuple[str, Any]]:
        rows = [("Shape:", "x".join(map(str, self.shape)))]
        rows += self._extra_info_rows()
        if self.n_obs is not None:
            rows.append(("Observations:", self.n_obs))
        rows.append(("Columns:", self._labels()))
        return rows

    def _extra_info_rows(self) -> list[tuple[str, Any]]:
        return []


@dataclass
class CorrelationMatrixResult(_MatrixResult):
    """Correlation matrix with labels and observation count.

    Behaves like the underlying ``ndarray``; adds ``.to_frame()`` and
    ``.style_heatmap()``.
    """

    matrix: np.ndarray
    columns: Optional[List[Any]] = None
    method: str = "pearson"
    n_obs: Optional[int] = None
    pairwise_counts: Optional[np.ndarray] = None

    _title = "Correlation Matrix"

    def _extra_info_rows(self) -> list[tuple[str, Any]]:
        return [("Method:", self.method)]

    def style_heatmap(self, cmap: str = "coolwarm"):
        """Render as a diverging background-gradient Styler (needs matplotlib)."""
        return self.to_frame().style.background_gradient(cmap=cmap, vmin=-1.0, vmax=1.0)


@dataclass
class CovarianceMatrixResult(_MatrixResult):
    """Covariance matrix with labels, ``ddof`` and observation count.

    Behaves like the underlying ``ndarray``; adds ``.to_frame()``.
    """

    matrix: np.ndarray
    columns: Optional[List[Any]] = None
    ddof: int = 1
    n_obs: Optional[int] = None
    pairwise_counts: Optional[np.ndarray] = None

    _title = "Covariance Matrix"

    def _extra_info_rows(self) -> list[tuple[str, Any]]:
        estimator = "sample (ddof=1)" if self.ddof == 1 else f"ddof={self.ddof}"
        return [("Estimator:", estimator)]

    def style_heatmap(self, cmap: str = "viridis"):
        """Render as a background-gradient Styler (needs matplotlib)."""
        return self.to_frame().style.background_gradient(cmap=cmap)
