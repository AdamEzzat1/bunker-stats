"""Rich result object for rolling-window aggregates.

:class:`RollingResult` bundles the arrays produced by
:meth:`bunker_stats.Rolling.aggregate` with the window configuration that made
them. It is dict-like over the statistic names (``result["mean"]``) and knows
how to become a labeled DataFrame or a quick plot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .._result import RichResult

__all__ = ["RollingResult"]


@dataclass
class RollingResult(RichResult):
    """Result of a rolling multi-statistic aggregation.

    Dict-like over the statistic names: ``result["mean"]`` returns that array,
    ``list(result)`` yields the stat names, ``len(result)`` counts them.
    """

    table: Dict[str, np.ndarray]
    window: int
    stats: Tuple[str, ...] = ()
    min_periods: Optional[int] = None
    alignment: str = "trailing"
    nan_policy: str = "propagate"
    axis: Optional[int] = None

    _title = "Rolling Aggregate"

    def __post_init__(self):
        if not self.stats:
            self.stats = tuple(self.table.keys())

    # -- dict-like over statistic names -----------------------------------
    def __getitem__(self, key: str) -> np.ndarray:
        return self.table[key]

    def __iter__(self):
        return iter(self.stats)

    def __len__(self) -> int:
        return len(self.stats)

    def keys(self):
        return self.table.keys()

    def values(self):
        return self.table.values()

    def items(self):
        return self.table.items()

    # -- conversions ------------------------------------------------------
    def to_frame(self, index=None):
        """Return the aggregates as a ``pandas.DataFrame`` (one column per stat).

        Only valid for 1-D input (each statistic is a 1-D array). ``index``, if
        given, labels the rows.
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'pandas is required for .to_frame(); install with '
                'pip install "bunker-stats-rs[notebook]"'
            ) from exc
        cols = {f"roll{self.window}_{name}": np.asarray(self.table[name]) for name in self.stats}
        return pd.DataFrame(cols, index=index)

    def to_dict(self, *, array: bool = False) -> Dict[str, Any]:
        """Config plus the ``table`` of arrays (lists unless ``array=True``)."""
        table = {
            name: (np.asarray(v) if array else np.asarray(v).tolist())
            for name, v in self.table.items()
        }
        return {
            "table": table,
            "window": self.window,
            "stats": list(self.stats),
            "min_periods": self.min_periods,
            "alignment": self.alignment,
            "nan_policy": self.nan_policy,
            "axis": self.axis,
        }

    def plot(self):
        """Line chart of each statistic as its own trace (Plotly Figure).

        Requires the ``notebook`` extra; 1-D input only. Returns the Figure —
        never calls ``.show()``.
        """
        from .._plotly import require_go

        go = require_go()
        fig = go.Figure()
        for name in self.stats:
            arr = np.asarray(self.table[name])
            if arr.ndim != 1:
                raise ValueError(
                    "plot() supports 1-D rolling results; "
                    f"stat {name!r} has shape {arr.shape}"
                )
            fig.add_trace(
                go.Scatter(
                    y=arr,
                    mode="lines",
                    name=name,
                    hovertemplate=(
                        f"{name}[%{{x}}] = %{{y:.6g}}"
                        f"<extra>window={self.window}, "
                        f"nan_policy={self.nan_policy}</extra>"
                    ),
                )
            )
        fig.update_layout(
            title=f"Rolling window = {self.window}",
            xaxis_title="index",
            yaxis_title="value",
        )
        return fig

    def info(self) -> str:
        lines = [self._title, "=" * 60]
        rows = [
            ("Window:", self.window),
            ("Statistics:", ", ".join(self.stats)),
            ("Min periods:", self.min_periods if self.min_periods is not None else self.window),
            ("Alignment:", self.alignment),
            ("NaN policy:", self.nan_policy),
        ]
        if self.axis is not None:
            rows.append(("Axis:", self.axis))
        first = next(iter(self.table.values()), None)
        if first is not None:
            rows.append(("Output length:", np.asarray(first).shape[0]))
        for label, value in rows:
            lines.append(f"{label:<20}{value}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"RollingResult(window={self.window}, stats={list(self.stats)})"
