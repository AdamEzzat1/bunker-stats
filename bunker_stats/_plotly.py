"""Lazy Plotly gate shared by every optional ``.plot_*`` method.

Plotly is part of the ``notebook`` extra, never a core dependency. Every
figure-producing method calls :func:`require_go` at call time, so importing
bunker-stats (and even constructing result/report objects) works without
Plotly installed; only invoking a plot method requires it, and the error tells
the user exactly how to fix it.

House rules for figures (enforced by the callers):

* return a ``plotly.graph_objects.Figure`` — never call ``.show()``;
* keep defaults simple and readable;
* put method/context metadata in hover labels where useful;
* figures must survive ``fig.to_json()`` (no numpy scalars in layout/traces
  that plotly cannot serialize — plotly handles ndarrays natively).
"""
from __future__ import annotations

_INSTALL_HINT = (
    "Install with pip install bunker-stats-rs[notebook] to use Plotly "
    "visualizations."
)


def require_go():
    """Return ``plotly.graph_objects`` or raise a helpful ImportError."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return go
