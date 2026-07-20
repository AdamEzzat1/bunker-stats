"""Inference submodule: rich result types for hypothesis tests.

The tests themselves live at the top level (``bunker_stats.t_test_2samp`` etc.);
this package holds the opt-in rich result objects returned when those tests are
called with ``rich=True``, alongside the facade that wires them up.

>>> import bunker_stats as bs
>>> res = bs.t_test_2samp(x, y, rich=True)   # doctest: +SKIP
>>> stat, pval = res                          # tuple unpacking
>>> print(res.info())
>>> res.conclusion(alpha=0.01)

The result classes can also be imported directly for type hints / isinstance
checks:

>>> from bunker_stats.infer import TTestResult   # doctest: +SKIP
"""
from __future__ import annotations

from .types import (
    ANOVAResult,
    ChiSquareResult,
    CorrelationTestResult,
    KSResult,
    MannWhitneyResult,
    NormalityResult,
    TTestResult,
)

__all__ = [
    "TTestResult",
    "ChiSquareResult",
    "MannWhitneyResult",
    "KSResult",
    "ANOVAResult",
    "CorrelationTestResult",
    "NormalityResult",
]
