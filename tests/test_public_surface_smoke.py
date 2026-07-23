"""Public-surface smoke tests.

Guards against API drift: every name advertised in ``bunker_stats.__all__``
must resolve to a real callable, not the ``_missing`` placeholder stub the
facade installs when the Rust extension does not export a name. A placeholder
only fails when called, so a plain ``hasattr`` check is not enough — we detect
the stub by its qualname.
"""
from __future__ import annotations

import numpy as np
import pytest

bs = pytest.importorskip("bunker_stats")


def _is_missing_stub(obj) -> bool:
    # facade `_missing(name)` returns a closure `_fn`; its qualname is
    # "_missing.<locals>._fn". Real exports never have that qualname.
    return "_missing.<locals>" in getattr(obj, "__qualname__", "")


def test_all_names_resolve_to_real_objects():
    missing = []
    for name in bs.__all__:
        obj = getattr(bs, name, None)
        if obj is None:
            missing.append((name, "attribute is None"))
        elif callable(obj) and _is_missing_stub(obj):
            missing.append((name, "unregistered placeholder stub"))
    assert not missing, f"{len(missing)} __all__ names are not real exports: {missing}"


def test_all_names_are_actually_exported_attributes():
    # Everything in __all__ must be a real attribute (star-import contract).
    absent = [n for n in bs.__all__ if not hasattr(bs, n)]
    assert not absent, f"__all__ lists names absent from the module: {absent}"


def test_previously_phantom_functions_are_callable():
    # These were exposed but unregistered before v0.3.0; each must now run.
    x = np.arange(50.0)
    assert np.asarray(bs.rolling_min(x, 5)).shape[0] > 0
    assert np.asarray(bs.rolling_max(x, 5)).shape[0] > 0
    assert np.asarray(bs.rolling_range(x, 5)).shape[0] > 0
    assert np.asarray(bs.rolling_cv(x, 5)).shape[0] > 0
    assert np.asarray(bs.rolling_count_above(x, threshold=10.0, window=5)).shape[0] > 0
    assert np.asarray(bs.rolling_pct_above(x, threshold=10.0, window=5)).shape[0] > 0
    grid, dens = bs.kde_gaussian(np.random.default_rng(0).normal(size=200))
    assert np.asarray(dens).shape[0] > 0


def test_removed_debug_export_is_absent():
    # kpss_test_debug was debug scaffolding; the real kpss_test remains.
    from bunker_stats import _rs
    assert not hasattr(_rs, "kpss_test_debug")
    assert hasattr(_rs, "kpss_test")
