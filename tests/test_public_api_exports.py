"""Public-API export integrity.

Guards the contract that no name in ``__all__`` points at a missing Rust
symbol or an unimportable attribute:

* every ``bunker_stats.__all__`` name resolves via ``getattr``;
* every resolved name is callable, a class, or a module (lazy submodules);
* no export is a ``_missing`` stub (the placeholder ``_get_rs`` installs when
  a Rust symbol is absent from the wheel — calling one raises AttributeError);
* the notebook/infer submodule ``__all__`` lists stay aligned with reality.
"""
from __future__ import annotations

import importlib
import types

import pytest

import bunker_stats as bs

pytestmark = []

LAZY_SUBMODULES = {"notebook", "pandas", "pandas_helpers"}


def _is_missing_stub(obj) -> bool:
    """True for the AttributeError-raising placeholder from ``_get_rs``."""
    return (
        isinstance(obj, types.FunctionType)
        and getattr(obj, "__name__", "") == "_fn"
        and getattr(obj, "__module__", "") == "bunker_stats"
    )


class TestRootExports:
    def test_all_names_resolve(self):
        unresolved = []
        for name in bs.__all__:
            try:
                getattr(bs, name)
            except Exception as exc:  # noqa: BLE001 - we report them all
                unresolved.append((name, repr(exc)))
        assert not unresolved, f"__all__ names failed to resolve: {unresolved}"

    def test_no_export_is_a_missing_rust_stub(self):
        stubs = [n for n in bs.__all__ if _is_missing_stub(getattr(bs, n))]
        assert not stubs, (
            f"__all__ exports backed by missing Rust symbols: {stubs}. "
            "Either export the symbol from src/lib.rs or remove the name."
        )

    def test_exports_are_callable_class_or_module(self):
        bad = []
        for name in bs.__all__:
            obj = getattr(bs, name)
            if name in LAZY_SUBMODULES or isinstance(obj, types.ModuleType):
                continue
            if not callable(obj):
                bad.append((name, type(obj).__name__))
        assert not bad, f"non-callable, non-module exports: {bad}"

    def test_all_has_no_duplicates(self):
        seen, dupes = set(), []
        for name in bs.__all__:
            if name in seen:
                dupes.append(name)
            seen.add(name)
        assert not dupes, f"duplicate __all__ entries: {dupes}"

    def test_p_adjust_facade_works_end_to_end(self):
        """The facade the repo findings flagged: p_adjust must be real."""
        import numpy as np

        out = np.asarray(bs.p_adjust(np.array([0.01, 0.02, 0.04]), "bh"))
        assert out.shape == (3,)
        assert np.all((out >= 0) & (out <= 1))

    @pytest.mark.parametrize("method", ["bonferroni", "holm", "bh"])
    def test_p_adjust_methods_accepted(self, method):
        import numpy as np

        out = np.asarray(bs.p_adjust(np.array([0.5, 0.1]), method))
        assert out.shape == (2,)

    def test_p_adjust_rejects_unknown_method(self):
        import numpy as np

        with pytest.raises(ValueError, match="method must be one of"):
            bs.p_adjust(np.array([0.5]), "not-a-method")


class TestSubmoduleExports:
    def test_infer_all_aligned(self):
        infer = importlib.import_module("bunker_stats.infer")
        for name in infer.__all__:
            assert callable(getattr(infer, name)), name

    def test_notebook_all_aligned(self):
        pytest.importorskip("pandas")
        nb = importlib.import_module("bunker_stats.notebook")
        for name in nb.__all__:
            assert callable(getattr(nb, name)), name

    def test_resampling_configs_importable(self):
        rs = importlib.import_module("bunker_stats.resampling")
        for name in ("BootstrapConfig", "PermutationConfig", "JackknifeConfig"):
            assert hasattr(rs, name), name

    def test_tsa_all_aligned(self):
        tsa = importlib.import_module("bunker_stats.tsa")
        missing = [n for n in tsa.__all__ if not hasattr(tsa, n)]
        assert not missing, f"tsa.__all__ names missing: {missing}"
