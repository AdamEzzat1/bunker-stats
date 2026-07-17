"""
tests/test_all_runner.py

A tiny "runner" test that makes it easy to smoke-check that every test module in
`tests/` imports cleanly from one entrypoint.

Usage:
    pytest -q tests/test_all_runner.py

The module list is discovered dynamically from the files actually present, so it
never goes stale. Previously it hardcoded three modules that do not exist
(`test_resampling`, `test_advanced_ops`, `test_nan_behavior`), which made this
test fail on every run with ModuleNotFoundError.
"""
from __future__ import annotations

import importlib
import pathlib

import pytest

_TESTS_DIR = pathlib.Path(__file__).parent
_SELF = pathlib.Path(__file__).stem

# Discover every `test_*.py` in this directory except this runner itself.
_TEST_MODULES = sorted(
    f"tests.{p.stem}"
    for p in _TESTS_DIR.glob("test_*.py")
    if p.stem != _SELF
)


@pytest.mark.parametrize("mod", _TEST_MODULES)
def test_test_module_imports_cleanly(mod: str):
    """Importing each discovered test module must succeed (or skip cleanly if it
    guards a missing optional dependency via pytest.importorskip)."""
    try:
        importlib.import_module(mod)
    except pytest.skip.Exception:
        pytest.skip(f"{mod} skipped at import (guarded optional dependency)")
