"""
tests/test_all_runner.py

A tiny "runner" test that makes it easy to run the whole suite from one entrypoint
while still keeping tests split across multiple files.

Usage:
    pytest -q tests/test_all_runner.py
    pytest -q tests/test_all_runner.py -s
    pytest -q tests/test_all_runner.py -k nan
"""

from __future__ import annotations

import importlib

# List your test modules here (module path, without ".py")
# Keep this aligned with your actual files in tests/.
_TEST_MODULES = [
    "tests.test_dist_debug",
    "tests.test_matrix",
    "tests.test_nan_coverage",
    "tests.test_property_nan_and_shapes",
    "tests.test_resampling",
    "tests.test_robust_parity",
    "tests.test_rolling_axis0",
    "tests.test_sandbox_payload_smoke_and_parity",
    "tests.test_advanced_ops",
    "tests.test_nan_behavior",
]


def test_all_test_modules_import_cleanly():
    """
    Smoke test: importing each test module should succeed.

    This catches:
      - missing optional deps that *should* be guarded by importorskip
      - accidental syntax errors
      - broken relative imports (e.g., tests._hyp_helpers)
    """
    for mod in _TEST_MODULES:
        importlib.import_module(mod)
