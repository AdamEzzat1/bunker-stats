"""Python binding confirmation for the compiled `bunker_stats_rs` extension.

This is the mandatory "it compiles != it works from Python" gate. It:

  (a) imports the extension and enumerates the whole registered public surface,
  (b) asserts every registered export is callable,
  (c) spot-checks numeric parity against numpy for a few kernels,
  (d) is a NaN-abort regression: functions that used to `partial_cmp().unwrap()`
      under `panic = "abort"` must now return NaN / raise, never SIGABRT. Each is
      run in a SUBPROCESS so an abort is an observable non-zero exit code rather
      than a hang or a torn-down test session,
  (e) confirms the zivot_andrews out-of-bounds fix: the call must return, not
      abort, for series long enough to exercise the lag regressor columns.

Build the extension first:  python -m maturin develop --release
"""
from __future__ import annotations

import subprocess
import sys
import types

import numpy as np
import pytest

bs = pytest.importorskip(
    "bunker_stats_rs",
    reason="build the extension first: `python -m maturin develop --release`",
)

# Functions that historically sorted raw user data with partial_cmp().unwrap()
# and therefore aborted the interpreter on NaN under panic="abort".
NAN_ABORT_TARGETS = [
    ("median_np", "b.median_np(x)"),
    ("mad_np", "b.mad_np(x)"),
    ("mad_std_np", "b.mad_std_np(x)"),
    ("iqr_np", "b.iqr_np(x)"),
    ("percentile_np", "b.percentile_np(x, 50.0)"),
    ("quantile_bins_np", "b.quantile_bins_np(x, 4)"),
    ("ecdf_np", "b.ecdf_np(x)"),
    ("qn_scale_np", "b.qn_scale_np(x)"),
    ("biweight_midvariance_np", "b.biweight_midvariance_np(x)"),
    ("huber_location_np", "b.huber_location_np(x)"),
    ("trimmed_mean_np", "b.trimmed_mean_np(x, 0.1)"),
    ("trimmed_std_np", "b.trimmed_std_np(x, 0.1)"),
    ("winsorized_mean_np", "b.winsorized_mean_np(x, 0.05, 0.95)"),
    ("robust_scale_np", "b.robust_scale_np(x, 1.4826)"),
]


def _public_names() -> list[str]:
    # Exclude the pyo3 module self-reference (bs.bunker_stats_rs is the module
    # object itself), which is not a registered function.
    return [
        n
        for n in dir(bs)
        if not n.startswith("_")
        and not isinstance(getattr(bs, n), types.ModuleType)
    ]


def test_public_surface_is_present_and_callable():
    """(a)+(b): every registered export resolves and is callable."""
    names = _public_names()
    # 190 wrap_pyfunction! registrations + the RobustStats class, minus the
    # module self-reference filtered above.
    print(f"\ncallable public exports: {len(names)}")
    assert len(names) >= 180, f"only {len(names)} public exports; expected ~189"
    non_callable = [n for n in names if not callable(getattr(bs, n))]
    assert not non_callable, f"registered but not callable: {non_callable}"


def test_registered_targets_exist():
    """The specific functions this hardening pass touched must be present."""
    for name, _call in NAN_ABORT_TARGETS:
        assert hasattr(bs, name), f"missing export: {name}"
    for name in ("zivot_andrews_test", "bg_test"):
        assert hasattr(bs, name), f"missing export: {name}"


def test_numeric_parity_spot_checks():
    """(c): a few kernels must match numpy/closed-form references."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(bs.median_np(x) - 3.0) < 1e-12
    assert abs(bs.mad_np(x) - 1.0) < 1e-12
    # Biweight midvariance pinned to the astropy formula (c=9, raw MAD).
    assert abs(bs.biweight_midvariance_np(x) - 2.297063991357617) < 1e-9


def _run_subprocess(body: str) -> subprocess.CompletedProcess:
    code = (
        "import numpy as np, bunker_stats_rs as b\n"
        "x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])\n"
        f"{body}\n"
        "print('OK')\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )


@pytest.mark.parametrize("name,call", NAN_ABORT_TARGETS, ids=[n for n, _ in NAN_ABORT_TARGETS])
def test_nan_input_does_not_abort(name, call):
    """(d): NaN input must not SIGABRT (run in a subprocess to observe the exit)."""
    r = _run_subprocess(call)
    assert r.returncode == 0 and "OK" in r.stdout, (
        f"{name} aborted/failed on NaN input: rc={r.returncode} err={r.stderr[-300:]}"
    )


def test_zivot_andrews_does_not_abort():
    """(e): regression for the n_regressors out-of-bounds write (SIGABRT)."""
    code = (
        "import numpy as np, bunker_stats_rs as b\n"
        "rng = np.random.default_rng(0)\n"
        "for n in (20, 25, 30, 50):\n"
        "    x = np.cumsum(rng.standard_normal(n))\n"
        "    stat, brk, pval = b.zivot_andrews_test(x)\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "OK" in r.stdout, (
        f"zivot_andrews_test aborted: rc={r.returncode} err={r.stderr[-300:]}"
    )
