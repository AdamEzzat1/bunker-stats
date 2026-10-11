"""Regression tests for the last round of review items.

* ``TestSkipNaCorrMatrix``     - ``corr_matrix(skipna=True)`` standardized
                                 each column over its own rows, not the
                                 pair's shared rows.
* ``TestSubnormalPValues``     - chi-square p-values between 4.9e-324 and
                                 2.2e-308 were reported as exactly 0 (the t
                                 and F tests here are guards: they already
                                 reached that range).
* ``TestPermutationInterrupt`` - a huge ``n_permutations`` could not be
                                 interrupted.
* ``TestRollingFullLength``    - trailing ``Rolling`` output had no rows for
                                 ``min_periods`` to act on.
* ``TestSmallerItems``         - ``suffix=`` on ``scale_compare_style``; the
                                 small-sample bootstrap note on the plain
                                 path; the ``best_time`` fixture.

References are pandas / numpy / scipy / closed forms. Small quantities are
compared with ``close`` (no absolute tolerance).
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pytest

import bunker_stats as bs

PACKAGE_DIR = str(pathlib.Path(bs.__file__).resolve().parent)


def close(value: float, reference: float, rel: float) -> bool:
    """Relative comparison with no absolute tolerance."""
    return value == pytest.approx(reference, rel=rel, abs=0)


@pytest.fixture(scope="module")
def pd():
    return pytest.importorskip("pandas")


# ======================================================================
# corr_matrix(skipna=True)
# ======================================================================

def gappy(seed: int, n: int = 60, p: int = 5, holes: int = 25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p)) + rng.normal(size=(n, 1))   # correlated columns
    x[rng.integers(0, n, holes), rng.integers(0, p, holes)] = np.nan
    return np.ascontiguousarray(x)


def pairwise_reference(x: np.ndarray) -> np.ndarray:
    """Correlation of each pair on the rows where both are finite, with each
    column shifted by its first finite value so that large offsets cost no
    precision (pandas itself loses ~1e-5 at an offset of 1e9)."""
    p = x.shape[1]
    out = np.full((p, p), np.nan)
    for i in range(p):
        for j in range(i, p):
            both = np.isfinite(x[:, i]) & np.isfinite(x[:, j])
            a, b = x[both, i], x[both, j]
            if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
                continue
            out[i, j] = out[j, i] = np.corrcoef(a - a[0], b - b[0])[0, 1]
    return out


class TestSkipNaCorrMatrix:
    @pytest.mark.parametrize("seed", range(6))
    def test_matches_pandas(self, seed, pd):
        x = gappy(seed)
        got = np.asarray(bs.corr_matrix(x, skipna=True))
        np.testing.assert_allclose(got, pd.DataFrame(x).corr().to_numpy(), rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(got, pairwise_reference(x), rtol=1e-11, atol=1e-14)

    def test_the_old_normalisation_was_measurably_different(self, pd):
        """Guard against a vacuous comparison: on this input the two
        definitions differ in the second decimal."""
        x = gappy(0, n=12, p=3, holes=8)
        frame = pd.DataFrame(x)
        pairwise = frame.corr().to_numpy()
        cov = frame.cov().to_numpy()
        sd = np.sqrt(np.diag(cov))
        assert np.abs(cov / np.outer(sd, sd) - pairwise).max() > 1e-2
        got = np.asarray(bs.corr_matrix(x, skipna=True))
        np.testing.assert_allclose(got, pairwise, rtol=1e-11, atol=1e-14)

    def test_infinities_are_skipped(self, pd):
        x = gappy(3)
        x[5, 1], x[9, 3] = np.inf, -np.inf
        got = np.asarray(bs.corr_matrix(x, skipna=True))
        ref = pd.DataFrame(x).replace([np.inf, -np.inf], np.nan).corr().to_numpy()
        np.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-14)

    def test_complete_data_equals_the_strict_kernel(self, pd):
        x = np.ascontiguousarray(np.random.default_rng(1).normal(size=(50, 4)))
        loose = np.asarray(bs.corr_matrix(x, skipna=True))
        np.testing.assert_allclose(loose, np.asarray(bs.corr_matrix(x)), rtol=1e-12)
        np.testing.assert_allclose(loose, pd.DataFrame(x).corr().to_numpy(), rtol=1e-11)

    def test_undefined_pairs(self, pd):
        nan = np.nan
        x = np.array([
            [1.0, nan, 5.0], [2.0, nan, 5.0], [nan, 3.0, 5.0], [nan, 4.0, 6.0], [3.0, 9.0, nan],
        ])
        got = np.asarray(bs.corr_matrix(x, skipna=True))
        np.testing.assert_array_equal(np.isnan(got), np.isnan(pd.DataFrame(x).corr().to_numpy()))
        assert np.isnan(got[0, 1]) and np.isnan(got[0, 2])

    def test_large_offset(self):
        x = gappy(4) * 1e-3 + 1e9
        got = np.asarray(bs.corr_matrix(x, skipna=True))
        np.testing.assert_allclose(got, pairwise_reference(x), rtol=1e-6)

    def test_every_entry_is_a_correlation(self):
        """The old rule could leave [-1, 1]; a correlation of one set of rows
        cannot."""
        for seed in range(20):
            got = np.asarray(bs.corr_matrix(gappy(seed, n=8, p=4, holes=12), skipna=True))
            finite = got[np.isfinite(got)]
            assert np.all(np.abs(finite) <= 1.0)

    def test_corr_df_and_the_kernel_agree(self, pd):
        import bunker_stats.pandas as bsp

        x = gappy(2)
        np.testing.assert_array_equal(
            bsp.corr_df(pd.DataFrame(x)).to_numpy(), np.asarray(bs.corr_matrix(x, skipna=True))
        )


# ======================================================================
# p-values below the smallest normal float
# ======================================================================

class TestSubnormalPValues:
    """2.2e-308 is the smallest *normal* float; values down to 4.9e-324 are
    still representable. The chi-square tail used to report exactly 0 there."""

    @pytest.mark.parametrize("n", [710, 720, 730, 740])
    def test_chi_square_with_two_degrees_of_freedom_is_an_exponential(self, n):
        """All `n` observations in one of three cells: the statistic is 2n and
        the tail of a chi-square(2) is exactly exp(-stat / 2) = exp(-n)."""
        res = bs.chi2_gof(np.array([float(n), 0.0, 0.0]))
        assert close(res["statistic"], 2.0 * n, 1e-12)
        want = float(np.exp(-float(n)))
        assert 0.0 < want < 2.3e-308
        # a subnormal this small has few bits; allow one unit in the last place
        assert res["pvalue"] > 0.0
        assert abs(res["pvalue"] - want) <= 1e-9 * want + 1e-323, (res["pvalue"], want)

    def test_past_the_last_subnormal_is_zero(self):
        assert bs.chi2_gof(np.array([760.0, 0.0, 0.0]))["pvalue"] == 0.0

    @pytest.mark.parametrize("target", [45.5, 46.0, 46.5])
    def test_t_tail(self, target):
        stats = pytest.importorskip("scipy.stats")
        z = np.random.default_rng(0).normal(size=2001)
        z = (z - z.mean()) / z.std(ddof=1)                 # mean 0, sd 1 exactly
        x = z + target / np.sqrt(z.size)                    # so t is about `target`
        res = bs.t_test_1samp(x, 0.0)
        assert 0.0 < res["pvalue"] < 2.3e-308, res["pvalue"]
        # scipy is a usable reference here only in some releases: 1.15 returns
        # the subnormal value, 1.17 returns 0.0. (The Rust tests pin these
        # tails against 60-digit references on every platform.)
        ref = 2.0 * stats.t.sf(res["statistic"], res["df"])
        if ref > 0.0:
            assert abs(res["pvalue"] - ref) <= 1e-8 * ref + 1e-323, (res["pvalue"], ref)

    def test_f_tail(self):
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(1)
        groups = [rng.normal(loc, 1.0, 600) for loc in (0.0, 3.0)]
        res = bs.f_test_oneway(groups)
        assert 0.0 < res["pvalue"] < 2.3e-308, res["pvalue"]
        ref = stats.f.sf(res["statistic"], res["df_between"], res["df_within"])
        if ref > 0.0:   # see test_t_tail
            assert abs(res["pvalue"] - ref) <= 1e-8 * ref + 1e-323, (res["pvalue"], ref)

    def test_ordinary_p_values_are_unchanged(self):
        stats = pytest.importorskip("scipy.stats")
        obs = np.array([18.0, 22.0, 27.0, 33.0])
        assert close(bs.chi2_gof(obs)["pvalue"], stats.chisquare(obs).pvalue, 1e-12)
        x = np.random.default_rng(2).normal(0.2, 1.0, 40)
        assert close(bs.t_test_1samp(x, 0.0)["pvalue"], stats.ttest_1samp(x, 0.0).pvalue, 1e-10)


# ======================================================================
# Permutation tests: interruptible, and unchanged
# ======================================================================

class TestPermutationInterrupt:
    # The child reports when it is about to start, then the parent sends it a
    # real signal. (An in-process `_thread.interrupt_main()` from a timer
    # thread cannot test this: the kernel holds the GIL, so the timer thread
    # never runs. An OS signal needs no GIL to be recorded.)
    BODY = (
        "import sys, pathlib, signal, time\n"
        "sys.path.insert(0, {root!r})\n"
        "import numpy as np, bunker_stats as bs\n"
        "assert str(pathlib.Path(bs.__file__).resolve().parent) == {pkg!r}, bs.__file__\n"
        "if hasattr(signal, 'SIGBREAK'):\n"
        "    # Windows: Ctrl-C cannot be sent to another process group, Ctrl-Break\n"
        "    # can; give it the handler Ctrl-C has.\n"
        "    signal.signal(signal.SIGBREAK, signal.default_int_handler)\n"
        "x = np.random.default_rng(0).normal(size=200)\n"
        "y = np.random.default_rng(1).normal(size=200)\n"
        "print('ready', flush=True)\n"
        "start = time.perf_counter()\n"
        "try:\n"
        "    bs._rs.{name}(x, y, n_permutations=10**12)\n"
        "    print('finished', flush=True)\n"
        "except KeyboardInterrupt:\n"
        "    print('interrupted after %.1f s' % (time.perf_counter() - start), flush=True)\n"
    )

    @pytest.mark.parametrize("name", ["permutation_mean_diff_test", "permutation_corr_test"])
    def test_an_interrupt_stops_a_huge_permutation_count(self, name):
        """10**12 permutations would run for hours. A pending interrupt must
        stop it at the next batch boundary."""
        import signal
        import time

        code = self.BODY.format(
            root=str(pathlib.Path(PACKAGE_DIR).parent), pkg=PACKAGE_DIR, name=name
        )
        windows = sys.platform == "win32"
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if windows else 0,
        )
        try:
            assert proc.stdout.readline().strip() == "ready", proc.stderr.read()[-300:]
            time.sleep(1.0)                                  # let it get going
            proc.send_signal(signal.CTRL_BREAK_EVENT if windows else signal.SIGINT)
            out, err = proc.communicate(timeout=120)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        assert proc.returncode == 0, err[-300:]
        assert out.strip().startswith("interrupted after"), (out, err[-300:])

    def test_results_do_not_depend_on_where_the_batches_fall(self):
        """Each permutation is seeded from its own index, so the count of
        extreme permutations over 0..N is the sum over any split of it. The
        add-one p-value is (extreme + 1) / (N + 1); recover `extreme` for two
        prefixes and check the totals are consistent with a longer run. With
        n = 2_000_000 a batch holds 33 permutations, so 200 of them cross six
        batch boundaries."""
        rng = np.random.default_rng(0)
        x, y = rng.normal(size=1_000_000), rng.normal(0.001, 1.0, size=1_000_000)

        def extreme(count):
            _, p = bs._rs.permutation_mean_diff_test(x, y, n_permutations=count, random_state=7)
            return round(p * (count + 1) - 1)

        # counts are cumulative in the permutation index, hence non-decreasing
        a, b, c = extreme(20), extreme(100), extreme(200)
        assert 0 <= a <= b <= c <= 200
        assert c - b <= 100 and b - a <= 80

    def test_small_sample_matches_a_direct_count(self):
        """For two tiny samples every relabelling can be enumerated: the
        permutation p-value must be close to the exact one."""
        from itertools import combinations

        x, y = np.array([1.0, 2.0, 3.0, 4.0, 10.0]), np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        pooled = np.r_[x, y]
        observed = abs(x.mean() - y.mean())
        diffs = []
        for idx in combinations(range(10), 5):
            mask = np.zeros(10, dtype=bool)
            mask[list(idx)] = True
            diffs.append(abs(pooled[mask].mean() - pooled[~mask].mean()))
        exact = float(np.mean(np.array(diffs) >= observed - 1e-12))
        _, p = bs._rs.permutation_mean_diff_test(x, y, n_permutations=200_000, random_state=1)
        assert abs(p - exact) < 0.005, (p, exact)


# ======================================================================
# Rolling(full_length=True)
# ======================================================================

class TestRollingFullLength:
    STATS = ("mean", "std", "var", "min", "max")

    @pytest.fixture
    def x(self):
        x = np.random.default_rng(0).normal(size=30)
        x[[4, 5, 17]] = np.nan
        return x

    @pytest.mark.parametrize(
        "window, min_periods", [(5, 2), (5, 1), (5, None), (7, 3), (1, 1), (30, 10), (40, 3)]
    )
    def test_matches_pandas_on_every_row(self, x, pd, window, min_periods):
        got = bs.Rolling(
            x, window=window, min_periods=min_periods, nan_policy="ignore", full_length=True
        ).aggregate(*self.STATS, "count")
        roll = pd.Series(x).rolling(window, min_periods=min_periods or window)
        for stat in self.STATS:
            ref = getattr(roll, stat)().to_numpy()
            assert got[stat].shape == (x.size,)
            np.testing.assert_array_equal(np.isnan(got[stat]), np.isnan(ref), err_msg=stat)
            np.testing.assert_allclose(got[stat], ref, rtol=1e-10, atol=1e-13, equal_nan=True)
        valid = pd.Series(x).notna().rolling(window, min_periods=1).sum().to_numpy()
        np.testing.assert_array_equal(got["count"], valid)

    def test_two_dimensional(self, pd):
        x = np.random.default_rng(1).normal(size=(20, 3))
        x[3, 1] = np.nan
        got = bs.Rolling(x, window=4, min_periods=2, nan_policy="ignore", full_length=True).mean()
        ref = pd.DataFrame(x).rolling(4, min_periods=2).mean().to_numpy()
        assert got.shape == x.shape
        np.testing.assert_allclose(got, ref, rtol=1e-12, equal_nan=True)

    def test_propagate_leaves_windows_with_a_nan_empty(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        r = bs.Rolling(x, window=5, min_periods=2, full_length=True)
        np.testing.assert_array_equal(
            np.isnan(r.mean()), [True, False, True, True, True, True, True, False]
        )
        np.testing.assert_array_equal(r.count(), [1, 2, 2, 3, 4, 4, 4, 5])

    def test_tail_equals_the_default_output(self, x):
        for policy in ("propagate", "ignore"):
            short = bs.Rolling(x, window=5, min_periods=2, nan_policy=policy).mean()
            full = bs.Rolling(x, window=5, min_periods=2, nan_policy=policy, full_length=True).mean()
            assert short.shape == (26,) and full.shape == (30,)
            np.testing.assert_array_equal(full[4:], short)

    def test_default_is_unchanged_and_centered_is_unaffected(self, x):
        assert bs.Rolling(x, window=5).mean().shape == (26,)
        centered = bs.Rolling(x, window=5, alignment="centered").mean()
        flagged = bs.Rolling(x, window=5, alignment="centered", full_length=True).mean()
        np.testing.assert_array_equal(centered, flagged)

    def test_result_object(self, x):
        res = bs.Rolling(x, window=5, min_periods=2, nan_policy="ignore", full_length=True).result("mean")
        assert len(res["mean"]) == x.size


# ======================================================================
# Smaller items
# ======================================================================

class TestSmallerItems:
    def test_scale_compare_takes_suffix(self, pd):
        from bunker_stats import notebook as nb

        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 40.0]})
        want = {"zscore": "a_c", "robust_z": "a_r"}
        new = nb.scale_compare_style(frame, "a", suffix=("_c", "_r"), rich=True)
        old = nb.scale_compare_style(frame, "a", suffixes=("_c", "_r"), rich=True)
        assert new.metadata["derived_columns"] == old.metadata["derived_columns"] == want
        default = nb.scale_compare_style(frame, "a", rich=True)
        assert default.metadata["derived_columns"] == {"zscore": "a_zscore", "robust_z": "a_robust_z"}
        nb.scale_compare_style(frame, "a", suffix=("_c", "_r"), suffixes=("_c", "_r"))
        with pytest.raises(TypeError, match="pass only `suffix`"):
            nb.scale_compare_style(frame, "a", suffix=("_c", "_r"), suffixes=("_x", "_y"))
        with pytest.raises(TypeError, match="must be a pair of strings"):
            nb.scale_compare_style(frame, "a", suffix="_z")
        with pytest.raises(ValueError, match="two different strings"):
            nb.scale_compare_style(frame, "a", suffix=("_z", "_z"))

    def test_small_sample_bootstrap_warns_without_rich(self, pd):
        from bunker_stats import notebook as nb

        rng = np.random.default_rng(0)
        small = pd.DataFrame({"a": rng.normal(size=12)})
        with pytest.warns(UserWarning, match=r"small sample \(n < 30\) in column\(s\): a") as rec:
            nb.bootstrap_ci_report(small, n_resamples=500)
        assert rec[0].filename == __file__
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            nb.bootstrap_ci_report(pd.DataFrame({"a": rng.normal(size=30)}), n_resamples=500)
            report = nb.bootstrap_ci_report(small, n_resamples=500, rich=True)
        assert any("small sample" in w for w in report.warnings)

    def test_best_time_takes_the_minimum(self, best_time):
        import time

        calls = []

        def work():
            calls.append(1)
            time.sleep(0.05 if len(calls) == 1 else 0.0)   # a slow first run

        assert best_time(work, repeats=4) < 0.04
        assert len(calls) == 4
