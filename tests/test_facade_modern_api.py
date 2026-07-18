"""Verification for the v0.3 modern facade layer.

Checks the four adopted conventions:
1. Clean names + keyword defaults (`bs.t_test_2samp(x, y)` works bare).
2. `skipna=` keyword dispatches to the same kernels as the *_skipna twins.
3. Explicit-unit keywords (`winsorize(lower_q=, upper_q=)`).
4. `bunker_stats.pandas` labeled-DataFrame layer.
"""
from __future__ import annotations

import numpy as np
import pytest

bs = pytest.importorskip("bunker_stats")

RNG = np.random.default_rng(777)


def _xy(n=60):
    x = RNG.normal(2.0, 1.5, n)
    y = 0.8 * x + RNG.normal(0, 1, n)
    x[[4, 19]] = np.nan
    y[[7]] = np.nan
    return x, y


class TestSkipnaDispatch:
    """bs.f(x, skipna=True) must equal the *_skipna twin; default must be strict."""

    def test_scalar_reducers(self):
        x, _ = _xy()
        for name in ["mean", "std", "var", "median", "mad", "iqr", "zscore"]:
            f = getattr(bs, name)
            twin = getattr(bs, f"{name}_skipna")
            got, ref = f(x, skipna=True), twin(x)
            assert np.allclose(np.asarray(got), np.asarray(ref), equal_nan=True), name
        # strict default: NaN input -> NaN out (never an abort, never silently skipped)
        assert np.isnan(bs.mean(x))
        assert np.isclose(bs.mean(x, skipna=True), np.nanmean(x))

    def test_trimmed_mean_kwarg(self):
        x, _ = _xy()
        assert np.isclose(
            bs.trimmed_mean(x, 0.1, skipna=True), bs.trimmed_mean_skipna(x, 0.1)
        )

    def test_pairwise_and_matrix(self):
        x, y = _xy()
        assert np.isclose(bs.cov(x, y, skipna=True), bs.cov_skipna(x, y))
        assert np.isclose(bs.corr(x, y, skipna=True), bs.corr_skipna(x, y))
        X = RNG.normal(size=(40, 3))
        assert np.allclose(
            np.asarray(bs.cov_matrix(X, skipna=True)),
            np.asarray(bs.cov_matrix_skipna(X)),
        )
        assert np.allclose(
            np.asarray(bs.cov_matrix(X)), np.cov(X, rowvar=False, ddof=1)
        )

    def test_rolling(self):
        x, y = _xy()
        w = 8
        assert np.allclose(
            np.asarray(bs.rolling_mean(x, w, skipna=True)),
            np.asarray(bs.rolling_mean_skipna(x, w)),
            equal_nan=True,
        )
        assert np.allclose(
            np.asarray(bs.rolling_cov(x, y, w, skipna=True)),
            np.asarray(bs.rolling_cov_skipna(x, y, w)),
            equal_nan=True,
        )
        # strict rolling on clean data unchanged by the wrapper
        clean = RNG.normal(size=30)
        assert len(bs.rolling_std(clean, w)) == len(clean) - w + 1


class TestKeywordErgonomics:
    def test_t_test_2samp_bare_call(self):
        a, b_ = RNG.normal(0, 1, 40), RNG.normal(0.5, 1, 40)
        r_default = bs.t_test_2samp(a, b_)                      # no TypeError
        r_welch = bs.t_test_2samp(a, b_, equal_var=False)
        r_student = bs.t_test_2samp(a, b_, equal_var=True)
        assert r_default == r_student                           # default = pooled t
        assert r_welch != r_student

    def test_effect_sizes_keyword_only(self):
        a, b_ = RNG.normal(0, 1, 30), RNG.normal(1, 1, 35)
        d = bs.cohens_d_2samp(a, b_)
        g = bs.hedges_g_2samp(a, b_)
        n = len(a) + len(b_)
        assert np.isclose(g, d * (1 - 3 / (4 * (n - 2) - 1)), rtol=1e-6)

    def test_winsorize_explicit_units(self):
        x = RNG.normal(size=200)
        out_default = np.asarray(bs.winsorize(x))               # 5% / 95% defaults
        out_kw = np.asarray(bs.winsorize(x, lower_q=0.05, upper_q=0.95))
        assert np.allclose(out_default, out_kw)
        tight = np.asarray(bs.winsorize(x, lower_q=0.25, upper_q=0.75))
        assert tight.std() < out_default.std()                  # tighter clip


class TestPandasNamespace:
    def test_cov_corr_df_labeled(self):
        pd = pytest.importorskip("pandas")
        import bunker_stats.pandas as bsp

        df = pd.DataFrame(
            {"a": RNG.normal(size=50), "b": RNG.normal(size=50),
             "c": RNG.normal(size=50), "label": ["x"] * 50}
        )
        C = bsp.corr_df(df)
        assert list(C.columns) == ["a", "b", "c"] == list(C.index)
        assert np.allclose(np.diag(C.to_numpy()), 1.0)
        V = bsp.cov_df(df)
        ref = df[["a", "b", "c"]].cov()
        assert np.allclose(V.to_numpy(), ref.to_numpy())
