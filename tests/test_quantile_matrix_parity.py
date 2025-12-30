import numpy as np
import pytest

# Try importing your Python facade
try:
    import bunker_stats as bs
except Exception as e:
    raise RuntimeError(
        "Could not import bunker_stats. Make sure you're running tests in the venv where "
        "bunker-stats is installed (maturin develop / pip install -e .)."
    ) from e


def _np_percentile_clamped(x: np.ndarray, q: float) -> float:
    # Your Rust percentile clamps q into [0, 100]
    q = float(np.clip(q, 0.0, 100.0))
    return float(np.percentile(x, q))


@pytest.mark.parametrize("n", [1, 2, 10, 101])
@pytest.mark.parametrize("q", [-10.0, 0.0, 50.0, 100.0, 120.0])
def test_percentile_np_matches_numpy_random(n, q):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n).astype(np.float64)

    got = bs.percentile_np(x, q)
    exp = _np_percentile_clamped(x, q)

    # Exact-ish for small n, but use tolerance for safety
    assert np.isfinite(got)
    assert got == pytest.approx(exp, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("q", [0.0, 25.0, 50.0, 75.0, 100.0])
def test_percentile_np_sorted_input(q):
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float64)
    got = bs.percentile_np(x, q)
    exp = float(np.percentile(x, q))
    assert got == pytest.approx(exp, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("q", [0.0, 50.0, 100.0])
def test_percentile_np_repeated_values(q):
    x = np.array([5.0] * 100, dtype=np.float64)
    got = bs.percentile_np(x, q)
    assert got == 5.0


def test_percentile_np_nan_propagates():
    x = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    got = bs.percentile_np(x, 50.0)
    assert np.isnan(got)


def test_iqr_np_matches_numpy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=257).astype(np.float64)

    q1, q3, iqr = bs.iqr_np(x)

    exp_q1 = float(np.percentile(x, 25.0))
    exp_q3 = float(np.percentile(x, 75.0))
    exp_iqr = exp_q3 - exp_q1

    assert q1 == pytest.approx(exp_q1, rel=1e-12, abs=1e-12)
    assert q3 == pytest.approx(exp_q3, rel=1e-12, abs=1e-12)
    assert iqr == pytest.approx(exp_iqr, rel=1e-12, abs=1e-12)


def test_iqr_np_nan_propagates():
    x = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float64)
    q1, q3, iqr = bs.iqr_np(x)
    assert np.isnan(q1)
    assert np.isnan(q3)
    assert np.isnan(iqr)


@pytest.mark.parametrize(
    "lower_q, upper_q",
    [
        (0.0, 100.0),
        (5.0, 95.0),
        (-10.0, 120.0),  # clamp behavior should be fine
        (25.0, 75.0),
    ],
)
def test_winsorize_np_matches_numpy_bounds(lower_q, upper_q):
    rng = np.random.default_rng(2)
    x = rng.normal(size=1000).astype(np.float64)

    out = np.asarray(bs.winsorize_np(x, lower_q, upper_q), dtype=np.float64)

    lo = _np_percentile_clamped(x, lower_q)
    hi = _np_percentile_clamped(x, upper_q)

    # Output must be clamped into [lo, hi]
    assert out.min() >= lo - 1e-12
    assert out.max() <= hi + 1e-12

    # Values below lo become lo, above hi become hi
    below = x < lo
    above = x > hi
    if below.any():
        assert np.allclose(out[below], lo, rtol=0, atol=1e-12)
    if above.any():
        assert np.allclose(out[above], hi, rtol=0, atol=1e-12)

    # Values already inside range remain unchanged
    mid = (~below) & (~above)
    assert np.allclose(out[mid], x[mid], rtol=0, atol=1e-12)


def test_cov_matrix_np_matches_numpy():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(200, 5)).astype(np.float64)

    got = np.asarray(bs.cov_matrix_np(a), dtype=np.float64)
    exp = np.cov(a, rowvar=False, bias=False)

    assert got.shape == exp.shape
    assert np.allclose(got, exp, rtol=1e-10, atol=1e-10)
    # symmetry
    assert np.allclose(got, got.T, rtol=0, atol=1e-12)


def test_corr_matrix_np_matches_numpy_and_symmetry():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(250, 6)).astype(np.float64)

    got = np.asarray(bs.corr_matrix_np(a), dtype=np.float64)
    exp = np.corrcoef(a, rowvar=False)

    assert got.shape == exp.shape
    # numpy corrcoef can be a tiny bit noisy; allow slightly looser tol
    assert np.allclose(got, exp, rtol=1e-8, atol=1e-8)
    # symmetry
    assert np.allclose(got, got.T, rtol=0, atol=1e-12)
    # diagonal should be ~1
    assert np.allclose(np.diag(got), 1.0, rtol=0, atol=1e-12)


def test_corr_matrix_constant_column_gives_nan_like_numpy():
    # If a column has zero std, numpy produces NaNs in correlation involving that col.
    rng = np.random.default_rng(5)
    x = rng.normal(size=(100, 3)).astype(np.float64)
    x[:, 1] = 7.0  # constant column

    got = np.asarray(bs.corr_matrix_np(x), dtype=np.float64)
    exp = np.corrcoef(x, rowvar=False)

    # Where numpy is NaN, we should also be NaN (or extremely close behavior)
    nan_mask = np.isnan(exp)
    assert np.array_equal(np.isnan(got), nan_mask)


