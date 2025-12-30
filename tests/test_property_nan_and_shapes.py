import numpy as np
import pytest
import warnings
from hypothesis import given, settings, strategies as st

import bunker_stats as bs


# Constrained float64 strategy:
# - finite floats bounded to avoid overflow in numpy reductions
# - plus explicit NaN coverage (Hypothesis forbids allow_nan with bounds)
finite_float64 = st.floats(
    min_value=-1e150,
    max_value=1e150,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)

float64 = st.one_of(
    finite_float64,
    st.just(np.nan),
)

arr_1d = st.lists(float64, min_size=0, max_size=500).map(
    lambda xs: np.array(xs, dtype=np.float64)
)


@settings(max_examples=200, deadline=None)
@given(arr_1d)
def test_mean_nan_matches_numpy_nanmean(x):
    out = bs.mean_nan_np(x)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Mean of empty slice",
            category=RuntimeWarning,
        )
        ref = np.nanmean(x) if x.size else np.nan

    if np.isnan(ref):
        assert np.isnan(out)
    else:
        assert np.isclose(out, ref, rtol=1e-12, atol=1e-12)


@settings(max_examples=200, deadline=None)
@given(arr_1d, st.integers(min_value=1, max_value=50))
def test_rolling_mean_nan_shapes(x, w):
    # bunker-stats contract: truncated output length n-w+1 for non-NaN-aware;
    # NaN-aware rolling may be full length (depending on your policy).
    # Here we just assert it doesn't crash and the dtype is float64.
    if x.size < w:
        return

    out = bs.rolling_mean_np(x, w)
    out = np.asarray(out)

    assert out.dtype == np.float64
    assert out.shape == (x.size - w + 1,)
