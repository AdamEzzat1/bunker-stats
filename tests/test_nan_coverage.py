import numpy as np
import pandas as pd
import bunker_stats as bs


def rand_with_nans(n=1000, p_nan=0.1, seed=123):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    mask = rng.random(n) < p_nan
    x[mask] = np.nan
    return x


def test_cov_corr_nan():
    x = rand_with_nans()
    y = rand_with_nans()

    # reference
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_f = x[mask]
    y_f = y[mask]

    if x_f.size >= 2:
        cov_ref = np.cov(x_f, y_f, ddof=1)[0, 1]
        corr_ref = np.corrcoef(x_f, y_f)[0, 1]
    else:
        cov_ref = np.nan
        corr_ref = np.nan

    cov_bs = bs.cov_nan_np(x, y)
    corr_bs = bs.corr_nan_np(x, y)

    if np.isnan(cov_ref):
        assert np.isnan(cov_bs)
    else:
        assert np.allclose(cov_bs, cov_ref, atol=1e-12)

    if np.isnan(corr_ref):
        assert np.isnan(corr_bs)
    else:
        assert np.allclose(corr_bs, corr_ref, atol=1e-12)


def test_rolling_cov_corr_nan():
    x = rand_with_nans()
    y = rand_with_nans()
    window = 20

    s_x = pd.Series(x)
    s_y = pd.Series(y)

    cov_ref = s_x.rolling(window).cov(s_y, ddof=1).to_numpy()
    corr_ref = s_x.rolling(window).corr(s_y).to_numpy()

    cov_bs = np.asarray(bs.rolling_cov_nan_np(x, y, window))
    corr_bs = np.asarray(bs.rolling_corr_nan_np(x, y, window))

    mask = ~(np.isnan(cov_ref) & np.isnan(cov_bs))
    assert np.allclose(cov_bs[mask], cov_ref[mask], atol=1e-9)

    mask = ~(np.isnan(corr_ref) & np.isnan(corr_bs))
    assert np.allclose(corr_bs[mask], corr_ref[mask], atol=1e-9)
