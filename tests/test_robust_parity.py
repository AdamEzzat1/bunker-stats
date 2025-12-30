import numpy as np
import pytest
import bunker_stats as bs

def test_mad_np_matches_manual():
    rng = np.random.default_rng(0)
    x = rng.normal(size=501).astype(np.float64)

    med = np.median(x)
    exp = np.median(np.abs(x - med))
    got = bs.mad_np(x)

    assert got == pytest.approx(exp, rel=1e-12, abs=1e-12)

def test_trimmed_mean_matches_manual():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200).astype(np.float64)
    p = 0.1

    xs = np.sort(x)
    cut = int(np.floor(len(xs) * p))
    exp = xs[cut:len(xs)-cut].mean()

    got = bs.trimmed_mean_np(x, p)
    assert got == pytest.approx(exp, rel=1e-12, abs=1e-12)
