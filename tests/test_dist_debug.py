"""
COMPREHENSIVE PYTEST SUITE FOR DISTRIBUTION KERNELS

This suite provides rigorous testing including:
- ✅ Numerical accuracy validation against SciPy
- ✅ Edge case handling (NaN, Inf, boundary values)
- ✅ Parameter validation and error handling
- ✅ Mathematical property verification
- ✅ Consistency checks (e.g., CDF + SF = 1)
- ✅ Relationship verification (e.g., log(pdf) = logpdf)
- ✅ PPF/CDF inverse relationship testing
- ✅ Large dataset stress tests
- ✅ Vectorization and performance
- ✅ Special case correctness

Run with: pytest test_dist.py -v
"""

import pytest
import numpy as np
from scipy import stats
import sys


def assert_allclose_debug(got, ref, *, rtol, atol, name=""):
    """
    Like np.testing.assert_allclose, but on failure prints the worst absolute/relative error
    and the corresponding values to speed up debugging.
    """
    got = np.asarray(got)
    ref = np.asarray(ref)

    # mask out NaNs/Infs so we can compute errors safely
    mask = np.isfinite(got) & np.isfinite(ref)
    if not np.any(mask):
        np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)
        return

    abs_err = np.abs(got[mask] - ref[mask])
    denom = np.maximum(np.abs(ref[mask]), atol)
    rel_err = abs_err / denom

    i_abs = int(np.argmax(abs_err))
    i_rel = int(np.argmax(rel_err))

    msg = (
        f"{name}\n"
        f"max_abs_err={abs_err[i_abs]:.3e} at masked_index={i_abs} "
        f"(got={got[mask][i_abs]:.17g}, ref={ref[mask][i_abs]:.17g})\n"
        f"max_rel_err={rel_err[i_rel]:.3e} at masked_index={i_rel} "
        f"(got={got[mask][i_rel]:.17g}, ref={ref[mask][i_rel]:.17g})\n"
    )

    np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol, err_msg=msg)


# Build first: maturin develop --release
try:
    import bunker_stats as bs
except ImportError:
    print("ERROR: bunker_stats not installed. Run: maturin develop --release")
    sys.exit(1)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def rng():
    """Fixed random state for reproducibility"""
    return np.random.RandomState(42)


@pytest.fixture
def standard_x():
    """Standard test points for distribution evaluation"""
    return np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0])


@pytest.fixture
def positive_x():
    """Positive test points for distributions with x >= 0"""
    return np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])


@pytest.fixture
def unit_x():
    """Test points in unit interval"""
    return np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])


@pytest.fixture
def quantiles():
    """Standard quantile values for PPF testing"""
    return np.array([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])


@pytest.fixture
def edge_quantiles():
    """Edge case quantiles"""
    return np.array([0.0, 1e-10, 0.5, 1.0 - 1e-10, 1.0])


@pytest.fixture
def large_x(rng):
    """Large array for performance testing"""
    return rng.uniform(-10, 10, 10000)


# ==============================================================================
# EXPONENTIAL DISTRIBUTION
# ==============================================================================

class TestExponentialPDF:
    """Tests for exp_pdf"""
    
    def test_against_scipy(self, positive_x):
        """Compare with scipy.stats.expon"""
        lam = 1.5
        scipy_pdf = stats.expon.pdf(positive_x, scale=1/lam)
        bunker_pdf = bs.exp_pdf(positive_x, lam=lam)
        
        np.testing.assert_allclose(bunker_pdf, scipy_pdf, rtol=1e-14)
    
    def test_negative_values_are_zero(self):
        """PDF should be 0 for x < 0"""
        x = np.array([-5.0, -1.0, -0.1])
        pdf = bs.exp_pdf(x, lam=1.0)
        
        np.testing.assert_array_equal(pdf, np.zeros_like(x))
    
    def test_at_zero(self):
        """PDF at x=0 should equal λ"""
        lam = 2.5
        x = np.array([0.0])
        pdf = bs.exp_pdf(x, lam=lam)
        
        np.testing.assert_allclose(pdf, np.array([lam]), rtol=1e-14)
    
    def test_parameter_validation(self):
        """Should error on invalid parameters"""
        x = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="lam must be positive"):
            bs.exp_pdf(x, lam=0.0)
        
        with pytest.raises(ValueError, match="lam must be positive"):
            bs.exp_pdf(x, lam=-1.0)
    
    def test_vectorization(self, large_x):
        """Should handle large arrays efficiently"""
        x_positive = np.abs(large_x)
        pdf = bs.exp_pdf(x_positive, lam=1.0)
        
        assert pdf.shape == x_positive.shape
        assert np.all(pdf >= 0)
        assert np.all(np.isfinite(pdf))


class TestExponentialLogPDF:
    """Tests for exp_logpdf"""
    
    def test_equals_log_of_pdf(self, positive_x):
        """logpdf should equal log(pdf)"""
        lam = 1.2
        pdf = bs.exp_pdf(positive_x, lam=lam)
        logpdf = bs.exp_logpdf(positive_x, lam=lam)
        
        expected_logpdf = np.log(pdf)
        np.testing.assert_allclose(logpdf, expected_logpdf, rtol=1e-14)
    
    def test_against_scipy(self, positive_x):
        """Compare with scipy.stats.expon.logpdf"""
        lam = 2.0
        scipy_logpdf = stats.expon.logpdf(positive_x, scale=1/lam)
        bunker_logpdf = bs.exp_logpdf(positive_x, lam=lam)
        
        np.testing.assert_allclose(bunker_logpdf, scipy_logpdf, rtol=1e-14)
    
    def test_negative_values_are_neg_inf(self):
        """logpdf should be -inf for x < 0"""
        x = np.array([-5.0, -1.0, -0.1])
        logpdf = bs.exp_logpdf(x, lam=1.0)
        
        np.testing.assert_array_equal(logpdf, np.full_like(x, -np.inf))
    
    def test_nan_handling(self):
        """NaN input should return NaN"""
        x = np.array([0.5, np.nan, 1.5])
        logpdf = bs.exp_logpdf(x, lam=1.0)
        
        assert np.isnan(logpdf[1])
        assert np.isfinite(logpdf[0])
        assert np.isfinite(logpdf[2])


class TestExponentialCDF:
    """Tests for exp_cdf"""
    
    def test_against_scipy(self, positive_x):
        """Compare with scipy.stats.expon.cdf"""
        lam = 1.5
        scipy_cdf = stats.expon.cdf(positive_x, scale=1/lam)
        bunker_cdf = bs.exp_cdf(positive_x, lam=lam)
        
        np.testing.assert_allclose(bunker_cdf, scipy_cdf, rtol=1e-14)
    
    def test_bounds(self, positive_x):
        """CDF should be in [0, 1]"""
        cdf = bs.exp_cdf(positive_x, lam=1.0)
        
        assert np.all(cdf >= 0)
        assert np.all(cdf <= 1)
    
    def test_monotonic_increasing(self):
        """CDF should be monotonically increasing"""
        x = np.linspace(0, 10, 100)
        cdf = bs.exp_cdf(x, lam=1.0)
        
        assert np.all(np.diff(cdf) >= 0)
    
    def test_negative_values_are_zero(self):
        """CDF should be 0 for x < 0"""
        x = np.array([-5.0, -1.0, -0.1])
        cdf = bs.exp_cdf(x, lam=1.0)
        
        np.testing.assert_array_equal(cdf, np.zeros_like(x))
    
    def test_approaches_one(self):
        """CDF should approach 1 as x → ∞"""
        x = np.array([10.0, 50.0, 100.0])
        cdf = bs.exp_cdf(x, lam=1.0)
        
        assert cdf[-1] > 0.99999


class TestExponentialSF:
    """Tests for exp_sf (survival function)"""
    
    def test_equals_one_minus_cdf(self, positive_x):
        """sf(x) should equal 1 - cdf(x)"""
        lam = 1.5
        cdf = bs.exp_cdf(positive_x, lam=lam)
        sf = bs.exp_sf(positive_x, lam=lam)
        
        # Allow small numerical errors in floating point arithmetic
        np.testing.assert_allclose(sf, 1.0 - cdf, rtol=1e-14, atol=1e-15)
    
    def test_against_scipy(self, positive_x):
        """Compare with scipy.stats.expon.sf"""
        lam = 2.0
        scipy_sf = stats.expon.sf(positive_x, scale=1/lam)
        bunker_sf = bs.exp_sf(positive_x, lam=lam)
        
        np.testing.assert_allclose(bunker_sf, scipy_sf, rtol=1e-14)
    
    def test_negative_values_are_one(self):
        """SF should be 1 for x < 0"""
        x = np.array([-5.0, -1.0, -0.1])
        sf = bs.exp_sf(x, lam=1.0)
        
        np.testing.assert_array_equal(sf, np.ones_like(x))
    
    def test_monotonic_decreasing(self):
        """SF should be monotonically decreasing"""
        x = np.linspace(0, 10, 100)
        sf = bs.exp_sf(x, lam=1.0)
        
        assert np.all(np.diff(sf) <= 0)


class TestExponentialLogSF:
    """Tests for exp_logsf"""
    
    def test_equals_log_of_sf(self, positive_x):
        """logsf should equal log(sf)"""
        lam = 1.2
        sf = bs.exp_sf(positive_x, lam=lam)
        logsf = bs.exp_logsf(positive_x, lam=lam)
        
        expected_logsf = np.log(sf)
        np.testing.assert_allclose(logsf, expected_logsf, rtol=1e-14)
    
    def test_against_scipy(self, positive_x):
        """Compare with scipy.stats.expon.logsf"""
        lam = 2.0
        scipy_logsf = stats.expon.logsf(positive_x, scale=1/lam)
        bunker_logsf = bs.exp_logsf(positive_x, lam=lam)
        
        np.testing.assert_allclose(bunker_logsf, scipy_logsf, rtol=1e-14)
    
    def test_negative_values_are_zero(self):
        """logsf should be 0 for x < 0 (since sf=1)"""
        x = np.array([-5.0, -1.0, -0.1])
        logsf = bs.exp_logsf(x, lam=1.0)
        
        np.testing.assert_array_equal(logsf, np.zeros_like(x))
    
    def test_nan_handling(self):
        """NaN input should return NaN"""
        x = np.array([0.5, np.nan, 1.5])
        logsf = bs.exp_logsf(x, lam=1.0)
        
        assert np.isnan(logsf[1])
        assert np.isfinite(logsf[0])
        assert np.isfinite(logsf[2])


class TestExponentialCumHazard:
    """Tests for exp_cumhazard"""
    
    def test_equals_negative_logsf(self, positive_x):
        """H(x) should equal -log(sf(x))"""
        lam = 1.5
        logsf = bs.exp_logsf(positive_x, lam=lam)
        cumhazard = bs.exp_cumhazard(positive_x, lam=lam)
        
        np.testing.assert_allclose(cumhazard, -logsf, rtol=1e-14)
    
    def test_exponential_property(self, positive_x):
        """For exponential: H(x) = λx"""
        lam = 2.0
        cumhazard = bs.exp_cumhazard(positive_x, lam=lam)
        expected = lam * positive_x
        
        np.testing.assert_allclose(cumhazard, expected, rtol=1e-14)
    
    def test_monotonic_increasing(self):
        """Cumulative hazard should be monotonically increasing"""
        x = np.linspace(0, 10, 100)
        cumhazard = bs.exp_cumhazard(x, lam=1.0)
        
        assert np.all(np.diff(cumhazard) >= 0)
    
    def test_nan_handling(self):
        """NaN input should return NaN"""
        x = np.array([0.5, np.nan, 1.5])
        cumhazard = bs.exp_cumhazard(x, lam=1.0)
        
        assert np.isnan(cumhazard[1])
        assert np.isfinite(cumhazard[0])
        assert np.isfinite(cumhazard[2])


class TestExponentialPPF:
    """Tests for exp_ppf (inverse CDF)"""
    
    def test_inverse_of_cdf(self, quantiles):
        """ppf(cdf(x)) should equal x"""
        lam = 1.5
        x_original = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        
        cdf_vals = bs.exp_cdf(x_original, lam=lam)
        x_recovered = bs.exp_ppf(cdf_vals, lam=lam)
        
        np.testing.assert_allclose(x_recovered, x_original, rtol=1e-12)
    
    def test_against_scipy(self, quantiles):
        """Compare with scipy.stats.expon.ppf"""
        lam = 2.0
        scipy_ppf = stats.expon.ppf(quantiles, scale=1/lam)
        bunker_ppf = bs.exp_ppf(quantiles, lam=lam)
        
        np.testing.assert_allclose(bunker_ppf, scipy_ppf, rtol=1e-14)
    
    def test_boundary_values(self):
        """Test special quantile values"""
        lam = 1.0
        
        # q=0 should give x=0
        assert bs.exp_ppf(np.array([0.0]), lam=lam)[0] == 0.0
        
        # q=1 should give x=inf
        assert bs.exp_ppf(np.array([1.0]), lam=lam)[0] == np.inf
        
        # q=0.5 should give median
        median = bs.exp_ppf(np.array([0.5]), lam=lam)[0]
        np.testing.assert_allclose(median, np.log(2) / lam, rtol=1e-14)
    
    def test_range_validation(self):
        """Should error on quantiles outside [0, 1]"""
        with pytest.raises(ValueError, match="q values must be in"):
            bs.exp_ppf(np.array([-0.1, 0.5]), lam=1.0)
        
        with pytest.raises(ValueError, match="q values must be in"):
            bs.exp_ppf(np.array([0.5, 1.1]), lam=1.0)
    
    def test_nan_handling(self):
        """NaN input should return NaN without error"""
        q = np.array([0.25, np.nan, 0.75])
        ppf = bs.exp_ppf(q, lam=1.0)
        
        assert np.isnan(ppf[1])
        assert np.isfinite(ppf[0])
        assert np.isfinite(ppf[2])
    
    def test_monotonic_increasing(self):
        """PPF should be monotonically increasing"""
        q = np.linspace(0, 1, 100)
        ppf = bs.exp_ppf(q, lam=1.0)
        
        assert np.all(np.diff(ppf) >= 0)


# ==============================================================================
# NORMAL DISTRIBUTION
# ==============================================================================

class TestNormalPDF:
    """Tests for norm_pdf"""
    
    def test_against_scipy(self, standard_x):
        """Compare with scipy.stats.norm.pdf"""
        mu, sigma = 1.5, 2.0
        scipy_pdf = stats.norm.pdf(standard_x, loc=mu, scale=sigma)
        bunker_pdf = bs.norm_pdf(standard_x, mu=mu, sigma=sigma)
        
        np.testing.assert_allclose(bunker_pdf, scipy_pdf, rtol=1e-14)
    
    def test_standard_normal(self, standard_x):
        """Test standard normal (μ=0, σ=1)"""
        scipy_pdf = stats.norm.pdf(standard_x)
        bunker_pdf = bs.norm_pdf(standard_x)
        
        np.testing.assert_allclose(bunker_pdf, scipy_pdf, rtol=1e-14)
    
    def test_symmetry(self):
        """PDF should be symmetric around mean"""
        mu = 3.0
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pdf = bs.norm_pdf(x, mu=mu, sigma=1.0)
        
        # f(μ-δ) = f(μ+δ)
        np.testing.assert_allclose(pdf[0], pdf[4], rtol=1e-14)
        np.testing.assert_allclose(pdf[1], pdf[3], rtol=1e-14)
    
    def test_peak_at_mean(self):
        """PDF should peak at the mean"""
        mu = 2.5
        x = np.array([1.0, 2.0, 2.5, 3.0, 4.0])
        pdf = bs.norm_pdf(x, mu=mu, sigma=1.0)
        
        assert pdf[2] == np.max(pdf)
    
    def test_parameter_validation(self):
        """Should error on invalid parameters"""
        x = np.array([0.0, 1.0])
        
        with pytest.raises(ValueError, match="sigma must be positive"):
            bs.norm_pdf(x, mu=0.0, sigma=0.0)
        
        with pytest.raises(ValueError, match="sigma must be positive"):
            bs.norm_pdf(x, mu=0.0, sigma=-1.0)


class TestNormalLogPDF:
    """Tests for norm_logpdf"""
    
    def test_equals_log_of_pdf(self, standard_x):
        """logpdf should equal log(pdf)"""
        mu, sigma = 1.0, 1.5
        pdf = bs.norm_pdf(standard_x, mu=mu, sigma=sigma)
        logpdf = bs.norm_logpdf(standard_x, mu=mu, sigma=sigma)
        
        expected_logpdf = np.log(pdf)
        np.testing.assert_allclose(logpdf, expected_logpdf, rtol=1e-14)
    
    def test_against_scipy(self, standard_x):
        """Compare with scipy.stats.norm.logpdf"""
        mu, sigma = 2.0, 1.5
        scipy_logpdf = stats.norm.logpdf(standard_x, loc=mu, scale=sigma)
        bunker_logpdf = bs.norm_logpdf(standard_x, mu=mu, sigma=sigma)
        
        np.testing.assert_allclose(bunker_logpdf, scipy_logpdf, rtol=1e-14)
    
    def test_nan_handling(self):
        """NaN input should return NaN"""
        x = np.array([0.0, np.nan, 1.0])
        logpdf = bs.norm_logpdf(x)
        
        assert np.isnan(logpdf[1])
        assert np.isfinite(logpdf[0])
        assert np.isfinite(logpdf[2])


class TestNormalCDF:
    """Tests for norm_cdf"""
    
    def test_against_scipy(self, standard_x):
        """Compare with scipy.stats.norm.cdf"""
        mu, sigma = 1.5, 2.0
        scipy_cdf = stats.norm.cdf(standard_x, loc=mu, scale=sigma)
        bunker_cdf = bs.norm_cdf(standard_x, mu=mu, sigma=sigma)
        
        # statrs may have slightly different precision than scipy
        assert_allclose_debug(bunker_cdf, scipy_cdf, rtol=1e-12, atol=1e-15, name="norm_cdf vs scipy")
    
    def test_standard_normal_symmetry(self):
        """Standard normal: Φ(-x) = 1 - Φ(x)"""
        x = np.array([0.5, 1.0, 2.0])
        cdf_pos = bs.norm_cdf(x)
        cdf_neg = bs.norm_cdf(-x)
        
        np.testing.assert_allclose(cdf_neg, 1.0 - cdf_pos, rtol=1e-14)
    
    def test_at_mean(self):
        """CDF at mean should be 0.5"""
        mu = 3.5
        cdf = bs.norm_cdf(np.array([mu]), mu=mu, sigma=1.0)
        
        np.testing.assert_allclose(cdf, 0.5, rtol=1e-14)
    
    def test_monotonic_increasing(self):
        """CDF should be monotonically increasing"""
        x = np.linspace(-5, 5, 100)
        cdf = bs.norm_cdf(x)
        
        assert np.all(np.diff(cdf) >= 0)


class TestNormalSF:
    """Tests for norm_sf (survival function)"""
    
    def test_equals_one_minus_cdf(self, standard_x):
        """sf(x) should equal 1 - cdf(x)"""
        mu, sigma = 1.0, 2.0
        cdf = bs.norm_cdf(standard_x, mu=mu, sigma=sigma)
        sf = bs.norm_sf(standard_x, mu=mu, sigma=sigma)
        
        np.testing.assert_allclose(sf, 1.0 - cdf, rtol=1e-14)
    
    def test_against_scipy(self, standard_x):
        """Compare with scipy.stats.norm.sf"""
        mu, sigma = 2.0, 1.5
        scipy_sf = stats.norm.sf(standard_x, loc=mu, scale=sigma)
        bunker_sf = bs.norm_sf(standard_x, mu=mu, sigma=sigma)
        
        # statrs may have slightly different precision than scipy
        assert_allclose_debug(bunker_sf, scipy_sf, rtol=1e-12, atol=1e-15, name="norm_sf vs scipy")


class TestNormalLogSF:
    """Tests for norm_logsf"""
    
    def test_equals_log_of_sf(self, standard_x):
        """logsf should equal log(sf)"""
        mu, sigma = 1.0, 1.5
        sf = bs.norm_sf(standard_x, mu=mu, sigma=sigma)
        logsf = bs.norm_logsf(standard_x, mu=mu, sigma=sigma)
        
        # Handle cases where sf is very small
        mask = sf > 0
        np.testing.assert_allclose(logsf[mask], np.log(sf[mask]), rtol=1e-14)
        
        # Where sf=0, logsf should be -inf
        assert np.all(logsf[~mask] == -np.inf)
    
    def test_against_scipy(self, standard_x):
        """Compare with scipy.stats.norm.logsf"""
        mu, sigma = 2.0, 1.5
        scipy_logsf = stats.norm.logsf(standard_x, loc=mu, scale=sigma)
        bunker_logsf = bs.norm_logsf(standard_x, mu=mu, sigma=sigma)
        
        # statrs may have slightly different precision than scipy
        assert_allclose_debug(bunker_logsf, scipy_logsf, rtol=1e-12, atol=1e-15, name="norm_logsf vs scipy")


class TestNormalCumHazard:
    """Tests for norm_cumhazard"""
    
    def test_equals_negative_logsf(self, standard_x):
        """H(x) should equal -log(sf(x))"""
        mu, sigma = 1.0, 1.5
        logsf = bs.norm_logsf(standard_x, mu=mu, sigma=sigma)
        cumhazard = bs.norm_cumhazard(standard_x, mu=mu, sigma=sigma)
        
        np.testing.assert_allclose(cumhazard, -logsf, rtol=1e-14)
    
    def test_monotonic_increasing(self):
        """Cumulative hazard should be monotonically increasing"""
        x = np.linspace(-5, 5, 100)
        cumhazard = bs.norm_cumhazard(x)
        
        assert np.all(np.diff(cumhazard) >= 0)


class TestNormalPPF:
    """Tests for norm_ppf (inverse CDF)"""
    
    def test_inverse_of_cdf(self, quantiles):
        """ppf(cdf(x)) should equal x"""
        mu, sigma = 2.0, 1.5
        x_original = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
        
        cdf_vals = bs.norm_cdf(x_original, mu=mu, sigma=sigma)
        x_recovered = bs.norm_ppf(cdf_vals, mu=mu, sigma=sigma)
        
        # statrs precision may cause small roundtrip errors
        np.testing.assert_allclose(x_recovered, x_original, rtol=1e-10, atol=1e-12)
    
    def test_against_scipy(self, quantiles):
        """Compare with scipy.stats.norm.ppf"""
        mu, sigma = 1.5, 2.0
        scipy_ppf = stats.norm.ppf(quantiles, loc=mu, scale=sigma)
        bunker_ppf = bs.norm_ppf(quantiles, mu=mu, sigma=sigma)
        
        np.testing.assert_allclose(bunker_ppf, scipy_ppf, rtol=1e-14)
    
    def test_median(self):
        """ppf(0.5) should equal the mean"""
        mu = 3.5
        median = bs.norm_ppf(np.array([0.5]), mu=mu, sigma=1.0)
        
        np.testing.assert_allclose(median, mu, rtol=1e-14)
    
    def test_range_validation(self):
        """Should error on quantiles outside [0, 1]"""
        with pytest.raises(ValueError, match="q values must be in"):
            bs.norm_ppf(np.array([-0.1, 0.5]))
        
        with pytest.raises(ValueError, match="q values must be in"):
            bs.norm_ppf(np.array([0.5, 1.1]))
    
    def test_nan_handling(self):
        """NaN input should return NaN without error"""
        q = np.array([0.25, np.nan, 0.75])
        ppf = bs.norm_ppf(q)
        
        assert np.isnan(ppf[1])
        assert np.isfinite(ppf[0])
        assert np.isfinite(ppf[2])


# ==============================================================================
# UNIFORM DISTRIBUTION
# ==============================================================================

class TestUniformPDF:
    """Tests for unif_pdf"""
    
    def test_against_scipy(self, unit_x):
        """Compare with scipy.stats.uniform.pdf"""
        a, b = 0.0, 1.0
        scipy_pdf = stats.uniform.pdf(unit_x, loc=a, scale=b-a)
        bunker_pdf = bs.unif_pdf(unit_x, a=a, b=b)
        
        np.testing.assert_allclose(bunker_pdf, scipy_pdf, rtol=1e-14)
    
    def test_constant_in_support(self):
        """PDF should be constant 1/(b-a) in [a, b]"""
        a, b = 2.0, 5.0
        x = np.linspace(a, b, 20)
        pdf = bs.unif_pdf(x, a=a, b=b)
        
        expected = 1.0 / (b - a)
        np.testing.assert_allclose(pdf, expected, rtol=1e-14)
    
    def test_zero_outside_support(self):
        """PDF should be 0 outside [a, b]"""
        a, b = 1.0, 3.0
        x = np.array([0.0, 0.5, 3.5, 4.0])
        pdf = bs.unif_pdf(x, a=a, b=b)
        
        np.testing.assert_array_equal(pdf, np.zeros_like(x))
    
    def test_parameter_validation(self):
        """Should error on invalid parameters"""
        x = np.array([0.5, 1.5])
        
        with pytest.raises(ValueError, match="b must be greater than a"):
            bs.unif_pdf(x, a=1.0, b=1.0)
        
        with pytest.raises(ValueError, match="b must be greater than a"):
            bs.unif_pdf(x, a=2.0, b=1.0)


class TestUniformLogPDF:
    """Tests for unif_logpdf"""
    
    def test_equals_log_of_pdf(self, unit_x):
        """logpdf should equal log(pdf)"""
        a, b = 0.0, 1.0
        pdf = bs.unif_pdf(unit_x, a=a, b=b)
        logpdf = bs.unif_logpdf(unit_x, a=a, b=b)
        
        # Inside support
        mask = (unit_x >= a) & (unit_x <= b)
        np.testing.assert_allclose(logpdf[mask], np.log(pdf[mask]), rtol=1e-14)
        
        # Outside support
        assert np.all(logpdf[~mask] == -np.inf)
    
    def test_against_scipy(self, unit_x):
        """Compare with scipy.stats.uniform.logpdf"""
        a, b = 0.0, 1.0
        scipy_logpdf = stats.uniform.logpdf(unit_x, loc=a, scale=b-a)
        bunker_logpdf = bs.unif_logpdf(unit_x, a=a, b=b)
        
        np.testing.assert_allclose(bunker_logpdf, scipy_logpdf, rtol=1e-14)
    
    def test_nan_handling(self):
        """NaN input should return NaN"""
        x = np.array([0.3, np.nan, 0.7])
        logpdf = bs.unif_logpdf(x, a=0.0, b=1.0)
        
        assert np.isnan(logpdf[1])
        assert np.isfinite(logpdf[0])
        assert np.isfinite(logpdf[2])


class TestUniformCDF:
    """Tests for unif_cdf"""
    
    def test_against_scipy(self, unit_x):
        """Compare with scipy.stats.uniform.cdf"""
        a, b = 0.0, 1.0
        scipy_cdf = stats.uniform.cdf(unit_x, loc=a, scale=b-a)
        bunker_cdf = bs.unif_cdf(unit_x, a=a, b=b)
        
        np.testing.assert_allclose(bunker_cdf, scipy_cdf, rtol=1e-14)
    
    def test_linear_in_support(self):
        """CDF should be linear in [a, b]"""
        a, b = 1.0, 4.0
        x = np.array([1.0, 2.0, 3.0, 4.0])
        cdf = bs.unif_cdf(x, a=a, b=b)
        expected = np.array([0.0, 1/3, 2/3, 1.0])
        
        np.testing.assert_allclose(cdf, expected, rtol=1e-14)
    
    def test_zero_below_support(self):
        """CDF should be 0 for x < a"""
        x = np.array([0.0, 0.5])
        cdf = bs.unif_cdf(x, a=1.0, b=3.0)
        
        np.testing.assert_array_equal(cdf, np.zeros_like(x))
    
    def test_one_above_support(self):
        """CDF should be 1 for x >= b"""
        x = np.array([3.0, 4.0, 5.0])
        cdf = bs.unif_cdf(x, a=1.0, b=3.0)
        
        np.testing.assert_array_equal(cdf, np.ones_like(x))
    
    def test_monotonic_increasing(self):
        """CDF should be monotonically increasing"""
        x = np.linspace(0, 2, 100)
        cdf = bs.unif_cdf(x, a=0.5, b=1.5)
        
        assert np.all(np.diff(cdf) >= 0)


class TestUniformSF:
    """Tests for unif_sf (survival function)"""
    
    def test_equals_one_minus_cdf(self, unit_x):
        """sf(x) should equal 1 - cdf(x)"""
        a, b = 0.0, 1.0
        cdf = bs.unif_cdf(unit_x, a=a, b=b)
        sf = bs.unif_sf(unit_x, a=a, b=b)
        
        np.testing.assert_allclose(sf, 1.0 - cdf, rtol=1e-14)
    
    def test_against_scipy(self, unit_x):
        """Compare with scipy.stats.uniform.sf"""
        a, b = 0.0, 1.0
        scipy_sf = stats.uniform.sf(unit_x, loc=a, scale=b-a)
        bunker_sf = bs.unif_sf(unit_x, a=a, b=b)
        
        np.testing.assert_allclose(bunker_sf, scipy_sf, rtol=1e-14)
    
    def test_linear_decreasing(self):
        """SF should decrease linearly in [a, b]"""
        a, b = 1.0, 4.0
        x = np.array([1.0, 2.0, 3.0, 4.0])
        sf = bs.unif_sf(x, a=a, b=b)
        expected = np.array([1.0, 2/3, 1/3, 0.0])
        
        np.testing.assert_allclose(sf, expected, rtol=1e-14)


class TestUniformLogSF:
    """Tests for unif_logsf"""
    
    def test_equals_log_of_sf(self, unit_x):
        """logsf should equal log(sf)"""
        a, b = 0.0, 1.0
        sf = bs.unif_sf(unit_x, a=a, b=b)
        logsf = bs.unif_logsf(unit_x, a=a, b=b)
        
        # Where sf > 0
        mask = sf > 0
        np.testing.assert_allclose(logsf[mask], np.log(sf[mask]), rtol=1e-14)
        
        # Where sf = 0, logsf should be -inf
        assert np.all(logsf[~mask] == -np.inf)
    
    def test_against_scipy(self, unit_x):
        """Compare with scipy.stats.uniform.logsf"""
        a, b = 0.0, 1.0
        scipy_logsf = stats.uniform.logsf(unit_x, loc=a, scale=b-a)
        bunker_logsf = bs.unif_logsf(unit_x, a=a, b=b)
        
        np.testing.assert_allclose(bunker_logsf, scipy_logsf, rtol=1e-14)
    
    def test_nan_handling(self):
        """NaN input should return NaN"""
        x = np.array([0.3, np.nan, 0.7])
        logsf = bs.unif_logsf(x, a=0.0, b=1.0)
        
        assert np.isnan(logsf[1])
        assert np.isfinite(logsf[0])
        assert np.isfinite(logsf[2])


class TestUniformCumHazard:
    """Tests for unif_cumhazard"""
    
    def test_equals_negative_logsf(self, unit_x):
        """H(x) should equal -log(sf(x))"""
        a, b = 0.0, 1.0
        logsf = bs.unif_logsf(unit_x, a=a, b=b)
        cumhazard = bs.unif_cumhazard(unit_x, a=a, b=b)
        
        np.testing.assert_allclose(cumhazard, -logsf, rtol=1e-14)
    
    def test_monotonic_increasing(self):
        """Cumulative hazard should be monotonically increasing"""
        x = np.linspace(0, 1, 100)
        cumhazard = bs.unif_cumhazard(x, a=0.2, b=0.8)
        
        assert np.all(np.diff(cumhazard) >= 0)
    
    def test_nan_handling(self):
        """NaN input should return NaN"""
        x = np.array([0.3, np.nan, 0.7])
        cumhazard = bs.unif_cumhazard(x, a=0.0, b=1.0)
        
        assert np.isnan(cumhazard[1])
        assert np.isfinite(cumhazard[0])
        assert np.isfinite(cumhazard[2])


class TestUniformPPF:
    """Tests for unif_ppf (inverse CDF)"""
    
    def test_inverse_of_cdf(self, quantiles):
        """ppf(cdf(x)) should equal x"""
        a, b = 1.0, 4.0
        x_original = np.linspace(a, b, 10)
        
        cdf_vals = bs.unif_cdf(x_original, a=a, b=b)
        x_recovered = bs.unif_ppf(cdf_vals, a=a, b=b)
        
        np.testing.assert_allclose(x_recovered, x_original, rtol=1e-12)
    
    def test_against_scipy(self, quantiles):
        """Compare with scipy.stats.uniform.ppf"""
        a, b = 2.0, 5.0
        scipy_ppf = stats.uniform.ppf(quantiles, loc=a, scale=b-a)
        bunker_ppf = bs.unif_ppf(quantiles, a=a, b=b)
        
        np.testing.assert_allclose(bunker_ppf, scipy_ppf, rtol=1e-14)
    
    def test_linear_mapping(self):
        """PPF should map [0,1] linearly to [a,b]"""
        a, b = 1.0, 5.0
        q = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        ppf = bs.unif_ppf(q, a=a, b=b)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        np.testing.assert_allclose(ppf, expected, rtol=1e-14)
    
    def test_range_validation(self):
        """Should error on quantiles outside [0, 1]"""
        with pytest.raises(ValueError, match="q values must be in"):
            bs.unif_ppf(np.array([-0.1, 0.5]), a=0.0, b=1.0)
        
        with pytest.raises(ValueError, match="q values must be in"):
            bs.unif_ppf(np.array([0.5, 1.1]), a=0.0, b=1.0)
    
    def test_nan_handling(self):
        """NaN input should return NaN without error"""
        q = np.array([0.25, np.nan, 0.75])
        ppf = bs.unif_ppf(q, a=0.0, b=1.0)
        
        assert np.isnan(ppf[1])
        assert np.isfinite(ppf[0])
        assert np.isfinite(ppf[2])
    
    def test_monotonic_increasing(self):
        """PPF should be monotonically increasing"""
        q = np.linspace(0, 1, 100)
        ppf = bs.unif_ppf(q, a=0.0, b=1.0)
        
        assert np.all(np.diff(ppf) >= 0)


# ==============================================================================
# CROSS-DISTRIBUTION PROPERTY TESTS
# ==============================================================================

class TestMathematicalProperties:
    """Test mathematical properties that should hold for all distributions"""
    
    def test_cdf_plus_sf_equals_one_exponential(self, positive_x):
        """CDF + SF should equal 1"""
        lam = 1.5
        cdf = bs.exp_cdf(positive_x, lam=lam)
        sf = bs.exp_sf(positive_x, lam=lam)
        
        np.testing.assert_allclose(cdf + sf, 1.0, rtol=1e-14)
    
    def test_cdf_plus_sf_equals_one_normal(self, standard_x):
        """CDF + SF should equal 1"""
        mu, sigma = 1.0, 2.0
        cdf = bs.norm_cdf(standard_x, mu=mu, sigma=sigma)
        sf = bs.norm_sf(standard_x, mu=mu, sigma=sigma)
        
        np.testing.assert_allclose(cdf + sf, 1.0, rtol=1e-14)
    
    def test_cdf_plus_sf_equals_one_uniform(self, unit_x):
        """CDF + SF should equal 1"""
        a, b = 0.0, 1.0
        cdf = bs.unif_cdf(unit_x, a=a, b=b)
        sf = bs.unif_sf(unit_x, a=a, b=b)
        
        np.testing.assert_allclose(cdf + sf, 1.0, rtol=1e-14)
    
    def test_pdf_integrates_to_one_exponential(self):
        """∫ PDF dx should equal 1 (numerical integration)"""
        from scipy.integrate import quad
        
        lam = 1.0
        # Integrate using scipy's quad on numpy wrapper
        def pdf_func(x):
            return bs.exp_pdf(np.array([x]), lam=lam)[0]
        
        integral, _ = quad(pdf_func, 0, 20)  # Use large upper bound
        np.testing.assert_allclose(integral, 1.0, rtol=1e-5)
    
    def test_cumhazard_from_survival_exponential(self, positive_x):
        """H(x) = -ln(S(x))"""
        lam = 2.0
        sf = bs.exp_sf(positive_x, lam=lam)
        cumhazard = bs.exp_cumhazard(positive_x, lam=lam)
        
        expected_cumhazard = -np.log(sf)
        np.testing.assert_allclose(cumhazard, expected_cumhazard, rtol=1e-14)


# ==============================================================================
# PERFORMANCE AND STRESS TESTS
# ==============================================================================

class TestPerformance:
    """Performance and large-scale tests"""
    
    def test_large_array_exponential(self, rng):
        """Should handle large arrays efficiently"""
        x = rng.exponential(1.0, 100000)
        
        import time
        start = time.time()
        pdf = bs.exp_pdf(x, lam=1.0)
        elapsed = time.time() - start
        
        assert elapsed < 0.1  # Should be very fast
        assert pdf.shape == x.shape
        assert np.all(pdf >= 0)
    
    def test_large_array_normal(self, rng):
        """Should handle large arrays efficiently"""
        x = rng.normal(0, 1, 100000)
        
        import time
        start = time.time()
        pdf = bs.norm_pdf(x)
        elapsed = time.time() - start
        
        assert elapsed < 0.1  # Should be very fast
        assert pdf.shape == x.shape
        assert np.all(pdf >= 0)
    
    def test_large_array_uniform(self, rng):
        """Should handle large arrays efficiently"""
        x = rng.uniform(0, 1, 100000)
        
        import time
        start = time.time()
        pdf = bs.unif_pdf(x, a=0.0, b=1.0)
        elapsed = time.time() - start
        
        assert elapsed < 0.1  # Should be very fast
        assert pdf.shape == x.shape


# ==============================================================================
# SUMMARY REPORT
# ==============================================================================

def test_summary_report(capsys):
    """Print summary of distribution functions tested"""
    print("\n" + "="*80)
    print("DISTRIBUTION KERNEL TEST SUMMARY")
    print("="*80)
    print("\nFunctions Tested:")
    print("\nExponential (7 functions):")
    print("  ✓ exp_pdf, exp_logpdf")
    print("  ✓ exp_cdf, exp_sf, exp_logsf")
    print("  ✓ exp_cumhazard, exp_ppf")
    print("\nNormal (7 functions):")
    print("  ✓ norm_pdf, norm_logpdf")
    print("  ✓ norm_cdf, norm_sf, norm_logsf")
    print("  ✓ norm_cumhazard, norm_ppf")
    print("\nUniform (7 functions):")
    print("  ✓ unif_pdf, unif_logpdf")
    print("  ✓ unif_cdf, unif_sf, unif_logsf")
    print("  ✓ unif_cumhazard, unif_ppf")
    print("\nTotal: 21 functions")
    print("="*80)
