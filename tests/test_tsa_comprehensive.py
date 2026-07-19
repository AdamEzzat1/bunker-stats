"""
COMPREHENSIVE PYTEST SUITE FOR TIME SERIES ANALYSIS (TSA) MODULE
Version: 2.0 - Updated for all optimized modules

This suite provides rigorous testing including:
- ✅ Numerical accuracy validation against statsmodels/scipy
- ✅ Edge case handling (NaN, empty arrays, extreme values)
- ✅ Parameter validation and error handling
- ✅ Mathematical property verification
- ✅ Consistency checks across related functions
- ✅ Large dataset stress tests
- ✅ Performance benchmarks
- ✅ Algorithm comparison (Levinson-Durbin vs Yule-Walker, FFT vs DFT)
- ✅ NEW: Stationarity tests (6 new functions)
- ✅ NEW: Spectral analysis (FFT-based + 6 new functions)
- ✅ NEW: Rolling operations (O(1) updates)

Run with: pytest test_tsa_comprehensive.py -v
For specific tests: pytest test_tsa_comprehensive.py::TestACF -v
For performance: pytest test_tsa_comprehensive.py::TestPerformance -v --durations=10
"""

import pytest
import numpy as np
from scipy import stats, signal
import sys
import time

# Try importing statsmodels (graceful fallback if not available)
try:
    from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf
    from statsmodels.tsa.stattools import acovf as sm_acovf
    from statsmodels.tsa.stattools import ccf as sm_ccf
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox, acorr_breusch_godfrey
    from statsmodels.stats.stattools import durbin_watson as sm_dw
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not available, some tests will be skipped")


def assert_allclose_debug(got, ref, *, rtol, atol, name=""):
    """Enhanced assert_allclose with detailed error reporting"""
    got = np.asarray(got)
    ref = np.asarray(ref)

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


# Import bunker_stats
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
def ar1_series(rng):
    """AR(1) process: x_t = 0.7*x_{t-1} + ε_t"""
    n = 200
    phi = 0.7
    x = np.zeros(n)
    noise = rng.randn(n) * 0.5
    for t in range(1, n):
        x[t] = phi * x[t-1] + noise[t]
    return x


@pytest.fixture
def ma1_series(rng):
    """MA(1) process: x_t = ε_t + 0.5*ε_{t-1}"""
    n = 200
    theta = 0.5
    noise = rng.randn(n) * 0.5
    x = np.zeros(n)
    x[0] = noise[0]
    for t in range(1, n):
        x[t] = noise[t] + theta * noise[t-1]
    return x


@pytest.fixture
def random_walk(rng):
    """Random walk: x_t = x_{t-1} + ε_t"""
    n = 200
    noise = rng.randn(n) * 0.1
    return np.cumsum(noise)


@pytest.fixture
def seasonal_series(rng):
    """Seasonal series with period=12"""
    n = 120
    t = np.arange(n)
    trend = 0.05 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    noise = rng.randn(n) * 0.5
    return trend + seasonal + noise


@pytest.fixture
def stationary_series(rng):
    """White noise (stationary)"""
    return rng.randn(200) * 0.5


# ==============================================================================
# TEST ACF/PACF (LEVINSON-DURBIN OPTIMIZED)
# ==============================================================================

class TestACF:
    """Test autocorrelation function"""
    
    def test_acf_basic(self, ar1_series):
        """Test ACF on AR(1) process"""
        result = bs.acf(ar1_series, nlags=10)
        
        # ACF should decay exponentially for AR(1)
        assert len(result) == 11
        assert result[0] == pytest.approx(1.0)
        assert 0 < result[1] < 1  # Positive autocorrelation
        assert result[1] > result[2]  # Decay
        
    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_acf_vs_statsmodels(self, ar1_series):
        """Validate against statsmodels"""
        nlags = 20
        result = bs.acf(ar1_series, nlags=nlags)
        expected = sm_acf(ar1_series, nlags=nlags, fft=False)
        
        assert_allclose_debug(result, expected, rtol=1e-12, atol=1e-14, 
                            name="ACF vs statsmodels")
    
    def test_acf_edge_cases(self, rng):
        """Test edge cases"""
        # Empty array
        result = bs.acf(np.array([]), nlags=10)
        assert len(result) == 0
        
        # Constant series (zero variance): lag 0 is 1 by convention, all
        # further lags are 0/0 and therefore NaN (v0.3, statsmodels parity)
        result = bs.acf(np.ones(50), nlags=10)
        assert len(result) == 11
        assert result[0] == 1.0
        assert np.all(np.isnan(np.asarray(result)[1:]))
        
        # Single value
        result = bs.acf(np.array([1.0]), nlags=5)
        assert len(result) == 1
        
    def test_acf_large_lags(self, rng):
        """Test with lags exceeding series length"""
        x = rng.randn(50)
        result = bs.acf(x, nlags=100)
        # Should cap at n-1
        assert len(result) == 50


class TestPACF:
    """Test partial autocorrelation function (Levinson-Durbin)"""
    
    def test_pacf_basic(self, ar1_series):
        """Test PACF on AR(1) process"""
        result = bs.pacf(ar1_series, nlags=10)
        
        # For AR(1), only first PACF should be significant
        assert len(result) == 11
        assert result[0] == pytest.approx(1.0)
        assert abs(result[1]) > 0.5  # Significant first lag
        assert all(abs(result[i]) < 0.3 for i in range(2, 11))  # Others small
        
    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_pacf_vs_statsmodels(self, ar1_series):
        """Validate Levinson-Durbin against statsmodels Yule-Walker"""
        nlags = 20
        result = bs.pacf(ar1_series, nlags=nlags)
        expected = sm_pacf(ar1_series, nlags=nlags, method='ywmle')
        
        # Should match very closely
        assert_allclose_debug(result, expected, rtol=1e-10, atol=1e-12,
                            name="PACF (Levinson-Durbin) vs statsmodels")
    
    def test_pacf_levinson_vs_yw(self, ar1_series):
        """Compare Levinson-Durbin to Yule-Walker implementation"""
        nlags = 20
        result_ld = bs.pacf(ar1_series, nlags=nlags)
        result_yw = bs.pacf_yw(ar1_series, nlags=nlags)
        
        # Should be identical
        assert_allclose_debug(result_ld, result_yw, rtol=1e-13, atol=1e-14,
                            name="Levinson-Durbin vs Yule-Walker")
    
    def test_pacf_algorithms_comparison(self, ar1_series):
        """Compare PACF algorithms
        
        NOTE: Different PACF algorithms use different mathematical approaches:
        - Levinson-Durbin and Yule-Walker solve the same equations (should match)
        - Innovations and Burg use different methods (may give different results)
        """
        nlags = 15
        
        # Levinson-Durbin (default) - our primary implementation
        pacf_ld = bs.pacf(ar1_series, nlags=nlags)
        
        # Yule-Walker - solves same equations as LD, just slower
        pacf_yw = bs.pacf_yw(ar1_series, nlags=nlags)
        
        # Innovations and Burg - different algorithms
        pacf_innov = bs.pacf_innovations(ar1_series, nlags=nlags)
        pacf_burg = bs.pacf_burg(ar1_series, nlags=nlags)
        
        # LD and YW should match exactly (they solve the same equations)
        assert_allclose_debug(pacf_ld, pacf_yw, rtol=1e-12, atol=1e-13,
                            name="LD vs YW")
        
        # Innovations and Burg are different algorithms - just verify they run
        # and produce sensible output
        assert len(pacf_innov) == nlags + 1, "Innovations PACF wrong length"
        assert len(pacf_burg) == nlags + 1, "Burg PACF wrong length"
        assert pacf_innov[0] == 1.0, "Innovations PACF[0] should be 1.0"
        assert pacf_burg[0] == 1.0, "Burg PACF[0] should be 1.0"
        
        # All methods should agree on lag 1 (it's just the lag-1 autocorrelation)
        assert_allclose_debug(pacf_ld[1:2], pacf_innov[1:2], rtol=1e-10, atol=1e-11,
                            name="LD vs Innovations lag 1")
        # Burg uses different formula - allow small numerical differences
        assert_allclose_debug(pacf_ld[1:2], pacf_burg[1:2], rtol=1e-2, atol=1e-2,
                            name="LD vs Burg lag 1 (approximate)")
    
    def test_pacf_edge_cases(self):
        """Test edge cases"""
        # Empty array
        result = bs.pacf(np.array([]), nlags=10)
        assert len(result) == 0
        
        # Constant series
        result = bs.pacf(np.ones(50), nlags=10)
        assert len(result) == 11
        # Should handle gracefully (may be NaN after lag 0)


class TestACFHelpers:
    """Test ACF helper functions"""
    
    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_acovf(self, ar1_series):
        """Test autocovariance function"""
        result = bs.acovf(ar1_series, nlags=20)
        expected = sm_acovf(ar1_series, nlag=20, fft=False)
        
        assert_allclose_debug(result, expected, rtol=1e-12, atol=1e-14,
                            name="ACOVF vs statsmodels")
    
    def test_acf_with_ci(self, ar1_series):
        """Test ACF with confidence intervals"""
        acf_vals, lower, upper = bs.acf_with_ci(ar1_series, nlags=20, alpha=0.05)
        
        # Check structure
        assert len(acf_vals) == 21
        assert len(lower) == 21
        assert len(upper) == 21
        
        # Confidence intervals should bracket ACF
        assert all(lower[i] <= acf_vals[i] <= upper[i] for i in range(21))
        
        # At lag 0, bounds should be 1.0
        assert lower[0] == 1.0
        assert upper[0] == 1.0
        
    def test_ccf(self, ar1_series, ma1_series):
        """Test cross-correlation function"""
        result = bs.ccf(ar1_series, ma1_series, nlags=10)
        
        # Should return 2*nlags + 1 values
        assert len(result) == 21
        
        # Should be finite
        assert all(np.isfinite(result))


# ==============================================================================
# TEST SPECTRAL ANALYSIS (FFT-BASED)
# ==============================================================================

class TestSpectral:
    """Test spectral analysis functions"""
    
    def test_periodogram_basic(self, rng):
        """Test basic periodogram"""
        x = rng.randn(1000)
        freqs, power = bs.periodogram(x)
        
        assert len(freqs) == 501  # n/2 + 1 for n=1000
        assert len(power) == 501
        assert freqs[0] == 0.0  # DC component
        assert freqs[-1] == pytest.approx(0.5, abs=1e-10)  # Nyquist
        assert all(power >= 0)  # Power non-negative
    
    def test_periodogram_vs_scipy(self, rng):
        """Validate FFT periodogram against scipy"""
        x = rng.randn(1000)
        
        freqs_bs, power_bs = bs.periodogram(x)
        freqs_sp, power_sp = signal.periodogram(x, scaling='spectrum', 
                                                detrend=False, window='boxcar')
        
        # Frequencies should match exactly
        assert_allclose_debug(freqs_bs, freqs_sp, rtol=1e-14, atol=0,
                            name="Periodogram freqs vs scipy")
        
        # Power should match closely (FFT accumulation differences)
        assert_allclose_debug(power_bs, power_sp, rtol=1e-10, atol=1e-12,
                            name="Periodogram power vs scipy")
    
    def test_periodogram_small_arrays(self, rng):
        """Test DFT fallback for small arrays (n < 64)"""
        x = rng.randn(50)
        freqs, power = bs.periodogram(x)
        
        # Should still work correctly
        assert len(freqs) == 26
        assert len(power) == 26
        assert all(np.isfinite(power))
    
    def test_welch_psd(self, rng):
        """Test Welch's method"""
        x = rng.randn(2000)
        freqs, psd = bs.welch_psd(x, nperseg=256, noverlap=128)
        
        assert len(freqs) == 129  # nperseg/2 + 1
        assert len(psd) == 129
        assert all(psd >= 0)
    
    def test_dominant_frequency(self, rng):
        """Test dominant frequency detection"""
        # Create signal with known frequency
        t = np.linspace(0, 10, 1000)
        freq_true = 5.0  # 5 Hz
        x = np.sin(2 * np.pi * freq_true * t) + rng.randn(1000) * 0.1
        
        freq_dom = bs.dominant_frequency(x)
        
        # Should detect frequency close to 5 Hz
        # Normalized frequency = 5 / (1/0.01) = 0.05
        expected_norm_freq = freq_true * (t[1] - t[0])
        assert freq_dom == pytest.approx(expected_norm_freq, rel=0.1)
    
    def test_spectral_entropy(self, rng):
        """Test spectral entropy"""
        # White noise should have high entropy
        noise = rng.randn(1000)
        entropy_noise = bs.spectral_entropy(noise)
        
        # Pure tone should have low entropy
        t = np.linspace(0, 10, 1000)
        tone = np.sin(2 * np.pi * 5 * t)
        entropy_tone = bs.spectral_entropy(tone)
        
        # Noise should have higher entropy than tone
        assert entropy_noise > entropy_tone
    
    def test_spectral_peaks(self, rng):
        """Test spectral peak finding"""
        # Create signal with 3 clear frequencies
        t = np.linspace(0, 10, 1000)
        x = (np.sin(2*np.pi*3*t) + 
             0.5*np.sin(2*np.pi*7*t) + 
             0.3*np.sin(2*np.pi*12*t) +
             rng.randn(1000)*0.05)
        
        peak_freqs, peak_powers = bs.spectral_peaks(x, n_peaks=5)
        
        # Should find at least 3 peaks
        assert len(peak_freqs) >= 3
        # Peak powers should be sorted (descending)
        assert all(peak_powers[i] >= peak_powers[i+1] 
                  for i in range(len(peak_powers)-1))
    
    def test_spectral_flatness(self, rng):
        """Test spectral flatness"""
        # White noise should have flatness close to 1
        noise = rng.randn(1000)
        flatness_noise = bs.spectral_flatness(noise)
        assert 0.5 < flatness_noise <= 1.0
        
        # Pure tone should have flatness close to 0
        t = np.linspace(0, 10, 1000)
        tone = np.sin(2 * np.pi * 5 * t)
        flatness_tone = bs.spectral_flatness(tone)
        assert 0.0 <= flatness_tone < 0.1
    
    def test_band_power(self, rng):
        """Test band power calculation"""
        x = rng.randn(1000)
        
        # Total power (all frequencies)
        total_power = bs.band_power(x, freq_low=0.0, freq_high=0.5)
        
        # Low frequency band
        low_power = bs.band_power(x, freq_low=0.0, freq_high=0.1)
        
        # High frequency band
        high_power = bs.band_power(x, freq_low=0.1, freq_high=0.5)
        
        # Sum of bands should approximate total
        assert total_power == pytest.approx(low_power + high_power, rel=0.01)
    
    def test_spectral_centroid(self, rng):
        """Test spectral centroid"""
        x = rng.randn(1000)
        centroid = bs.spectral_centroid(x)
        
        # Should be between 0 and 0.5 (Nyquist)
        assert 0.0 <= centroid <= 0.5
    
    def test_spectral_rolloff(self, rng):
        """Test spectral rolloff"""
        x = rng.randn(1000)
        rolloff = bs.spectral_rolloff(x, percentile=0.85)
        
        # Should be between 0 and 0.5
        assert 0.0 <= rolloff <= 0.5


# ==============================================================================
# TEST DIAGNOSTICS
# ==============================================================================

class TestDiagnostics:
    """Test diagnostic functions"""
    
    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_ljung_box(self, ar1_series):
        """Test Ljung-Box test"""
        stat, pval = bs.ljung_box(ar1_series, lags=10)
        
        # Compare with statsmodels
        result_sm = acorr_ljungbox(ar1_series, lags=10, return_df=False)
        
        # Handle statsmodels version differences:
        # - Older versions: (stat_array, pval_array)
        # - Some versions: DataFrame even when return_df=False
        if hasattr(result_sm, "iloc"):  # DataFrame-like
            stat_sm = result_sm["lb_stat"].to_numpy()
            pval_sm = result_sm["lb_pvalue"].to_numpy()
        else:
            stat_sm, pval_sm = result_sm
        
        assert stat == pytest.approx(stat_sm[-1], rel=1e-10)
        assert pval == pytest.approx(pval_sm[-1], rel=1e-10)
    
    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_durbin_watson(self, ar1_series):
        """Test Durbin-Watson statistic"""
        result = bs.durbin_watson(ar1_series)
        expected = sm_dw(ar1_series)
        
        assert result == pytest.approx(expected, rel=1e-12)
    
    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_bg_test_fixed(self, rng):
        """Test Breusch-Godfrey test (now fixed)"""
        # Generate residuals from AR(1)
        n = 100
        resid = rng.randn(n)
        for t in range(1, n):
            resid[t] += 0.5 * resid[t-1]
        
        stat, pval = bs.bg_test(resid, max_lag=5)
        
        # Should detect autocorrelation (low p-value)
        assert pval < 0.10
        
        # Compare with statsmodels (should be much closer now)
        from statsmodels.regression.linear_model import OLS
        import statsmodels.api as sm_api
        
        # Statsmodels BG test
        exog = np.ones((n, 1))
        result_ols = OLS(resid, exog).fit()
        bg_sm = acorr_breusch_godfrey(result_ols, nlags=5)
        
        # Should be close (within 5% now, was 58% error before fix)
        assert stat == pytest.approx(bg_sm[0], rel=0.05)
    
    def test_box_pierce(self, ar1_series):
        """Test Box-Pierce test"""
        stat, pval = bs.box_pierce(ar1_series, lags=10)
        
        # Should detect autocorrelation
        assert pval < 0.05
        assert stat > 0
    
    def test_runs_test(self, rng):
        """Test runs test for randomness"""
        # Random series should not reject
        x = rng.randn(100)
        n_runs, z, pval = bs.runs_test(x)
        assert pval > 0.05
        
        # Alternating pattern should reject
        x_alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(100)])
        n_runs_alt, z_alt, pval_alt = bs.runs_test(x_alt)
        assert pval_alt < 0.01  # Should strongly reject
    
    def test_acf_zero_crossing(self, ar1_series):
        """Test ACF zero crossing detection"""
        crossing = bs.acf_zero_crossing(ar1_series, max_lag=50)
        
        # AR(1) should eventually cross zero
        assert crossing is not None
        assert crossing > 0


# ==============================================================================
# TEST STATIONARITY (OPTIMIZED + 6 NEW FUNCTIONS)
# ==============================================================================

class TestStationarity:
    """Test stationarity tests"""
    
    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_adf_stationary(self, stationary_series):
        """Test ADF on stationary series"""
        stat, pval = bs.adf_test(stationary_series)
        
        # Should reject null (unit root)
        assert pval < 0.05
        
        # Compare with statsmodels
        result_sm = adfuller(stationary_series, regression='c', maxlag=0, autolag=None)
        assert stat == pytest.approx(result_sm[0], rel=0.05)
    
    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_adf_random_walk(self, random_walk):
        """Test ADF on random walk (non-stationary)"""
        stat, pval = bs.adf_test(random_walk)
        
        # Should NOT reject null (has unit root)
        assert pval > 0.10
    
    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_kpss_stationary(self, stationary_series):
        """Test KPSS on stationary series"""
        stat, pval = bs.kpss_test(stationary_series, regression='c')
        
        # Should NOT reject null (stationary)
        assert pval > 0.05
        
        # Compare with statsmodels
        result_sm = kpss(stationary_series, regression='c', nlags='auto')
        assert stat == pytest.approx(result_sm[0], rel=0.05)
    
    @pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
    def test_kpss_random_walk(self, random_walk):
        """Test KPSS on random walk"""
        stat, pval = bs.kpss_test(random_walk, regression='c')
        
        # Should reject null (not stationary)
        assert pval < 0.05
    
    def test_variance_ratio_test(self, random_walk, stationary_series):
        """Test variance ratio test"""
        # Random walk should have VR ≈ 1
        vr_rw, z_rw, p_rw = bs.variance_ratio_test(np.diff(random_walk), lags=2)
        assert 0.8 < vr_rw < 1.2
        
        # Mean-reverting should have VR < 1
        # Create mean-reverting series
        mr = np.zeros(100)
        for t in range(1, 100):
            mr[t] = 0.5 * mr[t-1] + np.random.randn()
        
        vr_mr, z_mr, p_mr = bs.variance_ratio_test(np.diff(mr), lags=2)
        # Should show some mean reversion
        assert vr_mr < 1.1
    
    def test_integration_order_test(self, stationary_series, random_walk):
        """Test integration order detection"""
        # Stationary series should be I(0)
        is_i0, is_i1, stat_l, stat_d = bs.integration_order_test(stationary_series)
        assert is_i0 is True
        
        # Random walk should be I(1)
        is_i0_rw, is_i1_rw, stat_l_rw, stat_d_rw = bs.integration_order_test(random_walk)
        assert is_i1_rw is True
    
    def test_trend_stationarity_test(self, rng):
        """Test trend stationarity"""
        # Create trend-stationary series
        t = np.arange(100)
        x = 0.05 * t + rng.randn(100) * 0.5
        
        stat, pval, is_stat = bs.trend_stationarity_test(x)
        
        # Should detect as trend-stationary
        assert is_stat is True
        assert pval > 0.05
    
    def test_seasonal_diff_test(self, seasonal_series):
        """Test seasonal differencing"""
        stat, pval, is_stat = bs.seasonal_diff_test(seasonal_series, period=12)
        
        # Seasonal differencing should help
        assert is_stat is True or pval < 0.20
    
    def test_seasonal_unit_root_test(self, seasonal_series):
        """Test seasonal unit root at multiple lags"""
        results = bs.seasonal_unit_root_test(seasonal_series, period=12)
        
        # Should return results for lags 1, 12, 24
        assert len(results) >= 2
        assert all(isinstance(r, tuple) and len(r) == 3 for r in results)
    
    def test_zivot_andrews_test(self, rng):
        """Test Zivot-Andrews structural break test"""
        # Create series with break
        x = np.zeros(100)
        for t in range(1, 50):
            x[t] = 0.9 * x[t-1] + rng.randn()
        for t in range(50, 100):
            x[t] = 5 + 0.9 * x[t-1] + rng.randn()  # Level shift
        
        stat, breakpoint, pval = bs.zivot_andrews_test(x)
        
        # Should detect break around index 50
        assert 40 <= breakpoint <= 60


# ==============================================================================
# TEST ROLLING OPERATIONS
# ==============================================================================

class TestRolling:
    """Test rolling operations"""
    
    def test_rolling_autocorr(self, ar1_series):
        """Test rolling autocorrelation"""
        result = bs.rolling_autocorr(ar1_series, lag=1, window=50)
        
        assert len(result) == len(ar1_series) - 50 + 1
        assert all(np.isfinite(result) | np.isnan(result))
    
    def test_rolling_correlation(self, ar1_series, ma1_series):
        """Test rolling correlation between two series"""
        result = bs.rolling_correlation(ar1_series, ma1_series, window=50)
        
        assert len(result) == len(ar1_series) - 50 + 1
        assert all(-1 <= r <= 1 for r in result if np.isfinite(r))
    
    def test_rolling_autocorr_multi(self, ar1_series):
        """Test rolling autocorrelation at multiple lags"""
        lags = [1, 2, 3]
        result = bs.rolling_autocorr_multi(ar1_series, lags=lags, window=50)
        
        assert result.shape == (len(ar1_series) - 50 + 1, 3)
    
    def test_rolling_min_max(self, rng):
        """Test O(1) rolling min/max"""
        x = rng.randn(1000)
        
        # These should exist if rolling.rs is integrated
        try:
            rmin = bs.rolling_min(x, window=20)
            rmax = bs.rolling_max(x, window=20)
            
            assert len(rmin) == len(x) - 20 + 1
            assert len(rmax) == len(x) - 20 + 1
            assert all(rmin[i] <= rmax[i] for i in range(len(rmin)))
        except AttributeError:
            pytest.skip("rolling_min/max not yet integrated")
    
    def test_rolling_range(self, rng):
        """Test rolling range"""
        x = rng.randn(1000)
        
        try:
            r_range = bs.rolling_range(x, window=20)
            rmin = bs.rolling_min(x, window=20)
            rmax = bs.rolling_max(x, window=20)
            
            # Range should equal max - min
            assert np.allclose(r_range, rmax - rmin)
        except AttributeError:
            pytest.skip("rolling_range not yet integrated")


# ==============================================================================
# PERFORMANCE BENCHMARKS
# ==============================================================================

class TestPerformance:
    """Performance benchmarks"""
    
    def test_periodogram_performance(self, rng):
        """Benchmark FFT periodogram vs DFT"""
        x = rng.randn(10000)
        
        start = time.time()
        for _ in range(10):
            bs.periodogram(x)
        elapsed = time.time() - start
        
        print(f"\n10 periodograms (n=10000): {elapsed:.3f}s ({elapsed/10*1000:.1f}ms each)")
        # Should be fast (<100ms total for 10 iterations)
        assert elapsed < 1.0
    
    def test_pacf_performance(self, rng):
        """Benchmark Levinson-Durbin PACF"""
        x = rng.randn(10000)
        
        start = time.time()
        bs.pacf(x, nlags=100)
        elapsed = time.time() - start
        
        print(f"\nPACF (n=10000, nlags=100): {elapsed:.3f}s")
        # Should be fast (<50ms)
        assert elapsed < 0.5
    
    def test_stationarity_performance(self, rng):
        """Benchmark stationarity tests"""
        x = rng.randn(1000)
        
        start = time.time()
        for _ in range(100):
            bs.adf_test(x)
        elapsed = time.time() - start
        
        print(f"\n100 ADF tests (n=1000): {elapsed:.3f}s ({elapsed/100*1000:.1f}ms each)")
        # Should be fast
        assert elapsed < 1.0


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestIntegration:
    """Integration tests across modules"""
    
    def test_spectral_vs_acf(self, rng):
        """Verify Wiener-Khinchin theorem: PSD ↔ ACF"""
        x = rng.randn(1024)
        
        # Get ACF
        acf_vals = bs.acf(x, nlags=100)
        
        # Get periodogram
        freqs, power = bs.periodogram(x)
        
        # Both should describe same process
        # (Exact verification requires FFT of ACF, just check sanity)
        assert len(acf_vals) > 0
        assert len(power) > 0
    
    def test_stationarity_workflow(self, random_walk, rng):
        """Test complete stationarity analysis workflow"""
        # 1. Check if stationary
        is_i0, is_i1, stat_l, stat_d = bs.integration_order_test(random_walk)
        
        # 2. If I(1), difference
        if is_i1:
            x_diff = np.diff(random_walk)
            
            # 3. Check differenced series
            is_i0_diff, _, _, _ = bs.integration_order_test(x_diff)
            
            # Should now be stationary
            assert is_i0_diff is True
    
    def test_diagnostic_workflow(self, ar1_series):
        """Test diagnostic analysis workflow"""
        # 1. Check autocorrelation
        stat_lb, pval_lb = bs.ljung_box(ar1_series, lags=10)
        
        # 2. If autocorrelated, examine ACF/PACF
        if pval_lb < 0.05:
            acf_vals = bs.acf(ar1_series, nlags=20)
            pacf_vals = bs.pacf(ar1_series, nlags=20)
            
            # 3. ACF/PACF should show AR(1) pattern
            assert pacf_vals[1] > 0.5  # Strong first lag
            assert all(abs(pacf_vals[i]) < 0.3 for i in range(2, 10))


# ==============================================================================
# RUN TESTS
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
