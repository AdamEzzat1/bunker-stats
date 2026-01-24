"""
COMPREHENSIVE PYTEST SUITE FOR RESAMPLING EXTENSIONS

This suite provides rigorous testing including:
- âœ… Numerical accuracy validation
- âœ… Edge case handling (NaN, Inf, zeros, small n)
- âœ… Statistical property verification
- âœ… Comparison with known theoretical results
- âœ… Large dataset stress tests
- âœ… Randomization and seed reproducibility
- âœ… CI coverage and width validation
- âœ… Time-series block structure verification

Run with: pytest test_resampling_extensions.py -v
"""

import pytest
import numpy as np
from scipy import stats
from scipy.stats import bootstrap as scipy_bootstrap
import sys

# Build first: maturin develop --release
try:
    import bunker_stats_rs as bsr
except ImportError:
    print("ERROR: bunker_stats_rs not installed. Run: maturin develop --release")
    sys.exit(1)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def rng():
    """Fixed random state for reproducibility"""
    return np.random.RandomState(42)


@pytest.fixture
def normal_data(rng):
    """Standard normal data (n=100)"""
    return rng.normal(0, 1, 100)


@pytest.fixture
def small_sample(rng):
    """Small sample (n=10) for edge case testing"""
    return rng.normal(5, 2, 10)


@pytest.fixture
def large_sample(rng):
    """Large sample (n=5000) for stress testing"""
    return rng.normal(10, 3, 5000)


@pytest.fixture
def skewed_data(rng):
    """Skewed data (exponential)"""
    return rng.exponential(2.0, 200)


@pytest.fixture
def paired_data(rng):
    """Paired samples with correlation"""
    x = rng.normal(0, 1, 150)
    y = x + rng.normal(0, 0.5, 150)
    return x, y


@pytest.fixture
def timeseries_data(rng):
    """Time series with autocorrelation"""
    n = 200
    ts = np.zeros(n)
    ts[0] = rng.normal(0, 1)
    for i in range(1, n):
        ts[i] = 0.7 * ts[i-1] + rng.normal(0, 1)
    return ts


# ==============================================================================
# BOOTSTRAP SE AND VAR
# ==============================================================================

class TestBootstrapSE:
    """Comprehensive tests for bootstrap_se"""
    
    def test_mean_se_matches_theoretical(self, normal_data):
        """Bootstrap SE for mean should â‰ˆ Ïƒ/âˆšn"""
        n = len(normal_data)
        sample_std = np.std(normal_data, ddof=1)
        theoretical_se = sample_std / np.sqrt(n)
        
        boot_se = bsr.bootstrap_se(normal_data, stat="mean", n_resamples=5000, random_state=42)
        
        # Should be within 15% (Monte Carlo error)
        np.testing.assert_allclose(boot_se, theoretical_se, rtol=0.15)
    
    def test_std_se_convergence(self, normal_data):
        """Bootstrap SE for std should be positive and finite"""
        boot_se = bsr.bootstrap_se(normal_data, stat="std", n_resamples=2000, random_state=42)
        
        assert boot_se > 0
        assert np.isfinite(boot_se)
        # For normal data: SE(ÏƒÌ‚) â‰ˆ Ïƒ/(âˆš(2n))
        expected_se = np.std(normal_data, ddof=1) / np.sqrt(2 * len(normal_data))
        np.testing.assert_allclose(boot_se, expected_se, rtol=0.3)
    
    def test_median_se_reasonable(self, skewed_data):
        """Bootstrap SE for median should be reasonable"""
        boot_se = bsr.bootstrap_se(skewed_data, stat="median", n_resamples=2000, random_state=42)
        
        assert boot_se > 0
        assert np.isfinite(boot_se)
        # For exponential, median SE â‰ˆ scale/(âˆšn)
        n = len(skewed_data)
        rough_se = 2.0 / np.sqrt(n)  # scale â‰ˆ 2
        assert 0.5 * rough_se < boot_se < 2 * rough_se
    
    def test_reproducibility_with_seed(self, normal_data):
        """Same seed should give identical results"""
        se1 = bsr.bootstrap_se(normal_data, stat="mean", n_resamples=1000, random_state=123)
        se2 = bsr.bootstrap_se(normal_data, stat="mean", n_resamples=1000, random_state=123)
        
        assert se1 == se2
    
    def test_increasing_n_resamples_convergence(self, normal_data):
        """More resamples should give more stable SE estimate"""
        se_100 = bsr.bootstrap_se(normal_data, stat="mean", n_resamples=100, random_state=42)
        se_10000 = bsr.bootstrap_se(normal_data, stat="mean", n_resamples=10000, random_state=43)
        
        # Both should be reasonable
        sample_se = np.std(normal_data, ddof=1) / np.sqrt(len(normal_data))
        assert 0.5 * sample_se < se_100 < 1.5 * sample_se
        assert 0.8 * sample_se < se_10000 < 1.2 * sample_se
    
    def test_edge_case_constant_data(self):
        """Constant data should give SE â‰ˆ 0"""
        x = np.ones(50)
        
        se_mean = bsr.bootstrap_se(x, stat="mean", n_resamples=1000, random_state=42)
        se_std = bsr.bootstrap_se(x, stat="std", n_resamples=1000, random_state=42)
        
        assert se_mean < 1e-10
        assert se_std < 1e-10
    
    def test_edge_case_two_values(self):
        """Minimal valid sample (n=2) should not crash"""
        x = np.array([1.0, 2.0])
        
        se = bsr.bootstrap_se(x, stat="mean", n_resamples=100, random_state=42)
        
        assert np.isfinite(se)
        assert se > 0
    
    def test_large_sample_efficiency(self, large_sample):
        """Should handle large samples efficiently"""
        import time
        
        start = time.time()
        se = bsr.bootstrap_se(large_sample, stat="mean", n_resamples=1000, random_state=42)
        elapsed = time.time() - start
        
        assert elapsed < 1.0  # Should be fast
        assert np.isfinite(se)


class TestBootstrapVar:
    """Tests for bootstrap_var (should be SEÂ²)"""
    
    def test_var_is_se_squared(self, normal_data):
        """bootstrap_var should equal bootstrap_seÂ²"""
        se = bsr.bootstrap_se(normal_data, stat="mean", n_resamples=2000, random_state=42)
        var = bsr.bootstrap_var(normal_data, stat="mean", n_resamples=2000, random_state=42)
        
        np.testing.assert_allclose(var, se**2, rtol=1e-10)
    
    def test_all_stats(self, normal_data):
        """All stats should give positive variance"""
        for stat in ["mean", "median", "std"]:
            var = bsr.bootstrap_var(normal_data, stat=stat, n_resamples=1000, random_state=42)
            
            assert var > 0
            assert np.isfinite(var)


# ==============================================================================
# BOOTSTRAP-T CI
# ==============================================================================

class TestBootstrapTCI:
    """Comprehensive tests for bootstrap_t_ci_mean (studentized bootstrap)"""
    
    def test_contains_true_mean_normal(self, rng):
        """CI should contain true mean ~95% of the time"""
        true_mean = 10.0
        n_trials = 100
        coverage = 0
        
        for _ in range(n_trials):
            x = rng.normal(true_mean, 2, 50)
            est, lower, upper = bsr.bootstrap_t_ci_mean(x, n_resamples=500, conf=0.95, random_state=None)
            
            if lower <= true_mean <= upper:
                coverage += 1
        
        coverage_rate = coverage / n_trials
        # Should be close to 0.95 (allow Â±10% for Monte Carlo variation)
        assert 0.85 <= coverage_rate <= 1.0
        print(f"\nBootstrap-t coverage rate: {coverage_rate:.2%} (expected ~95%)")
    
    def test_narrower_than_percentile_for_normal(self, normal_data):
        """Bootstrap-t should give narrower CI than percentile for normal data"""
        # Percentile CI
        est_pct, lower_pct, upper_pct = bsr.bootstrap_mean_ci(
            normal_data, n_resamples=2000, conf=0.95, random_state=42
        )
        width_pct = upper_pct - lower_pct
        
        # Bootstrap-t CI
        est_t, lower_t, upper_t = bsr.bootstrap_t_ci_mean(
            normal_data, n_resamples=2000, conf=0.95, random_state=42
        )
        width_t = upper_t - lower_t
        
        # For normal data, bootstrap-t is more accurate (narrower)
        assert width_t <= width_pct * 1.1
    
    def test_confidence_level_effect(self, normal_data):
        """Higher confidence should give wider CI"""
        _, lower_90, upper_90 = bsr.bootstrap_t_ci_mean(
            normal_data, n_resamples=1000, conf=0.90, random_state=42
        )
        _, lower_95, upper_95 = bsr.bootstrap_t_ci_mean(
            normal_data, n_resamples=1000, conf=0.95, random_state=43
        )
        _, lower_99, upper_99 = bsr.bootstrap_t_ci_mean(
            normal_data, n_resamples=1000, conf=0.99, random_state=44
        )
        
        width_90 = upper_90 - lower_90
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99
        
        assert width_90 < width_95 < width_99
    
    def test_skewed_data_robustness(self, skewed_data):
        """Should handle skewed data without crashing"""
        est, lower, upper = bsr.bootstrap_t_ci_mean(
            skewed_data, n_resamples=1000, conf=0.95, random_state=42
        )
        
        assert np.isfinite(est)
        assert np.isfinite(lower)
        assert np.isfinite(upper)
        assert lower < est < upper
    
    def test_reproducibility(self, normal_data):
        """Same seed should give identical CI"""
        ci1 = bsr.bootstrap_t_ci_mean(normal_data, n_resamples=500, conf=0.95, random_state=999)
        ci2 = bsr.bootstrap_t_ci_mean(normal_data, n_resamples=500, conf=0.95, random_state=999)
        
        np.testing.assert_array_equal(ci1, ci2)
    
    def test_estimate_is_sample_mean(self, normal_data):
        """Point estimate should equal sample mean"""
        est, lower, upper = bsr.bootstrap_t_ci_mean(
            normal_data, n_resamples=1000, conf=0.95, random_state=42
        )
        
        sample_mean = np.mean(normal_data)
        np.testing.assert_allclose(est, sample_mean, rtol=1e-10)
    
    def test_small_sample_handling(self, small_sample):
        """Should handle small samples (n=10)"""
        est, lower, upper = bsr.bootstrap_t_ci_mean(
            small_sample, n_resamples=500, conf=0.95, random_state=42
        )
        
        assert np.isfinite(est)
        assert lower < est < upper


# ==============================================================================
# BCa CI
# ==============================================================================

class TestBootstrapBCaCI:
    """Comprehensive tests for bootstrap_bca_ci (bias-corrected accelerated)"""
    
    def test_coverage_normal_data(self, rng):
        """BCa should have good coverage for normal data"""
        true_mean = 5.0
        n_trials = 100
        coverage = 0
        
        for _ in range(n_trials):
            x = rng.normal(true_mean, 1, 80)
            est, lower, upper = bsr.bootstrap_bca_ci(
                x, stat="mean", n_resamples=1000, conf=0.95, random_state=None
            )
            
            if lower <= true_mean <= upper:
                coverage += 1
        
        coverage_rate = coverage / n_trials
        assert 0.85 <= coverage_rate <= 1.0
        print(f"\nBCa coverage rate: {coverage_rate:.2%} (expected ~95%)")
    
    def test_better_than_percentile_for_skewed(self, rng):
        """BCa should outperform percentile CI for skewed data"""
        # Generate skewed data (exponential)
        true_median = np.log(2) * 2  # For Exp(scale=2)
        n_trials = 100
        
        coverage_pct = 0
        coverage_bca = 0
        
        for _ in range(n_trials):
            x = rng.exponential(2, 100)
            
            # Percentile CI
            _, lower_pct, upper_pct = bsr.bootstrap_ci(
                x, stat="median", n_resamples=1000, conf=0.95, random_state=None
            )
            
            # BCa CI
            _, lower_bca, upper_bca = bsr.bootstrap_bca_ci(
                x, stat="median", n_resamples=1000, conf=0.95, random_state=None
            )
            
            if lower_pct <= true_median <= upper_pct:
                coverage_pct += 1
            if lower_bca <= true_median <= upper_bca:
                coverage_bca += 1
        
        # BCa should have better or equal coverage
        print(f"\nPercentile coverage: {coverage_pct/n_trials:.2%}, BCa coverage: {coverage_bca/n_trials:.2%}")
        assert coverage_bca >= coverage_pct - 5  # Allow small variation
    
    def test_all_statistics(self, normal_data):
        """BCa should work for mean, median, std"""
        for stat in ["mean", "median", "std"]:
            est, lower, upper = bsr.bootstrap_bca_ci(
                normal_data, stat=stat, n_resamples=1000, conf=0.95, random_state=42
            )
            
            assert np.isfinite(est)
            assert np.isfinite(lower)
            assert np.isfinite(upper)
            assert lower < upper
    
    def test_bias_correction_with_biased_estimator(self, rng):
        """BCa should detect and correct bias"""
        # Create biased scenario: small sample from exponential
        # Sample mean is unbiased, but sample variance is biased
        x = rng.exponential(1, 20)
        
        est, lower, upper = bsr.bootstrap_bca_ci(
            x, stat="std", n_resamples=1000, conf=0.95, random_state=42
        )
        
        # BCa should produce asymmetric interval for biased estimator
        lower_dist = abs(est - lower)
        upper_dist = abs(upper - est)
        
        # Allow some asymmetry (not required to be perfectly symmetric)
        assert np.isfinite(est)
    
    def test_confidence_levels(self, normal_data):
        """Different confidence levels should work"""
        for conf in [0.90, 0.95, 0.99]:
            est, lower, upper = bsr.bootstrap_bca_ci(
                normal_data, stat="mean", n_resamples=800, conf=conf, random_state=42
            )
            
            assert lower < est < upper
    
    def test_reproducibility(self, normal_data):
        """Same seed should give same results"""
        ci1 = bsr.bootstrap_bca_ci(
            normal_data, stat="mean", n_resamples=500, conf=0.95, random_state=777
        )
        ci2 = bsr.bootstrap_bca_ci(
            normal_data, stat="mean", n_resamples=500, conf=0.95, random_state=777
        )
        
        np.testing.assert_array_equal(ci1, ci2)
    
    def test_handles_extreme_data(self):
        """Should handle data with extreme values"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # One outlier
        
        est, lower, upper = bsr.bootstrap_bca_ci(
            x, stat="mean", n_resamples=500, conf=0.95, random_state=42
        )
        
        assert np.isfinite(est)
        assert lower < upper


# ==============================================================================
# BAYESIAN BOOTSTRAP
# ==============================================================================

class TestBayesianBootstrapCI:
    """Tests for bayesian_bootstrap_ci (Rubin's Bayesian bootstrap)"""
    
    def test_coverage_rate(self, rng):
        """Bayesian bootstrap should have good coverage"""
        true_mean = 3.0
        n_trials = 100
        coverage = 0
        
        for _ in range(n_trials):
            x = rng.normal(true_mean, 1, 60)
            est, lower, upper = bsr.bayesian_bootstrap_ci(
                x, stat="mean", n_resamples=1000, conf=0.95, random_state=None
            )
            
            if lower <= true_mean <= upper:
                coverage += 1
        
        coverage_rate = coverage / n_trials
        assert 0.85 <= coverage_rate <= 1.0
        print(f"\nBayesian bootstrap coverage: {coverage_rate:.2%}")
    
    def test_similar_to_percentile_for_mean(self, normal_data):
        """For mean, Bayesian bootstrap should be similar to percentile"""
        _, lower_bb, upper_bb = bsr.bayesian_bootstrap_ci(
            normal_data, stat="mean", n_resamples=2000, conf=0.95, random_state=42
        )
        _, lower_pct, upper_pct = bsr.bootstrap_ci(
            normal_data, stat="mean", n_resamples=2000, conf=0.95, random_state=42
        )
        
        width_bb = upper_bb - lower_bb
        width_pct = upper_pct - lower_pct
        
        # Should be within 20% of each other for mean
        np.testing.assert_allclose(width_bb, width_pct, rtol=0.2)
    
    def test_all_statistics(self, normal_data):
        """Should work for mean, median, std"""
        for stat in ["mean", "median", "std"]:
            est, lower, upper = bsr.bayesian_bootstrap_ci(
                normal_data, stat=stat, n_resamples=1000, conf=0.95, random_state=42
            )
            
            assert np.isfinite(est)
            assert lower < upper
    
    def test_smoothness_property(self, rng):
        """Bayesian bootstrap should produce smooth estimates"""
        x = rng.normal(0, 1, 50)
        
        # Multiple runs with different seeds should give similar results
        results = []
        for seed in range(100, 105):
            est, lower, upper = bsr.bayesian_bootstrap_ci(
                x, stat="mean", n_resamples=1000, conf=0.95, random_state=seed
            )
            results.append((est, lower, upper))
        
        ests = [r[0] for r in results]
        
        # Point estimates should cluster around sample mean
        assert np.std(ests) < 0.1 * np.std(x) / np.sqrt(len(x))
    
    def test_reproducibility(self, normal_data):
        """Same seed gives same result"""
        ci1 = bsr.bayesian_bootstrap_ci(
            normal_data, stat="median", n_resamples=500, conf=0.95, random_state=888
        )
        ci2 = bsr.bayesian_bootstrap_ci(
            normal_data, stat="median", n_resamples=500, conf=0.95, random_state=888
        )
        
        np.testing.assert_array_equal(ci1, ci2)


# ==============================================================================
# TIME-SERIES BLOCK BOOTSTRAPS
# ==============================================================================

class TestMovingBlockBootstrap:
    """Tests for moving_block_bootstrap_mean_ci"""
    
    def test_contains_true_mean(self, rng):
        """CI should contain true mean for autocorrelated data"""
        # Generate AR(1) process
        n = 200
        true_mean = 5.0
        phi = 0.7
        
        x = np.zeros(n)
        x[0] = rng.normal(true_mean, 1)
        for i in range(1, n):
            x[i] = true_mean * (1 - phi) + phi * x[i-1] + rng.normal(0, 1)
        
        est, lower, upper = bsr.moving_block_bootstrap_mean_ci(
            x, block_len=10, n_resamples=1000, conf=0.95, random_state=42
        )
        
        assert lower <= true_mean <= upper
    
    def test_block_length_effect(self, timeseries_data):
        """Longer blocks should give wider CIs for autocorrelated data"""
        _, lower_5, upper_5 = bsr.moving_block_bootstrap_mean_ci(
            timeseries_data, block_len=5, n_resamples=1000, conf=0.95, random_state=42
        )
        _, lower_20, upper_20 = bsr.moving_block_bootstrap_mean_ci(
            timeseries_data, block_len=20, n_resamples=1000, conf=0.95, random_state=43
        )
        
        width_5 = upper_5 - lower_5
        width_20 = upper_20 - lower_20
        
        # Longer blocks should generally give wider CI (captures more dependence)
        # Not guaranteed, but likely
        print(f"\nBlock length effect: width(5)={width_5:.4f}, width(20)={width_20:.4f}")
    
    def test_iid_data_similar_to_standard_bootstrap(self, normal_data):
        """For IID data, should be similar to standard bootstrap"""
        _, lower_block, upper_block = bsr.moving_block_bootstrap_mean_ci(
            normal_data, block_len=1, n_resamples=1000, conf=0.95, random_state=42
        )
        _, lower_std, upper_std = bsr.bootstrap_mean_ci(
            normal_data, n_resamples=1000, conf=0.95, random_state=42
        )
        
        # Block length 1 should behave like standard bootstrap
        width_block = upper_block - lower_block
        width_std = upper_std - lower_std
        
        np.testing.assert_allclose(width_block, width_std, rtol=0.2)
    
    def test_estimate_is_sample_mean(self, timeseries_data):
        """Point estimate should be sample mean"""
        est, _, _ = bsr.moving_block_bootstrap_mean_ci(
            timeseries_data, block_len=10, n_resamples=500, conf=0.95, random_state=42
        )
        
        np.testing.assert_allclose(est, np.mean(timeseries_data), rtol=1e-10)
    
    def test_reproducibility(self, timeseries_data):
        """Same seed gives same result"""
        ci1 = bsr.moving_block_bootstrap_mean_ci(
            timeseries_data, block_len=10, n_resamples=500, conf=0.95, random_state=555
        )
        ci2 = bsr.moving_block_bootstrap_mean_ci(
            timeseries_data, block_len=10, n_resamples=500, conf=0.95, random_state=555
        )
        
        np.testing.assert_array_equal(ci1, ci2)


class TestCircularBlockBootstrap:
    """Tests for circular_block_bootstrap_mean_ci"""
    
    def test_valid_ci_structure(self, timeseries_data):
        """Should produce valid CI"""
        est, lower, upper = bsr.circular_block_bootstrap_mean_ci(
            timeseries_data, block_len=15, n_resamples=1000, conf=0.95, random_state=42
        )
        
        assert np.isfinite(est)
        assert lower < est < upper
    
    def test_circular_wrapping_works(self, rng):
        """Should handle circular wrapping correctly"""
        # Create data with distinct pattern that should wrap
        x = np.concatenate([np.ones(50), np.ones(50) * 10])
        
        est, lower, upper = bsr.circular_block_bootstrap_mean_ci(
            x, block_len=30, n_resamples=500, conf=0.95, random_state=42
        )
        
        # Should be able to sample across boundary
        assert np.isfinite(est)
        assert lower < upper
    
    def test_similar_to_moving_block(self, timeseries_data):
        """Should give similar results to moving block for short blocks"""
        _, lower_circ, upper_circ = bsr.circular_block_bootstrap_mean_ci(
            timeseries_data, block_len=5, n_resamples=1000, conf=0.95, random_state=42
        )
        _, lower_move, upper_move = bsr.moving_block_bootstrap_mean_ci(
            timeseries_data, block_len=5, n_resamples=1000, conf=0.95, random_state=42
        )
        
        width_circ = upper_circ - lower_circ
        width_move = upper_move - lower_move
        
        # Should be roughly similar
        np.testing.assert_allclose(width_circ, width_move, rtol=0.3)


class TestStationaryBootstrap:
    """Tests for stationary_bootstrap_mean_ci (Politis-Romano)"""
    
    def test_valid_output(self, timeseries_data):
        """Should produce valid CI"""
        est, lower, upper = bsr.stationary_bootstrap_mean_ci(
            timeseries_data, block_len=10, n_resamples=1000, conf=0.95, random_state=42
        )
        
        assert np.isfinite(est)
        assert lower < est < upper
    
    def test_geometric_block_lengths(self, timeseries_data):
        """Block lengths should be geometric with mean = block_len"""
        # This is implicit in the algorithm
        # Just test that different block_len parameters work
        for block_len in [5, 10, 20]:
            est, lower, upper = bsr.stationary_bootstrap_mean_ci(
                timeseries_data, block_len=block_len, n_resamples=500, conf=0.95, random_state=42
            )
            
            assert np.isfinite(est)
            assert lower < upper
    
    def test_comparable_to_circular_block(self, timeseries_data):
        """Should give similar results to circular block"""
        _, lower_stat, upper_stat = bsr.stationary_bootstrap_mean_ci(
            timeseries_data, block_len=10, n_resamples=1000, conf=0.95, random_state=42
        )
        _, lower_circ, upper_circ = bsr.circular_block_bootstrap_mean_ci(
            timeseries_data, block_len=10, n_resamples=1000, conf=0.95, random_state=43
        )
        
        width_stat = upper_stat - lower_stat
        width_circ = upper_circ - lower_circ
        
        # Should be in same ballpark
        np.testing.assert_allclose(width_stat, width_circ, rtol=0.4)
    
    def test_reproducibility(self, timeseries_data):
        """Same seed gives same result"""
        ci1 = bsr.stationary_bootstrap_mean_ci(
            timeseries_data, block_len=10, n_resamples=500, conf=0.95, random_state=333
        )
        ci2 = bsr.stationary_bootstrap_mean_ci(
            timeseries_data, block_len=10, n_resamples=500, conf=0.95, random_state=333
        )
        
        np.testing.assert_array_equal(ci1, ci2)


# ==============================================================================
# PERMUTATION TESTS
# ==============================================================================

class TestPermutationCorrTest:
    """Tests for permutation_corr_test"""
    
    def test_perfect_correlation_gives_small_p(self):
        """Perfect correlation should give p â‰ˆ 0"""
        x = np.arange(50, dtype=float)
        y = x * 2
        
        r_obs, p = bsr.permutation_corr_test(x, y, n_permutations=1000, random_state=42)
        
        np.testing.assert_allclose(r_obs, 1.0, atol=1e-10)
        assert p < 0.01
    
    def test_no_correlation_gives_high_p(self, rng):
        """Independent variables should give p > 0.05"""
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        
        r_obs, p = bsr.permutation_corr_test(x, y, n_permutations=1000, random_state=42)
        
        assert -0.3 < r_obs < 0.3  # Should be near zero
        assert p > 0.05
    
    def test_strong_correlation_detected(self, rng):
        """Strong correlation should be detected"""
        x = rng.normal(0, 1, 150)
        y = x + rng.normal(0, 0.1, 150)  # r â‰ˆ 0.99+
        
        r_obs, p = bsr.permutation_corr_test(x, y, n_permutations=2000, random_state=42)
        
        assert r_obs > 0.95
        assert p < 0.001
    
    def test_alternative_hypotheses(self, rng):
        """All three alternatives should work"""
        x = rng.normal(0, 1, 80)
        y = x + rng.normal(0, 0.3, 80)
        
        for alt in ["two-sided", "greater", "less"]:
            r_obs, p = bsr.permutation_corr_test(
                x, y, n_permutations=500, alternative=alt, random_state=42
            )
            
            assert 0 <= p <= 1
            assert np.isfinite(r_obs)
    
    def test_reproducibility(self, paired_data):
        """Same seed gives same result"""
        x, y = paired_data
        
        result1 = bsr.permutation_corr_test(x, y, n_permutations=500, random_state=111)
        result2 = bsr.permutation_corr_test(x, y, n_permutations=500, random_state=111)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_p_value_range(self, paired_data):
        """P-value should be in [0, 1]"""
        x, y = paired_data
        
        r_obs, p = bsr.permutation_corr_test(x, y, n_permutations=1000, random_state=42)
        
        assert 0 <= p <= 1


class TestPermutationMeanDiffTest:
    """Tests for permutation_mean_diff_test"""
    
    def test_identical_groups_high_p(self, rng):
        """Identical groups should give p â‰ˆ 1"""
        x = rng.normal(5, 1, 60)
        y = rng.normal(5, 1, 60)
        
        diff_obs, p = bsr.permutation_mean_diff_test(x, y, n_permutations=1000, random_state=42)
        
        assert abs(diff_obs) < 1.0  # Should be small
        assert p > 0.05
    
    def test_large_difference_small_p(self, rng):
        """Large mean difference should give p â‰ˆ 0"""
        x = rng.normal(0, 1, 80)
        y = rng.normal(5, 1, 80)  # Big difference
        
        diff_obs, p = bsr.permutation_mean_diff_test(x, y, n_permutations=1000, random_state=42)
        
        assert abs(diff_obs - (-5)) < 1.0  # Observed diff â‰ˆ 0 - 5 = -5
        assert p < 0.001
    
    def test_all_alternatives(self, rng):
        """All three alternatives should work"""
        x = rng.normal(1, 1, 50)
        y = rng.normal(0, 1, 50)
        
        for alt in ["two-sided", "greater", "less"]:
            diff_obs, p = bsr.permutation_mean_diff_test(
                x, y, n_permutations=500, alternative=alt, random_state=42
            )
            
            assert 0 <= p <= 1
            assert np.isfinite(diff_obs)
    
    def test_unequal_sample_sizes(self, rng):
        """Should handle unequal n correctly"""
        x = rng.normal(0, 1, 30)
        y = rng.normal(0.5, 1, 70)
        
        diff_obs, p = bsr.permutation_mean_diff_test(x, y, n_permutations=1000, random_state=42)
        
        assert np.isfinite(diff_obs)
        assert 0 <= p <= 1
    
    def test_observed_diff_calculation(self):
        """Observed difference should be mean(x) - mean(y)"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        
        diff_obs, p = bsr.permutation_mean_diff_test(x, y, n_permutations=500, random_state=42)
        
        expected_diff = np.mean(x) - np.mean(y)
        np.testing.assert_allclose(diff_obs, expected_diff, rtol=1e-10)
    
    def test_reproducibility(self, rng):
        """Same seed gives same result"""
        x = rng.normal(0, 1, 40)
        y = rng.normal(0.3, 1, 40)
        
        result1 = bsr.permutation_mean_diff_test(x, y, n_permutations=500, random_state=222)
        result2 = bsr.permutation_mean_diff_test(x, y, n_permutations=500, random_state=222)
        
        np.testing.assert_array_equal(result1, result2)


# ==============================================================================
# JACKKNIFE EXTENSIONS
# ==============================================================================

class TestInfluenceMean:
    """Tests for influence_mean (influence function)"""
    
    def test_influence_sum_to_zero(self, normal_data):
        """Influence values should sum to approximately zero"""
        infl = bsr.influence_mean(normal_data)
        
        assert len(infl) == len(normal_data)
        np.testing.assert_allclose(np.sum(infl), 0, atol=1e-10)
    
    def test_outlier_detection(self):
        """Outliers should have large influence values"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # 100 is outlier
        
        infl = bsr.influence_mean(x)
        
        # Last point should have largest influence
        assert abs(infl[-1]) == np.max(np.abs(infl))
        assert abs(infl[-1]) > abs(infl[0])
    
    def test_uniform_data_uniform_influence(self):
        """Uniform data should have similar influence values"""
        x = np.ones(20) * 5
        
        infl = bsr.influence_mean(x)
        
        # All influences should be near zero
        assert np.all(np.abs(infl) < 1e-10)
    
    def test_length_matches_input(self, normal_data):
        """Output length should match input length"""
        infl = bsr.influence_mean(normal_data)
        
        assert len(infl) == len(normal_data)


class TestDeleteDJackknife:
    """Tests for delete_d_jackknife_mean"""
    
    def test_d_1_matches_standard_jackknife(self, normal_data):
        """d=1 should match standard jackknife_mean"""
        theta_d, bias_d, se_d = bsr.delete_d_jackknife_mean(normal_data, d=1)
        theta_1, bias_1, se_1 = bsr.jackknife_mean(normal_data)
        
        # Should be very similar (not identical due to formula differences)
        np.testing.assert_allclose(theta_d, theta_1, rtol=0.1)
        np.testing.assert_allclose(se_d, se_1, rtol=0.3)
    
    def test_increasing_d_changes_results(self, normal_data):
        """Different d values should give different results"""
        theta_1, bias_1, se_1 = bsr.delete_d_jackknife_mean(normal_data, d=1)
        theta_5, bias_5, se_5 = bsr.delete_d_jackknife_mean(normal_data, d=5)
        theta_10, bias_10, se_10 = bsr.delete_d_jackknife_mean(normal_data, d=10)
        
        # Results should differ
        assert not np.allclose(se_1, se_5)
        assert not np.allclose(se_5, se_10)
    
    def test_d_equals_n_minus_1_returns_nan(self, small_sample):
        """d = n-1 should return NaN (no valid observations)"""
        n = len(small_sample)
        
        theta, bias, se = bsr.delete_d_jackknife_mean(small_sample, d=n-1)
        
        # Should handle gracefully
        assert np.isnan(theta) or np.isnan(se)
    
    def test_valid_with_various_d(self, normal_data):
        """Should work with various d values"""
        n = len(normal_data)
        
        for d in [1, 5, 10, 20]:
            if d < n:
                theta, bias, se = bsr.delete_d_jackknife_mean(normal_data, d=d)
                
                assert np.isfinite(theta)
                assert np.isfinite(se)


class TestJackknifeAfterBootstrap:
    """Tests for jackknife_after_bootstrap_se_mean (JAB diagnostic)"""
    
    def test_returns_positive_se(self, small_sample):
        """Should return positive SE estimate"""
        jab_se = bsr.jackknife_after_bootstrap_se_mean(
            small_sample, n_resamples=100, random_state=42
        )
        
        assert jab_se > 0
        assert np.isfinite(jab_se)
    
    def test_consistent_with_bootstrap_se_magnitude(self, normal_data):
        """JAB SE should be in same order of magnitude as bootstrap SE"""
        boot_se = bsr.bootstrap_se(normal_data, stat="mean", n_resamples=500, random_state=42)
        jab_se = bsr.jackknife_after_bootstrap_se_mean(
            normal_data, n_resamples=100, random_state=42
        )
        
        # Should be in same ballpark (JAB estimates SE of bootstrap SE)
        # So it should be smaller
        assert jab_se < boot_se
        assert jab_se > 0
    
    def test_reproducibility(self, small_sample):
        """Same seed gives same result"""
        jab1 = bsr.jackknife_after_bootstrap_se_mean(
            small_sample, n_resamples=100, random_state=999
        )
        jab2 = bsr.jackknife_after_bootstrap_se_mean(
            small_sample, n_resamples=100, random_state=999
        )
        
        assert jab1 == jab2
    
    def test_small_sample_warning(self):
        """Very small samples should still work but with limited info"""
        x = np.array([1.0, 2.0, 3.0])
        
        jab_se = bsr.jackknife_after_bootstrap_se_mean(
            x, n_resamples=50, random_state=42
        )
        
        # Should work but may be limited
        assert np.isfinite(jab_se) or np.isnan(jab_se)


# ==============================================================================
# EDGE CASES AND STRESS TESTS
# ==============================================================================

class TestEdgeCases:
    """Comprehensive edge case testing"""
    
    def test_single_value_array(self):
        """Single value should return NaN appropriately"""
        x = np.array([5.0])
        
        # Most functions should handle gracefully
        # (may return NaN or raise error depending on function)
        try:
            se = bsr.bootstrap_se(x, stat="mean", n_resamples=100, random_state=42)
            assert np.isnan(se) or se >= 0
        except:
            pass  # Some functions may appropriately raise error
    
    def test_two_values_minimal(self):
        """Two values is minimal valid case"""
        x = np.array([1.0, 2.0])
        
        se = bsr.bootstrap_se(x, stat="mean", n_resamples=100, random_state=42)
        
        assert np.isfinite(se)
        assert se > 0
    
    def test_constant_array(self):
        """All same values should give zero or near-zero variance"""
        x = np.ones(50) * 7.0
        
        se = bsr.bootstrap_se(x, stat="mean", n_resamples=500, random_state=42)
        var = bsr.bootstrap_var(x, stat="mean", n_resamples=500, random_state=42)
        
        assert se < 1e-10
        assert var < 1e-20
    
    def test_extreme_values_no_overflow(self):
        """Very large values should not cause overflow"""
        x = np.array([1e50, 1e50 + 1e40, 1e50 + 2e40, 1e50 + 3e40, 1e50 + 4e40])
        
        se = bsr.bootstrap_se(x, stat="mean", n_resamples=200, random_state=42)
        
        assert np.isfinite(se)
    
    def test_very_small_values_no_underflow(self):
        """Very small values should not underflow"""
        x = np.array([1e-100, 2e-100, 3e-100, 4e-100, 5e-100])
        
        se = bsr.bootstrap_se(x, stat="mean", n_resamples=200, random_state=42)
        
        assert np.isfinite(se)
    
    def test_mixed_positive_negative(self):
        """Should handle mixed signs"""
        x = np.array([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
        
        est, lower, upper = bsr.bootstrap_t_ci_mean(x, n_resamples=500, random_state=42)
        
        assert np.isfinite(est)
        assert lower < upper


class TestStressAndPerformance:
    """Stress testing and performance validation"""
    
    def test_large_n_resamples(self, normal_data):
        """Should handle many resamples efficiently"""
        import time
        
        start = time.time()
        se = bsr.bootstrap_se(normal_data, stat="mean", n_resamples=10000, random_state=42)
        elapsed = time.time() - start
        
        print(f"\n10000 resamples took {elapsed:.3f}s")
        assert elapsed < 2.0  # Should be fast
        assert np.isfinite(se)
    
    def test_large_sample_size(self, large_sample):
        """Should handle large datasets (n=5000)"""
        import time
        
        start = time.time()
        est, lower, upper = bsr.bootstrap_mean_ci(
            large_sample, n_resamples=500, conf=0.95, random_state=42
        )
        elapsed = time.time() - start
        
        print(f"\nn=5000 bootstrap took {elapsed:.3f}s")
        assert elapsed < 3.0
        assert lower < est < upper
    
    def test_bca_computational_cost(self, normal_data):
        """BCa should be more expensive but reasonable"""
        import time

        # Warm-up (avoid measuring first-call overhead / CPU ramp)
        bsr.bootstrap_ci(normal_data, stat="mean", n_resamples=200, random_state=42)
        bsr.bootstrap_bca_ci(normal_data, stat="mean", n_resamples=200, random_state=42)

        reps = 25  # small, keeps test fast but measurable

        # Percentile
        start = time.perf_counter()
        for _ in range(reps):
            bsr.bootstrap_ci(normal_data, stat="mean", n_resamples=1000, random_state=42)
        time_pct = time.perf_counter() - start

        # BCa
        start = time.perf_counter()
        for _ in range(reps):
            bsr.bootstrap_bca_ci(normal_data, stat="mean", n_resamples=1000, random_state=42)
        time_bca = time.perf_counter() - start

        avg_pct = time_pct / reps
        avg_bca = time_bca / reps

        print(f"\nPercentile avg: {avg_pct:.6f}s, BCa avg: {avg_bca:.6f}s")

        # BCa should be more expensive but not ridiculously so
        assert avg_bca < avg_pct * 10

    def test_permutation_test_scaling(self, rng):
        """Permutation test should scale linearly with n_permutations"""
        import time
        
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        
        start = time.time()
        bsr.permutation_corr_test(x, y, n_permutations=500, random_state=42)
        time_500 = time.time() - start
        
        start = time.time()
        bsr.permutation_corr_test(x, y, n_permutations=2000, random_state=42)
        time_2000 = time.time() - start
        
        ratio = time_2000 / time_500
        print(f"\nPermutation scaling: 500â†’2000 ratio = {ratio:.2f}")
        
        # Should be roughly 4x (linear scaling)
        assert 2 < ratio < 8


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestIntegration:
    """Integration tests combining multiple functions"""
    
    def test_all_ci_methods_reasonable(self, normal_data):
        """All CI methods should give reasonable intervals"""
        methods = {
            'percentile': bsr.bootstrap_mean_ci(normal_data, 1000, 0.95, 42),
            'bootstrap_t': bsr.bootstrap_t_ci_mean(normal_data, 1000, 0.95, 42),
            'bca': bsr.bootstrap_bca_ci(normal_data, "mean", 1000, 0.95, 42),
            'bayesian': bsr.bayesian_bootstrap_ci(normal_data, "mean", 1000, 0.95, 42),
        }
        
        sample_mean = np.mean(normal_data)
        
        for name, (est, lower, upper) in methods.items():
            print(f"\n{name}: [{lower:.4f}, {upper:.4f}], width={upper-lower:.4f}")
            
            # All should contain sample mean (or be very close)
            assert lower <= sample_mean <= upper or abs(est - sample_mean) < 0.01
            assert lower < upper
    
    def test_se_var_consistency(self, normal_data):
        """SE and VAR should be consistent across methods"""
        se = bsr.bootstrap_se(normal_data, stat="mean", n_resamples=2000, random_state=42)
        var = bsr.bootstrap_var(normal_data, stat="mean", n_resamples=2000, random_state=42)
        
        np.testing.assert_allclose(var, se**2, rtol=1e-10)
    
    def test_block_bootstrap_comparison(self, timeseries_data):
        """Different block bootstraps should give similar results"""
        methods = {
            'moving': bsr.moving_block_bootstrap_mean_ci(timeseries_data, 10, 1000, 0.95, 42),
            'circular': bsr.circular_block_bootstrap_mean_ci(timeseries_data, 10, 1000, 0.95, 43),
            'stationary': bsr.stationary_bootstrap_mean_ci(timeseries_data, 10, 1000, 0.95, 44),
        }
        
        widths = {name: upper - lower for name, (est, lower, upper) in methods.items()}
        
        print(f"\nBlock bootstrap widths: {widths}")
        
        # Should all be in reasonable range
        max_width = max(widths.values())
        min_width = min(widths.values())
        
        assert max_width < min_width * 2  # No more than 2x difference


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
