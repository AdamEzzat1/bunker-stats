# tests/test_resampling_config.py
"""
Test suite for bunker_stats.resampling config layer.

Validates:
1. Input validation catches errors with helpful messages
2. Config objects produce identical results to flat Rust functions
3. NaN policy works correctly
4. Reproducibility with random_state
5. Error messages are human-readable
"""

import numpy as np
import pytest
from bunker_stats.resampling import (
    BootstrapConfig,
    BootstrapCorrConfig,
    PermutationConfig,
    JackknifeConfig,
    bootstrap,
    bootstrap_corr,
    permutation_test,
    jackknife,
)
import bunker_stats as bsr


# ======================================================================================
# 1. INPUT VALIDATION TESTS
# ======================================================================================

def test_bootstrap_config_validates_n_resamples():
    """Ensure n_resamples >= 1"""
    with pytest.raises(ValueError, match="n_resamples must be >= 1"):
        BootstrapConfig(n_resamples=0)
    
    with pytest.raises(ValueError, match="n_resamples must be >= 1"):
        BootstrapConfig(n_resamples=-10)


def test_bootstrap_config_validates_conf():
    """Ensure conf in (0, 1)"""
    with pytest.raises(ValueError, match="conf must be in \\(0, 1\\)"):
        BootstrapConfig(conf=0.0)
    
    with pytest.raises(ValueError, match="conf must be in \\(0, 1\\)"):
        BootstrapConfig(conf=1.0)
    
    with pytest.raises(ValueError, match="conf must be in \\(0, 1\\)"):
        BootstrapConfig(conf=1.2)


def test_bootstrap_config_validates_stat():
    """Ensure stat is supported"""
    with pytest.raises(ValueError, match="stat must be one of"):
        BootstrapConfig(stat="variance")  # not yet supported


def test_permutation_config_validates_alternative():
    """Ensure alternative is valid"""
    with pytest.raises(ValueError, match="alternative must be one of"):
        PermutationConfig(alternative="not-equal")


def test_empty_array_rejected():
    """Empty arrays should raise helpful error"""
    config = BootstrapConfig()
    
    with pytest.raises(ValueError, match="array is empty"):
        config.run(np.array([]))


def test_multidimensional_array_rejected():
    """2D arrays should raise helpful error"""
    config = BootstrapConfig()
    data_2d = np.random.randn(10, 3)
    
    with pytest.raises(ValueError, match="expected 1D array"):
        config.run(data_2d)


def test_length_mismatch_corr():
    """Paired arrays must have same length"""
    config = BootstrapCorrConfig()
    x = np.random.randn(10)
    y = np.random.randn(15)
    
    with pytest.raises(ValueError, match="must have same length"):
        config.run(x, y)


# ======================================================================================
# 2. EQUIVALENCE TESTS (Config objects match flat Rust functions)
# ======================================================================================

def test_bootstrap_config_matches_rust_function():
    """Config object should produce identical results to direct Rust call"""
    np.random.seed(42)
    data = np.random.randn(100)
    
    # Using config object
    config = BootstrapConfig(n_resamples=1000, conf=0.95, stat="mean", random_state=42)
    result_config = config.run(data)
    
    # Using flat Rust function
    result_rust = bsr.bootstrap_ci(data, stat="mean", n_resamples=1000, conf=0.95, random_state=42)
    
    # Should be identical
    np.testing.assert_array_equal(result_config, result_rust)


def test_bootstrap_corr_config_matches_rust():
    """Bootstrap correlation config should match Rust function"""
    np.random.seed(42)
    x = np.random.randn(100)
    y = x + np.random.randn(100) * 0.5
    
    config = BootstrapCorrConfig(n_resamples=1000, conf=0.95, random_state=42)
    result_config = config.run(x, y)
    
    result_rust = bsr.bootstrap_corr(x, y, n_resamples=1000, conf=0.95, random_state=42)
    
    np.testing.assert_array_equal(result_config, result_rust)


def test_permutation_config_matches_rust_corr():
    """Permutation test config should match Rust function for correlation"""
    np.random.seed(42)
    x = np.random.randn(100)
    y = np.random.randn(100)
    
    config = PermutationConfig(n_permutations=1000, alternative="two-sided", random_state=42)
    result_config = config.run_corr(x, y)
    
    result_rust = bsr.permutation_test_corr(x, y, n_permutations=1000, alternative="two-sided", random_state=42)
    
    np.testing.assert_array_equal(result_config, result_rust)


def test_permutation_config_matches_rust_mean_diff():
    """Permutation test config should match Rust function for mean diff"""
    np.random.seed(42)
    x = np.random.randn(50) + 0.5
    y = np.random.randn(50)
    
    config = PermutationConfig(n_permutations=1000, alternative="greater", random_state=42)
    result_config = config.run_mean_diff(x, y)
    
    result_rust = bsr.permutation_mean_diff_test(x, y, n_permutations=1000, alternative="greater", random_state=42)
    
    np.testing.assert_array_equal(result_config, result_rust)


def test_jackknife_config_matches_rust():
    """Jackknife config should match Rust function"""
    np.random.seed(42)
    data = np.random.randn(100)
    
    config = JackknifeConfig(conf=0.95)
    result_config = config.run_mean_ci(data)
    
    result_rust = bsr.jackknife_mean_ci(data, conf=0.95)
    
    np.testing.assert_array_equal(result_config, result_rust)


# ======================================================================================
# 3. NaN POLICY TESTS
# ======================================================================================

def test_nan_policy_propagate_default():
    """Default nan_policy='propagate' should pass NaNs through to Rust"""
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    
    config = BootstrapConfig(n_resamples=100, random_state=42)
    result = config.run(data)
    
    # Result should contain NaNs (Rust propagates them)
    assert np.isnan(result[0]) or np.isnan(result[1]) or np.isnan(result[2])


def test_nan_policy_omit_single_array():
    """nan_policy='omit' should filter NaNs and produce valid results"""
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    
    config = BootstrapConfig(n_resamples=100, random_state=42, nan_policy="omit")
    
    # Should warn about NaN removal
    with pytest.warns(UserWarning, match="removed 1 NaN"):
        result = config.run(data)
    
    # Result should be finite (NaNs filtered out)
    assert np.all(np.isfinite(result))
    
    # Should match result on clean data
    data_clean = np.array([1.0, 2.0, 4.0, 5.0])
    config_clean = BootstrapConfig(n_resamples=100, random_state=42, nan_policy="propagate")
    result_clean = config_clean.run(data_clean)
    
    np.testing.assert_array_almost_equal(result, result_clean, decimal=10)


def test_nan_policy_omit_paired_arrays():
    """nan_policy='omit' should do pairwise deletion for correlation"""
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([2.0, np.nan, 6.0, 8.0, 10.0])
    
    config = BootstrapCorrConfig(n_resamples=100, random_state=42, nan_policy="omit")
    
    # Should warn about pair removal
    with pytest.warns(UserWarning, match="removed 2 pair"):
        result = config.run(x, y)
    
    # Should be finite
    assert np.all(np.isfinite(result))
    
    # Should match result on clean paired data (indices 0, 3, 4)
    x_clean = np.array([1.0, 4.0, 5.0])
    y_clean = np.array([2.0, 8.0, 10.0])
    config_clean = BootstrapCorrConfig(n_resamples=100, random_state=42)
    result_clean = config_clean.run(x_clean, y_clean)
    
    np.testing.assert_array_almost_equal(result, result_clean, decimal=10)


def test_nan_policy_omit_two_sample():
    """nan_policy='omit' should filter each sample independently for mean_diff"""
    x = np.array([1.0, np.nan, 3.0, 4.0])
    y = np.array([5.0, 6.0, np.nan, 8.0])
    
    config = PermutationConfig(n_permutations=100, random_state=42, nan_policy="omit")
    
    with pytest.warns(UserWarning, match="removed 1 NaN"):
        result = config.run_mean_diff(x, y)
    
    assert np.all(np.isfinite(result))


def test_all_nans_raises_error():
    """All NaN array should raise clear error"""
    data = np.array([np.nan, np.nan, np.nan])
    
    config = BootstrapConfig(nan_policy="omit")
    
    with pytest.raises(ValueError, match="all values are NaN"):
        config.run(data)


# ======================================================================================
# 4. REPRODUCIBILITY TESTS
# ======================================================================================

def test_random_state_reproducibility():
    """Same random_state should give identical results"""
    np.random.seed(42)
    data = np.random.randn(100)
    
    config = BootstrapConfig(n_resamples=1000, random_state=12345)
    
    result1 = config.run(data)
    result2 = config.run(data)
    
    np.testing.assert_array_equal(result1, result2)


def test_different_random_state_different_results():
    """Different random_state should give different results"""
    np.random.seed(42)
    data = np.random.randn(100)
    
    config1 = BootstrapConfig(n_resamples=1000, random_state=12345)
    config2 = BootstrapConfig(n_resamples=1000, random_state=67890)
    
    result1 = config1.run(data)
    result2 = config2.run(data)
    
    # Results should differ (with high probability)
    assert not np.allclose(result1, result2)


def test_random_state_none_is_deterministic():
    """random_state=None should use deterministic default (0)"""
    np.random.seed(42)
    data = np.random.randn(100)
    
    config_none = BootstrapConfig(n_resamples=1000, random_state=None)
    config_zero = BootstrapConfig(n_resamples=1000, random_state=0)
    
    result_none = config_none.run(data)
    result_zero = config_zero.run(data)
    
    # Should be identical (both use seed=0)
    np.testing.assert_array_equal(result_none, result_zero)


# ======================================================================================
# 5. CONVENIENCE FUNCTION TESTS
# ======================================================================================

def test_bootstrap_convenience_function():
    """Convenience function should work identically to config object"""
    np.random.seed(42)
    data = np.random.randn(100)
    
    # Using convenience function
    result_func = bootstrap(data, stat="mean", n_resamples=500, conf=0.95, random_state=42)
    
    # Using config object
    config = BootstrapConfig(stat="mean", n_resamples=500, conf=0.95, random_state=42)
    result_config = config.run(data)
    
    np.testing.assert_array_equal(result_func, result_config)


def test_permutation_test_convenience_function():
    """Convenience function should route to correct test type"""
    np.random.seed(42)
    x = np.random.randn(50)
    y = np.random.randn(50)
    
    # Correlation test
    result_corr = permutation_test(x, y, test="corr", n_permutations=500, random_state=42)
    
    config_corr = PermutationConfig(n_permutations=500, random_state=42)
    expected_corr = config_corr.run_corr(x, y)
    
    np.testing.assert_array_equal(result_corr, expected_corr)
    
    # Mean diff test
    result_diff = permutation_test(x, y, test="mean_diff", n_permutations=500, random_state=42)
    
    expected_diff = config_corr.run_mean_diff(x, y)
    
    np.testing.assert_array_equal(result_diff, expected_diff)


def test_jackknife_convenience_function():
    """Jackknife convenience function should route correctly"""
    np.random.seed(42)
    data = np.random.randn(100)
    
    # mean_ci method
    result_ci = jackknife(data, method="mean_ci", conf=0.95)
    
    config = JackknifeConfig(conf=0.95)
    expected_ci = config.run_mean_ci(data)
    
    np.testing.assert_array_equal(result_ci, expected_ci)
    
    # mean method
    result_mean = jackknife(data, method="mean")
    expected_mean = config.run_mean(data)
    
    np.testing.assert_array_equal(result_mean, expected_mean)


# ======================================================================================
# 6. CALLABLE SHORTHAND TESTS
# ======================================================================================

def test_config_callable_shorthand():
    """Config objects should work with __call__ syntax"""
    np.random.seed(42)
    data = np.random.randn(100)
    
    config = BootstrapConfig(n_resamples=500, random_state=42)
    
    result_run = config.run(data)
    result_call = config(data)
    
    np.testing.assert_array_equal(result_run, result_call)


def test_corr_config_callable():
    """Correlation config should support __call__"""
    np.random.seed(42)
    x = np.random.randn(100)
    y = np.random.randn(100)
    
    config = BootstrapCorrConfig(n_resamples=500, random_state=42)
    
    result_run = config.run(x, y)
    result_call = config(x, y)
    
    np.testing.assert_array_equal(result_run, result_call)


# ======================================================================================
# PYTEST CONFIGURATION
# ======================================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
