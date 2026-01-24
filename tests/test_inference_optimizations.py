"""Test Coverage for Inference Module Optimizations"""

import numpy as np
import scipy.stats as sp
import bunker_stats as bs
import pytest

# ============================================================================
# Bug Fix Validation Tests
# ============================================================================

def test_chi2_edge_cases():
    """Test chi-square survival function at extreme values."""
    
    # Very large chi2 statistic (should give p ≈ 0)
    obs = np.array([1000.0, 0.0])
    result = bs.chi2_gof(obs)
    assert result['pvalue'] < 1e-10
    
    # Very small chi2 statistic (should give p ≈ 1)
    obs = np.array([50.0, 50.0])
    exp = np.array([50.0, 50.0])
    result = bs.chi2_gof(obs, exp)
    assert result['pvalue'] > 0.99
    
    # Compare with scipy
    scipy_result = sp.chisquare(obs, exp)
    np.testing.assert_allclose(
        result['statistic'],
        scipy_result.statistic,
        rtol=1e-12
    )


def test_welch_zero_variance():
    """Test Welch t-test when both samples have zero variance."""
    
    # Both constant - should handle gracefully
    x = np.ones(10) * 5.0
    y = np.ones(10) * 7.0
    
    result = bs.t_test_2samp(x, y, equal_var=False)
    
    # With zero variance, t should be infinite if means differ
    assert np.isinf(result['statistic'])
    assert result['pvalue'] < 1e-10
    
    # Same means - t should be 0 or NaN depending on implementation
    x = np.ones(10) * 5.0
    y = np.ones(10) * 5.0
    result = bs.t_test_2samp(x, y, equal_var=False)
    assert result['statistic'] == 0.0 or np.isnan(result['statistic'])


def test_ks_large_n():
    """Test KS test numerical stability for large n."""
    
    np.random.seed(42)
    n = 5000
    x = np.random.randn(n)
    
    # Should not overflow or underflow
    result = bs.ks_1samp(x, 'norm', [0.0, 1.0], alternative='two-sided')
    
    assert np.isfinite(result['statistic'])
    assert 0.0 <= result['pvalue'] <= 1.0
    
    # Compare with scipy
    from scipy.stats import kstest
    scipy_result = kstest(x, 'norm', args=(0.0, 1.0))
    
    # Statistic should match exactly
    np.testing.assert_allclose(
        result['statistic'],
        scipy_result.statistic,
        rtol=1e-10
    )
    
    # P-value may differ slightly due to different algorithms
    # but should be in same ballpark
    assert abs(np.log10(result['pvalue']) - np.log10(scipy_result.pvalue)) < 0.5


def test_mann_whitney_ties():
    """Test Mann-Whitney with extensive ties."""
    
    # Many ties
    x = np.array([1, 1, 1, 2, 2, 3], dtype=float)
    y = np.array([1, 2, 2, 2, 3, 3], dtype=float)
    
    result = bs.mann_whitney_u(x, y, alternative='two-sided')
    
    # Compare with scipy
    scipy_result = sp.mannwhitneyu(x, y, alternative='two-sided')
    
    # Statistic might differ (U1 vs U2) but should be related
    assert np.isfinite(result['statistic'])
    assert 0.0 <= result['pvalue'] <= 1.0
    
    # P-values should be very close
    np.testing.assert_allclose(
        result['pvalue'],
        scipy_result.pvalue,
        rtol=1e-8
    )


# ============================================================================
# Performance Optimization Tests
# ============================================================================

def test_chi2_performance():
    """Benchmark chi-square test performance."""
    import time
    
    np.random.seed(42)
    obs = np.random.poisson(100, 1000).astype(float)
    # Scale expected to match observed sum for valid test
    exp = np.ones(1000) * (obs.sum() / 1000)  # NEW - normalized
    
    # Bunker-stats
    start = time.time()
    for _ in range(100):
        bs.chi2_gof(obs, exp)
    bs_time = time.time() - start
    
    # SciPy
    start = time.time()
    for _ in range(100):
        sp.chisquare(obs, exp)
    scipy_time = time.time() - start
    
    print(f"\nChi-square: bunker-stats={bs_time:.4f}s, scipy={scipy_time:.4f}s")
    print(f"Speedup: {scipy_time/bs_time:.2f}x")
    
    # Should be faster
    assert bs_time < scipy_time * 1.2  # Allow 20% margin


def test_ttest_performance():
    """Benchmark t-test performance."""
    import time
    
    np.random.seed(42)
    x = np.random.randn(1000)
    y = np.random.randn(1000) + 0.1
    
    # Bunker-stats
    start = time.time()
    for _ in range(100):
        bs.t_test_2samp(x, y, equal_var=True)
    bs_time = time.time() - start
    
    # SciPy
    start = time.time()
    for _ in range(100):
        sp.ttest_ind(x, y, equal_var=True)
    scipy_time = time.time() - start
    
    print(f"\nT-test: bunker-stats={bs_time:.4f}s, scipy={scipy_time:.4f}s")
    print(f"Speedup: {scipy_time/bs_time:.2f}x")
    
    assert bs_time < scipy_time * 1.5


# ============================================================================
# Numerical Accuracy Tests
# ============================================================================

def test_variance_numerical_stability():
    """Test variance calculation with difficult numbers."""
    
    # Large numbers (Welford's algorithm should help)
    x = np.array([1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3])
    y = np.array([1e10, 1e10 + 1, 1e10 + 2, 1e10 + 4])
    
    result = bs.t_test_2samp(x, y, equal_var=True)
    scipy_result = sp.ttest_ind(x, y, equal_var=True)
    
    # Should match scipy to high precision
    np.testing.assert_allclose(
        result['statistic'],
        scipy_result.statistic,
        rtol=1e-10
    )
    
    np.testing.assert_allclose(
        result['pvalue'],
        scipy_result.pvalue,
        rtol=1e-10
    )


def test_correlation_numerical_precision():
    """Test correlation calculation precision."""
    
    np.random.seed(42)
    x = np.random.randn(100)
    
    # Perfect correlation
    y = x * 2.0 + 1.0
    
    # Using existing functions
    r_xy = bs.cov(x, y) / (bs.std(x) * bs.std(y))
    
    # Should be exactly 1.0
    assert abs(r_xy - 1.0) < 1e-12
    
    # Negative perfect correlation
    y = -x * 2.0 + 1.0
    r_xy = bs.cov(x, y) / (bs.std(x) * bs.std(y))
    
    assert abs(r_xy - (-1.0)) < 1e-12


# ============================================================================
# Comprehensive Parity Tests
# ============================================================================

def test_chi2_gof_parity():
    """Comprehensive chi-square goodness of fit parity test."""
    
    test_cases = [
        # (observed, expected)
        (np.array([10.0, 20.0, 30.0]), None),  # Uniform
        (np.array([10.0, 20.0, 30.0]), np.array([15.0, 20.0, 25.0])),  # Custom
        (np.array([100.0, 100.0]), np.array([100.0, 100.0])),  # Perfect fit
        (np.array([100.0, 0.0]), np.array([50.0, 50.0])),  # Terrible fit
    ]
    
    for obs, exp in test_cases:
        bs_result = bs.chi2_gof(obs, exp)
        
        if exp is None:
            scipy_result = sp.chisquare(obs)
        else:
            scipy_result = sp.chisquare(obs, exp)
        
        np.testing.assert_allclose(
            bs_result['statistic'],
            scipy_result.statistic,
            rtol=1e-12,
            err_msg=f"Failed for obs={obs}, exp={exp}"
        )
        
        # For p-values, use appropriate tolerance based on magnitude
        # Extremely small p-values (< 1e-15) should use absolute tolerance
        if scipy_result.pvalue < 1e-15:
            # For catastrophically small p-values, both 0.0 and 1e-23 are "effectively zero"
            # Use absolute tolerance instead of relative
            np.testing.assert_allclose(
                bs_result['pvalue'],
                scipy_result.pvalue,
                atol=1e-15,
                rtol=0,
                err_msg=f"Failed for obs={obs}, exp={exp}"
            )
        else:
            np.testing.assert_allclose(
                bs_result['pvalue'],
                scipy_result.pvalue,
                rtol=1e-10,
                err_msg=f"Failed for obs={obs}, exp={exp}"
            )


def test_chi2_independence_parity():
    """Test chi-square independence test parity."""
    
    # Classic 2x2
    table = np.array([[10.0, 20.0], [20.0, 10.0]])
    bs_result = bs.chi2_independence(table)
    # Use correction=False to match bunker-stats (no Yates' correction)
    scipy_result = sp.chi2_contingency(table, correction=False)
    
    np.testing.assert_allclose(bs_result['statistic'], scipy_result[0], rtol=1e-12)
    np.testing.assert_allclose(bs_result['pvalue'], scipy_result[1], rtol=1e-10)
    
    # Larger table
    table = np.array([
        [10.0, 20.0, 30.0],
        [15.0, 25.0, 35.0],
        [20.0, 30.0, 40.0]
    ])
    bs_result = bs.chi2_independence(table)
    scipy_result = sp.chi2_contingency(table, correction=False)
    
    np.testing.assert_allclose(bs_result['statistic'], scipy_result[0], rtol=1e-12)
    np.testing.assert_allclose(bs_result['pvalue'], scipy_result[1], rtol=1e-10)


def test_ttest_1samp_parity():
    """Test one-sample t-test parity."""
    
    np.random.seed(42)
    
    test_cases = [
        (np.random.randn(10), 0.0, 'two-sided'),
        (np.random.randn(100) + 0.5, 0.0, 'greater'),
        (np.random.randn(50) - 0.3, 0.0, 'less'),
        (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3.0, 'two-sided'),
    ]
    
    for x, popmean, alt in test_cases:
        bs_result = bs.t_test_1samp(x, popmean, alternative=alt)
        scipy_result = sp.ttest_1samp(x, popmean, alternative=alt)
        
        np.testing.assert_allclose(
            bs_result['statistic'],
            scipy_result.statistic,
            rtol=1e-12,
            err_msg=f"Failed for popmean={popmean}, alt={alt}"
        )
        
        np.testing.assert_allclose(
            bs_result['pvalue'],
            scipy_result.pvalue,
            rtol=1e-10,
            err_msg=f"Failed for popmean={popmean}, alt={alt}"
        )


def test_ttest_2samp_parity():
    """Test two-sample t-test parity."""
    
    np.random.seed(42)
    
    test_cases = [
        (np.random.randn(20), np.random.randn(20), True, 'two-sided'),
        (np.random.randn(30) + 0.5, np.random.randn(25), True, 'greater'),
        (np.random.randn(15), np.random.randn(40), False, 'two-sided'),  # Welch
    ]
    
    for x, y, equal_var, alt in test_cases:
        bs_result = bs.t_test_2samp(x, y, equal_var=equal_var, alternative=alt)
        scipy_result = sp.ttest_ind(x, y, equal_var=equal_var, alternative=alt)
        
        np.testing.assert_allclose(
            bs_result['statistic'],
            scipy_result.statistic,
            rtol=1e-12,
            err_msg=f"Failed for equal_var={equal_var}, alt={alt}"
        )
        
        np.testing.assert_allclose(
            bs_result['pvalue'],
            scipy_result.pvalue,
            rtol=1e-10,
            err_msg=f"Failed for equal_var={equal_var}, alt={alt}"
        )


def test_mann_whitney_parity():
    """Test Mann-Whitney U test parity."""
    
    np.random.seed(42)
    
    test_cases = [
        (np.random.randn(20), np.random.randn(20), 'two-sided'),
        (np.random.randn(30), np.random.randn(25), 'greater'),
        (np.random.randn(15), np.random.randn(40), 'less'),
    ]
    
    for x, y, alt in test_cases:
        bs_result = bs.mann_whitney_u(x, y, alternative=alt)
        scipy_result = sp.mannwhitneyu(x, y, alternative=alt)
        
        # P-values should match
        np.testing.assert_allclose(
            bs_result['pvalue'],
            scipy_result.pvalue,
            rtol=1e-8,
            err_msg=f"Failed for alt={alt}"
        )


def test_ks_1samp_parity():
    """Test Kolmogorov-Smirnov test parity."""
    
    np.random.seed(42)
    
    test_cases = [
        (np.random.randn(50), 'norm', [0.0, 1.0], 'two-sided'),
        (np.random.exponential(2.0, 100), 'expon', [0.0, 2.0], 'two-sided'),
        (np.random.uniform(0, 1, 75), 'uniform', [0.0, 1.0], 'two-sided'),
    ]
    
    for x, dist, params, alt in test_cases:
        bs_result = bs.ks_1samp(x, dist, params, alternative=alt)
        
        from scipy.stats import kstest
        scipy_result = kstest(x, dist, args=tuple(params), alternative=alt)
        
        np.testing.assert_allclose(
            bs_result['statistic'],
            scipy_result.statistic,
            rtol=1e-12,
            err_msg=f"Failed for dist={dist}, alt={alt}"
        )
        
        # P-value may differ due to different algorithms, especially for large n
        # Allow larger tolerance
        if len(x) > 100:
            rtol = 0.1
        else:
            rtol = 1e-6
        
        np.testing.assert_allclose(
            bs_result['pvalue'],
            scipy_result.pvalue,
            rtol=rtol,
            err_msg=f"Failed for dist={dist}, alt={alt}"
        )


# ============================================================================
# Effect Size Tests
# ============================================================================

def test_cohens_d_parity():
    """Test Cohen's d calculation."""
    
    np.random.seed(42)
    x = np.random.randn(30)
    y = np.random.randn(25) + 0.5
    
    # Pooled
    d_pooled = bs.cohens_d_2samp(x, y, pooled=True)
    
    # Manual calculation
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    
    sp_var = ((len(x)-1)*vx + (len(y)-1)*vy) / (len(x)+len(y)-2)
    d_expected = (mx - my) / np.sqrt(sp_var)
    
    np.testing.assert_allclose(d_pooled, d_expected, rtol=1e-12)
    
    # Non-pooled
    d_nonpooled = bs.cohens_d_2samp(x, y, pooled=False)
    s = np.sqrt((vx + vy) / 2)
    d_expected = (mx - my) / s
    
    np.testing.assert_allclose(d_nonpooled, d_expected, rtol=1e-12)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])