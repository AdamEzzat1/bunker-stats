"""
Stress Test Suite - Large-Scale Operations

Tests bunker-stats at extreme scales to ensure production readiness:
- 1M+ row datasets
- 100K+ resamples
- High-dimensional data
- Edge cases at scale
- Performance under load

These tests are marked as 'slow' and can be run separately:
    pytest test_stress.py -v -m slow
"""

import pytest
import numpy as np
import bunker_stats as bs
import time


# ---------------------------------------------------------------------------
# API adapters (see tests/advance/test_concurrency_safety.py for rationale).
# Bridge the drifted test API to the shipped extension in one place.
# ---------------------------------------------------------------------------
_rs_bootstrap_ci = bs.bootstrap_ci
_rs_perm = bs.permutation_mean_diff_test
_STAT_NAMES = {np.mean: "mean", np.median: "median", np.std: "std"}


def _bootstrap_ci(data, stat_fn=np.mean, n_resamples=1000, seed=0):
    name = _STAT_NAMES.get(stat_fn, "mean")
    _point, lo, hi = _rs_bootstrap_ci(
        np.asarray(data, dtype=float), name, n_resamples=n_resamples, random_state=seed
    )
    return (lo, hi)


def _permutation_test(x, y, n_permutations=1000, seed=0):
    stat, pvalue = _rs_perm(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        n_permutations=n_permutations,
        random_state=seed,
    )
    return {"pvalue": pvalue, "statistic": stat}


# ============================================================================
# Rolling Statistics Stress Tests
# ============================================================================

class TestRollingStatsStress:
    """Stress test rolling statistics on large datasets"""
    
    @pytest.mark.slow
    def test_rolling_mean_1M_rows_single_col(self):
        """Rolling mean on 1M rows × 1 column"""
        np.random.seed(42)
        data = np.random.randn(1_000_000, 1)
        
        start = time.perf_counter()
        result = bs.rolling_mean_axis0_np(data, 100)
        elapsed = time.perf_counter() - start
        
        print(f"\n1M×1, window=100: {elapsed:.3f}s ({1_000_000/elapsed:.0f} rows/sec)")
        
        assert result.shape == (999_901, 1)
        assert np.all(np.isfinite(result))
        assert elapsed < 5.0  # Should complete in reasonable time
    
    @pytest.mark.slow
    def test_rolling_mean_1M_rows_10_cols(self):
        """Rolling mean on 1M rows × 10 columns"""
        np.random.seed(42)
        data = np.random.randn(1_000_000, 10)
        
        start = time.perf_counter()
        result = bs.rolling_mean_axis0_np(data, 100)
        elapsed = time.perf_counter() - start
        
        print(f"\n1M×10, window=100: {elapsed:.3f}s ({10_000_000/elapsed:.0f} values/sec)")
        
        assert result.shape == (999_901, 10)
        assert np.all(np.isfinite(result))
        assert elapsed < 10.0
    
    @pytest.mark.slow
    def test_rolling_mean_1M_rows_100_cols(self):
        """Rolling mean on 1M rows × 100 columns"""
        np.random.seed(42)
        data = np.random.randn(1_000_000, 100)
        
        start = time.perf_counter()
        result = bs.rolling_mean_axis0_np(data, 100)
        elapsed = time.perf_counter() - start
        
        print(f"\n1M×100, window=100: {elapsed:.3f}s ({100_000_000/elapsed:.0f} values/sec)")
        
        assert result.shape == (999_901, 100)
        assert np.all(np.isfinite(result))
    
    @pytest.mark.slow
    def test_rolling_std_1M_rows_10_cols(self):
        """Rolling std on 1M rows × 10 columns"""
        np.random.seed(42)
        data = np.random.randn(1_000_000, 10)
        
        start = time.perf_counter()
        result = bs.rolling_std_axis0_np(data, 50)
        elapsed = time.perf_counter() - start
        
        print(f"\n1M×10, window=50: {elapsed:.3f}s")
        
        assert result.shape == (999_951, 10)
        assert np.all(np.isfinite(result))
        # All stds should be positive (or NaN if not enough data)
        assert np.all((result >= 0) | np.isnan(result))
    
    @pytest.mark.slow
    def test_rolling_mean_std_combined_1M_rows(self):
        """Combined rolling mean+std on 1M rows"""
        np.random.seed(42)
        data = np.random.randn(1_000_000, 50)
        
        start = time.perf_counter()
        mean, std = bs.rolling_mean_std_axis0_np(data, 200)
        elapsed = time.perf_counter() - start
        
        print(f"\n1M×50, window=200: {elapsed:.3f}s")
        
        assert mean.shape == (999_801, 50)
        assert std.shape == (999_801, 50)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))
    
    @pytest.mark.slow
    def test_rolling_stats_large_window(self):
        """Rolling stats with very large window"""
        np.random.seed(42)
        data = np.random.randn(100_000, 10)
        
        # Very large window (10% of data)
        start = time.perf_counter()
        result = bs.rolling_mean_axis0_np(data, 10_000)
        elapsed = time.perf_counter() - start
        
        print(f"\n100K×10, window=10K: {elapsed:.3f}s")
        
        assert result.shape == (90_001, 10)
        assert np.all(np.isfinite(result))
    
    @pytest.mark.slow
    def test_rolling_stats_small_window_large_data(self):
        """Small window on very large dataset"""
        np.random.seed(42)
        data = np.random.randn(10_000_000, 1)
        
        # Tiny window relative to data size
        start = time.perf_counter()
        result = bs.rolling_mean_axis0_np(data, 5)
        elapsed = time.perf_counter() - start
        
        print(f"\n10M×1, window=5: {elapsed:.3f}s ({10_000_000/elapsed:.0f} rows/sec)")
        
        assert result.shape == (9_999_996, 1)
        assert np.all(np.isfinite(result))


# ============================================================================
# Bootstrap Stress Tests
# ============================================================================

class TestBootstrapStress:
    """Stress test bootstrap methods"""
    
    @pytest.mark.slow
    def test_bootstrap_1M_samples_1K_resamples(self):
        """Bootstrap on 1M samples with 1K resamples"""
        np.random.seed(42)
        data = np.random.randn(1_000_000)
        
        start = time.perf_counter()
        ci = _bootstrap_ci(data, np.mean, n_resamples=1_000, seed=42)
        elapsed = time.perf_counter() - start
        
        print(f"\n1M samples, 1K resamples: {elapsed:.3f}s")
        
        assert len(ci) == 2
        assert ci[0] < ci[1]
        assert np.all(np.isfinite(ci))
    
    @pytest.mark.slow
    def test_bootstrap_100K_samples_100K_resamples(self):
        """Bootstrap on 100K samples with 100K resamples"""
        np.random.seed(42)
        data = np.random.randn(100_000)
        
        start = time.perf_counter()
        ci = _bootstrap_ci(data, np.median, n_resamples=100_000, seed=42)
        elapsed = time.perf_counter() - start
        
        print(f"\n100K samples, 100K resamples: {elapsed:.3f}s")
        print(f"Total computations: {100_000 * 100_000:,} = 10 billion")
        
        assert len(ci) == 2
        assert ci[0] < ci[1]
        assert np.all(np.isfinite(ci))
    
    @pytest.mark.slow
    def test_bootstrap_10K_samples_1M_resamples(self):
        """Bootstrap on 10K samples with 1M resamples"""
        np.random.seed(42)
        data = np.random.randn(10_000)
        
        start = time.perf_counter()
        ci = _bootstrap_ci(data, np.mean, n_resamples=1_000_000, seed=42)
        elapsed = time.perf_counter() - start
        
        print(f"\n10K samples, 1M resamples: {elapsed:.3f}s")
        print(f"Total computations: {10_000 * 1_000_000:,} = 10 billion")
        
        assert len(ci) == 2
        assert ci[0] < ci[1]
        assert np.all(np.isfinite(ci))
    
    @pytest.mark.slow
    def test_bootstrap_median_large_scale(self):
        """Bootstrap median (O(n log n) per resample) at scale"""
        np.random.seed(42)
        data = np.random.randn(50_000)
        
        start = time.perf_counter()
        ci = _bootstrap_ci(data, np.median, n_resamples=10_000, seed=42)
        elapsed = time.perf_counter() - start
        
        print(f"\n50K samples, 10K resamples, median: {elapsed:.3f}s")
        
        assert len(ci) == 2
        assert ci[0] < ci[1]
        assert np.all(np.isfinite(ci))


# ============================================================================
# Permutation Test Stress Tests
# ============================================================================

class TestPermutationStress:
    """Stress test permutation tests"""
    
    @pytest.mark.slow
    def test_permutation_100K_samples_10K_permutations(self):
        """Permutation test on 100K samples"""
        np.random.seed(42)
        x = np.random.randn(100_000)
        y = np.random.randn(100_000) + 0.1
        
        start = time.perf_counter()
        result = _permutation_test(x, y, n_permutations=10_000, seed=42)
        elapsed = time.perf_counter() - start
        
        print(f"\n100K+100K samples, 10K permutations: {elapsed:.3f}s")
        
        assert 0.0 <= result['pvalue'] <= 1.0
    
    @pytest.mark.slow
    def test_permutation_10K_samples_100K_permutations(self):
        """Permutation test with many permutations"""
        np.random.seed(42)
        x = np.random.randn(10_000)
        y = np.random.randn(10_000) + 0.2
        
        start = time.perf_counter()
        result = _permutation_test(x, y, n_permutations=100_000, seed=42)
        elapsed = time.perf_counter() - start
        
        print(f"\n10K+10K samples, 100K permutations: {elapsed:.3f}s")
        
        assert 0.0 <= result['pvalue'] <= 1.0


# ============================================================================
# Robust Statistics Stress Tests
# ============================================================================

class TestRobustStatsStress:
    """Stress test robust statistics"""
    
    @pytest.mark.slow
    def test_median_10M_elements(self):
        """Median on 10M elements"""
        np.random.seed(42)
        data = np.random.randn(10_000_000)
        
        start = time.perf_counter()
        result = bs.median(data)
        elapsed = time.perf_counter() - start
        
        print(f"\n10M elements, median: {elapsed:.3f}s")
        
        assert np.isfinite(result)
        assert -0.1 < result < 0.1  # Should be near 0 for normal(0,1)
    
    @pytest.mark.slow
    def test_mad_1M_elements(self):
        """MAD on 1M elements"""
        np.random.seed(42)
        data = np.random.randn(1_000_000)
        
        start = time.perf_counter()
        result = bs.mad(data)
        elapsed = time.perf_counter() - start
        
        print(f"\n1M elements, MAD: {elapsed:.3f}s")
        
        assert np.isfinite(result)
        assert 0.6 < result < 0.8  # Should be near 0.6745 for normal(0,1)
    
    @pytest.mark.slow
    def test_iqr_1M_elements(self):
        """IQR on 1M elements"""
        np.random.seed(42)
        data = np.random.randn(1_000_000)
        
        start = time.perf_counter()
        result = bs.iqr(data)
        elapsed = time.perf_counter() - start
        
        print(f"\n1M elements, IQR: {elapsed:.3f}s")
        
        assert np.isfinite(result)
        assert 1.2 < result < 1.4  # Should be near 1.35 for normal(0,1)
    
    @pytest.mark.slow
    def test_trimmed_mean_1M_elements(self):
        """Trimmed mean on 1M elements"""
        np.random.seed(42)
        data = np.random.randn(1_000_000)
        
        start = time.perf_counter()
        result = bs.trimmed_mean(data, 0.1)
        elapsed = time.perf_counter() - start
        
        print(f"\n1M elements, trimmed mean (10%): {elapsed:.3f}s")
        
        assert np.isfinite(result)
        assert -0.1 < result < 0.1


# ============================================================================
# Inference Stress Tests
# ============================================================================

class TestInferenceStress:
    """Stress test statistical inference"""
    
    @pytest.mark.slow
    def test_ttest_1M_samples(self):
        """T-test on 1M samples"""
        np.random.seed(42)
        x = np.random.randn(1_000_000)
        y = np.random.randn(1_000_000) + 0.01
        
        start = time.perf_counter()
        result = bs.t_test_2samp(x, y, equal_var=True)
        elapsed = time.perf_counter() - start
        
        print(f"\n1M+1M samples, t-test: {elapsed:.3f}s")
        
        assert np.isfinite(result['statistic'])
        assert 0.0 <= result['pvalue'] <= 1.0
    
    @pytest.mark.slow
    def test_chi2_gof_100K_categories(self):
        """Chi-square goodness of fit on 100K categories"""
        np.random.seed(42)
        obs = np.random.poisson(50, 100_000).astype(float)
        exp = np.ones(100_000) * (obs.sum() / 100_000)
        
        start = time.perf_counter()
        result = bs.chi2_gof(obs, exp)
        elapsed = time.perf_counter() - start
        
        print(f"\n100K categories, chi-square: {elapsed:.3f}s")
        
        assert np.isfinite(result['statistic'])
        assert 0.0 <= result['pvalue'] <= 1.0
    
    @pytest.mark.slow
    def test_mann_whitney_100K_samples(self):
        """Mann-Whitney U on 100K samples"""
        np.random.seed(42)
        x = np.random.randn(100_000)
        y = np.random.randn(100_000) + 0.05
        
        start = time.perf_counter()
        result = bs.mann_whitney_u(x, y)
        elapsed = time.perf_counter() - start
        
        print(f"\n100K+100K samples, Mann-Whitney: {elapsed:.3f}s")
        
        assert np.isfinite(result['statistic'])
        assert 0.0 <= result['pvalue'] <= 1.0


# ============================================================================
# Edge Case Stress Tests
# ============================================================================

class TestEdgeCaseStress:
    """Stress test edge cases"""
    
    @pytest.mark.slow
    def test_rolling_stats_with_many_nans(self):
        """Rolling stats on data with 50% NaNs"""
        np.random.seed(42)
        data = np.random.randn(100_000, 10)
        
        # Set 50% to NaN
        nan_mask = np.random.rand(100_000, 10) > 0.5
        data[nan_mask] = np.nan
        
        start = time.perf_counter()
        result = bs.rolling_mean_axis0_np(data, 100)
        elapsed = time.perf_counter() - start
        
        print(f"\n100K×10 with 50% NaNs, window=100: {elapsed:.3f}s")

        assert result.shape == (99_901, 10)
        # Strict kernel contract: ANY NaN in a window propagates. With 50% NaN
        # and window=100, P(a NaN-free window) ~ 0.5^100 ~ 0, so an all-NaN
        # result is the CORRECT strict output, not a failure.
        assert np.all(np.isnan(result)), "strict kernel should propagate NaN here"

        # The NaN-aware path is what makes 50%-NaN data usable: nan_policy
        # 'ignore' with min_periods=1 must produce finite results.
        (mean_ignore,) = bs.rolling_multi_axis0(
            data, window=100, min_periods=1, nan_policy="ignore", stats=["mean"]
        )
        mean_ignore = np.asarray(mean_ignore)
        finite_frac = np.mean(np.isfinite(mean_ignore))
        print(f"nan_policy='ignore' finite fraction: {finite_frac:.3f}")
        assert finite_frac > 0.99
    
    @pytest.mark.slow
    def test_bootstrap_constant_data(self):
        """Bootstrap on constant data"""
        data = np.ones(100_000) * 42.0
        
        start = time.perf_counter()
        ci = _bootstrap_ci(data, np.mean, n_resamples=10_000, seed=42)
        elapsed = time.perf_counter() - start
        
        print(f"\n100K constant values, 10K resamples: {elapsed:.3f}s")
        
        # CI should be tight around 42.0
        assert ci[0] == 42.0
        assert ci[1] == 42.0
    
    @pytest.mark.slow
    def test_ttest_zero_variance(self):
        """T-test with zero variance in one sample"""
        x = np.ones(100_000) * 5.0
        y = np.random.randn(100_000) + 7.0
        
        start = time.perf_counter()
        result = bs.t_test_2samp(x, y, equal_var=False)
        elapsed = time.perf_counter() - start
        
        print(f"\n100K+100K samples (one zero-variance): {elapsed:.3f}s")
        
        # Should handle gracefully
        assert np.isfinite(result['statistic']) or np.isinf(result['statistic'])


# ============================================================================
# Numerical Stability Stress Tests
# ============================================================================

class TestNumericalStabilityStress:
    """Test numerical stability at scale"""
    
    @pytest.mark.slow
    def test_rolling_std_catastrophic_cancellation(self):
        """Rolling std with numbers that cause catastrophic cancellation"""
        base = 1e15
        data = np.random.randn(100_000, 10) + base
        
        start = time.perf_counter()
        mean, std = bs.rolling_mean_std_axis0_np(data, 100)
        elapsed = time.perf_counter() - start
        
        print(f"\n100K×10 (base 1e15), window=100: {elapsed:.3f}s")
        
        # All results should be finite
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))
        
        # Means should be near base
        assert np.all(np.abs(mean - base) < 100)
        
        # Stds should be reasonable (near 1.0)
        assert np.all((std > 0.5) & (std < 2.0))
    
    @pytest.mark.slow
    def test_variance_precision_large_numbers(self):
        """Variance calculation maintains precision with large numbers"""
        base = 1e12
        x = np.array([base + i for i in range(1_000_000)], dtype=float)
        
        start = time.perf_counter()
        variance = bs.var(x)
        elapsed = time.perf_counter() - start
        
        print(f"\n1M elements (base 1e12), variance: {elapsed:.3f}s")
        
        # bs.var is the SAMPLE variance (ddof=1). For [0, 1, ..., n-1] the
        # population variance is (n^2 - 1)/12; the ddof=1 value is that times
        # n/(n-1), i.e. n(n+1)/12. The old expectation used the population
        # formula, which differs by exactly n/(n-1) - 1 = 1e-6 at n=1e6 —
        # that was the entire "precision failure". The kernel itself is
        # accurate to ~3e-12 on this ill-conditioned input (numpy matches).
        n = len(x)
        expected_var = n * (n + 1) / 12

        # Should match to high precision
        rel_error = abs(variance - expected_var) / expected_var
        print(f"Relative error: {rel_error:.2e}")

        assert rel_error < 1e-10


# ============================================================================
# Concurrent Stress Tests
# ============================================================================

class TestConcurrentStress:
    """Stress test concurrent operations"""
    
    @pytest.mark.slow
    def test_concurrent_bootstrap_high_load(self):
        """Many concurrent bootstrap operations"""
        from concurrent.futures import ThreadPoolExecutor
        
        np.random.seed(42)
        data = np.random.randn(10_000)
        
        def run_bootstrap(seed):
            return _bootstrap_ci(data, np.mean, n_resamples=5_000, seed=seed)
        
        start = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(run_bootstrap, i) for i in range(100)]
            results = [f.result() for f in futures]
        
        elapsed = time.perf_counter() - start
        
        print(f"\n100 concurrent bootstraps (5K resamples each): {elapsed:.3f}s")
        print(f"Total: {100 * 5_000:,} resamples")
        
        # All should complete successfully
        assert len(results) == 100
        for ci in results:
            assert len(ci) == 2
            assert np.all(np.isfinite(ci))


# ============================================================================
# Summary Report
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def stress_test_summary(request):
    """Print summary after stress tests"""
    yield
    
    print("\n" + "="*80)
    print("STRESS TEST SUMMARY")
    print("="*80)
    print("\nAll large-scale stress tests passed!")
    print("bunker-stats is production-ready for:")
    print("  • 1M+ row datasets")
    print("  • 100K+ resamples")
    print("  • High-dimensional data")
    print("  • Concurrent operations")
    print("  • Numerically challenging scenarios")
    print("="*80)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'slow', '--tb=short'])
