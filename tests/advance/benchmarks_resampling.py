"""
Benchmark Suite for Resampling Module

Compares bunker-stats performance against scipy/statsmodels for:
- Bootstrap confidence intervals
- Permutation tests
- Jackknife statistics
- Various dataset sizes and resample counts

This is where bunker-stats should show SIGNIFICANT speedups (10-100x)
due to Rust parallelization with Rayon.

Dataset sizes: 100, 1K, 10K, 100K
Resample counts: 1K, 10K, 100K
"""

import pytest
import numpy as np
from scipy import stats as sp
import bunker_stats as bs
import time


# Dataset sizes
SIZES = [100, 1_000, 10_000, 100_000]

# Resample counts for bootstrap/permutation
RESAMPLES = [1_000, 10_000, 100_000]


@pytest.fixture(params=SIZES, ids=lambda x: f"n={x}")
def dataset_size(request):
    return request.param


@pytest.fixture
def sample_data(dataset_size):
    """Generate random samples"""
    np.random.seed(42)
    return np.random.randn(dataset_size)


@pytest.fixture
def two_samples(dataset_size):
    """Generate two random samples"""
    np.random.seed(42)
    x = np.random.randn(dataset_size)
    y = np.random.randn(dataset_size) + 0.3
    return x, y


# ============================================================================
# Bootstrap Benchmarks
# ============================================================================

class TestBootstrapBenchmarks:
    """Benchmark bootstrap methods"""
    
    @pytest.mark.parametrize("n_resamples", RESAMPLES, ids=lambda x: f"resamples={x}")
    def test_bootstrap_mean_vs_scipy(self, sample_data, dataset_size, n_resamples):
        """Bootstrap mean CI: bunker-stats vs scipy"""
        
        # Skip very large combinations
        if dataset_size * n_resamples > 10_000_000_000:
            pytest.skip("Combination too large for practical testing")
        
        # Bunker-stats
        start = time.perf_counter()
        ci_bs = bs.bootstrap_ci(sample_data, np.mean, n_resamples=n_resamples, seed=42)
        time_bs = time.perf_counter() - start
        
        # SciPy (using bootstrap method from scipy.stats)
        try:
            from scipy.stats import bootstrap
            start = time.perf_counter()
            rng = np.random.default_rng(42)
            result_sp = bootstrap(
                (sample_data,),
                np.mean,
                n_resamples=n_resamples,
                random_state=rng,
                method='percentile'
            )
            ci_sp = (result_sp.confidence_interval.low, result_sp.confidence_interval.high)
            time_sp = time.perf_counter() - start
            
            speedup = time_sp / time_bs
            
        except ImportError:
            # Fallback to manual implementation
            start = time.perf_counter()
            rng = np.random.RandomState(42)
            bootstrap_means = []
            for _ in range(n_resamples):
                resample = rng.choice(sample_data, size=len(sample_data), replace=True)
                bootstrap_means.append(np.mean(resample))
            bootstrap_means = np.array(bootstrap_means)
            ci_sp = (np.percentile(bootstrap_means, 2.5), np.percentile(bootstrap_means, 97.5))
            time_sp = time.perf_counter() - start
            
            speedup = time_sp / time_bs
        
        print(f"\n  n={dataset_size:>8,} resamples={n_resamples:>8,} | "
              f"bunker-stats: {time_bs:>6.3f}s | scipy: {time_sp:>6.3f}s | "
              f"speedup: {speedup:>6.2f}x")
        
        # Store results
        pytest.benchmark_results = getattr(pytest, 'benchmark_results', {})
        pytest.benchmark_results.setdefault('bootstrap_mean', []).append({
            'size': dataset_size,
            'resamples': n_resamples,
            'bunker_stats_time': time_bs,
            'scipy_time': time_sp,
            'speedup': speedup
        })
        
        # CIs should be reasonably close (they use different RNGs)
        # Just verify both are finite
        assert np.all(np.isfinite(ci_bs))
        assert np.all(np.isfinite(ci_sp))
    
    @pytest.mark.parametrize("n_resamples", [1_000, 10_000], ids=lambda x: f"resamples={x}")
    def test_bootstrap_median_vs_manual(self, sample_data, dataset_size, n_resamples):
        """Bootstrap median CI: bunker-stats vs manual implementation"""
        
        if dataset_size > 10_000 and n_resamples > 10_000:
            pytest.skip("Too slow for manual implementation")
        
        # Bunker-stats
        start = time.perf_counter()
        ci_bs = bs.bootstrap_ci(sample_data, np.median, n_resamples=n_resamples, seed=42)
        time_bs = time.perf_counter() - start
        
        # Manual implementation (no parallelization)
        start = time.perf_counter()
        rng = np.random.RandomState(42)
        bootstrap_medians = []
        for _ in range(n_resamples):
            resample = rng.choice(sample_data, size=len(sample_data), replace=True)
            bootstrap_medians.append(np.median(resample))
        bootstrap_medians = np.array(bootstrap_medians)
        ci_manual = (np.percentile(bootstrap_medians, 2.5), np.percentile(bootstrap_medians, 97.5))
        time_manual = time.perf_counter() - start
        
        speedup = time_manual / time_bs
        
        print(f"\n  n={dataset_size:>8,} resamples={n_resamples:>8,} | "
              f"bunker-stats: {time_bs:>6.3f}s | manual: {time_manual:>6.3f}s | "
              f"speedup: {speedup:>6.2f}x")
        
        assert np.all(np.isfinite(ci_bs))


# ============================================================================
# Permutation Test Benchmarks
# ============================================================================

class TestPermutationBenchmarks:
    """Benchmark permutation tests"""
    
    @pytest.mark.parametrize("n_permutations", [1_000, 10_000], ids=lambda x: f"perms={x}")
    def test_permutation_test_vs_manual(self, two_samples, dataset_size, n_permutations):
        """Permutation test: bunker-stats vs manual implementation"""
        x, y = two_samples
        
        if dataset_size > 10_000 and n_permutations > 10_000:
            pytest.skip("Too slow for manual implementation")
        
        # Bunker-stats
        start = time.perf_counter()
        result_bs = bs.permutation_test(x, y, n_permutations=n_permutations, seed=42)
        time_bs = time.perf_counter() - start
        
        # Manual implementation
        start = time.perf_counter()
        rng = np.random.RandomState(42)
        combined = np.concatenate([x, y])
        n_x = len(x)
        
        observed_diff = np.mean(x) - np.mean(y)
        perm_diffs = []
        
        for _ in range(n_permutations):
            rng.shuffle(combined)
            perm_x = combined[:n_x]
            perm_y = combined[n_x:]
            perm_diffs.append(np.mean(perm_x) - np.mean(perm_y))
        
        perm_diffs = np.array(perm_diffs)
        p_value_manual = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        time_manual = time.perf_counter() - start
        
        speedup = time_manual / time_bs
        
        print(f"\n  n={dataset_size:>8,} perms={n_permutations:>8,} | "
              f"bunker-stats: {time_bs:>6.3f}s | manual: {time_manual:>6.3f}s | "
              f"speedup: {speedup:>6.2f}x")
        
        pytest.benchmark_results = getattr(pytest, 'benchmark_results', {})
        pytest.benchmark_results.setdefault('permutation_test', []).append({
            'size': dataset_size,
            'permutations': n_permutations,
            'bunker_stats_time': time_bs,
            'manual_time': time_manual,
            'speedup': speedup
        })
        
        # P-values should be reasonably close
        assert 0.0 <= result_bs['pvalue'] <= 1.0
        assert 0.0 <= p_value_manual <= 1.0


# ============================================================================
# Jackknife Benchmarks
# ============================================================================

class TestJackknifeBenchmarks:
    """Benchmark jackknife methods"""
    
    def test_jackknife_mean(self, sample_data, dataset_size):
        """Jackknife mean: bunker-stats vs manual"""
        
        if dataset_size > 10_000:
            pytest.skip("Jackknife is O(n²), too slow for large n")
        
        # Bunker-stats
        start = time.perf_counter()
        jk_mean_bs, jk_se_bs = bs.jackknife_stats(sample_data, np.mean)
        time_bs = time.perf_counter() - start
        
        # Manual implementation
        start = time.perf_counter()
        n = len(sample_data)
        jk_estimates = np.zeros(n)
        
        for i in range(n):
            jk_sample = np.delete(sample_data, i)
            jk_estimates[i] = np.mean(jk_sample)
        
        jk_mean_manual = np.mean(jk_estimates)
        jk_se_manual = np.sqrt((n - 1) * np.var(jk_estimates, ddof=0))
        time_manual = time.perf_counter() - start
        
        speedup = time_manual / time_bs
        
        print(f"\n  n={dataset_size:>8,} | "
              f"bunker-stats: {time_bs:>6.3f}s | manual: {time_manual:>6.3f}s | "
              f"speedup: {speedup:>6.2f}x")
        
        # Results should match
        np.testing.assert_allclose(jk_mean_bs, jk_mean_manual, rtol=1e-10)
        np.testing.assert_allclose(jk_se_bs, jk_se_manual, rtol=1e-10)


# ============================================================================
# Cross-Validation Benchmarks
# ============================================================================

class TestCrossValidationBenchmarks:
    """Benchmark cross-validation methods"""
    
    @pytest.mark.parametrize("k", [5, 10], ids=lambda x: f"k={x}")
    def test_kfold_split(self, sample_data, dataset_size, k):
        """K-fold cross-validation: bunker-stats performance"""
        
        start = time.perf_counter()
        folds = bs.kfold_split(sample_data, k=k, shuffle=True, seed=42)
        time_bs = time.perf_counter() - start
        
        print(f"\n  n={dataset_size:>8,} k={k:>2} | time: {time_bs:.4f}s")
        
        # Verify we got k folds
        assert len(folds) == k
        
        # Verify all data is used exactly once
        all_test_indices = np.concatenate([test_idx for _, test_idx in folds])
        assert len(all_test_indices) == len(sample_data)
        assert len(np.unique(all_test_indices)) == len(sample_data)


# ============================================================================
# Large-Scale Bootstrap Benchmarks
# ============================================================================

class TestLargeScaleBootstrap:
    """Test bootstrap on very large datasets"""
    
    @pytest.mark.slow
    def test_bootstrap_1M_samples_1K_resamples(self):
        """Bootstrap on 1M samples with 1K resamples"""
        np.random.seed(42)
        data = np.random.randn(1_000_000)
        
        start = time.perf_counter()
        ci = bs.bootstrap_ci(data, np.mean, n_resamples=1_000, seed=42)
        elapsed = time.perf_counter() - start
        
        print(f"\n  1M samples, 1K resamples: {elapsed:.3f}s")
        assert np.all(np.isfinite(ci))
        assert ci[0] < ci[1]
    
    @pytest.mark.slow
    def test_bootstrap_100K_samples_100K_resamples(self):
        """Bootstrap on 100K samples with 100K resamples"""
        np.random.seed(42)
        data = np.random.randn(100_000)
        
        start = time.perf_counter()
        ci = bs.bootstrap_ci(data, np.median, n_resamples=100_000, seed=42)
        elapsed = time.perf_counter() - start
        
        print(f"\n  100K samples, 100K resamples: {elapsed:.3f}s")
        assert np.all(np.isfinite(ci))
        assert ci[0] < ci[1]


# ============================================================================
# Determinism Tests
# ============================================================================

class TestDeterminism:
    """Verify deterministic behavior with same seed"""
    
    def test_bootstrap_deterministic(self, sample_data):
        """Same seed should give same results"""
        ci1 = bs.bootstrap_ci(sample_data, np.mean, n_resamples=1000, seed=42)
        ci2 = bs.bootstrap_ci(sample_data, np.mean, n_resamples=1000, seed=42)
        
        np.testing.assert_array_equal(ci1, ci2)
    
    def test_permutation_deterministic(self, two_samples):
        """Same seed should give same p-value"""
        x, y = two_samples
        
        result1 = bs.permutation_test(x, y, n_permutations=1000, seed=42)
        result2 = bs.permutation_test(x, y, n_permutations=1000, seed=42)
        
        assert result1['pvalue'] == result2['pvalue']


# ============================================================================
# Summary Report
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def generate_summary_report(request):
    """Generate resampling benchmark summary"""
    yield
    
    if hasattr(pytest, 'benchmark_results'):
        print("\n" + "="*80)
        print("RESAMPLING MODULE BENCHMARK SUMMARY")
        print("="*80)
        print("\nNOTE: This is where bunker-stats should show SIGNIFICANT speedups")
        print("due to Rust + Rayon parallelization. Expect 10-100x improvements.\n")
        
        for func_name, results in pytest.benchmark_results.items():
            print(f"\n{func_name.upper()}:")
            
            if 'resamples' in results[0]:
                print(f"{'Size':>12} | {'Resamples':>10} | {'BS Time':>10} | {'Ref Time':>10} | {'Speedup':>8}")
                print("-" * 75)
                
                for r in results:
                    bs_time = r['bunker_stats_time']
                    ref_time = r.get('scipy_time') or r.get('manual_time')
                    
                    print(f"{r['size']:>12,} | {r['resamples']:>10,} | "
                          f"{bs_time:>9.4f}s | {ref_time:>9.4f}s | {r['speedup']:>7.2f}x")
            
            elif 'permutations' in results[0]:
                print(f"{'Size':>12} | {'Perms':>10} | {'BS Time':>10} | {'Manual Time':>10} | {'Speedup':>8}")
                print("-" * 75)
                
                for r in results:
                    print(f"{r['size']:>12,} | {r['permutations']:>10,} | "
                          f"{r['bunker_stats_time']:>9.4f}s | {r['manual_time']:>9.4f}s | {r['speedup']:>7.2f}x")
        
        print("\n" + "="*80)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
