"""
Benchmark Suite for Robust Statistics Module

Compares bunker-stats performance against scipy.stats for:
- Median, MAD, trimmed statistics
- Robust location/scale estimators
- Various dataset sizes: 100, 1K, 10K, 100K, 1M

Usage:
    pytest benchmarks_robust_stats.py -v --benchmark-only
    pytest benchmarks_robust_stats.py -v -k "median"
"""

import pytest
import numpy as np
from scipy import stats
import bunker_stats as bs
import time


# Dataset sizes to benchmark
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000]

# Number of iterations for timing
N_ITERS = {
    100: 10_000,
    1_000: 5_000,
    10_000: 1_000,
    100_000: 100,
    1_000_000: 10,
}


@pytest.fixture(params=SIZES, ids=lambda x: f"n={x}")
def dataset_size(request):
    """Parametrized dataset sizes"""
    return request.param


@pytest.fixture
def normal_data(dataset_size):
    """Generate normal random data"""
    np.random.seed(42)
    return np.random.randn(dataset_size)


@pytest.fixture
def contaminated_data(dataset_size):
    """Generate normal data with 5% outliers"""
    np.random.seed(42)
    data = np.random.randn(dataset_size)
    n_outliers = max(1, dataset_size // 20)
    outlier_indices = np.random.choice(dataset_size, n_outliers, replace=False)
    data[outlier_indices] = np.random.randn(n_outliers) * 100
    return data


# ============================================================================
# Median Benchmarks
# ============================================================================

class TestMedianBenchmarks:
    """Benchmark median computation"""
    
    def test_median_vs_numpy(self, normal_data, dataset_size):
        """Compare median: bunker-stats vs numpy"""
        n_iters = N_ITERS[dataset_size]
        
        # Bunker-stats
        start = time.perf_counter()
        for _ in range(n_iters):
            result_bs = bs.median(normal_data)
        time_bs = time.perf_counter() - start
        
        # NumPy
        start = time.perf_counter()
        for _ in range(n_iters):
            result_np = np.median(normal_data)
        time_np = time.perf_counter() - start
        
        # Verify correctness
        np.testing.assert_allclose(result_bs, result_np, rtol=1e-10)
        
        speedup = time_np / time_bs
        print(f"\n  n={dataset_size:>8,} | bunker-stats: {time_bs:.4f}s | numpy: {time_np:.4f}s | speedup: {speedup:.2f}x")
        
        # Store results for reporting
        pytest.benchmark_results = getattr(pytest, 'benchmark_results', {})
        pytest.benchmark_results.setdefault('median', []).append({
            'size': dataset_size,
            'bunker_stats_time': time_bs,
            'numpy_time': time_np,
            'speedup': speedup
        })


# ============================================================================
# MAD Benchmarks
# ============================================================================

class TestMADBenchmarks:
    """Benchmark Median Absolute Deviation"""
    
    def test_mad_vs_scipy(self, normal_data, dataset_size):
        """Compare MAD: bunker-stats vs scipy"""
        n_iters = N_ITERS[dataset_size]
        
        # Bunker-stats
        start = time.perf_counter()
        for _ in range(n_iters):
            result_bs = bs.mad(normal_data)
        time_bs = time.perf_counter() - start
        
        # SciPy
        start = time.perf_counter()
        for _ in range(n_iters):
            result_sp = stats.median_abs_deviation(normal_data, scale=1.0)
        time_sp = time.perf_counter() - start
        
        # Verify correctness
        np.testing.assert_allclose(result_bs, result_sp, rtol=1e-10)
        
        speedup = time_sp / time_bs
        print(f"\n  n={dataset_size:>8,} | bunker-stats: {time_bs:.4f}s | scipy: {time_sp:.4f}s | speedup: {speedup:.2f}x")
        
        pytest.benchmark_results = getattr(pytest, 'benchmark_results', {})
        pytest.benchmark_results.setdefault('mad', []).append({
            'size': dataset_size,
            'bunker_stats_time': time_bs,
            'scipy_time': time_sp,
            'speedup': speedup
        })
    
    def test_mad_std_vs_scipy(self, normal_data, dataset_size):
        """Compare MAD (scaled): bunker-stats vs scipy"""
        n_iters = N_ITERS[dataset_size]
        
        # Bunker-stats
        start = time.perf_counter()
        for _ in range(n_iters):
            result_bs = bs.mad_std(normal_data)
        time_bs = time.perf_counter() - start
        
        # SciPy
        start = time.perf_counter()
        for _ in range(n_iters):
            result_sp = stats.median_abs_deviation(normal_data, scale='normal')
        time_sp = time.perf_counter() - start
        
        # Verify correctness (relaxed tolerance due to median algorithm differences)
        np.testing.assert_allclose(result_bs, result_sp, rtol=1e-6)
        
        speedup = time_sp / time_bs
        print(f"\n  n={dataset_size:>8,} | bunker-stats: {time_bs:.4f}s | scipy: {time_sp:.4f}s | speedup: {speedup:.2f}x")


# ============================================================================
# Trimmed Statistics Benchmarks
# ============================================================================

class TestTrimmedStatsBenchmarks:
    """Benchmark trimmed mean and standard deviation"""
    
    @pytest.mark.parametrize("proportion", [0.1, 0.2, 0.25])
    def test_trimmed_mean_vs_scipy(self, normal_data, dataset_size, proportion):
        """Compare trimmed mean: bunker-stats vs scipy"""
        n_iters = N_ITERS[dataset_size]
        
        # Bunker-stats
        start = time.perf_counter()
        for _ in range(n_iters):
            result_bs = bs.trimmed_mean(normal_data, proportion)
        time_bs = time.perf_counter() - start
        
        # SciPy
        start = time.perf_counter()
        for _ in range(n_iters):
            result_sp = stats.trim_mean(normal_data, proportion)
        time_sp = time.perf_counter() - start
        
        # Verify correctness
        np.testing.assert_allclose(result_bs, result_sp, rtol=1e-10)
        
        speedup = time_sp / time_bs
        print(f"\n  n={dataset_size:>8,} trim={proportion:.2f} | bunker-stats: {time_bs:.4f}s | scipy: {time_sp:.4f}s | speedup: {speedup:.2f}x")


# ============================================================================
# IQR Benchmarks
# ============================================================================

class TestIQRBenchmarks:
    """Benchmark Interquartile Range"""
    
    def test_iqr_vs_scipy(self, normal_data, dataset_size):
        """Compare IQR: bunker-stats vs scipy"""
        n_iters = N_ITERS[dataset_size]
        
        # Bunker-stats
        start = time.perf_counter()
        for _ in range(n_iters):
            result_bs = bs.iqr(normal_data)
        time_bs = time.perf_counter() - start
        
        # SciPy
        start = time.perf_counter()
        for _ in range(n_iters):
            result_sp = stats.iqr(normal_data)
        time_sp = time.perf_counter() - start
        
        # Verify correctness (relaxed due to percentile interpolation)
        np.testing.assert_allclose(result_bs, result_sp, rtol=0.01)
        
        speedup = time_sp / time_bs
        print(f"\n  n={dataset_size:>8,} | bunker-stats: {time_bs:.4f}s | scipy: {time_sp:.4f}s | speedup: {speedup:.2f}x")


# ============================================================================
# Robust Location Estimators
# ============================================================================

class TestRobustLocationBenchmarks:
    """Benchmark robust location estimators"""
    
    def test_huber_location(self, contaminated_data, dataset_size):
        """Benchmark Huber M-estimator"""
        # Only test on smaller datasets (iterative algorithm)
        if dataset_size > 10_000:
            pytest.skip("Huber M-estimator too slow for large datasets")
        
        n_iters = min(100, N_ITERS[dataset_size])
        
        start = time.perf_counter()
        for _ in range(n_iters):
            result = bs.huber_location(contaminated_data)
        elapsed = time.perf_counter() - start
        
        print(f"\n  n={dataset_size:>8,} | time: {elapsed:.4f}s | per-call: {elapsed/n_iters*1000:.2f}ms")
        assert np.isfinite(result)


# ============================================================================
# Robust Scale Estimators
# ============================================================================

class TestRobustScaleBenchmarks:
    """Benchmark robust scale estimators"""
    
    def test_qn_scale(self, contaminated_data, dataset_size):
        """Benchmark Qn scale estimator"""
        # Qn is O(n log n) but with higher constant
        if dataset_size > 100_000:
            pytest.skip("Qn scale too slow for very large datasets")
        
        n_iters = min(100, N_ITERS[dataset_size])
        
        start = time.perf_counter()
        for _ in range(n_iters):
            result = bs.qn_scale(contaminated_data)
        elapsed = time.perf_counter() - start
        
        print(f"\n  n={dataset_size:>8,} | time: {elapsed:.4f}s | per-call: {elapsed/n_iters*1000:.2f}ms")
        assert np.isfinite(result)
    
    def test_biweight_midvariance(self, contaminated_data, dataset_size):
        """Benchmark biweight midvariance"""
        n_iters = min(1000, N_ITERS[dataset_size])
        
        start = time.perf_counter()
        for _ in range(n_iters):
            result = bs.biweight_midvariance(contaminated_data)
        elapsed = time.perf_counter() - start
        
        print(f"\n  n={dataset_size:>8,} | time: {elapsed:.4f}s | per-call: {elapsed/n_iters*1000:.2f}ms")
        assert np.isfinite(result)


# ============================================================================
# Skip-NA Variants Benchmarks
# ============================================================================

class TestSkipNABenchmarks:
    """Benchmark NaN-aware variants"""
    
    @pytest.fixture
    def data_with_nans(self, dataset_size):
        """Generate data with 10% NaNs"""
        np.random.seed(42)
        data = np.random.randn(dataset_size)
        n_nans = dataset_size // 10
        nan_indices = np.random.choice(dataset_size, n_nans, replace=False)
        data[nan_indices] = np.nan
        return data
    
    def test_median_skipna_vs_numpy(self, data_with_nans, dataset_size):
        """Compare NaN-aware median: bunker-stats vs numpy"""
        n_iters = N_ITERS[dataset_size]
        
        # Bunker-stats
        start = time.perf_counter()
        for _ in range(n_iters):
            result_bs = bs.median_skipna(data_with_nans)
        time_bs = time.perf_counter() - start
        
        # NumPy
        start = time.perf_counter()
        for _ in range(n_iters):
            result_np = np.nanmedian(data_with_nans)
        time_np = time.perf_counter() - start
        
        # Verify correctness
        np.testing.assert_allclose(result_bs, result_np, rtol=1e-10)
        
        speedup = time_np / time_bs
        print(f"\n  n={dataset_size:>8,} | bunker-stats: {time_bs:.4f}s | numpy: {time_np:.4f}s | speedup: {speedup:.2f}x")


# ============================================================================
# Summary Report Generation
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def generate_summary_report(request):
    """Generate benchmark summary after all tests"""
    yield
    
    if hasattr(pytest, 'benchmark_results'):
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY REPORT")
        print("="*80)
        
        for func_name, results in pytest.benchmark_results.items():
            print(f"\n{func_name.upper()}:")
            print(f"{'Size':>12} | {'BS Time':>10} | {'Ref Time':>10} | {'Speedup':>8}")
            print("-" * 60)
            
            for r in results:
                bs_time = r['bunker_stats_time']
                ref_time = r.get('scipy_time') or r.get('numpy_time')
                speedup = r['speedup']
                
                print(f"{r['size']:>12,} | {bs_time:>9.4f}s | {ref_time:>9.4f}s | {speedup:>7.2f}x")
        
        print("\n" + "="*80)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
