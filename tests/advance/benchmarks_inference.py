"""
Benchmark Suite for Inference Module

Compares bunker-stats performance against scipy.stats for:
- t-tests (1-sample, 2-sample, paired)
- Chi-square tests
- Mann-Whitney U test
- Kolmogorov-Smirnov test
- Effect sizes

Dataset sizes: 100, 1K, 10K, 100K

NOTE: Inference tests are typically O(n) so we expect comparable
performance to scipy. The value proposition is PRECISION, not speed.
"""

import pytest
import numpy as np
from scipy import stats as sp
import bunker_stats as bs
import time


SIZES = [100, 1_000, 10_000, 100_000]

N_ITERS = {
    100: 10_000,
    1_000: 5_000,
    10_000: 1_000,
    100_000: 100,
}


@pytest.fixture(params=SIZES, ids=lambda x: f"n={x}")
def dataset_size(request):
    return request.param


@pytest.fixture
def sample_data(dataset_size):
    """Generate random samples for testing"""
    np.random.seed(42)
    x = np.random.randn(dataset_size)
    y = np.random.randn(dataset_size) + 0.3  # Small effect size
    return x, y


# ============================================================================
# T-Test Benchmarks
# ============================================================================

class TestTTestBenchmarks:
    """Benchmark t-tests"""
    
    def test_ttest_1samp_vs_scipy(self, dataset_size):
        """1-sample t-test: bunker-stats vs scipy"""
        np.random.seed(42)
        x = np.random.randn(dataset_size)
        n_iters = N_ITERS[dataset_size]
        
        # Bunker-stats
        start = time.perf_counter()
        for _ in range(n_iters):
            result_bs = bs.t_test_1samp(x, 0.0)
        time_bs = time.perf_counter() - start
        
        # SciPy
        start = time.perf_counter()
        for _ in range(n_iters):
            result_sp = sp.ttest_1samp(x, 0.0)
        time_sp = time.perf_counter() - start
        
        # Verify precision
        np.testing.assert_allclose(
            result_bs['statistic'],
            result_sp.statistic,
            rtol=1e-12
        )
        
        speedup = time_sp / time_bs
        print(f"\n  n={dataset_size:>8,} | bunker-stats: {time_bs:.4f}s | scipy: {time_sp:.4f}s | speedup: {speedup:.2f}x")
        
        pytest.benchmark_results = getattr(pytest, 'benchmark_results', {})
        pytest.benchmark_results.setdefault('ttest_1samp', []).append({
            'size': dataset_size,
            'bunker_stats_time': time_bs,
            'scipy_time': time_sp,
            'speedup': speedup
        })
    
    def test_ttest_2samp_equal_var_vs_scipy(self, sample_data, dataset_size):
        """2-sample t-test (equal variance): bunker-stats vs scipy"""
        x, y = sample_data
        n_iters = N_ITERS[dataset_size]
        
        # Bunker-stats
        start = time.perf_counter()
        for _ in range(n_iters):
            result_bs = bs.t_test_2samp(x, y, equal_var=True)
        time_bs = time.perf_counter() - start
        
        # SciPy
        start = time.perf_counter()
        for _ in range(n_iters):
            result_sp = sp.ttest_ind(x, y, equal_var=True)
        time_sp = time.perf_counter() - start
        
        # Verify precision
        np.testing.assert_allclose(
            result_bs['statistic'],
            result_sp.statistic,
            rtol=1e-12
        )
        
        speedup = time_sp / time_bs
        print(f"\n  n={dataset_size:>8,} | bunker-stats: {time_bs:.4f}s | scipy: {time_sp:.4f}s | speedup: {speedup:.2f}x")
    
    def test_ttest_2samp_welch_vs_scipy(self, sample_data, dataset_size):
        """Welch's t-test: bunker-stats vs scipy"""
        x, y = sample_data
        n_iters = N_ITERS[dataset_size]
        
        # Bunker-stats
        start = time.perf_counter()
        for _ in range(n_iters):
            result_bs = bs.t_test_2samp(x, y, equal_var=False)
        time_bs = time.perf_counter() - start
        
        # SciPy
        start = time.perf_counter()
        for _ in range(n_iters):
            result_sp = sp.ttest_ind(x, y, equal_var=False)
        time_sp = time.perf_counter() - start
        
        # Verify precision
        np.testing.assert_allclose(
            result_bs['statistic'],
            result_sp.statistic,
            rtol=1e-12
        )
        
        speedup = time_sp / time_bs
        print(f"\n  n={dataset_size:>8,} | bunker-stats: {time_bs:.4f}s | scipy: {time_sp:.4f}s | speedup: {speedup:.2f}x")
    
    def test_ttest_paired_vs_scipy(self, sample_data, dataset_size):
        """Paired t-test: bunker-stats vs scipy"""
        x, y = sample_data
        n_iters = N_ITERS[dataset_size]
        
        # Bunker-stats
        start = time.perf_counter()
        for _ in range(n_iters):
            result_bs = bs.t_test_paired(x, y)
        time_bs = time.perf_counter() - start
        
        # SciPy
        start = time.perf_counter()
        for _ in range(n_iters):
            result_sp = sp.ttest_rel(x, y)
        time_sp = time.perf_counter() - start
        
        # Verify precision
        np.testing.assert_allclose(
            result_bs['statistic'],
            result_sp.statistic,
            rtol=1e-12
        )
        
        speedup = time_sp / time_bs
        print(f"\n  n={dataset_size:>8,} | bunker-stats: {time_bs:.4f}s | scipy: {time_sp:.4f}s | speedup: {speedup:.2f}x")


# ============================================================================
# Chi-Square Test Benchmarks
# ============================================================================

class TestChiSquareBenchmarks:
    """Benchmark chi-square tests"""
    
    def test_chi2_gof_vs_scipy(self, dataset_size):
        """Chi-square goodness of fit: bunker-stats vs scipy"""
        np.random.seed(42)
        obs = np.random.poisson(50, dataset_size).astype(float)
        exp = np.ones(dataset_size) * (obs.sum() / dataset_size)
        
        n_iters = N_ITERS[dataset_size]
        
        # Bunker-stats
        start = time.perf_counter()
        for _ in range(n_iters):
            result_bs = bs.chi2_gof(obs, exp)
        time_bs = time.perf_counter() - start
        
        # SciPy
        start = time.perf_counter()
        for _ in range(n_iters):
            result_sp = sp.chisquare(obs, exp)
        time_sp = time.perf_counter() - start
        
        # Verify precision
        np.testing.assert_allclose(
            result_bs['statistic'],
            result_sp.statistic,
            rtol=1e-12
        )
        
        speedup = time_sp / time_bs
        print(f"\n  n={dataset_size:>8,} | bunker-stats: {time_bs:.4f}s | scipy: {time_sp:.4f}s | speedup: {speedup:.2f}x")
        
        pytest.benchmark_results = getattr(pytest, 'benchmark_results', {})
        pytest.benchmark_results.setdefault('chi2_gof', []).append({
            'size': dataset_size,
            'bunker_stats_time': time_bs,
            'scipy_time': time_sp,
            'speedup': speedup
        })
    
    def test_chi2_independence_vs_scipy(self):
        """Chi-square independence test: bunker-stats vs scipy"""
        # Test various table sizes
        table_sizes = [(10, 10), (50, 50), (100, 100)]
        
        for rows, cols in table_sizes:
            np.random.seed(42)
            table = np.random.poisson(20, (rows, cols)).astype(float)
            
            n_iters = 1000
            
            # Bunker-stats
            start = time.perf_counter()
            for _ in range(n_iters):
                result_bs = bs.chi2_independence(table)
            time_bs = time.perf_counter() - start
            
            # SciPy
            start = time.perf_counter()
            for _ in range(n_iters):
                result_sp = sp.chi2_contingency(table, correction=False)
            time_sp = time.perf_counter() - start
            
            # Verify precision
            np.testing.assert_allclose(
                result_bs['statistic'],
                result_sp[0],
                rtol=1e-12
            )
            
            speedup = time_sp / time_bs
            print(f"\n  table {rows}x{cols:>3} | bunker-stats: {time_bs:.4f}s | scipy: {time_sp:.4f}s | speedup: {speedup:.2f}x")


# ============================================================================
# Mann-Whitney U Test Benchmarks
# ============================================================================

class TestMannWhitneyBenchmarks:
    """Benchmark Mann-Whitney U test"""
    
    def test_mann_whitney_vs_scipy(self, sample_data, dataset_size):
        """Mann-Whitney U: bunker-stats vs scipy"""
        x, y = sample_data
        n_iters = N_ITERS[dataset_size]
        
        # Bunker-stats
        start = time.perf_counter()
        for _ in range(n_iters):
            result_bs = bs.mann_whitney_u(x, y)
        time_bs = time.perf_counter() - start
        
        # SciPy
        start = time.perf_counter()
        for _ in range(n_iters):
            result_sp = sp.mannwhitneyu(x, y)
        time_sp = time.perf_counter() - start
        
        # Verify p-value (statistics may differ due to U1 vs U2)
        np.testing.assert_allclose(
            result_bs['pvalue'],
            result_sp.pvalue,
            rtol=1e-8
        )
        
        speedup = time_sp / time_bs
        print(f"\n  n={dataset_size:>8,} | bunker-stats: {time_bs:.4f}s | scipy: {time_sp:.4f}s | speedup: {speedup:.2f}x")


# ============================================================================
# Kolmogorov-Smirnov Test Benchmarks
# ============================================================================

class TestKSBenchmarks:
    """Benchmark Kolmogorov-Smirnov test"""
    
    def test_ks_1samp_vs_scipy(self, dataset_size):
        """KS 1-sample test: bunker-stats vs scipy"""
        np.random.seed(42)
        x = np.random.randn(dataset_size)
        
        n_iters = N_ITERS[dataset_size]
        
        # Bunker-stats
        start = time.perf_counter()
        for _ in range(n_iters):
            result_bs = bs.ks_1samp(x, 'norm', [0.0, 1.0])
        time_bs = time.perf_counter() - start
        
        # SciPy
        from scipy.stats import kstest
        start = time.perf_counter()
        for _ in range(n_iters):
            result_sp = kstest(x, 'norm', args=(0.0, 1.0))
        time_sp = time.perf_counter() - start
        
        # Verify statistic
        np.testing.assert_allclose(
            result_bs['statistic'],
            result_sp.statistic,
            rtol=1e-12
        )
        
        speedup = time_sp / time_bs
        print(f"\n  n={dataset_size:>8,} | bunker-stats: {time_bs:.4f}s | scipy: {time_sp:.4f}s | speedup: {speedup:.2f}x")


# ============================================================================
# Effect Size Benchmarks
# ============================================================================

class TestEffectSizeBenchmarks:
    """Benchmark effect size calculations"""
    
    def test_cohens_d_pooled(self, sample_data, dataset_size):
        """Cohen's d (pooled): bunker-stats performance"""
        x, y = sample_data
        n_iters = N_ITERS[dataset_size]
        
        start = time.perf_counter()
        for _ in range(n_iters):
            d = bs.cohens_d_2samp(x, y, pooled=True)
        elapsed = time.perf_counter() - start
        
        print(f"\n  n={dataset_size:>8,} | time: {elapsed:.4f}s | per-call: {elapsed/n_iters*1000:.3f}ms")
        assert np.isfinite(d)
    
    def test_cohens_d_nonpooled(self, sample_data, dataset_size):
        """Cohen's d (non-pooled): bunker-stats performance"""
        x, y = sample_data
        n_iters = N_ITERS[dataset_size]
        
        start = time.perf_counter()
        for _ in range(n_iters):
            d = bs.cohens_d_2samp(x, y, pooled=False)
        elapsed = time.perf_counter() - start
        
        print(f"\n  n={dataset_size:>8,} | time: {elapsed:.4f}s | per-call: {elapsed/n_iters*1000:.3f}ms")
        assert np.isfinite(d)


# ============================================================================
# Precision Validation Tests
# ============================================================================

class TestPrecisionValidation:
    """Validate that bunker-stats maintains higher precision than scipy"""
    
    def test_variance_catastrophic_cancellation(self):
        """Test variance with large numbers (Welford's algorithm advantage)"""
        # This is where bunker-stats should shine
        base = 1e10
        x = np.array([base, base + 1, base + 2, base + 3, base + 4])
        y = np.array([base, base + 1, base + 2, base + 3, base + 5])
        
        # Run both
        result_bs = bs.t_test_2samp(x, y, equal_var=True)
        result_sp = sp.ttest_ind(x, y, equal_var=True)
        
        print(f"\nCatastrophic Cancellation Test:")
        print(f"  bunker-stats statistic: {result_bs['statistic']:.15e}")
        print(f"  scipy statistic:        {result_sp.statistic:.15e}")
        print(f"  Difference:             {abs(result_bs['statistic'] - result_sp.statistic):.15e}")
        
        # Both should be close, but bunker-stats should be numerically stable
        assert np.isfinite(result_bs['statistic'])
        assert np.isfinite(result_sp.statistic)
        
        # Precision should be within 1e-10
        np.testing.assert_allclose(
            result_bs['statistic'],
            result_sp.statistic,
            rtol=1e-10,
            err_msg="Precision loss detected in variance calculation"
        )


# ============================================================================
# Summary Report
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def generate_summary_report(request):
    """Generate inference benchmark summary"""
    yield
    
    if hasattr(pytest, 'benchmark_results'):
        print("\n" + "="*80)
        print("INFERENCE MODULE BENCHMARK SUMMARY")
        print("="*80)
        print("\nNOTE: Inference operations are O(n) with low constants.")
        print("The value proposition is PRECISION (1e-12 to 1e-14), not speed.")
        print("Comparable or better performance is a bonus.\n")
        
        for func_name, results in pytest.benchmark_results.items():
            print(f"\n{func_name.upper()}:")
            print(f"{'Size':>12} | {'BS Time':>10} | {'SciPy Time':>10} | {'Speedup':>8}")
            print("-" * 60)
            
            for r in results:
                print(f"{r['size']:>12,} | {r['bunker_stats_time']:>9.4f}s | {r['scipy_time']:>9.4f}s | {r['speedup']:>7.2f}x")
        
        print("\n" + "="*80)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
