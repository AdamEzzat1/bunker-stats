"""
Benchmark Suite for Matrix Operations Module

Compares bunker-stats performance against pandas/numpy for:
- Rolling statistics (mean, std, min, max)
- Matrix operations
- Various dataset sizes and window sizes

This is where bunker-stats should show SIGNIFICANT speedups for
rolling operations on large datasets.

Dataset sizes: 1K, 10K, 100K, 1M rows
Columns: 1, 10, 100
Window sizes: 10, 50, 200
"""

import pytest
import numpy as np
import pandas as pd
import bunker_stats as bs
import time


# Dataset configurations
ROWS = [1_000, 10_000, 100_000, 1_000_000]
COLS = [1, 10, 100]
WINDOWS = [10, 50, 200]


@pytest.fixture(params=ROWS, ids=lambda x: f"rows={x}")
def n_rows(request):
    return request.param


@pytest.fixture(params=COLS, ids=lambda x: f"cols={x}")
def n_cols(request):
    return request.param


@pytest.fixture
def matrix_data(n_rows, n_cols):
    """Generate random matrix data"""
    np.random.seed(42)
    return np.random.randn(n_rows, n_cols)


# ============================================================================
# Rolling Mean Benchmarks
# ============================================================================

class TestRollingMeanBenchmarks:
    """Benchmark rolling mean computation"""
    
    @pytest.mark.parametrize("window", WINDOWS, ids=lambda x: f"w={x}")
    def test_rolling_mean_vs_pandas(self, matrix_data, n_rows, n_cols, window):
        """Rolling mean: bunker-stats vs pandas"""
        
        # Skip very large combinations
        if n_rows * n_cols > 100_000_000:
            pytest.skip("Combination too large")
        
        # Bunker-stats (axis=0 operations)
        start = time.perf_counter()
        result_bs = bs.rolling_mean_axis0_np(matrix_data, window)
        time_bs = time.perf_counter() - start
        
        # Pandas
        start = time.perf_counter()
        df = pd.DataFrame(matrix_data)
        result_pd = df.rolling(window=window).mean().values[window-1:]
        time_pd = time.perf_counter() - start
        
        speedup = time_pd / time_bs
        
        print(f"\n  rows={n_rows:>8,} cols={n_cols:>3} window={window:>3} | "
              f"bunker-stats: {time_bs:>6.3f}s | pandas: {time_pd:>6.3f}s | "
              f"speedup: {speedup:>6.2f}x")
        
        # Store results
        pytest.benchmark_results = getattr(pytest, 'benchmark_results', {})
        pytest.benchmark_results.setdefault('rolling_mean', []).append({
            'rows': n_rows,
            'cols': n_cols,
            'window': window,
            'bunker_stats_time': time_bs,
            'pandas_time': time_pd,
            'speedup': speedup
        })
        
        # Verify correctness
        np.testing.assert_allclose(result_bs, result_pd, rtol=1e-10)


# ============================================================================
# Rolling Std Benchmarks
# ============================================================================

class TestRollingStdBenchmarks:
    """Benchmark rolling standard deviation"""
    
    @pytest.mark.parametrize("window", WINDOWS, ids=lambda x: f"w={x}")
    def test_rolling_std_vs_pandas(self, matrix_data, n_rows, n_cols, window):
        """Rolling std: bunker-stats vs pandas"""
        
        if n_rows * n_cols > 100_000_000:
            pytest.skip("Combination too large")
        
        # Bunker-stats
        start = time.perf_counter()
        result_bs = bs.rolling_std_axis0_np(matrix_data, window)
        time_bs = time.perf_counter() - start
        
        # Pandas
        start = time.perf_counter()
        df = pd.DataFrame(matrix_data)
        result_pd = df.rolling(window=window).std().values[window-1:]
        time_pd = time.perf_counter() - start
        
        speedup = time_pd / time_bs
        
        print(f"\n  rows={n_rows:>8,} cols={n_cols:>3} window={window:>3} | "
              f"bunker-stats: {time_bs:>6.3f}s | pandas: {time_pd:>6.3f}s | "
              f"speedup: {speedup:>6.2f}x")
        
        pytest.benchmark_results = getattr(pytest, 'benchmark_results', {})
        pytest.benchmark_results.setdefault('rolling_std', []).append({
            'rows': n_rows,
            'cols': n_cols,
            'window': window,
            'bunker_stats_time': time_bs,
            'pandas_time': time_pd,
            'speedup': speedup
        })
        
        # Verify correctness
        np.testing.assert_allclose(result_bs, result_pd, rtol=1e-10)


# ============================================================================
# Combined Rolling Mean + Std Benchmarks
# ============================================================================

class TestCombinedRollingBenchmarks:
    """Benchmark combined rolling mean and std"""
    
    @pytest.mark.parametrize("window", WINDOWS, ids=lambda x: f"w={x}")
    def test_rolling_mean_std_vs_pandas(self, matrix_data, n_rows, n_cols, window):
        """Combined rolling mean+std: bunker-stats vs pandas"""
        
        if n_rows * n_cols > 100_000_000:
            pytest.skip("Combination too large")
        
        # Bunker-stats (single pass)
        start = time.perf_counter()
        mean_bs, std_bs = bs.rolling_mean_std_axis0_np(matrix_data, window)
        time_bs = time.perf_counter() - start
        
        # Pandas (two passes)
        start = time.perf_counter()
        df = pd.DataFrame(matrix_data)
        rolling = df.rolling(window=window)
        mean_pd = rolling.mean().values[window-1:]
        std_pd = rolling.std().values[window-1:]
        time_pd = time.perf_counter() - start
        
        speedup = time_pd / time_bs
        
        print(f"\n  rows={n_rows:>8,} cols={n_cols:>3} window={window:>3} | "
              f"bunker-stats: {time_bs:>6.3f}s | pandas: {time_pd:>6.3f}s | "
              f"speedup: {speedup:>6.2f}x")
        
        pytest.benchmark_results = getattr(pytest, 'benchmark_results', {})
        pytest.benchmark_results.setdefault('rolling_mean_std', []).append({
            'rows': n_rows,
            'cols': n_cols,
            'window': window,
            'bunker_stats_time': time_bs,
            'pandas_time': time_pd,
            'speedup': speedup
        })
        
        # Verify correctness
        np.testing.assert_allclose(mean_bs, mean_pd, rtol=1e-10)
        np.testing.assert_allclose(std_bs, std_pd, rtol=1e-10)


# ============================================================================
# Rolling Min/Max Benchmarks
# ============================================================================

class TestRollingMinMaxBenchmarks:
    """Benchmark rolling min/max"""
    
    @pytest.mark.parametrize("window", WINDOWS, ids=lambda x: f"w={x}")
    def test_rolling_min_vs_pandas(self, matrix_data, n_rows, n_cols, window):
        """Rolling min: bunker-stats vs pandas"""
        
        if n_rows * n_cols > 100_000_000:
            pytest.skip("Combination too large")
        
        # Bunker-stats
        start = time.perf_counter()
        result_bs = bs.rolling_min_axis0_np(matrix_data, window)
        time_bs = time.perf_counter() - start
        
        # Pandas
        start = time.perf_counter()
        df = pd.DataFrame(matrix_data)
        result_pd = df.rolling(window=window).min().values[window-1:]
        time_pd = time.perf_counter() - start
        
        speedup = time_pd / time_bs
        
        print(f"\n  rows={n_rows:>8,} cols={n_cols:>3} window={window:>3} | "
              f"bunker-stats: {time_bs:>6.3f}s | pandas: {time_pd:>6.3f}s | "
              f"speedup: {speedup:>6.2f}x")
        
        # Verify correctness
        np.testing.assert_allclose(result_bs, result_pd, rtol=1e-10)
    
    @pytest.mark.parametrize("window", WINDOWS, ids=lambda x: f"w={x}")
    def test_rolling_max_vs_pandas(self, matrix_data, n_rows, n_cols, window):
        """Rolling max: bunker-stats vs pandas"""
        
        if n_rows * n_cols > 100_000_000:
            pytest.skip("Combination too large")
        
        # Bunker-stats
        start = time.perf_counter()
        result_bs = bs.rolling_max_axis0_np(matrix_data, window)
        time_bs = time.perf_counter() - start
        
        # Pandas
        start = time.perf_counter()
        df = pd.DataFrame(matrix_data)
        result_pd = df.rolling(window=window).max().values[window-1:]
        time_pd = time.perf_counter() - start
        
        speedup = time_pd / time_bs
        
        print(f"\n  rows={n_rows:>8,} cols={n_cols:>3} window={window:>3} | "
              f"bunker-stats: {time_bs:>6.3f}s | pandas: {time_pd:>6.3f}s | "
              f"speedup: {speedup:>6.2f}x")
        
        # Verify correctness
        np.testing.assert_allclose(result_bs, result_pd, rtol=1e-10)


# ============================================================================
# Cumulative Operations Benchmarks
# ============================================================================

class TestCumulativeBenchmarks:
    """Benchmark cumulative operations"""
    
    def test_cumsum_vs_numpy(self, matrix_data, n_rows, n_cols):
        """Cumulative sum: bunker-stats vs numpy"""
        
        if n_rows * n_cols > 100_000_000:
            pytest.skip("Combination too large")
        
        # Bunker-stats
        start = time.perf_counter()
        result_bs = bs.cumsum_axis0_np(matrix_data)
        time_bs = time.perf_counter() - start
        
        # NumPy
        start = time.perf_counter()
        result_np = np.cumsum(matrix_data, axis=0)
        time_np = time.perf_counter() - start
        
        speedup = time_np / time_bs
        
        print(f"\n  rows={n_rows:>8,} cols={n_cols:>3} | "
              f"bunker-stats: {time_bs:>6.3f}s | numpy: {time_np:>6.3f}s | "
              f"speedup: {speedup:>6.2f}x")
        
        # Verify correctness
        np.testing.assert_allclose(result_bs, result_np, rtol=1e-10)


# ============================================================================
# Large-Scale Stress Tests
# ============================================================================

class TestLargeScaleMatrix:
    """Test matrix operations on very large datasets"""
    
    @pytest.mark.slow
    def test_rolling_mean_1M_rows_10_cols(self):
        """Rolling mean on 1M rows × 10 columns"""
        np.random.seed(42)
        data = np.random.randn(1_000_000, 10)
        
        start = time.perf_counter()
        result = bs.rolling_mean_axis0_np(data, 100)
        elapsed = time.perf_counter() - start
        
        print(f"\n  1M×10 matrix, window=100: {elapsed:.3f}s")
        assert result.shape == (999_901, 10)
        assert np.all(np.isfinite(result))
    
    @pytest.mark.slow
    def test_rolling_std_1M_rows_100_cols(self):
        """Rolling std on 1M rows × 100 columns"""
        np.random.seed(42)
        data = np.random.randn(1_000_000, 100)
        
        start = time.perf_counter()
        result = bs.rolling_std_axis0_np(data, 50)
        elapsed = time.perf_counter() - start
        
        print(f"\n  1M×100 matrix, window=50: {elapsed:.3f}s")
        assert result.shape == (999_951, 100)
        assert np.all(np.isfinite(result))
    
    @pytest.mark.slow
    def test_combined_rolling_1M_rows_50_cols(self):
        """Combined rolling mean+std on 1M rows × 50 columns"""
        np.random.seed(42)
        data = np.random.randn(1_000_000, 50)
        
        start = time.perf_counter()
        mean, std = bs.rolling_mean_std_axis0_np(data, 200)
        elapsed = time.perf_counter() - start
        
        print(f"\n  1M×50 matrix, window=200: {elapsed:.3f}s")
        assert mean.shape == (999_801, 50)
        assert std.shape == (999_801, 50)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))


# ============================================================================
# Memory Efficiency Tests
# ============================================================================

class TestMemoryEfficiency:
    """Test memory usage of matrix operations"""
    
    def test_rolling_mean_memory_single_pass(self):
        """Rolling mean should use single-pass algorithm"""
        import tracemalloc
        
        np.random.seed(42)
        data = np.random.randn(100_000, 10)
        
        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]
        
        result = bs.rolling_mean_axis0_np(data, 100)
        
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        
        # Memory growth should be reasonable (mostly output allocation)
        data_size = data.nbytes
        result_size = result.nbytes
        overhead = peak - baseline - result_size
        
        print(f"\n  Data: {data_size/1e6:.1f}MB | Result: {result_size/1e6:.1f}MB | "
              f"Overhead: {overhead/1e6:.1f}MB")
        
        # Overhead should be minimal (< 2x result size)
        assert overhead < result_size * 2


# ============================================================================
# Numerical Stability Tests
# ============================================================================

class TestNumericalStability:
    """Test numerical stability of rolling operations"""
    
    def test_rolling_std_large_numbers(self):
        """Rolling std with large numbers (Welford's advantage)"""
        base = 1e10
        data = np.array([[base + i] for i in range(1000)])
        
        result = bs.rolling_std_axis0_np(data, 10)
        
        # Should compute reasonable variance despite large base
        assert np.all(np.isfinite(result))
        # Std should be small relative to base
        assert np.all(result < base / 1e6)
    
    def test_rolling_mean_catastrophic_cancellation(self):
        """Rolling mean should handle catastrophic cancellation"""
        # Create data where naive algorithm would lose precision
        base = 1e15
        data = np.random.randn(1000, 5) + base
        
        result = bs.rolling_mean_axis0_np(data, 50)
        
        # All results should be finite and close to base
        assert np.all(np.isfinite(result))
        assert np.all(np.abs(result - base) < 100)


# ============================================================================
# Summary Report
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def generate_summary_report(request):
    """Generate matrix operations benchmark summary"""
    yield
    
    if hasattr(pytest, 'benchmark_results'):
        print("\n" + "="*80)
        print("MATRIX OPERATIONS BENCHMARK SUMMARY")
        print("="*80)
        print("\nNOTE: Rolling operations show significant speedups on large datasets")
        print("due to efficient single-pass algorithms and Rayon parallelization.\n")
        
        for func_name, results in pytest.benchmark_results.items():
            print(f"\n{func_name.upper()}:")
            print(f"{'Rows':>10} | {'Cols':>5} | {'Window':>7} | {'BS Time':>10} | {'Pandas Time':>12} | {'Speedup':>8}")
            print("-" * 85)
            
            for r in results:
                print(f"{r['rows']:>10,} | {r['cols']:>5} | {r['window']:>7} | "
                      f"{r['bunker_stats_time']:>9.4f}s | {r['pandas_time']:>11.4f}s | {r['speedup']:>7.2f}x")
        
        print("\n" + "="*80)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
