"""
Memory Safety Test Suite

Tests for memory leaks, memory efficiency, and proper memory management:
- No memory leaks in repeated operations
- Efficient memory usage (minimal allocations)
- Proper cleanup of temporary buffers
- In-place operations don't corrupt input
- Large dataset handling without excessive memory

CRITICAL: These tests ensure bunker-stats is production-ready for:
- Long-running processes
- High-frequency operations
- Memory-constrained environments
"""

import pytest
import numpy as np
import bunker_stats as bs
import gc
import sys


try:
    import tracemalloc
    HAS_TRACEMALLOC = True
except ImportError:
    HAS_TRACEMALLOC = False
    pytest.skip("tracemalloc not available", allow_module_level=True)


# ============================================================================
# Memory Leak Tests
# ============================================================================

class TestNoMemoryLeaks:
    """Test for memory leaks in repeated operations"""
    
    def test_bootstrap_no_leak(self):
        """Repeated bootstrap operations don't leak memory"""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        tracemalloc.start()
        gc.collect()
        
        # Establish baseline
        baseline = tracemalloc.get_traced_memory()[0]
        
        # Run many iterations
        for i in range(100):
            ci = bs.bootstrap_ci(data, np.mean, n_resamples=1000, seed=i)
            del ci
        
        gc.collect()
        final = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        # Memory should not grow significantly
        growth = final - baseline
        print(f"\nMemory growth: {growth:,} bytes ({growth/1024:.1f} KB)")
        
        # Allow 1MB growth max (should be near zero)
        assert growth < 1_000_000, f"Memory leak detected: {growth:,} bytes"
    
    def test_rolling_stats_no_leak(self):
        """Repeated rolling stats don't leak memory"""
        np.random.seed(42)
        data = np.random.randn(10000, 10)
        
        tracemalloc.start()
        gc.collect()
        
        baseline = tracemalloc.get_traced_memory()[0]
        
        # Run many iterations
        for _ in range(100):
            mean, std = bs.rolling_mean_std_axis0_np(data, 50)
            del mean, std
        
        gc.collect()
        final = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        growth = final - baseline
        print(f"\nMemory growth: {growth:,} bytes ({growth/1024:.1f} KB)")
        
        assert growth < 1_000_000, f"Memory leak detected: {growth:,} bytes"
    
    def test_inference_no_leak(self):
        """Repeated statistical tests don't leak memory"""
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        
        tracemalloc.start()
        gc.collect()
        
        baseline = tracemalloc.get_traced_memory()[0]
        
        # Run many iterations
        for _ in range(1000):
            result = bs.t_test_2samp(x, y, equal_var=True)
            del result
        
        gc.collect()
        final = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        growth = final - baseline
        print(f"\nMemory growth: {growth:,} bytes ({growth/1024:.1f} KB)")
        
        assert growth < 500_000, f"Memory leak detected: {growth:,} bytes"
    
    def test_robust_stats_no_leak(self):
        """Repeated robust stats don't leak memory"""
        np.random.seed(42)
        data = np.random.randn(10000)
        
        tracemalloc.start()
        gc.collect()
        
        baseline = tracemalloc.get_traced_memory()[0]
        
        # Run many iterations
        for _ in range(1000):
            m = bs.median(data)
            mad = bs.mad(data)
            iqr = bs.iqr(data)
            del m, mad, iqr
        
        gc.collect()
        final = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        growth = final - baseline
        print(f"\nMemory growth: {growth:,} bytes ({growth/1024:.1f} KB)")
        
        assert growth < 500_000, f"Memory leak detected: {growth:,} bytes"


# ============================================================================
# Memory Efficiency Tests
# ============================================================================

class TestMemoryEfficiency:
    """Test memory efficiency of operations"""
    
    def test_bootstrap_memory_usage(self):
        """Bootstrap memory usage is reasonable"""
        np.random.seed(42)
        data = np.random.randn(10000)
        
        tracemalloc.start()
        
        baseline = tracemalloc.get_traced_memory()[0]
        ci = bs.bootstrap_ci(data, np.mean, n_resamples=10000, seed=42)
        peak = tracemalloc.get_traced_memory()[1]
        
        tracemalloc.stop()
        
        data_size = data.nbytes
        peak_usage = peak - baseline
        
        print(f"\nData size: {data_size:,} bytes ({data_size/1024:.1f} KB)")
        print(f"Peak usage: {peak_usage:,} bytes ({peak_usage/1024:.1f} KB)")
        print(f"Ratio: {peak_usage/data_size:.2f}x")
        
        # Peak should be reasonable (< 100x data size for 10K resamples)
        # Note: This is conservative; actual usage should be much lower
        assert peak_usage < data_size * 100
    
    def test_rolling_mean_memory_single_pass(self):
        """Rolling mean uses single-pass algorithm (efficient memory)"""
        np.random.seed(42)
        data = np.random.randn(100000, 10)
        
        tracemalloc.start()
        
        baseline = tracemalloc.get_traced_memory()[0]
        result = bs.rolling_mean_axis0_np(data, 100)
        peak = tracemalloc.get_traced_memory()[1]
        
        tracemalloc.stop()
        
        data_size = data.nbytes
        result_size = result.nbytes
        overhead = peak - baseline - result_size
        
        print(f"\nData: {data_size/1e6:.2f}MB | Result: {result_size/1e6:.2f}MB | "
              f"Overhead: {overhead/1e6:.2f}MB")
        print(f"Overhead ratio: {overhead/result_size:.2f}x")
        
        # Overhead should be minimal (< 2x result size)
        # Single-pass algorithm shouldn't need much temporary storage
        assert overhead < result_size * 2
    
    def test_rolling_mean_std_combined_efficiency(self):
        """Combined rolling mean+std is more efficient than separate"""
        np.random.seed(42)
        data = np.random.randn(50000, 10)
        
        # Combined operation
        tracemalloc.start()
        baseline_combined = tracemalloc.get_traced_memory()[0]
        mean, std = bs.rolling_mean_std_axis0_np(data, 100)
        peak_combined = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        
        combined_peak = peak_combined - baseline_combined
        
        # Separate operations
        tracemalloc.start()
        baseline_separate = tracemalloc.get_traced_memory()[0]
        mean_sep = bs.rolling_mean_axis0_np(data, 100)
        std_sep = bs.rolling_std_axis0_np(data, 100)
        peak_separate = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        
        separate_peak = peak_separate - baseline_separate
        
        print(f"\nCombined peak: {combined_peak/1e6:.2f}MB")
        print(f"Separate peak: {separate_peak/1e6:.2f}MB")
        print(f"Efficiency: {separate_peak/combined_peak:.2f}x")
        
        # Combined should be more efficient (single pass vs two passes)
        # Allow some margin for measurement noise
        assert combined_peak <= separate_peak * 1.1
    
    def test_median_in_place_efficiency(self):
        """Median computation doesn't allocate excessively"""
        np.random.seed(42)
        data = np.random.randn(100000)
        
        tracemalloc.start()
        
        baseline = tracemalloc.get_traced_memory()[0]
        result = bs.median(data)
        peak = tracemalloc.get_traced_memory()[1]
        
        tracemalloc.stop()
        
        data_size = data.nbytes
        peak_usage = peak - baseline
        
        print(f"\nData size: {data_size/1e6:.2f}MB")
        print(f"Peak usage: {peak_usage/1e6:.2f}MB")
        print(f"Ratio: {peak_usage/data_size:.2f}x")
        
        # Median needs to sort, so might allocate ~1-2x data size
        assert peak_usage < data_size * 3


# ============================================================================
# Input Preservation Tests
# ============================================================================

class TestInputPreservation:
    """Verify operations don't modify input arrays"""
    
    def test_bootstrap_preserves_input(self):
        """Bootstrap doesn't modify input array"""
        np.random.seed(42)
        data = np.random.randn(1000)
        data_copy = data.copy()
        
        bs.bootstrap_ci(data, np.mean, n_resamples=1000, seed=42)
        
        np.testing.assert_array_equal(data, data_copy)
        assert data.flags['C_CONTIGUOUS'] == data_copy.flags['C_CONTIGUOUS']
    
    def test_rolling_stats_preserves_input(self):
        """Rolling stats don't modify input array"""
        np.random.seed(42)
        data = np.random.randn(10000, 10)
        data_copy = data.copy()
        
        bs.rolling_mean_std_axis0_np(data, 50)
        
        np.testing.assert_array_equal(data, data_copy)
        assert data.flags['C_CONTIGUOUS'] == data_copy.flags['C_CONTIGUOUS']
    
    def test_inference_preserves_inputs(self):
        """Inference tests don't modify input arrays"""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        x_copy = x.copy()
        y_copy = y.copy()
        
        bs.t_test_2samp(x, y, equal_var=True)
        
        np.testing.assert_array_equal(x, x_copy)
        np.testing.assert_array_equal(y, y_copy)
    
    def test_median_preserves_input(self):
        """Median doesn't modify input array"""
        np.random.seed(42)
        data = np.random.randn(1000)
        data_copy = data.copy()
        
        bs.median(data)
        
        np.testing.assert_array_equal(data, data_copy)
    
    def test_mad_preserves_input(self):
        """MAD doesn't modify input array"""
        np.random.seed(42)
        data = np.random.randn(1000)
        data_copy = data.copy()
        
        bs.mad(data)
        
        np.testing.assert_array_equal(data, data_copy)


# ============================================================================
# Large Dataset Memory Tests
# ============================================================================

class TestLargeDatasetMemory:
    """Test memory handling for large datasets"""
    
    @pytest.mark.slow
    def test_rolling_stats_10M_elements(self):
        """Rolling stats on 10M elements (1M rows × 10 cols)"""
        np.random.seed(42)
        data = np.random.randn(1_000_000, 10)
        
        data_size = data.nbytes
        print(f"\nData size: {data_size/1e6:.2f}MB")
        
        tracemalloc.start()
        
        baseline = tracemalloc.get_traced_memory()[0]
        mean, std = bs.rolling_mean_std_axis0_np(data, 100)
        peak = tracemalloc.get_traced_memory()[1]
        
        tracemalloc.stop()
        
        result_size = mean.nbytes + std.nbytes
        overhead = peak - baseline - result_size
        
        print(f"Result size: {result_size/1e6:.2f}MB")
        print(f"Peak overhead: {overhead/1e6:.2f}MB")
        print(f"Total peak: {(peak-baseline)/1e6:.2f}MB")
        
        # Should handle gracefully without excessive memory
        assert peak - baseline < data_size * 3
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))
    
    @pytest.mark.slow
    def test_bootstrap_100K_samples(self):
        """Bootstrap on 100K samples"""
        np.random.seed(42)
        data = np.random.randn(100_000)
        
        data_size = data.nbytes
        print(f"\nData size: {data_size/1e6:.2f}MB")
        
        tracemalloc.start()
        
        baseline = tracemalloc.get_traced_memory()[0]
        ci = bs.bootstrap_ci(data, np.median, n_resamples=10_000, seed=42)
        peak = tracemalloc.get_traced_memory()[1]
        
        tracemalloc.stop()
        
        peak_usage = peak - baseline
        
        print(f"Peak usage: {peak_usage/1e6:.2f}MB")
        print(f"Ratio: {peak_usage/data_size:.2f}x")
        
        assert np.all(np.isfinite(ci))
        assert ci[0] < ci[1]


# ============================================================================
# Memory Growth Pattern Tests
# ============================================================================

class TestMemoryGrowthPatterns:
    """Test memory growth patterns under various scenarios"""
    
    def test_increasing_dataset_size_linear_growth(self):
        """Memory usage grows linearly with dataset size"""
        sizes = [1000, 5000, 10000, 50000]
        memory_usage = []
        
        for size in sizes:
            np.random.seed(42)
            data = np.random.randn(size, 10)
            
            tracemalloc.start()
            baseline = tracemalloc.get_traced_memory()[0]
            
            result = bs.rolling_mean_axis0_np(data, 50)
            
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            
            usage = peak - baseline
            memory_usage.append(usage)
            
            del result
            gc.collect()
        
        # Memory should scale roughly linearly
        # Check that doubling size roughly doubles memory
        for i in range(len(sizes) - 1):
            size_ratio = sizes[i+1] / sizes[i]
            memory_ratio = memory_usage[i+1] / memory_usage[i]
            
            print(f"Size {sizes[i]} → {sizes[i+1]}: {size_ratio:.2f}x | "
                  f"Memory {memory_usage[i]/1e6:.2f}MB → {memory_usage[i+1]/1e6:.2f}MB: {memory_ratio:.2f}x")
            
            # Memory ratio should be close to size ratio (within 2x tolerance)
            assert 0.5 < memory_ratio / size_ratio < 2.0
    
    def test_increasing_resamples_bounded_growth(self):
        """Memory doesn't grow unbounded with resamples"""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        resample_counts = [1000, 5000, 10000]
        memory_usage = []
        
        for n_resamples in resample_counts:
            tracemalloc.start()
            baseline = tracemalloc.get_traced_memory()[0]
            
            ci = bs.bootstrap_ci(data, np.mean, n_resamples=n_resamples, seed=42)
            
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            
            usage = peak - baseline
            memory_usage.append(usage)
            
            del ci
            gc.collect()
        
        print(f"\nResamples: {resample_counts}")
        print(f"Memory (MB): {[m/1e6 for m in memory_usage]}")
        
        # Memory should grow with resamples but not excessively
        # 10x resamples shouldn't need 10x memory (due to streaming)
        assert memory_usage[-1] / memory_usage[0] < 20


# ============================================================================
# Cleanup Tests
# ============================================================================

class TestProperCleanup:
    """Test proper cleanup of resources"""
    
    def test_bootstrap_cleanup_after_exception(self):
        """Bootstrap cleans up even if statistic function fails"""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        def bad_statistic(x):
            if len(x) > 500:  # Fail on some resamples
                raise ValueError("Intentional error")
            return np.mean(x)
        
        tracemalloc.start()
        gc.collect()
        baseline = tracemalloc.get_traced_memory()[0]
        
        try:
            bs.bootstrap_ci(data, bad_statistic, n_resamples=100, seed=42)
        except:
            pass  # Expected to fail
        
        gc.collect()
        final = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        # Memory should return to baseline (no leaks from exception)
        growth = final - baseline
        print(f"\nMemory after exception: {growth:,} bytes")
        
        assert growth < 100_000  # Minimal residual


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
