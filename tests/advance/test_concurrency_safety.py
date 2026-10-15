"""
Concurrency Safety Test Suite

Tests thread-safety and deterministic behavior of bunker-stats operations:
- Deterministic results with same seed across multiple runs
- Thread-safe operations when called concurrently
- No race conditions in parallel operations
- Consistent RNG behavior

CRITICAL: These tests ensure bunker-stats can be safely used in:
- Multi-threaded applications
- Parallel data processing pipelines
- Concurrent statistical workflows
"""

import pytest
import numpy as np
import bunker_stats as bs
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing


# ---------------------------------------------------------------------------
# API adapters. These tests were written against an earlier/envisioned API
# (`bootstrap_ci(data, np.mean, seed=...)` returning a CI, and a dict-returning
# `permutation_test`). The shipped extension exposes `bootstrap_ci(x, "mean",
# random_state=...) -> (point, lo, hi)` and `permutation_mean_diff_test(x, y,
# random_state=...) -> (stat, pvalue)`. Bridge the two here in one place so the
# concurrency/determinism assertions run against the real kernels unchanged.
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


# ---------------------------------------------------------------------------
# Module-level workers for the ProcessPoolExecutor tests. Windows uses the
# `spawn` start method, which pickles the submitted callable — a function
# defined inside a test body is unpicklable there ("Can't pickle local
# object"), so these must live at module scope and receive the data as an
# argument instead of closing over it.
# ---------------------------------------------------------------------------

def _proc_bootstrap_worker(data, seed):
    return _bootstrap_ci(data, np.mean, n_resamples=1000, seed=seed)


def _proc_rolling_worker(data):
    return bs.rolling_mean_std_axis0_np(data, 50)


# ============================================================================
# Determinism Tests
# ============================================================================

class TestDeterministicBehavior:
    """Verify deterministic results with same seed"""
    
    def test_bootstrap_deterministic_sequential(self):
        """Bootstrap with same seed gives identical results sequentially"""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        results = []
        for _ in range(10):
            ci = _bootstrap_ci(data, np.mean, n_resamples=1000, seed=42)
            results.append(ci)
        
        # All results should be identical
        for r in results[1:]:
            np.testing.assert_array_equal(r, results[0])
    
    def test_permutation_deterministic_sequential(self):
        """Permutation test with same seed gives identical results"""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100) + 0.3
        
        results = []
        for _ in range(10):
            result = _permutation_test(x, y, n_permutations=1000, seed=42)
            results.append(result['pvalue'])
        
        # All p-values should be identical
        for p in results[1:]:
            assert p == results[0]
    
    def test_rolling_stats_deterministic(self):
        """Rolling stats should be deterministic (no randomness)"""
        np.random.seed(42)
        data = np.random.randn(10000, 10)
        
        results = []
        for _ in range(10):
            mean, std = bs.rolling_mean_std_axis0_np(data, 50)
            results.append((mean, std))
        
        # All results should be identical
        for mean, std in results[1:]:
            np.testing.assert_array_equal(mean, results[0][0])
            np.testing.assert_array_equal(std, results[0][1])
    
    def test_inference_deterministic(self):
        """Inference tests should be deterministic"""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100) + 0.3
        
        results = []
        for _ in range(10):
            result = bs.t_test_2samp(x, y, equal_var=True)
            results.append((result['statistic'], result['pvalue']))
        
        # All results should be identical
        for stat, pval in results[1:]:
            assert stat == results[0][0]
            assert pval == results[0][1]
    
    def test_robust_stats_deterministic(self):
        """Robust statistics should be deterministic"""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        results = []
        for _ in range(10):
            median = bs.median(data)
            mad = bs.mad(data)
            iqr = bs.iqr(data)
            results.append((median, mad, iqr))
        
        # All results should be identical
        for m, mad, iqr in results[1:]:
            assert m == results[0][0]
            assert mad == results[0][1]
            assert iqr == results[0][2]


# ============================================================================
# Thread Safety Tests
# ============================================================================

class TestThreadSafety:
    """Test thread-safe concurrent execution"""
    
    def test_bootstrap_thread_safe(self):
        """Bootstrap can be called concurrently from multiple threads"""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        def run_bootstrap(seed):
            return _bootstrap_ci(data, np.mean, n_resamples=1000, seed=seed)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Run with different seeds
            futures = [executor.submit(run_bootstrap, i) for i in range(10)]
            results = [f.result() for f in futures]
        
        # All results should be valid (different seeds = different results)
        for ci in results:
            assert len(ci) == 2
            assert ci[0] < ci[1]
            assert np.all(np.isfinite(ci))
    
    def test_bootstrap_same_seed_thread_safe(self):
        """Bootstrap with same seed from multiple threads gives consistent results"""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        def run_bootstrap():
            return _bootstrap_ci(data, np.mean, n_resamples=1000, seed=42)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Run with same seed
            futures = [executor.submit(run_bootstrap) for _ in range(10)]
            results = [f.result() for f in futures]
        
        # All results should be identical (same seed)
        for r in results[1:]:
            np.testing.assert_array_equal(r, results[0])
    
    def test_rolling_stats_thread_safe(self):
        """Rolling stats can be computed concurrently"""
        np.random.seed(42)
        data = np.random.randn(10000, 10)
        
        def compute_rolling():
            return bs.rolling_mean_std_axis0_np(data, 50)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(compute_rolling) for _ in range(10)]
            results = [f.result() for f in futures]
        
        # All results should be identical
        for mean, std in results[1:]:
            np.testing.assert_array_equal(mean, results[0][0])
            np.testing.assert_array_equal(std, results[0][1])
    
    def test_inference_thread_safe(self):
        """Statistical tests can be run concurrently"""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100) + 0.3
        
        def run_test():
            return bs.t_test_2samp(x, y, equal_var=True)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_test) for _ in range(10)]
            results = [f.result() for f in futures]
        
        # All results should be identical
        for r in results[1:]:
            assert r['statistic'] == results[0]['statistic']
            assert r['pvalue'] == results[0]['pvalue']
    
    def test_robust_stats_thread_safe(self):
        """Robust statistics can be computed concurrently"""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        def compute_robust():
            return (bs.median(data), bs.mad(data), bs.iqr(data))
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(compute_robust) for _ in range(10)]
            results = [f.result() for f in futures]
        
        # All results should be identical
        for m, mad, iqr in results[1:]:
            assert m == results[0][0]
            assert mad == results[0][1]
            assert iqr == results[0][2]


# ============================================================================
# Process Safety Tests
# ============================================================================

class TestProcessSafety:
    """Test multi-process safety"""
    
    def test_bootstrap_process_safe(self):
        """Bootstrap works correctly in multi-process context"""
        np.random.seed(42)
        data = np.random.randn(1000)

        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_proc_bootstrap_worker, data, i) for i in range(4)]
            results = [f.result() for f in futures]
        
        # All results should be valid
        for ci in results:
            assert len(ci) == 2
            assert ci[0] < ci[1]
            assert np.all(np.isfinite(ci))
    
    def test_rolling_stats_process_safe(self):
        """Rolling stats work in multi-process context"""
        np.random.seed(42)
        data = np.random.randn(10000, 10)

        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_proc_rolling_worker, data) for _ in range(4)]
            results = [f.result() for f in futures]
        
        # All results should be identical
        for mean, std in results[1:]:
            np.testing.assert_array_equal(mean, results[0][0])
            np.testing.assert_array_equal(std, results[0][1])


# ============================================================================
# Race Condition Tests
# ============================================================================

class TestNoRaceConditions:
    """Test for race conditions in concurrent operations"""
    
    def test_shared_data_no_corruption(self):
        """Concurrent operations on shared data don't corrupt results"""
        np.random.seed(42)
        shared_data = np.random.randn(10000)
        
        results = []
        lock = threading.Lock()
        
        def compute_stats():
            # Multiple operations on same data
            m = bs.mean(shared_data)
            s = bs.std(shared_data)
            med = bs.median(shared_data)
            
            with lock:
                results.append((m, s, med))
        
        threads = [threading.Thread(target=compute_stats) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All results should be identical
        for m, s, med in results[1:]:
            np.testing.assert_allclose(m, results[0][0], rtol=1e-12)
            np.testing.assert_allclose(s, results[0][1], rtol=1e-12)
            np.testing.assert_allclose(med, results[0][2], rtol=1e-12)
    
    def test_concurrent_bootstrap_different_data(self):
        """Concurrent bootstrap on different datasets"""
        np.random.seed(42)
        datasets = [np.random.randn(1000) for _ in range(4)]
        
        def run_bootstrap(data, seed):
            return _bootstrap_ci(data, np.mean, n_resamples=1000, seed=seed)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(run_bootstrap, data, i)
                for i, data in enumerate(datasets)
            ]
            results = [f.result() for f in futures]
        
        # All results should be valid and different
        for ci in results:
            assert len(ci) == 2
            assert ci[0] < ci[1]
            assert np.all(np.isfinite(ci))


# ============================================================================
# RNG State Isolation Tests
# ============================================================================

class TestRNGStateIsolation:
    """Test that RNG state is properly isolated"""
    
    def test_bootstrap_rng_isolation(self):
        """Bootstrap operations don't interfere with each other's RNG"""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        # Run two bootstrap operations with same seed sequentially
        ci1 = _bootstrap_ci(data, np.mean, n_resamples=1000, seed=42)
        ci2 = _bootstrap_ci(data, np.mean, n_resamples=1000, seed=42)
        
        # Results should be identical
        np.testing.assert_array_equal(ci1, ci2)
        
        # Run interleaved
        results = []
        for _ in range(5):
            ci_a = _bootstrap_ci(data, np.mean, n_resamples=1000, seed=42)
            ci_b = _bootstrap_ci(data, np.mean, n_resamples=1000, seed=42)
            results.append((ci_a, ci_b))
        
        # All results should be identical
        for ci_a, ci_b in results:
            np.testing.assert_array_equal(ci_a, ci1)
            np.testing.assert_array_equal(ci_b, ci1)
    
    def test_permutation_rng_isolation(self):
        """Permutation tests maintain independent RNG state"""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100) + 0.3
        
        # Sequential runs with same seed
        p1 = _permutation_test(x, y, n_permutations=1000, seed=42)['pvalue']
        p2 = _permutation_test(x, y, n_permutations=1000, seed=42)['pvalue']
        
        assert p1 == p2
        
        # Interleaved runs
        results = []
        for _ in range(5):
            p_a = _permutation_test(x, y, n_permutations=1000, seed=42)['pvalue']
            p_b = _permutation_test(x, y, n_permutations=1000, seed=42)['pvalue']
            results.append((p_a, p_b))
        
        # All should match
        for p_a, p_b in results:
            assert p_a == p1
            assert p_b == p1


# ============================================================================
# Stress Tests for Concurrent Access
# ============================================================================

class TestConcurrentStress:
    """Stress tests for concurrent operations"""
    
    @pytest.mark.slow
    def test_high_concurrency_bootstrap(self):
        """Many concurrent bootstrap operations"""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        def run_bootstrap(seed):
            return _bootstrap_ci(data, np.mean, n_resamples=1000, seed=seed)
        
        # Run 100 concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_bootstrap, i) for i in range(100)]
            results = [f.result() for f in futures]
        
        # All should complete successfully
        assert len(results) == 100
        for ci in results:
            assert len(ci) == 2
            assert np.all(np.isfinite(ci))
    
    @pytest.mark.slow
    def test_high_concurrency_rolling_stats(self):
        """Many concurrent rolling stats computations"""
        np.random.seed(42)
        data = np.random.randn(10000, 10)
        
        def compute_rolling(_):
            return bs.rolling_mean_std_axis0_np(data, 50)
        
        # Run 50 concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(compute_rolling, i) for i in range(50)]
            results = [f.result() for f in futures]
        
        # All should be identical
        for mean, std in results[1:]:
            np.testing.assert_array_equal(mean, results[0][0])
            np.testing.assert_array_equal(std, results[0][1])


# ============================================================================
# Data Isolation Tests
# ============================================================================

class TestDataIsolation:
    """Verify operations don't modify input data"""
    
    def test_bootstrap_preserves_input(self):
        """Bootstrap doesn't modify input data"""
        np.random.seed(42)
        data = np.random.randn(1000)
        data_copy = data.copy()
        
        _bootstrap_ci(data, np.mean, n_resamples=1000, seed=42)
        
        np.testing.assert_array_equal(data, data_copy)
    
    def test_rolling_stats_preserves_input(self):
        """Rolling stats don't modify input data"""
        np.random.seed(42)
        data = np.random.randn(10000, 10)
        data_copy = data.copy()
        
        bs.rolling_mean_std_axis0_np(data, 50)
        
        np.testing.assert_array_equal(data, data_copy)
    
    def test_inference_preserves_input(self):
        """Inference tests don't modify input data"""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        x_copy = x.copy()
        y_copy = y.copy()
        
        bs.t_test_2samp(x, y, equal_var=True)
        
        np.testing.assert_array_equal(x, x_copy)
        np.testing.assert_array_equal(y, y_copy)
    
    def test_robust_stats_preserves_input(self):
        """Robust stats don't modify input data"""
        np.random.seed(42)
        data = np.random.randn(1000)
        data_copy = data.copy()
        
        bs.median(data)
        bs.mad(data)
        bs.iqr(data)
        
        np.testing.assert_array_equal(data, data_copy)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
