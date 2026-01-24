"""
Test suite for robust statistics module

These tests verify:
1. Numerical accuracy against SciPy
2. Edge case handling (empty, NaN, single value)
3. Performance characteristics
4. Mathematical properties
"""

import pytest
import numpy as np
from scipy import stats
import bunker_stats as bs

EPSILON = 1e-10


class TestMedian:
    """Test median function"""
    
    def test_basic_odd(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bs.median(data)
        expected = np.median(data)
        assert abs(result - expected) < EPSILON
    
    def test_basic_even(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = bs.median(data)
        expected = np.median(data)
        assert abs(result - expected) < EPSILON
    
    def test_unsorted(self):
        data = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        result = bs.median(data)
        expected = np.median(data)
        assert abs(result - expected) < EPSILON
    
    def test_empty(self):
        data = np.array([])
        result = bs.median(data)
        assert np.isnan(result)
    
    def test_single(self):
        data = np.array([42.0])
        result = bs.median(data)
        assert abs(result - 42.0) < EPSILON
    
    def test_with_outliers(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 1000.0])
        result = bs.median(data)
        expected = np.median(data)
        assert abs(result - expected) < EPSILON
        # Median should be robust to the outlier
        assert abs(result - 3.0) < EPSILON


class TestMAD:
    """Test Median Absolute Deviation"""
    
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bs.mad(data)
        # Median is 3, deviations are [2, 1, 0, 1, 2], MAD is 1
        assert abs(result - 1.0) < EPSILON
    
    def test_constant(self):
        data = np.array([5.0, 5.0, 5.0, 5.0])
        result = bs.mad(data)
        assert abs(result - 0.0) < EPSILON
    
    def test_scipy_parity(self):
        """Compare with SciPy's median_abs_deviation"""
        data = np.random.randn(100)
        result = bs.mad(data)
        expected = stats.median_abs_deviation(data, scale=1.0)
        assert abs(result - expected) < EPSILON
    
    def test_empty(self):
        data = np.array([])
        result = bs.mad(data)
        assert np.isnan(result)


class TestMADStd:
    """Test MAD scaled to standard deviation"""
    
    def test_normal_consistency(self):
        """For normal data, MAD*1.4826 should approximate std"""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        mad_std = bs.mad_std(data)
        actual_std = np.std(data, ddof=1)
        
        # Should be within 10% for large normal sample
        assert abs(mad_std - actual_std) / actual_std < 0.1
    
    def test_scipy_parity(self):
        """Compare with SciPy's default MAD (includes scale factor)"""
        data = np.random.randn(100)
        result = bs.mad_std(data)
        expected = stats.median_abs_deviation(data, scale='normal')
        # Note: SciPy uses slightly different median algorithm, so tolerance is relaxed
        assert abs(result - expected) < 1e-6


class TestTrimmedMean:
    """Test trimmed mean"""
    
    def test_no_trim(self):
        """With 0% trim, should equal regular mean"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bs.trimmed_mean(data, 0.0)
        expected = np.mean(data)
        assert abs(result - expected) < EPSILON
    
    def test_10_percent_trim(self):
        """Test 10% trim on each end"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = bs.trimmed_mean(data, 0.1)
        # Removes 1 and 10, mean of [2..9] = 5.5
        assert abs(result - 5.5) < EPSILON
    
    def test_scipy_parity(self):
        """Compare with SciPy's trim_mean"""
        data = np.random.randn(100)
        proportion = 0.1
        
        result = bs.trimmed_mean(data, proportion)
        expected = stats.trim_mean(data, proportion)
        assert abs(result - expected) < EPSILON
    
    def test_outlier_resistance(self):
        """Trimmed mean should be robust to outliers"""
        clean_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        outlier_data = np.array([1.0, 2.0, 3.0, 4.0, 1000.0])
        
        clean_result = bs.trimmed_mean(clean_data, 0.2)
        outlier_result = bs.trimmed_mean(outlier_data, 0.2)
        
        # Both should trim the extreme value
        assert abs(clean_result - outlier_result) < 1.0
    
    def test_invalid_proportion(self):
        """Test that over-trimming returns NaN"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bs.trimmed_mean(data, 0.5)  # 50% from each end = no data left
        assert np.isnan(result)


class TestTrimmedStd:
    """Test trimmed standard deviation"""
    
    def test_no_trim(self):
        """With 0% trim, should approximate regular std"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bs.trimmed_std(data, 0.0)
        expected = np.std(data, ddof=1)
        assert abs(result - expected) < EPSILON
    
    def test_outlier_resistance(self):
        """Trimmed std should be more robust than regular std"""
        clean_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        outlier_data = np.array([1.0, 2.0, 3.0, 4.0, 1000.0])
        
        clean_std = np.std(clean_data, ddof=1)
        outlier_std = np.std(outlier_data, ddof=1)
        
        trimmed_clean = bs.trimmed_std(clean_data, 0.2)
        trimmed_outlier = bs.trimmed_std(outlier_data, 0.2)
        
        # Regular std is heavily affected by outlier
        assert outlier_std > 10 * clean_std
        
        # Trimmed std is resistant
        assert abs(trimmed_clean - trimmed_outlier) < 1.0


class TestWinsorizedMean:
    """Test Winsorized mean"""
    
    def test_basic(self):
        """Test basic Winsorization"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        result = bs.winsorized_mean(data, 10.0, 90.0)
        
        # Should clip 100 to something reasonable
        assert result < 20.0
        assert result > np.mean(data[:4])
    
    def test_no_winsorization(self):
        """With extreme percentiles, should approximate regular mean"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bs.winsorized_mean(data, 0.0, 100.0)
        expected = np.mean(data)
        assert abs(result - expected) < EPSILON


class TestIQR:
    """Test Interquartile Range"""
    
    def test_basic(self):
        """Test IQR calculation"""
        data = np.linspace(0, 100, 101)
        result = bs.iqr(data)
        expected = stats.iqr(data)
        # Allow some tolerance due to percentile interpolation differences
        assert abs(result - expected) < 1.0
    
    def test_scipy_parity(self):
        """Compare with SciPy on random data"""
        np.random.seed(42)
        data = np.random.randn(100)
        result = bs.iqr(data)
        expected = stats.iqr(data)
        # Note: Tolerance relaxed due to percentile interpolation method differences
        assert abs(result - expected) < 0.1
    
    def test_small_data(self):
        """Test with minimal data"""
        data = np.array([1.0])
        result = bs.iqr(data)
        assert np.isnan(result)


class TestBiweightMidvariance:
    """Test Biweight midvariance"""
    
    def test_basic(self):
        """Test basic functionality"""
        np.random.seed(42)
        data = np.random.randn(100)
        result = bs.biweight_midvariance(data)
        
        # Should be finite and positive
        assert np.isfinite(result)
        assert result > 0
    
    def test_tuning_constant(self):
        """Test different tuning constants"""
        data = np.random.randn(100)
        
        result1 = bs.biweight_midvariance(data, c=6.0)
        result2 = bs.biweight_midvariance(data, c=9.0)
        
        # Both should be finite
        assert np.isfinite(result1)
        assert np.isfinite(result2)


class TestQnScale:
    """Test Qn scale estimator"""
    
    def test_basic(self):
        """Test basic functionality"""
        np.random.seed(42)
        data = np.random.randn(100)
        result = bs.qn_scale(data)
        
        # Should approximate std for normal data
        actual_std = np.std(data, ddof=1)
        assert abs(result - actual_std) / actual_std < 0.3
    
    def test_small_sample(self):
        """Test with small sample"""
        data = np.array([1.0, 2.0, 3.0])
        result = bs.qn_scale(data)
        assert np.isfinite(result)
    
    def test_constant_data(self):
        """Constant data should give near-zero scale"""
        data = np.array([5.0] * 10)
        result = bs.qn_scale(data)
        # May not be exactly 0 due to numerical precision
        assert result < 0.1


class TestHuberLocation:
    """Test Huber M-estimator"""
    
    def test_normal_data(self):
        """On normal data, should approximate mean"""
        np.random.seed(42)
        data = np.random.randn(100)
        result = bs.huber_location(data)
        expected = np.mean(data)
        
        # Should be close to mean for clean data
        assert abs(result - expected) < 0.2
    
    def test_outlier_resistance(self):
        """Should be robust to outliers"""
        clean_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        outlier_data = np.array([1.0, 2.0, 3.0, 4.0, 1000.0])
        
        clean_result = bs.huber_location(clean_data)
        outlier_result = bs.huber_location(outlier_data)
        
        # Should be much closer than the means
        mean_diff = abs(np.mean(clean_data) - np.mean(outlier_data))
        huber_diff = abs(clean_result - outlier_result)
        
        assert huber_diff < mean_diff / 10


class TestSkipNAVariants:
    """Test NaN-aware variants"""
    
    def test_median_skipna(self):
        """Test median with NaN values"""
        data = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = bs.median_skipna(data)
        expected = np.nanmedian(data)
        assert abs(result - expected) < EPSILON
    
    def test_mad_skipna(self):
        """Test MAD with NaN values"""
        data = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = bs.mad_skipna(data)
        assert np.isfinite(result)
    
    def test_trimmed_mean_skipna(self):
        """Test trimmed mean with NaN values"""
        data = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = bs.trimmed_mean_skipna(data, 0.1)
        assert np.isfinite(result)
    
    def test_all_nan(self):
        """Test with all NaN values"""
        data = np.array([np.nan, np.nan, np.nan])
        
        assert np.isnan(bs.median_skipna(data))
        assert np.isnan(bs.mad_skipna(data))
        assert np.isnan(bs.trimmed_mean_skipna(data, 0.1))


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_array(self):
        """All functions should handle empty arrays"""
        data = np.array([])
        
        assert np.isnan(bs.median(data))
        assert np.isnan(bs.mad(data))
        assert np.isnan(bs.trimmed_mean(data, 0.1))
        assert np.isnan(bs.trimmed_std(data, 0.1))
        assert np.isnan(bs.iqr(data))
    
    def test_single_value(self):
        """Test with single-value arrays"""
        data = np.array([42.0])
        
        assert abs(bs.median(data) - 42.0) < EPSILON
        assert abs(bs.mad(data) - 0.0) < EPSILON
        assert abs(bs.trimmed_mean(data, 0.0) - 42.0) < EPSILON
    
    def test_two_values(self):
        """Test with two values"""
        data = np.array([1.0, 5.0])
        
        assert abs(bs.median(data) - 3.0) < EPSILON
        assert abs(bs.mad(data) - 2.0) < EPSILON


class TestPerformance:
    """Performance benchmarks (not strict tests, just monitoring)"""
    
    def test_median_performance(self, benchmark_if_available=False):
        """Median should be O(n log n)"""
        np.random.seed(42)
        data = np.random.randn(10000)
        
        # Just verify it completes in reasonable time
        result = bs.median(data)
        assert np.isfinite(result)
    
    def test_mad_performance(self, benchmark_if_available=False):
        """MAD should be O(n log n) with only 2 sorts"""
        np.random.seed(42)
        data = np.random.randn(10000)
        
        result = bs.mad(data)
        assert np.isfinite(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
