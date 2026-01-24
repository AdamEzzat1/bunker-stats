"""
Comprehensive tests for bunker-stats v0.2.9 rolling statistics module.

Tests cover:
- Configuration validation (RollingConfig)
- Window alignment (trailing vs centered)
- NaN policies (propagate, ignore, require_min_periods)
- Min periods behavior
- Edge cases (empty arrays, window too large, etc.)
- Multi-stat aggregation
- 2D axis-0 operations
- Backward compatibility
- Numerical accuracy
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal


# Import new v0.2.9 API
try:
    from bunker_stats import Rolling, RollingConfig
    from bunker_stats import rolling_multi, rolling_multi_axis0
    HAS_NEW_API = True
except ImportError:
    HAS_NEW_API = False
    pytest.skip("v0.2.9 rolling API not available", allow_module_level=True)

# Import legacy API for compatibility tests
from bunker_stats import rolling_mean, rolling_std, rolling_var


# ============================================================================
# Configuration Validation Tests
# ============================================================================

class TestRollingConfig:
    """Test RollingConfig validation."""
    
    def test_valid_config(self):
        """Valid configurations should work."""
        cfg = RollingConfig(window=5)
        assert cfg.window == 5
        assert cfg.min_periods is None
        assert cfg.alignment == "trailing"
        assert cfg.nan_policy == "propagate"
    
    def test_window_too_small(self):
        """Window < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="window must be >= 1"):
            RollingConfig(window=0)
    
    def test_min_periods_too_large(self):
        """min_periods > window should raise ValueError."""
        with pytest.raises(ValueError, match="cannot exceed window"):
            RollingConfig(window=5, min_periods=10)
    
    def test_min_periods_zero(self):
        """min_periods = 0 should raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            RollingConfig(window=5, min_periods=0)
    
    def test_invalid_alignment(self):
        """Invalid alignment should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid alignment"):
            Rolling(np.array([1, 2, 3]), window=2, alignment="invalid")
    
    def test_invalid_nan_policy(self):
        """Invalid nan_policy should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid nan_policy"):
            Rolling(np.array([1, 2, 3]), window=2, nan_policy="invalid")


# ============================================================================
# Alignment Tests (Trailing vs Centered)
# ============================================================================

class TestAlignment:
    """Test window alignment behavior."""
    
    def test_trailing_shape(self):
        """Trailing window produces (n - window + 1) output."""
        x = np.arange(10.0)
        
        # Window=3: 10 - 3 + 1 = 8 outputs
        r = Rolling(x, window=3, alignment='trailing')
        result = r.mean()
        assert result.shape == (8,)
        
        # Window=5: 10 - 5 + 1 = 6 outputs
        r = Rolling(x, window=5, alignment='trailing')
        result = r.mean()
        assert result.shape == (6,)
    
    def test_centered_shape(self):
        """Centered window produces (n,) output."""
        x = np.arange(10.0)
        
        # Centered always returns same length as input
        r = Rolling(x, window=3, alignment='centered')
        result = r.mean()
        assert result.shape == (10,)
        
        r = Rolling(x, window=5, alignment='centered')
        result = r.mean()
        assert result.shape == (10,)
    
    def test_trailing_values(self):
        """Trailing window computes correct values."""
        x = np.array([1., 2., 3., 4., 5.])
        r = Rolling(x, window=3, alignment='trailing')
        result = r.mean()
        
        # Window [1,2,3] -> mean = 2.0
        assert_allclose(result[0], 2.0)
        # Window [2,3,4] -> mean = 3.0
        assert_allclose(result[1], 3.0)
        # Window [3,4,5] -> mean = 4.0
        assert_allclose(result[2], 4.0)
    
    def test_centered_values(self):
        """Centered window computes correct values."""
        x = np.array([1., 2., 3., 4., 5.])
        r = Rolling(x, window=3, alignment='centered')
        result = r.mean()
        
        # Position 0: window [1, 2] (truncated left) -> mean = 1.5
        assert_allclose(result[0], 1.5, rtol=1e-10)
        # Position 1: window [1, 2, 3] -> mean = 2.0
        assert_allclose(result[1], 2.0, rtol=1e-10)
        # Position 2: window [2, 3, 4] (centered at 2) -> mean = 3.0
        # For centered window=3 at position 2: half=1, so [2-1, 2+2) = [1,4) = indices 1,2,3 = values [2,3,4]
        assert_allclose(result[2], 3.0, rtol=1e-10)
    
    def test_centered_edge_truncation(self):
        """Centered window truncates correctly at edges."""
        x = np.arange(10.0)  # [0, 1, 2, ..., 9]
        r = Rolling(x, window=5, alignment='centered')
        result = r.mean()
        
        # half = 5 // 2 = 2
        # Position 0: [0, 3) = [0, 1, 2] -> mean = 1.0
        assert_allclose(result[0], 1.0, rtol=1e-10)
        
        # Position 9: [7, 10) = [7, 8, 9] -> mean = 8.0
        assert_allclose(result[9], 8.0, rtol=1e-10)


# ============================================================================
# NaN Policy Tests
# ============================================================================

class TestNanPolicy:
    """Test NaN handling policies."""
    
    def test_propagate_any_nan_produces_nan(self):
        """Propagate policy: any NaN in window -> result NaN."""
        x = np.array([1., 2., np.nan, 4., 5.])
        r = Rolling(x, window=3, nan_policy='propagate')
        result = r.mean()
        
        # Window [1, 2, NaN] -> NaN
        assert np.isnan(result[0])
        # Window [2, NaN, 4] -> NaN
        assert np.isnan(result[1])
        # Window [NaN, 4, 5] -> NaN
        assert np.isnan(result[2])
    
    def test_propagate_no_nan_produces_values(self):
        """Propagate policy: no NaN -> normal values."""
        x = np.array([1., 2., 3., 4., 5.])
        r = Rolling(x, window=3, nan_policy='propagate')
        result = r.mean()
        
        assert_allclose(result[0], 2.0)
        assert_allclose(result[1], 3.0)
        assert_allclose(result[2], 4.0)
    
    def test_ignore_skips_nans(self):
        """Ignore policy: skip NaNs, use valid values."""
        x = np.array([1., 2., np.nan, 4., 5.])
        r = Rolling(x, window=3, min_periods=2, nan_policy='ignore')
        result = r.mean()
        
        # Window [1, 2, NaN]: valid = [1, 2] (count=2 >= min_periods=2) -> mean = 1.5
        assert_allclose(result[0], 1.5, rtol=1e-10)
        
        # Window [2, NaN, 4]: valid = [2, 4] (count=2 >= min_periods=2) -> mean = 3.0
        assert_allclose(result[1], 3.0, rtol=1e-10)
        
        # Window [NaN, 4, 5]: valid = [4, 5] (count=2 >= min_periods=2) -> mean = 4.5
        assert_allclose(result[2], 4.5, rtol=1e-10)
    
    def test_ignore_insufficient_valid_produces_nan(self):
        """Ignore policy: valid_count < min_periods -> NaN."""
        x = np.array([1., np.nan, np.nan, 4., 5.])
        r = Rolling(x, window=3, min_periods=3, nan_policy='ignore')
        result = r.mean()
        
        # Window [1, NaN, NaN]: valid_count = 1 < 3 -> NaN
        assert np.isnan(result[0])
        
        # Window [NaN, NaN, 4]: valid_count = 1 < 3 -> NaN
        assert np.isnan(result[1])
        
        # Window [NaN, 4, 5]: valid_count = 2 < 3 -> NaN
        assert np.isnan(result[2])
    
    def test_require_min_periods_same_as_ignore(self):
        """RequireMinPeriods policy is functionally same as Ignore."""
        x = np.array([1., 2., np.nan, 4., 5.])
        
        r_ignore = Rolling(x, window=3, min_periods=2, nan_policy='ignore')
        r_require = Rolling(x, window=3, min_periods=2, nan_policy='require_min_periods')
        
        result_ignore = r_ignore.mean()
        result_require = r_require.mean()
        
        assert_allclose(result_ignore, result_require)


# ============================================================================
# Min Periods Tests
# ============================================================================

class TestMinPeriods:
    """Test min_periods behavior."""
    
    def test_min_periods_default_equals_window(self):
        """Default min_periods should equal window (strict)."""
        x = np.array([1., 2., 3., 4., 5.])
        
        # Without min_periods specified, should default to window
        r = Rolling(x, window=3, nan_policy='propagate')
        result = r.mean()
        
        # All windows have 3 values, should work
        assert not np.any(np.isnan(result))
    
    def test_min_periods_with_centered_edge_truncation(self):
        """Centered window with min_periods handles edge truncation."""
        x = np.array([1., 2., 3., 4., 5.])
        
        # Window=5, centered -> edges have fewer values
        # With min_periods=5, edges should be NaN
        r = Rolling(x, window=5, min_periods=5, alignment='centered')
        result = r.mean()
        
        # Position 0: actual window [1, 2, 3] (3 values) < 5 -> NaN
        assert np.isnan(result[0])
        
        # Position 2: actual window [1, 2, 3, 4, 5] (5 values) = 5 -> mean = 3.0
        assert_allclose(result[2], 3.0, rtol=1e-10)
        
        # Position 4: actual window [3, 4, 5] (3 values) < 5 -> NaN
        assert np.isnan(result[4])
    
    def test_min_periods_allows_partial_windows(self):
        """min_periods < window allows partial windows."""
        x = np.array([1., 2., 3., 4., 5.])
        
        # Window=5, min_periods=3 with centered alignment
        r = Rolling(x, window=5, min_periods=3, alignment='centered')
        result = r.mean()
        
        # Position 0: [1, 2, 3] (3 values) >= 3 -> valid
        assert not np.isnan(result[0])
        
        # Position 4: [3, 4, 5] (3 values) >= 3 -> valid
        assert not np.isnan(result[4])


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_array(self):
        """Empty array should return empty result."""
        x = np.array([])
        r = Rolling(x, window=3)
        result = r.mean()
        assert result.shape == (0,)
    
    def test_window_too_large_trailing(self):
        """Window larger than array (trailing) returns empty."""
        x = np.array([1., 2., 3.])
        r = Rolling(x, window=5, alignment='trailing')
        result = r.mean()
        assert result.shape == (0,)
    
    def test_window_too_large_centered(self):
        """Window larger than array (centered) still returns n outputs."""
        x = np.array([1., 2., 3.])
        r = Rolling(x, window=5, alignment='centered')
        result = r.mean()
        # Centered always returns n outputs
        assert result.shape == (3,)
    
    def test_window_equals_length(self):
        """Window exactly equals array length."""
        x = np.array([1., 2., 3., 4., 5.])
        r = Rolling(x, window=5, alignment='trailing')
        result = r.mean()
        
        # Should return single value
        assert result.shape == (1,)
        assert_allclose(result[0], 3.0)
    
    def test_window_one(self):
        """Window=1 should return original values."""
        x = np.array([1., 2., 3., 4., 5.])
        r = Rolling(x, window=1, alignment='trailing')
        result = r.mean()
        
        assert_allclose(result, x)
    
    def test_all_nans_propagate(self):
        """All NaNs with propagate policy."""
        x = np.array([np.nan, np.nan, np.nan, np.nan])
        r = Rolling(x, window=2, nan_policy='propagate')
        result = r.mean()
        
        assert np.all(np.isnan(result))
    
    def test_all_nans_ignore(self):
        """All NaNs with ignore policy."""
        x = np.array([np.nan, np.nan, np.nan, np.nan])
        r = Rolling(x, window=2, min_periods=1, nan_policy='ignore')
        result = r.mean()
        
        # No valid values -> all NaN
        assert np.all(np.isnan(result))
    
    def test_single_value(self):
        """Single value array."""
        x = np.array([42.0])
        r = Rolling(x, window=1)
        result = r.mean()
        
        assert result.shape == (1,)
        assert_allclose(result[0], 42.0)
    
    def test_constant_array(self):
        """Array with all same values."""
        x = np.full(10, 5.0)
        r = Rolling(x, window=3)
        
        mean_result = r.mean()
        std_result = r.std()
        
        # Mean should be constant
        assert_allclose(mean_result, 5.0)
        
        # Std should be 0 (or very close due to numerical error)
        assert_allclose(std_result, 0.0, atol=1e-10)


# ============================================================================
# Multi-Stat Aggregation Tests
# ============================================================================

class TestMultiStatAggregation:
    """Test efficient multi-statistic computation."""
    
    def test_aggregate_single_stat(self):
        """Aggregate with single stat works."""
        x = np.arange(10.0)
        r = Rolling(x, window=3)
        result = r.aggregate("mean")
        
        assert "mean" in result
        assert result["mean"].shape == (8,)
    
    def test_aggregate_multiple_stats(self):
        """Aggregate with multiple stats returns all."""
        x = np.arange(10.0)
        r = Rolling(x, window=3)
        result = r.aggregate("mean", "std", "var")
        
        assert set(result.keys()) == {"mean", "std", "var"}
        assert all(v.shape == (8,) for v in result.values())
    
    def test_aggregate_all_stats(self):
        """Aggregate with all available stats."""
        x = np.arange(10.0)
        r = Rolling(x, window=3)
        result = r.aggregate("mean", "std", "var", "count", "min", "max")
        
        assert set(result.keys()) == {"mean", "std", "var", "count", "min", "max"}
    
    def test_aggregate_empty_raises(self):
        """Aggregate with no stats raises error."""
        x = np.arange(10.0)
        r = Rolling(x, window=3)
        
        with pytest.raises(ValueError, match="At least one statistic"):
            r.aggregate()
    
    def test_var_equals_std_squared(self):
        """Variance should equal std squared."""
        x = np.random.randn(20)
        r = Rolling(x, window=5)
        result = r.aggregate("std", "var")
        
        assert_allclose(result["var"], result["std"] ** 2)
    
    def test_count_correct(self):
        """Count statistic counts valid values."""
        x = np.array([1., 2., np.nan, 4., 5., np.nan, 7.])
        r = Rolling(x, window=3, nan_policy='ignore')
        result = r.aggregate("count")
        
        # Window [1, 2, NaN]: count = 2
        assert_allclose(result["count"][0], 2.0)
        
        # Window [2, NaN, 4]: count = 2
        assert_allclose(result["count"][1], 2.0)
    
    def test_min_max_correct(self):
        """Min and max statistics are correct."""
        x = np.array([5., 2., 8., 1., 9.])
        r = Rolling(x, window=3)
        result = r.aggregate("min", "max")
        
        # Window [5, 2, 8]: min=2, max=8
        assert_allclose(result["min"][0], 2.0)
        assert_allclose(result["max"][0], 8.0)
        
        # Window [2, 8, 1]: min=1, max=8
        assert_allclose(result["min"][1], 1.0)
        assert_allclose(result["max"][1], 8.0)
        
        # Window [8, 1, 9]: min=1, max=9
        assert_allclose(result["min"][2], 1.0)
        assert_allclose(result["max"][2], 9.0)


# ============================================================================
# 2D Axis-0 Tests
# ============================================================================

class TestAxis0Operations:
    """Test 2D column-wise rolling operations."""
    
    def test_2d_basic(self):
        """Basic 2D axis-0 rolling."""
        # 5x3 matrix
        x = np.array([
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
            [10., 11., 12.],
            [13., 14., 15.],
        ])
        
        r = Rolling(x, window=3, alignment='trailing')
        result = r.mean()
        
        # Output should be (5-3+1) x 3 = 3x3
        assert result.shape == (3, 3)
        
        # First row: mean of rows 0,1,2
        # Col 0: mean([1, 4, 7]) = 4.0
        assert_allclose(result[0, 0], 4.0)
        # Col 1: mean([2, 5, 8]) = 5.0
        assert_allclose(result[0, 1], 5.0)
        # Col 2: mean([3, 6, 9]) = 6.0
        assert_allclose(result[0, 2], 6.0)
    
    def test_2d_centered(self):
        """2D centered window."""
        x = np.array([
            [1., 10.],
            [2., 20.],
            [3., 30.],
            [4., 40.],
            [5., 50.],
        ])
        
        r = Rolling(x, window=3, alignment='centered')
        result = r.mean()
        
        # Centered returns same shape
        assert result.shape == (5, 2)
    
    def test_2d_multi_stat(self):
        """2D with multiple statistics."""
        x = np.random.randn(10, 4)
        r = Rolling(x, window=5)
        result = r.aggregate("mean", "std")
        
        # Both should have shape (10-5+1, 4) = (6, 4)
        assert result["mean"].shape == (6, 4)
        assert result["std"].shape == (6, 4)
    
    def test_2d_nan_handling(self):
        """2D with NaN values."""
        x = np.array([
            [1., np.nan],
            [2., 2.],
            [3., 3.],
        ])
        
        r = Rolling(x, window=2, nan_policy='propagate')
        result = r.mean()
        
        # Col 0: all valid
        assert not np.any(np.isnan(result[:, 0]))
        
        # Col 1: first window has NaN
        assert np.isnan(result[0, 1])
        assert not np.isnan(result[1, 1])


# ============================================================================
# Backward Compatibility Tests
# ============================================================================

class TestBackwardCompatibility:
    """Test that new API is compatible with legacy API."""
    
    def test_rolling_class_equals_legacy_mean(self):
        """Rolling.mean() equals rolling_mean()."""
        x = np.random.randn(20)
        window = 5
        
        # New API
        r = Rolling(x, window=window, alignment='trailing', nan_policy='propagate')
        new_result = r.mean()
        
        # Legacy API
        legacy_result = rolling_mean(x, window)
        
        assert_allclose(new_result, legacy_result)
    
    def test_rolling_class_equals_legacy_std(self):
        """Rolling.std() equals rolling_std()."""
        x = np.random.randn(20)
        window = 5
        
        # New API
        r = Rolling(x, window=window, alignment='trailing', nan_policy='propagate')
        new_result = r.std()
        
        # Legacy API
        legacy_result = rolling_std(x, window)
        
        assert_allclose(new_result, legacy_result)
    
    def test_rolling_class_equals_legacy_var(self):
        """Rolling.var() equals rolling_var()."""
        x = np.random.randn(20)
        window = 5
        
        # New API
        r = Rolling(x, window=window, alignment='trailing', nan_policy='propagate')
        new_result = r.var()
        
        # Legacy API
        legacy_result = rolling_var(x, window)
        
        assert_allclose(new_result, legacy_result)
    
    def test_legacy_functions_still_work(self):
        """Legacy functions are not broken."""
        x = np.arange(10.0)
        
        # These should not raise
        mean_result = rolling_mean(x, 3)
        std_result = rolling_std(x, 3)
        var_result = rolling_var(x, 3)
        
        assert mean_result.shape == (8,)
        assert std_result.shape == (8,)
        assert var_result.shape == (8,)


# ============================================================================
# Numerical Accuracy Tests
# ============================================================================

class TestNumericalAccuracy:
    """Test numerical accuracy and stability."""
    
    def test_kahan_summation_accuracy(self):
        """Test that Kahan summation prevents catastrophic cancellation."""
        # Large + small values that would lose precision with naive sum
        x = np.array([1e10, 1.0, -1e10, 1.0, 1.0])
        
        r = Rolling(x, window=5)
        result = r.mean()
        
        # Should be close to 3.0 / 5 = 0.6, not 0.0
        assert_allclose(result[0], 0.6, rtol=1e-9)
    
    def test_variance_never_negative(self):
        """Variance should never be negative (due to FP error clamping)."""
        # Use values that might cause slight negative variance from FP error
        x = np.full(20, 5.0)
        x[10] = 5.0 + 1e-15  # Tiny perturbation
        
        r = Rolling(x, window=5)
        result = r.var()
        
        # All variance values should be >= 0
        assert np.all(result >= 0)
    
    def test_large_values(self):
        """Test with large values."""
        x = np.random.randn(20) * 1e10
        r = Rolling(x, window=5)
        result = r.aggregate("mean", "std")
        
        # Should not overflow or produce NaN/Inf
        assert np.all(np.isfinite(result["mean"]))
        assert np.all(np.isfinite(result["std"]))
    
    def test_small_values(self):
        """Test with small values."""
        x = np.random.randn(20) * 1e-10
        r = Rolling(x, window=5)
        result = r.aggregate("mean", "std")
        
        # Should not underflow to zero
        assert not np.all(result["mean"] == 0)


# ============================================================================
# Low-Level API Tests
# ============================================================================

class TestLowLevelAPI:
    """Test low-level rolling_multi and rolling_multi_axis0 functions."""
    
    def test_rolling_multi_single_stat(self):
        """rolling_multi with single stat."""
        x = np.arange(10.0)
        
        result = rolling_multi(x, window=3, stats=["mean"])
        
        # Should return tuple with one array
        assert len(result) == 1
        assert result[0].shape == (8,)
    
    def test_rolling_multi_multiple_stats(self):
        """rolling_multi with multiple stats."""
        x = np.arange(10.0)
        
        result = rolling_multi(x, window=3, stats=["mean", "std", "var"])
        
        # Should return tuple with three arrays
        assert len(result) == 3
        assert all(arr.shape == (8,) for arr in result)
    
    def test_rolling_multi_centered(self):
        """rolling_multi with centered alignment."""
        x = np.arange(10.0)
        
        result = rolling_multi(x, window=3, alignment="centered", stats=["mean"])
        
        assert result[0].shape == (10,)
    
    def test_rolling_multi_axis0_basic(self):
        """rolling_multi_axis0 basic test."""
        x = np.random.randn(10, 3)
        
        result = rolling_multi_axis0(x, window=5, stats=["mean", "std"])
        
        # Should return tuple with two 2D arrays
        assert len(result) == 2
        assert result[0].shape == (6, 3)
        assert result[1].shape == (6, 3)


# ============================================================================
# Determinism Tests
# ============================================================================

class TestDeterminism:
    """Test that operations are deterministic."""
    
    def test_repeated_calls_identical(self):
        """Multiple calls with same input produce identical results."""
        x = np.random.randn(100)
        r = Rolling(x, window=10)
        
        result1 = r.mean()
        result2 = r.mean()
        result3 = r.mean()
        
        # Should be bit-identical
        assert_array_equal(result1, result2)
        assert_array_equal(result1, result3)
    
    def test_aggregate_deterministic(self):
        """Aggregate calls are deterministic."""
        x = np.random.randn(100)
        r = Rolling(x, window=10)
        
        result1 = r.aggregate("mean", "std", "var")
        result2 = r.aggregate("mean", "std", "var")
        
        for key in result1.keys():
            assert_array_equal(result1[key], result2[key])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])