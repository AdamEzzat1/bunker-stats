"""
Comprehensive tests for robust statistics v0.2.9

Tests:
- Policy dispatch matches legacy functions
- RobustStats class interface
- Determinism and repeatability
- Edge cases and error handling
- Huber estimator convergence
"""

import numpy as np
import pytest
from bunker_stats import (
    RobustStats,
    robust_fit,
    robust_score,
    median,
    mad,
    trimmed_mean,
    iqr,
    rolling_median,
)


class TestPolicyDispatch:
    """Verify policy-driven API matches legacy functions"""

    def test_median_mad_policy_matches_legacy(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Legacy
        legacy_loc = median(data)
        legacy_scale = mad(data)
        
        # Policy
        loc, scale = robust_fit(data, location="median", scale="mad", mad_consistent=False)
        
        assert loc == legacy_loc
        assert scale == legacy_scale

    def test_trimmed_mean_policy_matches_legacy(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        trim = 0.1
        
        legacy = trimmed_mean(data, trim)
        loc, _ = robust_fit(data, location="trimmed_mean", scale="mad", trim=trim)
        
        assert abs(loc - legacy) < 1e-12

    def test_iqr_policy_matches_legacy(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        legacy = iqr(data)
        _, scale = robust_fit(data, location="median", scale="iqr")
        
        assert abs(scale - legacy) < 1e-10


class TestRobustStatsClass:
    """Test RobustStats class interface"""

    def test_default_construction(self):
        rs = RobustStats()
        assert "median" in repr(rs)
        assert "mad" in repr(rs)

    def test_fit_returns_tuple(self):
        rs = RobustStats(location="median", scale="mad")
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        loc, scale = rs.fit(data)
        
        assert isinstance(loc, float)
        assert isinstance(scale, float)
        assert loc == 3.0
        assert abs(scale - 1.4826) < 1e-4  # MAD with consistency

    def test_score_returns_array(self):
        rs = RobustStats(location="median", scale="mad", mad_consistent=False)
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        
        scores = rs.score(data)
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(data)
        # Median=2, MAD=1, scores should be [-2, -1, 0, 1, 2]
        np.testing.assert_array_almost_equal(scores, [-2, -1, 0, 1, 2], decimal=10)

    def test_trimmed_mean_with_trim_param(self):
        rs = RobustStats(location="trimmed_mean", scale="iqr", trim=0.2)
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        
        loc, scale = rs.fit(data)
        
        # 20% trim removes 2 from each end, mean of [3,4,5,6,7,8] = 5.5
        assert abs(loc - 5.5) < 1e-10
        assert scale > 0

    def test_huber_estimator(self):
        rs = RobustStats(location="huber", scale="mad", c=1.345, max_iter=50)
        data = np.array([1, 2, 3, 4, 5, 100], dtype=float)  # outlier
        
        loc, scale = rs.fit(data)
        
        # Huber should be more robust to outlier than mean
        assert loc < 20  # Much less than mean=19.17
        assert loc > 2   # More than median=3.5
        assert scale > 0


class TestDeterminism:
    """Verify deterministic behavior and repeatability"""

    def test_fit_is_deterministic(self):
        data = np.random.RandomState(42).randn(100)
        rs = RobustStats(location="median", scale="qn")
        
        loc1, scale1 = rs.fit(data)
        loc2, scale2 = rs.fit(data)
        
        assert loc1 == loc2
        assert scale1 == scale2

    def test_score_is_deterministic(self):
        data = np.random.RandomState(42).randn(50)
        rs = RobustStats(location="trimmed_mean", scale="mad", trim=0.1)
        
        scores1 = rs.score(data)
        scores2 = rs.score(data)
        
        np.testing.assert_array_equal(scores1, scores2)

    def test_huber_is_deterministic(self):
        data = np.array([3.1, 1.4, 5.9, 2.6, 5.3, 8.9, 7.9])
        rs = RobustStats(
            location="huber",
            scale="mad",
            c=1.345,
            max_iter=100,
            tol=1e-8
        )
        
        results = [rs.fit(data) for _ in range(5)]
        
        # All runs should produce identical results
        for i in range(1, 5):
            assert results[i][0] == results[0][0]
            assert results[i][1] == results[0][1]

    def test_repeatability_within_tolerance(self):
        """Verify repeated calls are within numerical tolerance"""
        data = np.random.RandomState(123).randn(200)
        rs = RobustStats(location="huber", scale="qn", c=1.5)
        
        locs = [rs.fit(data)[0] for _ in range(10)]
        
        # All estimates within 1e-12 of each other
        assert np.std(locs) < 1e-12


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_array_returns_nan(self):
        rs = RobustStats()
        data = np.array([])
        
        loc, scale = rs.fit(data)
        
        assert np.isnan(loc)
        assert np.isnan(scale)

    def test_empty_array_score_returns_nan(self):
        rs = RobustStats()
        data = np.array([])
        
        scores = rs.score(data)
        
        assert len(scores) == 1
        assert np.isnan(scores[0])

    def test_invalid_trim_returns_nan(self):
        rs = RobustStats(location="trimmed_mean", trim=0.5)  # >= 0.5 invalid
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        
        loc, _ = rs.fit(data)
        
        assert np.isnan(loc)

    def test_over_trimming_returns_nan(self):
        # CORRECTED: Use n=5, trim=0.5 to actually over-trim
        # With n=5, trim=0.5: cut = floor(5*0.5) = 2, cut*2 = 4
        # But trim >= 0.5 should fail validation
        rs = RobustStats(location="trimmed_mean", trim=0.5)
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        
        loc, _ = rs.fit(data)
        
        assert np.isnan(loc)

    def test_single_value_array(self):
        rs = RobustStats()
        data = np.array([42.0])
        
        loc, scale = rs.fit(data)
        
        assert loc == 42.0
        # Scale might be NaN (IQR needs 2+ points)

    def test_zero_scale_score_returns_nan(self):
        rs = RobustStats()
        data = np.array([5.0, 5.0, 5.0, 5.0])  # no variation
        
        scores = rs.score(data)
        
        assert all(np.isnan(scores))

    def test_invalid_location_raises(self):
        with pytest.raises(ValueError):
            RobustStats(location="invalid")

    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError):
            RobustStats(scale="stddev")  # not implemented


class TestHuberConvergence:
    """Test Huber estimator behavior with different parameters"""

    def test_huber_c_affects_result(self):
        data = np.array([1, 2, 3, 4, 5, 100], dtype=float)
        
        rs_conservative = RobustStats(location="huber", c=1.0, max_iter=100)
        rs_liberal = RobustStats(location="huber", c=2.0, max_iter=100)
        
        loc_cons, _ = rs_conservative.fit(data)
        loc_lib, _ = rs_liberal.fit(data)
        
        # Different c should give different results
        assert loc_cons != loc_lib
        # Both should be finite
        assert np.isfinite(loc_cons)
        assert np.isfinite(loc_lib)

    def test_huber_max_iter_sufficient(self):
        data = np.random.RandomState(999).randn(100)
        
        rs_few = RobustStats(location="huber", max_iter=5)
        rs_many = RobustStats(location="huber", max_iter=200)
        
        loc_few, _ = rs_few.fit(data)
        loc_many, _ = rs_many.fit(data)
        
        # Should converge to similar values
        assert abs(loc_few - loc_many) < 0.1

    def test_huber_scale_policy_matters(self):
        data = np.array([1, 2, 3, 4, 5, 50], dtype=float)
        
        # Huber with MAD scale
        rs_mad = RobustStats(location="huber", scale="mad")
        # Huber with IQR scale
        rs_iqr = RobustStats(location="huber", scale="iqr")
        
        loc_mad, _ = rs_mad.fit(data)
        loc_iqr, _ = rs_iqr.fit(data)
        
        # Scale policy used internally affects Huber result
        # Both should be finite though
        assert np.isfinite(loc_mad)
        assert np.isfinite(loc_iqr)


class TestModuleFunctions:
    """Test module-level convenience functions"""

    def test_robust_fit_function(self):
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        
        loc, scale = robust_fit(data)
        
        assert loc == 3.0
        assert scale > 0

    def test_robust_score_function(self):
        data = np.array([0, 1, 2, 3, 4], dtype=float)
        
        scores = robust_score(data, mad_consistent=False)
        
        assert len(scores) == 5
        np.testing.assert_array_almost_equal(scores, [-2, -1, 0, 1, 2])

    def test_function_kwargs_work(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        
        loc, scale = robust_fit(
            data,
            location="trimmed_mean",
            scale="iqr",
            trim=0.2
        )
        
        assert abs(loc - 5.5) < 1e-10


class TestRollingMedian:
    """Test rolling median implementation"""

    def test_rolling_median_basic(self):
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        result = rolling_median(data, window=3)
        
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 2.0
        assert result[3] == 3.0
        assert result[4] == 4.0

    def test_rolling_median_window_1(self):
        data = np.array([1, 2, 3], dtype=float)
        result = rolling_median(data, window=1)
        
        np.testing.assert_array_equal(result, data)

    def test_rolling_median_even_window(self):
        data = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        result = rolling_median(data, window=2)
        
        assert np.isnan(result[0])
        assert result[1] == 1.5
        assert result[2] == 2.5
        assert result[3] == 3.5

    def test_rolling_median_deterministic(self):
        data = np.random.RandomState(777).randn(50)
        
        r1 = rolling_median(data, window=5)
        r2 = rolling_median(data, window=5)
        
        np.testing.assert_array_equal(r1, r2)


class TestNumericalStability:
    """Test numerical stability and precision"""

    def test_large_values(self):
        data = np.array([1e10, 2e10, 3e10, 4e10, 5e10])
        rs = RobustStats()
        
        loc, scale = rs.fit(data)
        
        assert np.isfinite(loc)
        assert np.isfinite(scale)
        assert abs(loc - 3e10) < 1e9

    def test_small_values(self):
        data = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
        rs = RobustStats()
        
        loc, scale = rs.fit(data)
        
        assert np.isfinite(loc)
        assert np.isfinite(scale)

    def test_mixed_scale_data(self):
        data = np.array([0.001, 0.002, 0.003, 1000.0])
        rs = RobustStats()
        
        scores = rs.score(data)
        
        assert all(np.isfinite(scores))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])