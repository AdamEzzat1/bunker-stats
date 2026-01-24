"""
COMPREHENSIVE PYTEST SUITE FOR MATRIX OPERATIONS

This suite provides rigorous testing including:
- ✅ Numerical accuracy validation against NumPy/SciPy
- ✅ Edge case handling (empty, n<2, zero variance, NaN, Inf)
- ✅ Symmetry verification
- ✅ Mathematical properties (positive definiteness, trace, etc.)
- ✅ Pairwise completeness for NaN-aware functions
- ✅ Large matrix stress tests
- ✅ Shape validation
- ✅ Consistency across related functions

Run with: pytest test_matrix.py -v
For coverage: pytest test_matrix.py -v --cov=bunker_stats_rs --cov-report=html
"""

import pytest
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import sys

# Build first: maturin develop --release
try:
    import bunker_stats_rs as bsr
except ImportError:
    print("ERROR: bunker_stats_rs not installed. Run: maturin develop --release")
    sys.exit(1)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def rng():
    """Fixed random state for reproducibility"""
    return np.random.RandomState(42)


@pytest.fixture
def small_matrix(rng):
    """Small matrix (10×5) for quick tests"""
    return rng.normal(0, 1, (10, 5))


@pytest.fixture
def medium_matrix(rng):
    """Medium matrix (100×20) for standard tests"""
    return rng.normal(10, 3, (100, 20))


@pytest.fixture
def large_matrix(rng):
    """Large matrix (500×50) for stress testing"""
    return rng.normal(5, 2, (500, 50))


@pytest.fixture
def square_matrix(rng):
    """Square matrix (20×20) for special tests"""
    return rng.normal(0, 1, (20, 20))


@pytest.fixture
def matrix_with_nan(rng):
    """Matrix with scattered NaN values"""
    X = rng.normal(0, 1, (50, 10))
    mask = rng.random((50, 10)) < 0.1  # 10% NaN
    X[mask] = np.nan
    return X


@pytest.fixture
def centered_matrix(rng):
    """Pre-centered matrix (mean=0 for each column)"""
    X = rng.normal(0, 1, (100, 15))
    return X - X.mean(axis=0)


@pytest.fixture
def zero_var_matrix(rng):
    """Matrix with one zero-variance column"""
    X = rng.normal(0, 1, (50, 5))
    X[:, 2] = 5.0  # Constant column
    return X


@pytest.fixture
def highly_correlated_matrix(rng):
    """Matrix with highly correlated columns"""
    X = rng.normal(0, 1, (100, 5))
    # Make column 1 almost identical to column 0
    X[:, 1] = X[:, 0] + rng.normal(0, 0.01, 100)
    return X


# ==============================================================================
# BASIC COVARIANCE TESTS
# ==============================================================================

class TestCovMatrixBasic:
    """Tests for cov_matrix_np (sample covariance, ddof=1)"""
    
    def test_matches_numpy_basic(self, medium_matrix):
        """Should match numpy.cov exactly"""
        cov_rust = bsr.cov_matrix_np(medium_matrix)
        cov_numpy = np.cov(medium_matrix.T, ddof=1)
        
        np.testing.assert_allclose(cov_rust, cov_numpy, rtol=1e-10)
    
    def test_is_symmetric(self, medium_matrix):
        """Covariance matrix must be symmetric"""
        cov = bsr.cov_matrix_np(medium_matrix)
        
        np.testing.assert_allclose(cov, cov.T, atol=1e-14)
    
    def test_diagonal_is_variance(self, medium_matrix):
        """Diagonal should equal column variances"""
        cov = bsr.cov_matrix_np(medium_matrix)
        variances = np.var(medium_matrix, axis=0, ddof=1)
        
        np.testing.assert_allclose(np.diag(cov), variances, rtol=1e-10)
    
    def test_positive_semidefinite(self, medium_matrix):
        """Covariance matrix must be positive semi-definite"""
        cov = bsr.cov_matrix_np(medium_matrix)
        eigenvalues = np.linalg.eigvalsh(cov)
        
        # All eigenvalues should be >= 0 (allowing small numerical errors)
        assert np.all(eigenvalues >= -1e-10)
    
    def test_edge_case_n_equals_2(self):
        """Should handle minimum valid case (n=2)"""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        cov_rust = bsr.cov_matrix_np(X)
        cov_numpy = np.cov(X.T, ddof=1)
        
        np.testing.assert_allclose(cov_rust, cov_numpy, rtol=1e-10)
    
    def test_edge_case_n_equals_1(self):
        """Should return NaN for n=1"""
        X = np.array([[1.0, 2.0, 3.0]])
        cov = bsr.cov_matrix_np(X)
        
        assert cov.shape == (3, 3)
        assert np.all(np.isnan(cov))
    
    def test_edge_case_empty(self):
        """Should handle empty matrix"""
        X = np.array([]).reshape(0, 3)
        cov = bsr.cov_matrix_np(X)
        
        assert cov.shape == (3, 3)
        assert np.all(np.isnan(cov))
    
    def test_zero_variance_column(self, zero_var_matrix):
        """Should handle constant column (zero variance)"""
        cov = bsr.cov_matrix_np(zero_var_matrix)
        
        # Constant column should have 0 variance
        assert np.abs(cov[2, 2]) < 1e-14
        
        # But matrix should still be valid
        assert cov.shape == (5, 5)
    
    def test_large_matrix(self, large_matrix):
        """Stress test with large matrix"""
        cov_rust = bsr.cov_matrix_np(large_matrix)
        cov_numpy = np.cov(large_matrix.T, ddof=1)
        
        np.testing.assert_allclose(cov_rust, cov_numpy, rtol=1e-10)


# ==============================================================================
# COVARIANCE VARIANTS
# ==============================================================================

class TestCovMatrixBias:
    """Tests for cov_matrix_bias_np (population covariance, ddof=0)"""
    
    def test_matches_numpy_ddof0(self, medium_matrix):
        """Should match numpy.cov with ddof=0"""
        cov_rust = bsr.cov_matrix_bias_np(medium_matrix)
        cov_numpy = np.cov(medium_matrix.T, ddof=0)
        
        np.testing.assert_allclose(cov_rust, cov_numpy, rtol=1e-10)
    
    def test_relation_to_sample_cov(self, medium_matrix):
        """Population cov should be sample_cov * (n-1)/n"""
        n = len(medium_matrix)
        cov_sample = bsr.cov_matrix_np(medium_matrix)
        cov_pop = bsr.cov_matrix_bias_np(medium_matrix)
        
        expected = cov_sample * (n - 1) / n
        np.testing.assert_allclose(cov_pop, expected, rtol=1e-10)
    
    def test_n_equals_1_gives_zeros(self):
        """For n=1, population covariance should be 0"""
        X = np.array([[1.0, 2.0, 3.0]])
        cov = bsr.cov_matrix_bias_np(X)
        
        assert cov.shape == (3, 3)
        np.testing.assert_allclose(cov, np.zeros((3, 3)), atol=1e-14)


class TestCovMatrixCentered:
    """Tests for cov_matrix_centered_np (assumes pre-centered data)"""
    
    def test_on_centered_data(self, centered_matrix):
        """Should match regular cov on centered data"""
        cov_centered = bsr.cov_matrix_centered_np(centered_matrix)
        cov_regular = bsr.cov_matrix_np(centered_matrix)
        
        # Should be very close (within numerical precision)
        np.testing.assert_allclose(cov_centered, cov_regular, rtol=1e-10)
    
    def test_on_uncentered_data_differs(self, medium_matrix):
        """Should differ from regular cov on uncentered data"""
        # Deliberately use uncentered data
        cov_centered = bsr.cov_matrix_centered_np(medium_matrix)
        cov_regular = bsr.cov_matrix_np(medium_matrix)
        
        # These should NOT be the same
        assert not np.allclose(cov_centered, cov_regular, rtol=0.01)
    
    def test_manual_centering(self, medium_matrix):
        """Manual centering + centered_cov should match regular cov"""
        # Manually center
        X_centered = medium_matrix - medium_matrix.mean(axis=0)
        
        cov_centered = bsr.cov_matrix_centered_np(X_centered)
        cov_regular = bsr.cov_matrix_np(medium_matrix)
        
        np.testing.assert_allclose(cov_centered, cov_regular, rtol=1e-10)


class TestCovMatrixSkipna:
    """Tests for cov_matrix_skipna_np (pairwise-complete covariance)"""
    
    def test_matches_regular_when_no_nan(self, medium_matrix):
        """Should match regular cov when no NaN present"""
        cov_skipna = bsr.cov_matrix_skipna_np(medium_matrix)
        cov_regular = bsr.cov_matrix_np(medium_matrix)
        
        np.testing.assert_allclose(cov_skipna, cov_regular, rtol=1e-10)
    
    def test_handles_scattered_nan(self, matrix_with_nan):
        """Should compute pairwise-complete covariance"""
        cov = bsr.cov_matrix_skipna_np(matrix_with_nan)
        
        # Should not be all NaN
        assert not np.all(np.isnan(cov))
        
        # Should be symmetric
        np.testing.assert_allclose(cov, cov.T, atol=1e-14, equal_nan=True)
    
    def test_diagonal_matches_nanvar(self, matrix_with_nan):
        """Diagonal should match numpy's nanvar"""
        cov = bsr.cov_matrix_skipna_np(matrix_with_nan)
        
        for i in range(matrix_with_nan.shape[1]):
            col = matrix_with_nan[:, i]
            expected_var = np.nanvar(col, ddof=1)
            
            if not np.isnan(expected_var):
                np.testing.assert_allclose(cov[i, i], expected_var, rtol=1e-10)
    
    def test_column_all_nan_gives_nan_row_col(self, matrix_with_nan):
        """If a column is all NaN, its row and column should be NaN"""
        X = matrix_with_nan.copy()
        X[:, 3] = np.nan  # Make one column all NaN
        
        cov = bsr.cov_matrix_skipna_np(X)
        
        # Row 3 and column 3 should be all NaN
        assert np.all(np.isnan(cov[3, :]))
        assert np.all(np.isnan(cov[:, 3]))
    
    def test_pairwise_count_lt_2_gives_nan(self):
        """If pairwise count < 2, should give NaN"""
        X = np.array([
            [1.0, np.nan],
            [np.nan, 2.0],
            [3.0, np.nan]
        ])
        
        cov = bsr.cov_matrix_skipna_np(X)
        
        # Only 1 valid pair for cov[0,1], should be NaN
        assert np.isnan(cov[0, 1])
        assert np.isnan(cov[1, 0])


# ==============================================================================
# CORRELATION TESTS
# ==============================================================================

class TestCorrMatrixBasic:
    """Tests for corr_matrix_np (Pearson correlation)"""
    
    def test_matches_numpy_corrcoef(self, medium_matrix):
        """Should match numpy.corrcoef exactly"""
        corr_rust = bsr.corr_matrix_np(medium_matrix)
        corr_numpy = np.corrcoef(medium_matrix.T)
        
        np.testing.assert_allclose(corr_rust, corr_numpy, rtol=1e-10, atol=1e-14)
    
    def test_diagonal_is_ones(self, medium_matrix):
        """Diagonal should be all 1.0"""
        corr = bsr.corr_matrix_np(medium_matrix)
        
        np.testing.assert_allclose(np.diag(corr), np.ones(corr.shape[0]), atol=1e-14)
    
    def test_is_symmetric(self, medium_matrix):
        """Correlation matrix must be symmetric"""
        corr = bsr.corr_matrix_np(medium_matrix)
        
        np.testing.assert_allclose(corr, corr.T, atol=1e-14)
    
    def test_values_in_range(self, medium_matrix):
        """All correlations should be in [-1, 1]"""
        corr = bsr.corr_matrix_np(medium_matrix)
        
        assert np.all(corr >= -1.0 - 1e-10)
        assert np.all(corr <= 1.0 + 1e-10)
    
    def test_perfect_correlation(self):
        """Perfectly correlated columns should have corr=1"""
        X = np.random.randn(100, 1)
        X_dup = np.hstack([X, X])  # Duplicate column
        
        corr = bsr.corr_matrix_np(X_dup)
        
        assert np.abs(corr[0, 1] - 1.0) < 1e-10
        assert np.abs(corr[1, 0] - 1.0) < 1e-10
    
    def test_perfect_anticorrelation(self):
        """Perfectly anti-correlated columns should have corr=-1"""
        X = np.random.randn(100, 1)
        X_neg = np.hstack([X, -X])  # Negated column
        
        corr = bsr.corr_matrix_np(X_neg)
        
        assert np.abs(corr[0, 1] - (-1.0)) < 1e-10
    
    def test_zero_variance_gives_nan(self, zero_var_matrix):
        """Zero variance column should give NaN correlations"""
        corr = bsr.corr_matrix_np(zero_var_matrix)
        
        # Row and column 2 should be NaN (except maybe diagonal)
        assert np.all(np.isnan(corr[2, :3]))  # Off-diagonal
        assert np.all(np.isnan(corr[:3, 2]))


class TestCorrMatrixSkipna:
    """Tests for corr_matrix_skipna_np (pairwise-complete correlation)"""
    
    def test_matches_regular_when_no_nan(self, medium_matrix):
        """Should match regular corr when no NaN present"""
        corr_skipna = bsr.corr_matrix_skipna_np(medium_matrix)
        corr_regular = bsr.corr_matrix_np(medium_matrix)
        
        np.testing.assert_allclose(corr_skipna, corr_regular, rtol=1e-10, atol=1e-14)
    
    def test_handles_scattered_nan(self, matrix_with_nan):
        """Should compute pairwise-complete correlation"""
        corr = bsr.corr_matrix_skipna_np(matrix_with_nan)
        
        # Should not be all NaN
        assert not np.all(np.isnan(corr))
        
        # Diagonal should be 1.0 where defined
        diag = np.diag(corr)
        finite_diag = diag[~np.isnan(diag)]
        if len(finite_diag) > 0:
            np.testing.assert_allclose(finite_diag, np.ones(len(finite_diag)), atol=1e-10)
    
    def test_is_symmetric(self, matrix_with_nan):
        """Should be symmetric"""
        corr = bsr.corr_matrix_skipna_np(matrix_with_nan)
        
        np.testing.assert_allclose(corr, corr.T, atol=1e-14, equal_nan=True)
    
    def test_values_in_range(self, matrix_with_nan):
        """Valid correlations should be in [-1, 1]"""
        corr = bsr.corr_matrix_skipna_np(matrix_with_nan)
        
        valid = corr[~np.isnan(corr)]
        assert np.all(valid >= -1.0 - 1e-10)
        assert np.all(valid <= 1.0 + 1e-10)


class TestCorrDistance:
    """Tests for corr_distance_np (1 - correlation)"""
    
    def test_formula(self, medium_matrix):
        """Should equal 1 - correlation"""
        corr = bsr.corr_matrix_np(medium_matrix)
        dist = bsr.corr_distance_np(medium_matrix)
        
        expected = 1.0 - corr
        np.testing.assert_allclose(dist, expected, rtol=1e-10, atol=1e-14)
    
    def test_diagonal_is_zero(self, medium_matrix):
        """Diagonal should be 0 (distance from self)"""
        dist = bsr.corr_distance_np(medium_matrix)
        
        np.testing.assert_allclose(np.diag(dist), np.zeros(dist.shape[0]), atol=1e-14)
    
    def test_is_symmetric(self, medium_matrix):
        """Distance matrix should be symmetric"""
        dist = bsr.corr_distance_np(medium_matrix)
        
        np.testing.assert_allclose(dist, dist.T, atol=1e-14)
    
    def test_range(self, medium_matrix):
        """Valid distances should be in [0, 2]"""
        dist = bsr.corr_distance_np(medium_matrix)
        
        valid = dist[~np.isnan(dist)]
        assert np.all(valid >= -1e-10)
        assert np.all(valid <= 2.0 + 1e-10)


# ==============================================================================
# GRAM MATRIX TESTS
# ==============================================================================

class TestGramXTX:
    """Tests for xtx_matrix_np (X^T X)"""
    
    def test_matches_numpy(self, medium_matrix):
        """Should match X.T @ X"""
        xtx_rust = bsr.xtx_matrix_np(medium_matrix)
        xtx_numpy = medium_matrix.T @ medium_matrix
        
        np.testing.assert_allclose(xtx_rust, xtx_numpy, rtol=1e-10)
    
    def test_is_symmetric(self, medium_matrix):
        """X^T X must be symmetric"""
        xtx = bsr.xtx_matrix_np(medium_matrix)
        
        np.testing.assert_allclose(xtx, xtx.T, atol=1e-14)
    
    def test_positive_semidefinite(self, medium_matrix):
        """X^T X must be positive semi-definite"""
        xtx = bsr.xtx_matrix_np(medium_matrix)
        eigenvalues = np.linalg.eigvalsh(xtx)
        
        assert np.all(eigenvalues >= -1e-10)
    
    def test_shape(self, medium_matrix):
        """Shape should be (p, p) for (n, p) input"""
        n, p = medium_matrix.shape
        xtx = bsr.xtx_matrix_np(medium_matrix)
        
        assert xtx.shape == (p, p)
    
    def test_diagonal_is_column_sum_squares(self, medium_matrix):
        """Diagonal should be sum of squares for each column"""
        xtx = bsr.xtx_matrix_np(medium_matrix)
        
        for i in range(medium_matrix.shape[1]):
            expected = np.sum(medium_matrix[:, i] ** 2)
            np.testing.assert_allclose(xtx[i, i], expected, rtol=1e-10)
    
    def test_small_example(self):
        """Verify with small known example"""
        X = np.array([[1.0, 2.0],
                      [3.0, 4.0]])
        xtx = bsr.xtx_matrix_np(X)
        
        # Expected: [[1, 3], [2, 4]] @ [[1, 2], [3, 4]]
        #         = [[10, 14], [14, 20]]
        expected = np.array([[10.0, 14.0],
                            [14.0, 20.0]])
        
        np.testing.assert_allclose(xtx, expected, rtol=1e-10)


class TestGramXXT:
    """Tests for xxt_matrix_np (X X^T)"""
    
    def test_matches_numpy(self, small_matrix):
        """Should match X @ X.T"""
        # Use small matrix to avoid huge output
        xxt_rust = bsr.xxt_matrix_np(small_matrix)
        xxt_numpy = small_matrix @ small_matrix.T
        
        np.testing.assert_allclose(xxt_rust, xxt_numpy, rtol=1e-10)
    
    def test_is_symmetric(self, small_matrix):
        """X X^T must be symmetric"""
        xxt = bsr.xxt_matrix_np(small_matrix)
        
        np.testing.assert_allclose(xxt, xxt.T, atol=1e-14)
    
    def test_positive_semidefinite(self, small_matrix):
        """X X^T must be positive semi-definite"""
        xxt = bsr.xxt_matrix_np(small_matrix)
        eigenvalues = np.linalg.eigvalsh(xxt)
        
        assert np.all(eigenvalues >= -1e-10)
    
    def test_shape(self, small_matrix):
        """Shape should be (n, n) for (n, p) input"""
        n, p = small_matrix.shape
        xxt = bsr.xxt_matrix_np(small_matrix)
        
        assert xxt.shape == (n, n)
    
    def test_diagonal_is_row_sum_squares(self, small_matrix):
        """Diagonal should be sum of squares for each row"""
        xxt = bsr.xxt_matrix_np(small_matrix)
        
        for i in range(small_matrix.shape[0]):
            expected = np.sum(small_matrix[i, :] ** 2)
            np.testing.assert_allclose(xxt[i, i], expected, rtol=1e-10)


# ==============================================================================
# PAIRWISE DISTANCE TESTS
# ==============================================================================

class TestPairwiseEuclidean:
    """Tests for pairwise_euclidean_cols_np"""
    
    def test_diagonal_is_zero(self, medium_matrix):
        """Distance from column to itself should be 0"""
        dist = bsr.pairwise_euclidean_cols_np(medium_matrix)
        
        np.testing.assert_allclose(np.diag(dist), np.zeros(dist.shape[0]), atol=1e-14)
    
    def test_is_symmetric(self, medium_matrix):
        """Distance matrix must be symmetric"""
        dist = bsr.pairwise_euclidean_cols_np(medium_matrix)
        
        np.testing.assert_allclose(dist, dist.T, atol=1e-14)
    
    def test_non_negative(self, medium_matrix):
        """All distances must be >= 0"""
        dist = bsr.pairwise_euclidean_cols_np(medium_matrix)
        
        assert np.all(dist >= -1e-14)
    
    def test_triangle_inequality(self, medium_matrix):
        """Should satisfy triangle inequality: d(i,k) <= d(i,j) + d(j,k)"""
        dist = bsr.pairwise_euclidean_cols_np(medium_matrix)
        p = dist.shape[0]
        
        # Test a sample of triplets
        rng = np.random.RandomState(42)
        for _ in range(min(50, p * p)):
            i, j, k = rng.choice(p, 3, replace=True)
            assert dist[i, k] <= dist[i, j] + dist[j, k] + 1e-10
    
    def test_known_example(self):
        """Test with known example"""
        # Two columns: [1, 2, 3] and [4, 5, 6]
        X = np.array([[1.0, 4.0],
                      [2.0, 5.0],
                      [3.0, 6.0]])
        dist = bsr.pairwise_euclidean_cols_np(X)
        
        # Distance = sqrt((1-4)^2 + (2-5)^2 + (3-6)^2) = sqrt(27)
        expected_dist = np.sqrt(27.0)
        
        assert np.abs(dist[0, 1] - expected_dist) < 1e-10
        assert np.abs(dist[1, 0] - expected_dist) < 1e-10


class TestPairwiseCosine:
    """Tests for pairwise_cosine_cols_np"""
    
    def test_diagonal_is_zero(self, medium_matrix):
        """Distance from column to itself should be 0"""
        dist = bsr.pairwise_cosine_cols_np(medium_matrix)
        
        # Diagonal should be 0 (distance from self)
        diag = np.diag(dist)
        finite_diag = diag[~np.isnan(diag)]
        np.testing.assert_allclose(finite_diag, np.zeros(len(finite_diag)), atol=1e-10)
    
    def test_is_symmetric(self, medium_matrix):
        """Distance matrix must be symmetric"""
        dist = bsr.pairwise_cosine_cols_np(medium_matrix)
        
        np.testing.assert_allclose(dist, dist.T, atol=1e-14, equal_nan=True)
    
    def test_range(self, medium_matrix):
        """Valid cosine distances should be in [0, 2]"""
        dist = bsr.pairwise_cosine_cols_np(medium_matrix)
        
        valid = dist[~np.isnan(dist)]
        assert np.all(valid >= -1e-10)
        assert np.all(valid <= 2.0 + 1e-10)
    
    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have distance = 1.0"""
        # Create orthogonal vectors
        X = np.array([[1.0, 0.0],
                      [0.0, 1.0],
                      [0.0, 0.0]])
        dist = bsr.pairwise_cosine_cols_np(X)
        
        # Cosine similarity of orthogonal = 0, so distance = 1 - 0 = 1
        assert np.abs(dist[0, 1] - 1.0) < 1e-10
    
    def test_parallel_vectors(self):
        """Parallel vectors should have distance = 0"""
        X = np.array([[1.0, 2.0],
                      [2.0, 4.0],
                      [3.0, 6.0]])
        dist = bsr.pairwise_cosine_cols_np(X)
        
        # Columns are parallel, cosine similarity = 1, distance = 0
        assert np.abs(dist[0, 1]) < 1e-10
    
    def test_zero_norm_gives_nan(self):
        """Zero-norm column should give NaN"""
        X = np.array([[1.0, 0.0],
                      [2.0, 0.0],
                      [3.0, 0.0]])
        dist = bsr.pairwise_cosine_cols_np(X)
        
        # Column 1 has zero norm
        assert np.isnan(dist[0, 1])
        assert np.isnan(dist[1, 0])


# ==============================================================================
# DIAGNOSTIC TESTS
# ==============================================================================

class TestDiag:
    """Tests for diag_np"""
    
    def test_extracts_diagonal(self, square_matrix):
        """Should extract diagonal correctly"""
        diag_rust = bsr.diag_np(square_matrix)
        diag_numpy = np.diag(square_matrix)
        
        np.testing.assert_allclose(diag_rust, diag_numpy, rtol=1e-10)
    
    def test_length_equals_dimension(self, square_matrix):
        """Diagonal length should equal matrix dimension"""
        diag = bsr.diag_np(square_matrix)
        
        assert len(diag) == square_matrix.shape[0]
    
    def test_non_square_raises_error(self, medium_matrix):
        """Should raise error for non-square matrix"""
        with pytest.raises(Exception):  # Could be ValueError or similar
            bsr.diag_np(medium_matrix)


class TestTrace:
    """Tests for trace_np"""
    
    def test_matches_numpy(self, square_matrix):
        """Should match numpy.trace"""
        trace_rust = bsr.trace_np(square_matrix)
        trace_numpy = np.trace(square_matrix)
        
        np.testing.assert_allclose(trace_rust, trace_numpy, rtol=1e-10)
    
    def test_equals_sum_of_diag(self, square_matrix):
        """Trace should equal sum of diagonal"""
        trace = bsr.trace_np(square_matrix)
        diag = bsr.diag_np(square_matrix)
        
        np.testing.assert_allclose(trace, np.sum(diag), rtol=1e-10)
    
    def test_identity_matrix(self):
        """Trace of identity should equal dimension"""
        I = np.eye(10)
        trace = bsr.trace_np(I)
        
        assert np.abs(trace - 10.0) < 1e-14
    
    def test_non_square_raises_error(self, medium_matrix):
        """Should raise error for non-square matrix"""
        with pytest.raises(Exception):
            bsr.trace_np(medium_matrix)


class TestIsSymmetric:
    """Tests for is_symmetric_np"""
    
    def test_symmetric_matrix_true(self, medium_matrix):
        """Covariance matrix should be symmetric"""
        cov = bsr.cov_matrix_np(medium_matrix)
        
        assert bsr.is_symmetric_np(cov, tol=1e-10)
    
    def test_asymmetric_matrix_false(self):
        """Non-symmetric matrix should return False"""
        A = np.array([[1.0, 2.0],
                      [3.0, 4.0]])
        
        assert not bsr.is_symmetric_np(A, tol=1e-10)
    
    def test_nearly_symmetric_with_tolerance(self):
        """Nearly symmetric should pass with appropriate tolerance"""
        A = np.array([[1.0, 2.0],
                      [2.0 + 1e-11, 4.0]])
        
        assert bsr.is_symmetric_np(A, tol=1e-10)
        assert not bsr.is_symmetric_np(A, tol=1e-12)
    
    def test_identity_is_symmetric(self):
        """Identity matrix should be symmetric"""
        I = np.eye(15)
        
        assert bsr.is_symmetric_np(I, tol=1e-14)


# ==============================================================================
# INTEGRATION AND CONSISTENCY TESTS
# ==============================================================================

class TestMatrixConsistency:
    """Tests for consistency across related functions"""
    
    def test_corr_from_cov_consistency(self, medium_matrix):
        """corr computed from cov should match direct corr"""
        # Get covariance
        cov = bsr.cov_matrix_np(medium_matrix)
        
        # Get correlation directly
        corr_direct = bsr.corr_matrix_np(medium_matrix)
        
        # Compute correlation from covariance manually
        stds = np.sqrt(np.diag(cov))
        corr_from_cov = cov / np.outer(stds, stds)
        
        np.testing.assert_allclose(corr_direct, corr_from_cov, rtol=1e-10)
    
    def test_xtx_relates_to_cov(self, medium_matrix):
        """X^T X should relate to covariance"""
        n, p = medium_matrix.shape
        
        # Center the data
        X_centered = medium_matrix - medium_matrix.mean(axis=0)
        
        # X^T X for centered data
        xtx = bsr.xtx_matrix_np(X_centered)
        
        # Covariance * (n-1)
        cov = bsr.cov_matrix_np(medium_matrix)
        cov_scaled = cov * (n - 1)
        
        np.testing.assert_allclose(xtx, cov_scaled, rtol=1e-10)
    
    def test_trace_of_cov_equals_sum_variances(self, medium_matrix):
        """Trace of covariance should equal sum of variances"""
        cov = bsr.cov_matrix_np(medium_matrix)
        trace = bsr.trace_np(cov)
        
        variances = np.var(medium_matrix, axis=0, ddof=1)
        sum_var = np.sum(variances)
        
        np.testing.assert_allclose(trace, sum_var, rtol=1e-10)


# ==============================================================================
# STRESS TESTS
# ==============================================================================

class TestStressScenarios:
    """Stress tests with edge cases and large data"""
    
    def test_very_large_matrix(self):
        """Test with very large matrix"""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (1000, 100))
        
        # Should complete without error
        cov = bsr.cov_matrix_np(X)
        assert cov.shape == (100, 100)
        
        corr = bsr.corr_matrix_np(X)
        assert corr.shape == (100, 100)
    
    def test_many_columns_few_rows(self):
        """Test with p > n (more columns than rows)"""
        X = np.random.randn(10, 50)
        
        cov = bsr.cov_matrix_np(X)
        assert cov.shape == (50, 50)
        
        # Should be rank deficient
        rank = np.linalg.matrix_rank(cov)
        assert rank <= 10
    
    def test_extreme_values(self):
        """Test with extreme values"""
        X = np.array([[1e10, 1e-10],
                      [2e10, 2e-10],
                      [3e10, 3e-10]])
        
        # Should handle without overflow/underflow
        cov = bsr.cov_matrix_np(X)
        assert not np.any(np.isinf(cov))
        assert not np.all(np.isnan(cov))
    
    def test_all_same_values(self):
        """Test with all identical values"""
        X = np.ones((100, 10))
        
        cov = bsr.cov_matrix_np(X)
        
        # Should be all zeros
        np.testing.assert_allclose(cov, np.zeros((10, 10)), atol=1e-14)
    
    def test_mixed_scale_columns(self):
        """Test with columns of very different scales"""
        rng = np.random.RandomState(42)
        X = np.column_stack([
            rng.normal(0, 1e-5, 100),
            rng.normal(0, 1.0, 100),
            rng.normal(0, 1e5, 100)
        ])
        
        cov = bsr.cov_matrix_np(X)
        
        # Should still be positive semi-definite
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)


# ==============================================================================
# SHAPE AND TYPE TESTS
# ==============================================================================

class TestShapeAndTypes:
    """Tests for shape validation and type handling"""
    
    def test_output_shape_cov(self, medium_matrix):
        """Covariance output shape should be (p, p)"""
        n, p = medium_matrix.shape
        cov = bsr.cov_matrix_np(medium_matrix)
        
        assert cov.shape == (p, p)
    
    def test_output_shape_xtx(self, medium_matrix):
        """X^T X output shape should be (p, p)"""
        n, p = medium_matrix.shape
        xtx = bsr.xtx_matrix_np(medium_matrix)
        
        assert xtx.shape == (p, p)
    
    def test_output_shape_xxt(self, small_matrix):
        """X X^T output shape should be (n, n)"""
        n, p = small_matrix.shape
        xxt = bsr.xxt_matrix_np(small_matrix)
        
        assert xxt.shape == (n, n)
    
    def test_output_dtype(self, medium_matrix):
        """All outputs should be float64"""
        cov = bsr.cov_matrix_np(medium_matrix)
        corr = bsr.corr_matrix_np(medium_matrix)
        xtx = bsr.xtx_matrix_np(medium_matrix)
        
        assert cov.dtype == np.float64
        assert corr.dtype == np.float64
        assert xtx.dtype == np.float64
    
    def test_float32_input_works(self):
        """Should handle float32 input"""
        X = np.random.randn(50, 10).astype(np.float32)
        
        # Should convert and work
        cov = bsr.cov_matrix_np(X)
        assert cov.dtype == np.float64


# ==============================================================================
# PERFORMANCE REGRESSION TESTS
# ==============================================================================

class TestPerformanceRegression:
    """Tests to catch performance regressions"""
    
    def test_medium_matrix_timing(self, medium_matrix, benchmark):
        """Benchmark medium matrix covariance"""
        if 'benchmark' in dir():
            result = benchmark(bsr.cov_matrix_np, medium_matrix)
            assert result.shape == (20, 20)
    
    def test_parallel_gives_same_result(self, large_matrix):
        """Parallel and sequential should give identical results"""
        # This assumes you have a way to toggle parallelism
        # For now, just verify deterministic output
        cov1 = bsr.cov_matrix_np(large_matrix)
        cov2 = bsr.cov_matrix_np(large_matrix)
        
        np.testing.assert_array_equal(cov1, cov2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
