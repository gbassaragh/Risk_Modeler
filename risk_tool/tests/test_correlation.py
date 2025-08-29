"""Tests for correlation modeling functionality.

Validates correlation implementation and Iman-Conover method.
"""

import numpy as np
import pytest
from scipy import stats
from risk_tool.engine.correlation import (
    CorrelationMatrix, 
    ImanConovierTransform, 
    CholekyTransform,
    CorrelationValidator,
    apply_correlations
)


class TestCorrelationMatrix:
    """Test correlation matrix management."""
    
    def test_correlation_matrix_creation(self):
        """Test basic correlation matrix creation."""
        variables = ['var1', 'var2', 'var3']
        corr_matrix = CorrelationMatrix(variables)
        
        assert corr_matrix.n_vars == 3
        assert corr_matrix.variable_names == variables
        
        # Should initialize as identity
        matrix = corr_matrix.get_correlation_matrix()
        expected = np.eye(3)
        np.testing.assert_array_equal(matrix, expected)
    
    def test_set_correlation(self):
        """Test setting correlations."""
        variables = ['var1', 'var2', 'var3']
        corr_matrix = CorrelationMatrix(variables)
        
        # Set correlation between var1 and var2
        corr_matrix.set_correlation('var1', 'var2', 0.5)
        
        matrix = corr_matrix.get_correlation_matrix()
        assert matrix[0, 1] == 0.5
        assert matrix[1, 0] == 0.5  # Should be symmetric
        assert matrix[0, 0] == 1.0  # Diagonal should remain 1
    
    def test_invalid_correlation_value(self):
        """Test validation of correlation values."""
        variables = ['var1', 'var2']
        corr_matrix = CorrelationMatrix(variables)
        
        # Test correlation > 1
        with pytest.raises(ValueError):
            corr_matrix.set_correlation('var1', 'var2', 1.5)
        
        # Test correlation < -1
        with pytest.raises(ValueError):
            corr_matrix.set_correlation('var1', 'var2', -1.5)
    
    def test_unknown_variable(self):
        """Test error handling for unknown variables."""
        variables = ['var1', 'var2']
        corr_matrix = CorrelationMatrix(variables)
        
        with pytest.raises(ValueError):
            corr_matrix.set_correlation('var1', 'unknown_var', 0.5)
    
    def test_positive_definite_check(self):
        """Test positive definite matrix checking."""
        variables = ['var1', 'var2', 'var3']
        corr_matrix = CorrelationMatrix(variables)
        
        # Identity matrix should be positive definite
        assert corr_matrix.is_positive_definite()
        
        # Create a valid correlation matrix
        corr_matrix.set_correlation('var1', 'var2', 0.5)
        corr_matrix.set_correlation('var1', 'var3', 0.3)
        corr_matrix.set_correlation('var2', 'var3', 0.2)
        
        assert corr_matrix.is_positive_definite()
    
    def test_make_positive_definite(self):
        """Test making matrix positive definite."""
        variables = ['var1', 'var2', 'var3']
        corr_matrix = CorrelationMatrix(variables)
        
        # Create potentially problematic correlations
        corr_matrix.set_correlation('var1', 'var2', 0.9)
        corr_matrix.set_correlation('var1', 'var3', 0.9)
        corr_matrix.set_correlation('var2', 'var3', 0.9)
        
        # Make positive definite
        corr_matrix.make_positive_definite()
        
        # Should still be positive definite
        assert corr_matrix.is_positive_definite()
        
        # Diagonal should still be 1
        matrix = corr_matrix.get_correlation_matrix()
        np.testing.assert_array_almost_equal(np.diag(matrix), np.ones(3))


class TestImanConovierTransform:
    """Test Iman-Conover correlation transformation."""
    
    @pytest.fixture
    def random_samples(self):
        """Generate independent random samples."""
        np.random.seed(42)
        n_samples = 1000
        n_vars = 3
        
        # Generate independent normal samples
        samples = np.random.normal(0, 1, (n_samples, n_vars))
        return samples
    
    def test_identity_correlation(self, random_samples):
        """Test with identity correlation matrix (no correlation)."""
        n_samples, n_vars = random_samples.shape
        identity_matrix = np.eye(n_vars)
        
        transformed = ImanConovierTransform.transform(
            random_samples, identity_matrix, "spearman"
        )
        
        # Should have same shape
        assert transformed.shape == random_samples.shape
        
        # Correlations should be close to zero
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                corr, _ = stats.spearmanr(transformed[:, i], transformed[:, j])
                assert abs(corr) < 0.1  # Should be close to 0
    
    def test_positive_correlation(self, random_samples):
        """Test with positive correlation."""
        n_vars = random_samples.shape[1]
        target_matrix = np.eye(n_vars)
        target_matrix[0, 1] = target_matrix[1, 0] = 0.6
        
        transformed = ImanConovierTransform.transform(
            random_samples, target_matrix, "spearman"
        )
        
        # Check achieved correlation
        achieved_corr, _ = stats.spearmanr(transformed[:, 0], transformed[:, 1])
        assert abs(achieved_corr - 0.6) < 0.1  # Should be close to target
    
    def test_negative_correlation(self, random_samples):
        """Test with negative correlation."""
        n_vars = random_samples.shape[1]
        target_matrix = np.eye(n_vars)
        target_matrix[0, 1] = target_matrix[1, 0] = -0.4
        
        transformed = ImanConovierTransform.transform(
            random_samples, target_matrix, "spearman"
        )
        
        # Check achieved correlation
        achieved_corr, _ = stats.spearmanr(transformed[:, 0], transformed[:, 1])
        assert abs(achieved_corr - (-0.4)) < 0.1  # Should be close to target
    
    def test_multiple_correlations(self, random_samples):
        """Test with multiple correlations."""
        n_vars = random_samples.shape[1]
        target_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ])
        
        transformed = ImanConovierTransform.transform(
            random_samples, target_matrix, "spearman"
        )
        
        # Check multiple correlations
        corr_01, _ = stats.spearmanr(transformed[:, 0], transformed[:, 1])
        corr_02, _ = stats.spearmanr(transformed[:, 0], transformed[:, 2])
        corr_12, _ = stats.spearmanr(transformed[:, 1], transformed[:, 2])
        
        assert abs(corr_01 - 0.5) < 0.1
        assert abs(corr_02 - 0.3) < 0.1
        assert abs(corr_12 - 0.2) < 0.1
    
    def test_invalid_matrix_size(self, random_samples):
        """Test error handling for mismatched matrix size."""
        wrong_size_matrix = np.eye(2)  # Wrong size
        
        with pytest.raises(ValueError):
            ImanConovierTransform.transform(
                random_samples, wrong_size_matrix, "spearman"
            )
    
    def test_non_symmetric_matrix(self, random_samples):
        """Test error handling for non-symmetric matrix."""
        n_vars = random_samples.shape[1]
        non_symmetric = np.eye(n_vars)
        non_symmetric[0, 1] = 0.5
        # non_symmetric[1, 0] = 0.3  # Different value - makes it non-symmetric
        
        with pytest.raises(ValueError):
            ImanConovierTransform.transform(
                random_samples, non_symmetric, "spearman"
            )


class TestCholekyTransform:
    """Test Cholesky decomposition method."""
    
    @pytest.fixture
    def normal_samples(self):
        """Generate independent standard normal samples."""
        np.random.seed(123)
        n_samples = 1000
        n_vars = 2
        return np.random.normal(0, 1, (n_samples, n_vars))
    
    def test_normal_transformation(self, normal_samples):
        """Test Cholesky transformation of normal samples."""
        target_matrix = np.array([
            [1.0, 0.7],
            [0.7, 1.0]
        ])
        
        transformed = CholekyTransform.transform_normal(
            normal_samples, target_matrix
        )
        
        # Check achieved correlation
        achieved_corr = np.corrcoef(transformed[:, 0], transformed[:, 1])[0, 1]
        assert abs(achieved_corr - 0.7) < 0.1
    
    def test_singular_matrix_regularization(self, normal_samples):
        """Test handling of singular matrices."""
        # Create singular matrix
        singular_matrix = np.array([
            [1.0, 1.0],
            [1.0, 1.0]  # Perfectly correlated - singular
        ])
        
        # Should handle gracefully (regularize)
        transformed = CholekyTransform.transform_normal(
            normal_samples, singular_matrix
        )
        
        assert transformed.shape == normal_samples.shape


class TestCorrelationValidator:
    """Test correlation validation functionality."""
    
    @pytest.fixture
    def correlated_samples(self):
        """Generate samples with known correlations."""
        np.random.seed(456)
        n_samples = 1000
        
        # Generate correlated samples
        x1 = np.random.normal(0, 1, n_samples)
        x2 = 0.6 * x1 + 0.8 * np.random.normal(0, 1, n_samples)  # Correlation ≈ 0.6
        
        return np.column_stack([x1, x2])
    
    def test_correlation_validation(self, correlated_samples):
        """Test correlation validation against targets."""
        target_correlations = {(0, 1): 0.6}
        
        validation_result = CorrelationValidator.validate_correlations(
            correlated_samples, target_correlations, "pearson", tolerance=0.1
        )
        
        assert validation_result['all_within_tolerance']
        assert validation_result['max_error'] < 0.1
        assert (0, 1) in validation_result['details']
        assert validation_result['details'][(0, 1)]['within_tolerance']
    
    def test_correlation_validation_failure(self, correlated_samples):
        """Test correlation validation with tight tolerance."""
        target_correlations = {(0, 1): 0.9}  # Wrong target
        
        validation_result = CorrelationValidator.validate_correlations(
            correlated_samples, target_correlations, "pearson", tolerance=0.05
        )
        
        assert not validation_result['all_within_tolerance']
        assert not validation_result['details'][(0, 1)]['within_tolerance']
    
    def test_compute_correlation_matrix(self, correlated_samples):
        """Test correlation matrix computation."""
        corr_matrix = CorrelationValidator.compute_correlation_matrix(
            correlated_samples, "pearson"
        )
        
        # Should be 2x2 matrix
        assert corr_matrix.shape == (2, 2)
        
        # Diagonal should be 1
        assert abs(corr_matrix[0, 0] - 1.0) < 1e-10
        assert abs(corr_matrix[1, 1] - 1.0) < 1e-10
        
        # Off-diagonal should match known correlation
        assert abs(corr_matrix[0, 1] - 0.6) < 0.1
        assert abs(corr_matrix[1, 0] - 0.6) < 0.1


class TestApplyCorrelations:
    """Test high-level correlation application function."""
    
    @pytest.fixture
    def independent_samples(self):
        """Generate independent samples."""
        np.random.seed(789)
        n_samples = 1000
        n_vars = 3
        return np.random.normal(0, 1, (n_samples, n_vars))
    
    def test_no_correlations(self, independent_samples):
        """Test with no correlations specified."""
        correlations = []
        variable_names = ['var1', 'var2', 'var3']
        
        result = apply_correlations(
            independent_samples, correlations, variable_names
        )
        
        # Should return original samples
        np.testing.assert_array_equal(result, independent_samples)
    
    def test_single_correlation(self, independent_samples):
        """Test with single correlation."""
        correlations = [
            {
                'pair': ['var1', 'var2'],
                'rho': 0.5,
                'method': 'spearman'
            }
        ]
        variable_names = ['var1', 'var2', 'var3']
        
        result = apply_correlations(
            independent_samples, correlations, variable_names
        )
        
        # Check that correlation was applied
        achieved_corr, _ = stats.spearmanr(result[:, 0], result[:, 1])
        assert abs(achieved_corr - 0.5) < 0.15
        
        # Check that other correlations remain close to zero
        other_corr, _ = stats.spearmanr(result[:, 0], result[:, 2])
        assert abs(other_corr) < 0.15
    
    def test_multiple_correlations(self, independent_samples):
        """Test with multiple correlations."""
        correlations = [
            {
                'pair': ['var1', 'var2'],
                'rho': 0.6,
                'method': 'spearman'
            },
            {
                'pair': ['var1', 'var3'],
                'rho': 0.3,
                'method': 'spearman'
            }
        ]
        variable_names = ['var1', 'var2', 'var3']
        
        result = apply_correlations(
            independent_samples, correlations, variable_names
        )
        
        # Check both correlations
        corr_12, _ = stats.spearmanr(result[:, 0], result[:, 1])
        corr_13, _ = stats.spearmanr(result[:, 0], result[:, 2])
        
        assert abs(corr_12 - 0.6) < 0.15
        assert abs(corr_13 - 0.3) < 0.15
    
    def test_invalid_variable_name(self, independent_samples):
        """Test error handling for invalid variable names."""
        correlations = [
            {
                'pair': ['var1', 'unknown_var'],
                'rho': 0.5,
                'method': 'spearman'
            }
        ]
        variable_names = ['var1', 'var2', 'var3']
        
        with pytest.raises(ValueError):
            apply_correlations(
                independent_samples, correlations, variable_names
            )


class TestCorrelationEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_perfect_correlation(self):
        """Test perfect correlation (rho = 1.0)."""
        np.random.seed(999)
        samples = np.random.normal(0, 1, (500, 2))
        target_matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
        
        # Should handle gracefully
        result = ImanConovierTransform.transform(samples, target_matrix, "spearman")
        
        # Check that result has reasonable correlation (may not be exactly 1 due to regularization)
        achieved_corr, _ = stats.spearmanr(result[:, 0], result[:, 1])
        assert achieved_corr > 0.8  # Should be high
    
    def test_small_sample_size(self):
        """Test with small sample size."""
        np.random.seed(111)
        small_samples = np.random.normal(0, 1, (10, 2))  # Very small
        target_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        result = ImanConovierTransform.transform(small_samples, target_matrix, "spearman")
        
        # Should complete without error
        assert result.shape == small_samples.shape
    
    def test_single_variable(self):
        """Test with single variable (edge case)."""
        np.random.seed(222)
        single_var = np.random.normal(0, 1, (100, 1))
        target_matrix = np.array([[1.0]])
        
        result = ImanConovierTransform.transform(single_var, target_matrix, "spearman")
        
        # Should return essentially the same
        np.testing.assert_array_almost_equal(
            np.sort(result[:, 0]), np.sort(single_var[:, 0]), decimal=10
        )


if __name__ == "__main__":
    pytest.main([__file__])