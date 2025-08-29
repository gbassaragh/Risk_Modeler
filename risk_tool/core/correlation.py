"""Correlation modeling using Iman-Conover and Cholesky methods.

Implements rank correlation for dependent Monte Carlo samples.
"""

import numpy as np
from scipy import stats
from scipy.linalg import cholesky, LinAlgError
from typing import Dict, List, Tuple, Optional, Union
import warnings


class CorrelationMatrix:
    """Manages correlation matrix and validation."""
    
    def __init__(self, variable_names: List[str]):
        """Initialize correlation matrix.
        
        Args:
            variable_names: List of variable names
        """
        self.variable_names = variable_names
        self.n_vars = len(variable_names)
        self.correlation_matrix = np.eye(self.n_vars)
        self.name_to_index = {name: i for i, name in enumerate(variable_names)}
    
    def set_correlation(self, var1: str, var2: str, correlation: float) -> None:
        """Set correlation between two variables.
        
        Args:
            var1: First variable name
            var2: Second variable name
            correlation: Correlation coefficient (-1 to 1)
        """
        if abs(correlation) > 1:
            raise ValueError(f"Correlation must be between -1 and 1, got {correlation}")
        
        if var1 not in self.name_to_index:
            raise ValueError(f"Variable {var1} not found")
        if var2 not in self.name_to_index:
            raise ValueError(f"Variable {var2} not found")
        
        i = self.name_to_index[var1]
        j = self.name_to_index[var2]
        
        self.correlation_matrix[i, j] = correlation
        self.correlation_matrix[j, i] = correlation
    
    def get_correlation_matrix(self) -> np.ndarray:
        """Get the correlation matrix."""
        return self.correlation_matrix.copy()
    
    def is_positive_definite(self) -> bool:
        """Check if correlation matrix is positive definite."""
        try:
            eigenvals = np.linalg.eigvals(self.correlation_matrix)
            return np.all(eigenvals > 1e-8)
        except LinAlgError:
            return False
    
    def make_positive_definite(self, regularization: float = 1e-6) -> None:
        """Make correlation matrix positive definite using eigenvalue regularization.
        
        Args:
            regularization: Minimum eigenvalue threshold
        """
        eigenvals, eigenvecs = np.linalg.eigh(self.correlation_matrix)
        
        # Regularize small eigenvalues
        eigenvals = np.maximum(eigenvals, regularization)
        
        # Reconstruct matrix
        self.correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Ensure diagonal is 1
        np.fill_diagonal(self.correlation_matrix, 1.0)


class ImanConovierTransform:
    """Iman-Conover rank correlation transformation."""
    
    @staticmethod
    def transform(samples: np.ndarray, 
                 correlation_matrix: np.ndarray,
                 method: str = "spearman") -> np.ndarray:
        """Apply Iman-Conover transformation to impose rank correlations.
        
        Args:
            samples: Independent samples of shape (n_samples, n_variables)
            correlation_matrix: Target correlation matrix
            method: Correlation method ("spearman" or "pearson")
            
        Returns:
            Transformed samples with target correlations
        """
        n_samples, n_vars = samples.shape
        
        if correlation_matrix.shape != (n_vars, n_vars):
            raise ValueError("Correlation matrix size doesn't match number of variables")
        
        # Check if correlation matrix is valid
        if not np.allclose(correlation_matrix, correlation_matrix.T):
            raise ValueError("Correlation matrix must be symmetric")
        
        # Convert to ranks
        ranks = np.zeros_like(samples)
        for i in range(n_vars):
            ranks[:, i] = stats.rankdata(samples[:, i])
        
        # Convert ranks to standard normal
        uniform_ranks = ranks / (n_samples + 1)
        normal_scores = stats.norm.ppf(uniform_ranks)
        
        # Apply Cholesky decomposition
        try:
            L = cholesky(correlation_matrix, lower=True)
        except LinAlgError:
            # Make matrix positive definite
            correlation_matrix_pd = ImanConovierTransform._make_positive_definite(correlation_matrix)
            L = cholesky(correlation_matrix_pd, lower=True)
        
        # Transform normal scores
        transformed_scores = normal_scores @ L.T
        
        # Convert back to ranks
        transformed_ranks = np.zeros_like(transformed_scores)
        for i in range(n_vars):
            transformed_ranks[:, i] = stats.norm.cdf(transformed_scores[:, i])
        
        # Apply to original samples
        result = np.zeros_like(samples)
        for i in range(n_vars):
            sorted_samples = np.sort(samples[:, i])
            ranks_for_var = transformed_ranks[:, i] * (n_samples - 1)
            
            # Interpolate to get final values
            result[:, i] = np.interp(ranks_for_var, 
                                   np.arange(n_samples), 
                                   sorted_samples)
        
        return result
    
    @staticmethod
    def _make_positive_definite(matrix: np.ndarray, 
                              regularization: float = 1e-6) -> np.ndarray:
        """Make matrix positive definite."""
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        eigenvals = np.maximum(eigenvals, regularization)
        
        result = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        np.fill_diagonal(result, 1.0)
        
        return result


class CholekyTransform:
    """Cholesky decomposition for normal copulas."""
    
    @staticmethod
    def transform_normal(samples: np.ndarray, 
                        correlation_matrix: np.ndarray) -> np.ndarray:
        """Transform independent normal samples using Cholesky decomposition.
        
        Args:
            samples: Independent standard normal samples
            correlation_matrix: Target correlation matrix
            
        Returns:
            Correlated normal samples
        """
        try:
            L = cholesky(correlation_matrix, lower=True)
        except LinAlgError:
            # Regularize matrix
            correlation_matrix = CholekyTransform._regularize_matrix(correlation_matrix)
            L = cholesky(correlation_matrix, lower=True)
        
        return samples @ L.T
    
    @staticmethod
    def _regularize_matrix(matrix: np.ndarray, 
                          regularization: float = 1e-8) -> np.ndarray:
        """Regularize matrix to make it positive definite."""
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        eigenvals = np.maximum(eigenvals, regularization)
        
        result = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        np.fill_diagonal(result, 1.0)
        
        return result


class CorrelationValidator:
    """Validates achieved correlations against targets."""
    
    @staticmethod
    def validate_correlations(samples: np.ndarray,
                            target_correlations: Dict[Tuple[int, int], float],
                            method: str = "spearman",
                            tolerance: float = 0.05) -> Dict[str, Union[bool, float, Dict]]:
        """Validate achieved correlations against targets.
        
        Args:
            samples: Sample array
            target_correlations: Dictionary of (i, j) -> correlation
            method: Correlation method ("spearman" or "pearson")
            tolerance: Acceptable tolerance
            
        Returns:
            Validation results
        """
        n_samples, n_vars = samples.shape
        
        # Calculate achieved correlations
        if method == "spearman":
            corr_func = stats.spearmanr
        elif method == "pearson":
            corr_func = stats.pearsonr
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        achieved = {}
        validation_results = {}
        all_within_tolerance = True
        max_error = 0.0
        
        for (i, j), target in target_correlations.items():
            if i >= n_vars or j >= n_vars:
                continue
                
            if method == "spearman":
                achieved_corr, _ = corr_func(samples[:, i], samples[:, j])
            else:
                achieved_corr, _ = corr_func(samples[:, i], samples[:, j])
            
            error = abs(achieved_corr - target)
            within_tolerance = error <= tolerance
            
            achieved[(i, j)] = achieved_corr
            validation_results[(i, j)] = {
                'target': target,
                'achieved': achieved_corr,
                'error': error,
                'within_tolerance': within_tolerance
            }
            
            if not within_tolerance:
                all_within_tolerance = False
            
            max_error = max(max_error, error)
        
        return {
            'all_within_tolerance': all_within_tolerance,
            'max_error': max_error,
            'tolerance': tolerance,
            'method': method,
            'n_samples': n_samples,
            'details': validation_results
        }
    
    @staticmethod
    def compute_correlation_matrix(samples: np.ndarray, 
                                 method: str = "spearman") -> np.ndarray:
        """Compute full correlation matrix from samples.
        
        Args:
            samples: Sample array
            method: Correlation method
            
        Returns:
            Correlation matrix
        """
        n_vars = samples.shape[1]
        corr_matrix = np.eye(n_vars)
        
        if method == "spearman":
            corr_func = lambda x, y: stats.spearmanr(x, y)[0]
        elif method == "pearson":
            corr_func = lambda x, y: stats.pearsonr(x, y)[0]
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                corr = corr_func(samples[:, i], samples[:, j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        
        return corr_matrix


def apply_correlations(samples: np.ndarray,
                      correlations: List[Dict[str, Union[str, float]]],
                      variable_names: List[str],
                      method: str = "iman_conover") -> np.ndarray:
    """Apply correlations to independent samples.
    
    Args:
        samples: Independent samples of shape (n_samples, n_variables)
        correlations: List of correlation specifications
        variable_names: Names of variables
        method: Correlation method ("iman_conover" or "cholesky")
        
    Returns:
        Correlated samples
    """
    if not correlations:
        return samples
    
    # Build correlation matrix
    corr_matrix = CorrelationMatrix(variable_names)
    
    for corr_spec in correlations:
        pair = corr_spec['pair']
        rho = corr_spec['rho']
        
        if len(pair) != 2:
            raise ValueError("Correlation pair must have exactly 2 elements")
        
        var1, var2 = pair
        corr_matrix.set_correlation(var1, var2, rho)
    
    # Ensure positive definite
    if not corr_matrix.is_positive_definite():
        corr_matrix.make_positive_definite()
    
    # Apply transformation
    target_matrix = corr_matrix.get_correlation_matrix()
    
    if method == "iman_conover":
        corr_method = correlations[0].get('method', 'spearman') if correlations else 'spearman'
        return ImanConovierTransform.transform(samples, target_matrix, corr_method)
    elif method == "cholesky":
        return CholekyTransform.transform_normal(samples, target_matrix)
    else:
        raise ValueError(f"Unknown correlation method: {method}")