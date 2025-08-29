"""Latent factor correlation model for coherent risk dependencies.

Implements factor models where observable variables are driven by
common latent factors (e.g., weather, commodity prices, labor market).
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple
import warnings
from dataclasses import dataclass


@dataclass
class LatentFactor:
    """Latent factor definition."""

    name: str
    distribution_config: Dict[str, Any]
    loadings: Dict[str, float]  # Variable name -> loading


@dataclass
class FactorStructure:
    """Complete factor structure with residual correlations."""

    factors: List[LatentFactor]
    residual_correlations: Dict[Tuple[str, str], float]
    variable_names: List[str]


class LatentFactorModel:
    """Latent factor model for correlation structure."""

    def __init__(
        self,
        factors: List[LatentFactor],
        residual_correlations: Optional[Dict[Tuple[str, str], float]] = None,
    ):
        """Initialize latent factor model.

        Args:
            factors: List of latent factors
            residual_correlations: Additional pairwise correlations not explained by factors
        """
        self.factors = factors
        self.residual_correlations = residual_correlations or {}

        # Extract all variable names
        self.variable_names = set()
        for factor in factors:
            self.variable_names.update(factor.loadings.keys())
        self.variable_names = sorted(list(self.variable_names))

        self.n_variables = len(self.variable_names)
        self.n_factors = len(factors)

        # Build loading matrix
        self.loading_matrix = self._build_loading_matrix()

        # Build residual correlation matrix
        self.residual_correlation_matrix = self._build_residual_correlation_matrix()

        # Validate the model
        self._validate_model()

    def _build_loading_matrix(self) -> np.ndarray:
        """Build factor loading matrix (variables x factors).

        Returns:
            Loading matrix L where L[i,j] is loading of variable i on factor j
        """
        loadings = np.zeros((self.n_variables, self.n_factors))

        for j, factor in enumerate(self.factors):
            for var_name, loading in factor.loadings.items():
                if var_name in self.variable_names:
                    i = self.variable_names.index(var_name)
                    loadings[i, j] = loading

        return loadings

    def _build_residual_correlation_matrix(self) -> np.ndarray:
        """Build residual correlation matrix.

        Returns:
            Residual correlation matrix for idiosyncratic components
        """
        residual_corr = np.eye(self.n_variables)

        for (var1, var2), corr in self.residual_correlations.items():
            if var1 in self.variable_names and var2 in self.variable_names:
                i = self.variable_names.index(var1)
                j = self.variable_names.index(var2)
                residual_corr[i, j] = corr
                residual_corr[j, i] = corr

        return residual_corr

    def _validate_model(self):
        """Validate the factor model specification."""
        # Check that residual correlation matrix is positive semi-definite
        try:
            eigenvals = np.linalg.eigvals(self.residual_correlation_matrix)
            if np.any(eigenvals < -1e-8):
                warnings.warn(
                    "Residual correlation matrix is not positive semi-definite"
                )
        except:
            warnings.warn("Could not validate residual correlation matrix")

        # Check that implied correlation matrix is valid
        implied_corr = self.get_implied_correlation_matrix()
        try:
            eigenvals = np.linalg.eigvals(implied_corr)
            if np.any(eigenvals < -1e-8):
                warnings.warn(
                    "Implied correlation matrix is not positive semi-definite"
                )
        except:
            warnings.warn("Could not validate implied correlation matrix")

    def get_implied_correlation_matrix(self) -> np.ndarray:
        """Calculate implied correlation matrix from factor structure.

        Returns:
            Implied correlation matrix
        """
        # Factor contribution: L @ L.T where L is loading matrix
        factor_contribution = self.loading_matrix @ self.loading_matrix.T

        # Total correlation = factor contribution + residual correlations
        # But we need to be careful about the diagonal elements

        # Calculate diagonal elements (total variance)
        factor_variances = np.diag(factor_contribution)
        residual_variances = np.diag(self.residual_correlation_matrix)
        total_variances = factor_variances + residual_variances

        # Implied covariance matrix
        implied_cov = factor_contribution.copy()

        # Add residual covariances (off-diagonal only)
        for i in range(self.n_variables):
            for j in range(self.n_variables):
                if i != j:
                    implied_cov[i, j] += self.residual_correlation_matrix[
                        i, j
                    ] * np.sqrt(residual_variances[i] * residual_variances[j])

        # Add residual variances to diagonal
        np.fill_diagonal(implied_cov, total_variances)

        # Convert to correlation matrix
        std_devs = np.sqrt(np.diag(implied_cov))
        implied_corr = implied_cov / np.outer(std_devs, std_devs)

        return implied_corr

    def generate_correlated_samples(
        self, n_samples: int, random_state: np.random.RandomState
    ) -> np.ndarray:
        """Generate correlated samples using factor structure.

        Args:
            n_samples: Number of samples
            random_state: Random state

        Returns:
            Array of shape (n_samples, n_variables) with correlated samples
        """
        from .distributions import DistributionSampler

        # Generate factor samples
        factor_samples = np.zeros((n_samples, self.n_factors))

        for j, factor in enumerate(self.factors):
            factor_samples[:, j] = DistributionSampler.sample(
                factor.distribution_config, n_samples, random_state
            )

        # Generate residual samples (standard normal)
        residual_samples = random_state.multivariate_normal(
            np.zeros(self.n_variables), self.residual_correlation_matrix, size=n_samples
        )

        # Combine factors and residuals
        # Variable_i = sum_j(loading_ij * factor_j) + residual_i
        variable_samples = (factor_samples @ self.loading_matrix.T) + residual_samples

        return variable_samples

    def apply_to_uniform_samples(self, uniform_samples: np.ndarray) -> np.ndarray:
        """Transform uniform samples to have the factor correlation structure.

        Args:
            uniform_samples: Uniform samples in [0,1]^n_variables

        Returns:
            Transformed samples with factor correlation structure
        """
        n_samples, n_vars = uniform_samples.shape

        if n_vars != self.n_variables:
            raise ValueError(f"Expected {self.n_variables} variables, got {n_vars}")

        # Convert uniform samples to standard normal
        standard_normal = stats.norm.ppf(uniform_samples)

        # Apply factor structure transformation
        # We need to find a transformation matrix such that when applied
        # to independent normal samples, we get the desired correlation structure

        # Get target correlation matrix
        target_corr = self.get_implied_correlation_matrix()

        # Use Cholesky decomposition for transformation
        try:
            L = np.linalg.cholesky(target_corr)
        except np.linalg.LinAlgError:
            # If not positive definite, use eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(target_corr)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive
            L = eigenvecs @ np.diag(np.sqrt(eigenvals))

        # Transform samples
        transformed = (L @ standard_normal.T).T

        # Convert back to uniform
        correlated_uniform = stats.norm.cdf(transformed)

        return correlated_uniform

    def get_factor_contributions(self) -> Dict[str, Dict[str, float]]:
        """Get factor contributions to each variable's variance.

        Returns:
            Dictionary mapping variable names to factor contributions
        """
        contributions = {}

        for i, var_name in enumerate(self.variable_names):
            var_contributions = {}

            # Factor contributions
            for j, factor in enumerate(self.factors):
                loading = self.loading_matrix[i, j]
                contribution = loading**2
                var_contributions[factor.name] = contribution

            # Residual contribution
            residual_variance = self.residual_correlation_matrix[i, i]
            var_contributions["residual"] = residual_variance

            contributions[var_name] = var_contributions

        return contributions

    def get_common_variance_explained(self) -> Dict[str, float]:
        """Get proportion of variance explained by common factors for each variable.

        Returns:
            Dictionary mapping variable names to proportion of common variance
        """
        variance_explained = {}

        for i, var_name in enumerate(self.variable_names):
            # Total factor loading squared
            factor_variance = np.sum(self.loading_matrix[i, :] ** 2)

            # Total variance (assuming unit variance for each variable)
            residual_variance = self.residual_correlation_matrix[i, i]
            total_variance = factor_variance + residual_variance

            if total_variance > 0:
                explained = factor_variance / total_variance
            else:
                explained = 0.0

            variance_explained[var_name] = explained

        return variance_explained


class FactorModelBuilder:
    """Builder for constructing factor models from high-level specifications."""

    def __init__(self):
        """Initialize builder."""
        self.factors = []
        self.residual_correlations = {}
        self.variable_names = set()

    def add_factor(
        self,
        name: str,
        distribution_config: Dict[str, Any],
        variable_loadings: Dict[str, float],
    ) -> "FactorModelBuilder":
        """Add a latent factor.

        Args:
            name: Factor name
            distribution_config: Factor distribution configuration
            variable_loadings: Variable loadings on this factor

        Returns:
            Self for chaining
        """
        factor = LatentFactor(name, distribution_config, variable_loadings)
        self.factors.append(factor)
        self.variable_names.update(variable_loadings.keys())
        return self

    def add_residual_correlation(
        self, var1: str, var2: str, correlation: float
    ) -> "FactorModelBuilder":
        """Add residual correlation between variables.

        Args:
            var1: First variable
            var2: Second variable
            correlation: Correlation coefficient

        Returns:
            Self for chaining
        """
        if abs(correlation) > 1:
            raise ValueError("Correlation must be between -1 and 1")

        key = tuple(sorted([var1, var2]))
        self.residual_correlations[key] = correlation
        return self

    def build(self) -> LatentFactorModel:
        """Build the factor model.

        Returns:
            LatentFactorModel instance
        """
        if not self.factors:
            raise ValueError("Must add at least one factor")

        return LatentFactorModel(self.factors, self.residual_correlations)


def create_commodity_factor_model(
    commodity_variables: List[str],
    steel_loading: float = 0.8,
    aluminum_loading: float = 0.7,
    copper_loading: float = 0.6,
) -> LatentFactorModel:
    """Create a commodity price factor model for T&D projects.

    Args:
        commodity_variables: List of variables affected by commodity prices
        steel_loading: Loading on steel factor
        aluminum_loading: Loading on aluminum factor
        copper_loading: Loading on copper factor

    Returns:
        LatentFactorModel for commodity prices
    """
    builder = FactorModelBuilder()

    # Steel factor
    steel_vars = [
        v
        for v in commodity_variables
        if "steel" in v.lower() or "structure" in v.lower()
    ]
    if steel_vars:
        steel_loadings = {var: steel_loading for var in steel_vars}
        builder.add_factor(
            "steel_commodity",
            {"type": "normal", "mean": 0, "stdev": 1},  # Standard normal factor
            steel_loadings,
        )

    # Aluminum factor
    aluminum_vars = [
        v
        for v in commodity_variables
        if "aluminum" in v.lower() or "conductor" in v.lower()
    ]
    if aluminum_vars:
        aluminum_loadings = {var: aluminum_loading for var in aluminum_vars}
        builder.add_factor(
            "aluminum_commodity",
            {"type": "normal", "mean": 0, "stdev": 1},
            aluminum_loadings,
        )

    # Copper factor
    copper_vars = [v for v in commodity_variables if "copper" in v.lower()]
    if copper_vars:
        copper_loadings = {var: copper_loading for var in copper_vars}
        builder.add_factor(
            "copper_commodity",
            {"type": "normal", "mean": 0, "stdev": 1},
            copper_loadings,
        )

    return builder.build()


def create_weather_factor_model(
    weather_variables: List[str], weather_loading: float = 0.7
) -> LatentFactorModel:
    """Create a weather factor model for construction activities.

    Args:
        weather_variables: List of variables affected by weather
        weather_loading: Loading on weather factor

    Returns:
        LatentFactorModel for weather impacts
    """
    builder = FactorModelBuilder()

    weather_loadings = {var: weather_loading for var in weather_variables}
    builder.add_factor(
        "weather",
        {"type": "normal", "mean": 0, "stdev": 1},  # Standard normal factor
        weather_loadings,
    )

    return builder.build()


def create_labor_market_factor_model(
    labor_variables: List[str], labor_loading: float = 0.6
) -> LatentFactorModel:
    """Create a labor market factor model.

    Args:
        labor_variables: List of variables affected by labor market
        labor_loading: Loading on labor factor

    Returns:
        LatentFactorModel for labor market impacts
    """
    builder = FactorModelBuilder()

    labor_loadings = {var: labor_loading for var in labor_variables}
    builder.add_factor(
        "labor_market",
        {"type": "normal", "mean": 0, "stdev": 1},  # Standard normal factor
        labor_loadings,
    )

    return builder.build()


def combine_factor_models(models: List[LatentFactorModel]) -> LatentFactorModel:
    """Combine multiple factor models into a single model.

    Args:
        models: List of factor models to combine

    Returns:
        Combined LatentFactorModel
    """
    if not models:
        raise ValueError("Must provide at least one model to combine")

    combined_factors = []
    combined_residuals = {}

    for model in models:
        combined_factors.extend(model.factors)
        combined_residuals.update(model.residual_correlations)

    return LatentFactorModel(combined_factors, combined_residuals)
