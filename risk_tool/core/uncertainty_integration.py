"""Integration utilities for two-layer uncertainty with risk modeling.

Provides convenient interfaces to use two-layer Monte Carlo with
existing risk modeling components like RiskModel and correlation structures.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import warnings

from .two_layer import (
    TwoLayerMonteCarlo,
    TwoLayerConfig,
    EpistemicParameter,
    TwoLayerResults,
    create_default_two_layer_config,
)
from .data_models import RiskItem, WBSItem, Distribution, ProjectInfo
from .risk_driver import RiskSimulator

# from .correlation import CorrelationMatrix  # Will be implemented if needed
from .latent_factors import LatentFactorModel
from .distributions import DistributionSampler


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty analysis integration."""

    # Two-layer configuration
    n_epistemic: int = 50
    n_aleatory: int = 2000
    target_percentiles: List[float] = None
    confidence_band: Tuple[float, float] = (5, 95)

    # Epistemic uncertainty settings
    distribution_uncertainty: float = (
        0.15  # Default uncertainty in distribution parameters
    )
    correlation_uncertainty: float = 0.10  # Uncertainty in correlation coefficients
    factor_loading_uncertainty: float = 0.05  # Uncertainty in factor loadings

    # Integration settings
    parallel_processing: bool = True
    preserve_structure: bool = True  # Preserve correlation/factor structure

    def __post_init__(self):
        if self.target_percentiles is None:
            self.target_percentiles = [50, 80, 90, 95]


class RiskModelUncertaintyAnalyzer:
    """Analyzes epistemic uncertainty in risk models."""

    def __init__(
        self, risk_model: RiskModel, uncertainty_config: UncertaintyConfig = None
    ):
        """Initialize uncertainty analyzer.

        Args:
            risk_model: Base risk model to analyze
            uncertainty_config: Uncertainty analysis configuration
        """
        self.risk_model = risk_model
        self.config = uncertainty_config or UncertaintyConfig()

        # Extract epistemic parameters from risk model
        self.epistemic_parameters = self._extract_epistemic_parameters()

    def _extract_epistemic_parameters(self) -> List[EpistemicParameter]:
        """Extract epistemic parameters from risk model components."""
        epistemic_params = []

        # Add distribution parameter uncertainty
        epistemic_params.extend(self._create_distribution_epistemic_params())

        # Add correlation uncertainty if correlations exist
        if (
            hasattr(self.risk_model, "correlation_matrix")
            and self.risk_model.correlation_matrix
        ):
            epistemic_params.extend(self._create_correlation_epistemic_params())

        # Add factor model uncertainty if factor model exists
        if hasattr(self.risk_model, "factor_model") and self.risk_model.factor_model:
            epistemic_params.extend(self._create_factor_epistemic_params())

        return epistemic_params

    def _create_distribution_epistemic_params(self) -> List[EpistemicParameter]:
        """Create epistemic parameters for distribution parameters."""
        epistemic_params = []

        # Process WBS items
        for item in self.risk_model.wbs_items:
            # Quantity distribution uncertainty
            if item.dist_quantity:
                param_name = f"wbs_{item.code}_quantity_{self._get_param_name(item.dist_quantity)}"
                ep_param = self._create_dist_epistemic_param(
                    param_name, item.dist_quantity, self.config.distribution_uncertainty
                )
                if ep_param:
                    epistemic_params.append(ep_param)

            # Unit cost distribution uncertainty
            if item.dist_unit_cost:
                param_name = f"wbs_{item.code}_unit_cost_{self._get_param_name(item.dist_unit_cost)}"
                ep_param = self._create_dist_epistemic_param(
                    param_name,
                    item.dist_unit_cost,
                    self.config.distribution_uncertainty,
                )
                if ep_param:
                    epistemic_params.append(ep_param)

        # Process risk items
        for item in self.risk_model.risk_items:
            if item.distribution:
                param_name = (
                    f"risk_{item.name}_{self._get_param_name(item.distribution)}"
                )
                ep_param = self._create_dist_epistemic_param(
                    param_name, item.distribution, self.config.distribution_uncertainty
                )
                if ep_param:
                    epistemic_params.append(ep_param)

        return epistemic_params

    def _get_param_name(self, distribution: Distribution) -> str:
        """Get parameter name for distribution."""
        dist_config = distribution.dict()
        dist_type = dist_config.get("type", "").lower()

        if dist_type in ["normal", "truncnorm"]:
            return "mean"
        elif dist_type == "triangular":
            return "mode"
        elif dist_type in ["lognormal", "pert"]:
            return "mean"
        else:
            return "param"

    def _create_dist_epistemic_param(
        self, name: str, distribution: Distribution, uncertainty_factor: float
    ) -> Optional[EpistemicParameter]:
        """Create epistemic parameter for distribution parameter."""
        try:
            dist_config = distribution.dict()
            dist_type = dist_config.get("type", "").lower()

            if dist_type == "normal":
                base_mean = dist_config["mean"]
                epistemic_config = {
                    "type": "normal",
                    "mean": base_mean,
                    "stdev": abs(base_mean * uncertainty_factor),
                }

            elif dist_type == "triangular":
                base_mode = dist_config["mode"]
                epistemic_config = {
                    "type": "normal",
                    "mean": base_mode,
                    "stdev": abs(base_mode * uncertainty_factor),
                }

            elif dist_type == "lognormal":
                base_mean = dist_config["mean"]
                epistemic_config = {
                    "type": "lognormal",
                    "mean": base_mean,
                    "sigma": dist_config.get("sigma", 0.5) + uncertainty_factor,
                }

            elif dist_type == "pert":
                base_ml = dist_config["most_likely"]
                epistemic_config = {
                    "type": "normal",
                    "mean": base_ml,
                    "stdev": abs(base_ml * uncertainty_factor),
                }

            else:
                warnings.warn(f"Epistemic uncertainty not supported for {dist_type}")
                return None

            return EpistemicParameter(name, epistemic_config)

        except Exception as e:
            warnings.warn(f"Failed to create epistemic parameter for {name}: {e}")
            return None

    def _create_correlation_epistemic_params(self) -> List[EpistemicParameter]:
        """Create epistemic parameters for correlation coefficients."""
        epistemic_params = []

        corr_matrix = self.risk_model.correlation_matrix
        if not hasattr(corr_matrix, "correlations") or not corr_matrix.correlations:
            return epistemic_params

        # Add uncertainty to correlation coefficients
        for (var1, var2), corr_value in corr_matrix.correlations.items():
            if abs(corr_value) > 1e-6:  # Skip near-zero correlations
                param_name = f"corr_{var1}_{var2}"

                # Transform correlation to unbounded scale (Fisher transform)
                fisher_z = np.arctanh(corr_value)
                uncertainty_std = self.config.correlation_uncertainty

                epistemic_config = {
                    "type": "normal",
                    "mean": fisher_z,
                    "stdev": uncertainty_std,
                }

                epistemic_params.append(
                    EpistemicParameter(param_name, epistemic_config)
                )

        return epistemic_params

    def _create_factor_epistemic_params(self) -> List[EpistemicParameter]:
        """Create epistemic parameters for factor model loadings."""
        epistemic_params = []

        factor_model = self.risk_model.factor_model
        if not hasattr(factor_model, "factors"):
            return epistemic_params

        # Add uncertainty to factor loadings
        for factor in factor_model.factors:
            for var_name, loading in factor.loadings.items():
                if abs(loading) > 1e-6:  # Skip near-zero loadings
                    param_name = f"loading_{factor.name}_{var_name}"

                    epistemic_config = {
                        "type": "normal",
                        "mean": loading,
                        "stdev": abs(loading * self.config.factor_loading_uncertainty),
                    }

                    epistemic_params.append(
                        EpistemicParameter(param_name, epistemic_config)
                    )

        return epistemic_params

    def analyze_uncertainty(self) -> TwoLayerResults:
        """Run full uncertainty analysis.

        Returns:
            TwoLayerResults containing uncertainty analysis
        """
        # Create two-layer configuration
        two_layer_config = TwoLayerConfig(
            n_epistemic=self.config.n_epistemic,
            n_aleatory=self.config.n_aleatory,
            target_percentiles=self.config.target_percentiles,
            confidence_band=self.config.confidence_band,
            parallel_processing=self.config.parallel_processing,
        )

        # Create two-layer Monte Carlo
        two_layer = TwoLayerMonteCarlo(
            two_layer_config, self.epistemic_parameters, random_seed=42
        )

        # Run simulation
        return two_layer.run_simulation(self._aleatory_model_function)

    def _aleatory_model_function(
        self,
        epistemic_params: Dict[str, float],
        n_samples: int,
        random_state: np.random.RandomState,
    ) -> np.ndarray:
        """Aleatory model function for two-layer simulation.

        Args:
            epistemic_params: Dictionary of epistemic parameter values
            n_samples: Number of aleatory samples
            random_state: Random state for aleatory simulation

        Returns:
            Array of total cost samples
        """
        # Create modified risk model with epistemic parameter values
        modified_model = self._create_modified_risk_model(epistemic_params)

        # Run aleatory simulation
        results = modified_model.simulate(
            n_samples=n_samples,
            method="LHS",  # Use LHS for inner loop
            random_seed=random_state.randint(0, 2**31),
        )

        return results.total_cost_samples

    def _create_modified_risk_model(
        self, epistemic_params: Dict[str, float]
    ) -> RiskModel:
        """Create modified risk model with epistemic parameter values."""
        # Start with copy of original model
        modified_items = []

        # Modify WBS items
        for item in self.risk_model.wbs_items:
            modified_item = item.copy(deep=True)

            # Update quantity distribution
            if item.dist_quantity:
                param_key = f"wbs_{item.code}_quantity_{self._get_param_name(item.dist_quantity)}"
                if param_key in epistemic_params:
                    modified_item.dist_quantity = self._update_distribution(
                        item.dist_quantity,
                        self._get_param_name(item.dist_quantity),
                        epistemic_params[param_key],
                    )

            # Update unit cost distribution
            if item.dist_unit_cost:
                param_key = f"wbs_{item.code}_unit_cost_{self._get_param_name(item.dist_unit_cost)}"
                if param_key in epistemic_params:
                    modified_item.dist_unit_cost = self._update_distribution(
                        item.dist_unit_cost,
                        self._get_param_name(item.dist_unit_cost),
                        epistemic_params[param_key],
                    )

            modified_items.append(modified_item)

        # Modify risk items
        modified_risks = []
        for item in self.risk_model.risk_items:
            modified_risk = item.copy(deep=True)

            if item.distribution:
                param_key = (
                    f"risk_{item.name}_{self._get_param_name(item.distribution)}"
                )
                if param_key in epistemic_params:
                    modified_risk.distribution = self._update_distribution(
                        item.distribution,
                        self._get_param_name(item.distribution),
                        epistemic_params[param_key],
                    )

            modified_risks.append(modified_risk)

        # Create modified correlations and factor model if needed
        modified_correlations = self._update_correlations(epistemic_params)
        modified_factor_model = self._update_factor_model(epistemic_params)

        # Create new risk model
        modified_model = RiskModel(
            wbs_items=modified_items,
            risk_items=modified_risks,
            correlation_matrix=modified_correlations,
            factor_model=modified_factor_model,
        )

        return modified_model

    def _update_distribution(
        self, original_dist: Distribution, param_name: str, new_value: float
    ) -> Distribution:
        """Update distribution with new parameter value."""
        dist_dict = original_dist.dict()

        # Update the specific parameter
        if param_name in dist_dict:
            dist_dict[param_name] = new_value

        # Recreate distribution
        return Distribution(**dist_dict)

    def _update_correlations(
        self, epistemic_params: Dict[str, float]
    ) -> Optional[CorrelationMatrix]:
        """Update correlation matrix with epistemic parameters."""
        if (
            not hasattr(self.risk_model, "correlation_matrix")
            or not self.risk_model.correlation_matrix
        ):
            return None

        original_corr = self.risk_model.correlation_matrix
        if not hasattr(original_corr, "correlations") or not original_corr.correlations:
            return original_corr

        # Update correlations with epistemic values
        updated_correlations = {}
        for (var1, var2), orig_corr in original_corr.correlations.items():
            param_key = f"corr_{var1}_{var2}"

            if param_key in epistemic_params:
                # Transform back from Fisher scale
                fisher_z = epistemic_params[param_key]
                updated_corr = np.tanh(fisher_z)  # Inverse Fisher transform
                # Ensure correlation is in valid range
                updated_corr = np.clip(updated_corr, -0.99, 0.99)
                updated_correlations[(var1, var2)] = updated_corr
            else:
                updated_correlations[(var1, var2)] = orig_corr

        # Create new correlation matrix
        return CorrelationMatrix(correlations=updated_correlations)

    def _update_factor_model(
        self, epistemic_params: Dict[str, float]
    ) -> Optional[LatentFactorModel]:
        """Update factor model with epistemic parameters."""
        if (
            not hasattr(self.risk_model, "factor_model")
            or not self.risk_model.factor_model
        ):
            return None

        original_model = self.risk_model.factor_model

        # Update factor loadings
        updated_factors = []
        for factor in original_model.factors:
            updated_loadings = {}

            for var_name, orig_loading in factor.loadings.items():
                param_key = f"loading_{factor.name}_{var_name}"

                if param_key in epistemic_params:
                    updated_loadings[var_name] = epistemic_params[param_key]
                else:
                    updated_loadings[var_name] = orig_loading

            # Create updated factor
            updated_factor = factor.copy()
            updated_factor.loadings = updated_loadings
            updated_factors.append(updated_factor)

        # Create new factor model
        return LatentFactorModel(updated_factors, original_model.residual_correlations)


# Convenience functions


def analyze_project_uncertainty(
    project_config: ProjectConfig,
    wbs_items: List[WBSItem],
    risk_items: List[RiskItem],
    uncertainty_config: UncertaintyConfig = None,
) -> TwoLayerResults:
    """Analyze uncertainty for a complete project.

    Args:
        project_config: Project configuration
        wbs_items: WBS items with distributions
        risk_items: Risk items
        uncertainty_config: Uncertainty analysis configuration

    Returns:
        TwoLayerResults with uncertainty analysis
    """
    # Create risk model
    risk_model = RiskModel(wbs_items=wbs_items, risk_items=risk_items)

    # Create uncertainty analyzer
    analyzer = RiskModelUncertaintyAnalyzer(risk_model, uncertainty_config)

    # Run analysis
    return analyzer.analyze_uncertainty()


def quick_uncertainty_analysis(
    risk_model: RiskModel, n_epistemic: int = 50, n_aleatory: int = 2000
) -> TwoLayerResults:
    """Quick uncertainty analysis with default settings.

    Args:
        risk_model: Risk model to analyze
        n_epistemic: Number of epistemic samples
        n_aleatory: Number of aleatory samples

    Returns:
        TwoLayerResults with uncertainty analysis
    """
    config = UncertaintyConfig(n_epistemic=n_epistemic, n_aleatory=n_aleatory)

    analyzer = RiskModelUncertaintyAnalyzer(risk_model, config)
    return analyzer.analyze_uncertainty()
