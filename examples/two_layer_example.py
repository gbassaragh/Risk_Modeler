"""Example demonstrating two-layer uncertainty analysis.

Shows how to use epistemic and aleatory uncertainty to understand
the range of possible outcomes and confidence bounds around percentiles.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from risk_tool.core import (
    TwoLayerMonteCarlo,
    TwoLayerConfig,
    EpistemicParameter,
    create_default_two_layer_config,
    analyze_parameter_uncertainty,
)
from risk_tool.core.distributions import DistributionSampler


def simple_cost_model(epistemic_params, n_samples, random_state, **kwargs):
    """Simple cost model for demonstration.

    Models total cost as sum of labor + materials with uncertainty.

    Args:
        epistemic_params: Dict with 'labor_rate_mean', 'material_cost_mean'
        n_samples: Number of aleatory samples
        random_state: Random state

    Returns:
        Array of total cost samples
    """
    # Extract epistemic parameter values
    labor_rate_mean = epistemic_params.get("labor_rate_mean", 150.0)
    material_cost_mean = epistemic_params.get("material_cost_mean", 50000.0)

    # Fixed project parameters
    labor_hours = 1000  # Fixed scope

    # Aleatory distributions (inherent variability)
    labor_rate_dist = {
        "type": "normal",
        "mean": labor_rate_mean,  # Epistemic parameter
        "stdev": labor_rate_mean * 0.15,  # 15% aleatory variation
    }

    material_cost_dist = {
        "type": "triangular",
        "low": material_cost_mean * 0.8,
        "mode": material_cost_mean,  # Epistemic parameter
        "high": material_cost_mean * 1.3,
    }

    # Sample aleatory variations
    labor_rates = DistributionSampler.sample(labor_rate_dist, n_samples, random_state)
    material_costs = DistributionSampler.sample(
        material_cost_dist, n_samples, random_state
    )

    # Calculate total costs
    labor_costs = labor_rates * labor_hours
    total_costs = labor_costs + material_costs

    return total_costs


def main():
    """Run two-layer uncertainty analysis example."""

    print("Two-Layer Uncertainty Analysis Example")
    print("=" * 45)
    print()

    # Define epistemic parameters (parameter uncertainty)
    epistemic_params = [
        EpistemicParameter(
            "labor_rate_mean",
            {
                "type": "normal",
                "mean": 150.0,  # Best estimate of mean labor rate
                "stdev": 15.0,  # Uncertainty in the mean (epistemic)
            },
        ),
        EpistemicParameter(
            "material_cost_mean",
            {
                "type": "normal",
                "mean": 50000.0,  # Best estimate of material cost
                "stdev": 5000.0,  # Uncertainty in the estimate (epistemic)
            },
        ),
    ]

    # Create two-layer configuration
    config = TwoLayerConfig(
        n_epistemic=50,  # Outer loop: parameter uncertainty
        n_aleatory=1000,  # Inner loop: inherent variability
        target_percentiles=[50, 80, 90, 95],
        confidence_band=(10, 90),  # 80% confidence band
        parallel_processing=True,
    )

    # Create two-layer Monte Carlo
    two_layer = TwoLayerMonteCarlo(
        config=config, epistemic_parameters=epistemic_params, random_seed=42
    )

    # Run simulation
    print("Running two-layer simulation...")
    print(f"- Epistemic samples: {config.n_epistemic}")
    print(f"- Aleatory samples per epistemic: {config.n_aleatory}")
    print(f"- Total samples: {config.n_epistemic * config.n_aleatory:,}")
    print()

    results = two_layer.run_simulation(simple_cost_model)

    # Display results
    print("Results Summary:")
    print("-" * 20)
    print(results.summary_table())

    print("\nInterpretation:")
    print("-" * 15)
    print(
        "The confidence bands show the range of percentiles due to parameter uncertainty."
    )
    print("For example, if P80 shows 200,000 | 210,000 | 220,000:")
    print("- The P80 could be as low as 200,000 or as high as 220,000")
    print("- Our best estimate of P80 is 210,000")
    print("- This reflects uncertainty in our input parameters")
    print()

    # Show percentile bands in detail
    print("Detailed Percentile Analysis:")
    print("-" * 32)

    for percentile in config.target_percentiles:
        lower, mean, upper = results.get_percentile_band(percentile)
        spread = upper - lower
        rel_spread = (spread / mean) * 100 if mean != 0 else 0

        print(
            f"P{int(percentile):2d}: ${lower:8,.0f} - ${mean:8,.0f} - ${upper:8,.0f}  "
            f"(spread: ${spread:6,.0f}, {rel_spread:.1f}%)"
        )

    print()

    # Uncertainty decomposition
    print("Uncertainty Decomposition:")
    print("-" * 27)
    epistemic_cv = results.epistemic_stats.get("cv", 0)
    aleatory_cv = results.aleatory_stats.get("cv", 0)

    print(f"Epistemic uncertainty (parameter):  CV = {epistemic_cv:.3f}")
    print(f"Aleatory uncertainty (inherent):    CV = {aleatory_cv:.3f}")

    if epistemic_cv > aleatory_cv:
        print("→ Parameter uncertainty dominates (improve estimates)")
    else:
        print("→ Inherent variability dominates (manage project execution)")

    print()
    print("Analysis complete!")


def advanced_example():
    """Example using the convenience function."""
    print("\nAdvanced Example: Using Convenience Function")
    print("=" * 45)

    # Define distributions with epistemic uncertainty
    distributions = {
        "foundation_cost": {"type": "lognormal", "mean": 100000, "sigma": 0.3},
        "equipment_cost": {
            "type": "triangular",
            "low": 50000,
            "mode": 75000,
            "high": 120000,
        },
    }

    # Define epistemic uncertainty factors
    uncertainty_factors = {
        "foundation_cost": 0.20,  # 20% uncertainty in foundation cost estimate
        "equipment_cost": 0.15,  # 15% uncertainty in equipment cost estimate
    }

    def project_model(epistemic_params, n_samples, random_state):
        """Project cost model using epistemic parameters."""

        # Update distributions with epistemic values
        foundation_dist = {
            "type": "lognormal",
            "mean": epistemic_params.get("foundation_cost_mean", 100000),
            "sigma": 0.3,
        }

        equipment_dist = {
            "type": "triangular",
            "low": 50000,
            "mode": epistemic_params.get("equipment_cost_mean", 75000),
            "high": 120000,
        }

        # Sample costs
        foundation_costs = DistributionSampler.sample(
            foundation_dist, n_samples, random_state
        )
        equipment_costs = DistributionSampler.sample(
            equipment_dist, n_samples, random_state
        )

        return foundation_costs + equipment_costs

    # Run analysis using convenience function
    print("Running convenience function analysis...")
    results = analyze_parameter_uncertainty(
        distributions=distributions,
        model_function=project_model,
        uncertainty_factors=uncertainty_factors,
        n_epistemic=40,
        n_aleatory=800,
    )

    print("\nConvenience Function Results:")
    print(results.summary_table())


if __name__ == "__main__":
    main()
    advanced_example()
