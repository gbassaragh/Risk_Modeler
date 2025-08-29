"""Example demonstrating distribution calibration and fitting.

Shows how to use MLE and Bayesian fitting, model selection,
and historical data processing for risk modeling calibration.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from risk_tool.fit import (
    fit_distribution_mle,
    fit_distribution_bayesian,
    compare_distributions,
    select_best_distribution,
    process_cost_data,
    detect_outliers,
    analyze_seasonality,
)


def generate_sample_cost_data():
    """Generate realistic cost data for demonstration."""
    np.random.seed(42)

    # Base costs (lognormal distribution)
    n_samples = 200
    mu = np.log(50000)  # Median cost around $50k
    sigma = 0.4  # Moderate variability
    base_costs = np.random.lognormal(mu, sigma, n_samples)

    # Add some outliers (5% of data)
    n_outliers = int(0.05 * n_samples)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    base_costs[outlier_indices] *= np.random.uniform(3, 8, n_outliers)  # High outliers

    # Add some seasonal pattern (for time series data)
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="W")
    seasonal_factor = 1 + 0.1 * np.sin(
        2 * np.pi * np.arange(n_samples) / 52
    )  # Annual cycle
    seasonal_costs = base_costs * seasonal_factor

    # Create DataFrame
    data = pd.DataFrame(
        {"date": dates, "cost": seasonal_costs, "base_cost": base_costs}
    )

    return data


def demonstrate_mle_fitting():
    """Demonstrate maximum likelihood estimation."""
    print("Maximum Likelihood Estimation (MLE)")
    print("=" * 40)

    # Generate sample data
    data = generate_sample_cost_data()
    costs = data["cost"].values

    # Fit normal distribution
    print("Fitting Normal Distribution:")
    normal_result = fit_distribution_mle(costs, "normal")
    print(normal_result.summary())
    print()

    # Fit lognormal distribution
    print("Fitting Lognormal Distribution:")
    lognormal_result = fit_distribution_mle(costs, "lognormal")
    print(lognormal_result.summary())
    print()

    # Fit gamma distribution
    print("Fitting Gamma Distribution:")
    gamma_result = fit_distribution_mle(costs, "gamma")
    print(gamma_result.summary())
    print()

    return [normal_result, lognormal_result, gamma_result]


def demonstrate_bayesian_fitting():
    """Demonstrate Bayesian parameter estimation."""
    print("Bayesian Parameter Estimation")
    print("=" * 30)

    try:
        # Generate smaller dataset for faster Bayesian fitting
        np.random.seed(42)
        costs = np.random.lognormal(np.log(50000), 0.4, 100)

        # Fit normal distribution with Bayesian inference
        print("Bayesian fitting of Normal distribution:")
        bayesian_result = fit_distribution_bayesian(costs, "normal", n_samples=1000)
        print(bayesian_result.summary())
        print()

        # Show parameter uncertainties
        if bayesian_result.parameter_uncertainties:
            print("Parameter Uncertainties:")
            for param, uncertainty in bayesian_result.parameter_uncertainties.items():
                mean_val = bayesian_result.parameters[param]
                print(f"  {param}: {mean_val:.2f} ± {uncertainty:.2f}")

        print()

        # Convergence diagnostics
        if bayesian_result.convergence_info:
            print("Convergence Diagnostics:")
            for metric, values in bayesian_result.convergence_info.items():
                print(f"  {metric}: {values}")

        print()
        return bayesian_result

    except RuntimeError as e:
        print(f"Bayesian fitting not available: {e}")
        print("Install PyMC with: pip install pymc")
        return None


def demonstrate_model_comparison():
    """Demonstrate comparing multiple distributions."""
    print("Model Comparison and Selection")
    print("=" * 30)

    # Generate sample data
    data = generate_sample_cost_data()
    costs = data["cost"].values

    # Compare multiple distributions
    distributions = ["normal", "lognormal", "gamma", "weibull"]
    print(f"Comparing distributions: {distributions}")
    print()

    comparison_results = compare_distributions(costs, distributions)

    # Show results ranked by AIC
    print("Results (ranked by AIC):")
    print("-" * 25)
    for i, result in enumerate(comparison_results, 1):
        print(f"{i}. {result.distribution_type}")
        print(f"   AIC: {result.aic:.2f}")
        print(f"   BIC: {result.bic:.2f}")
        print(f"   Log-likelihood: {result.log_likelihood:.2f}")
        if result.ks_pvalue:
            print(f"   KS test p-value: {result.ks_pvalue:.4f}")
        print()

    # Best distribution
    best = comparison_results[0]
    print(f"Best distribution: {best.distribution_type}")
    print(f"Best AIC: {best.aic:.2f}")

    return comparison_results


def demonstrate_automatic_selection():
    """Demonstrate automatic model selection."""
    print("Automatic Model Selection")
    print("=" * 25)

    # Generate sample data
    data = generate_sample_cost_data()
    costs = data["cost"].values

    # Automatic selection with default criteria
    selection_result = select_best_distribution(costs)

    print("Selection Results:")
    print(selection_result.summary())
    print()

    # Show detailed comparison
    if selection_result.delta_aic:
        print("ΔAIC (difference from best):")
        for dist, delta in selection_result.delta_aic.items():
            print(f"  {dist}: {delta:.2f}")
        print()

    return selection_result


def demonstrate_data_processing():
    """Demonstrate historical data processing."""
    print("Historical Data Processing")
    print("=" * 26)

    # Generate sample data with quality issues
    data = generate_sample_cost_data()

    # Add some missing values
    missing_indices = np.random.choice(len(data), 10, replace=False)
    data.loc[missing_indices, "cost"] = np.nan

    # Add some zero values
    zero_indices = np.random.choice(len(data), 5, replace=False)
    data.loc[zero_indices, "cost"] = 0

    # Process the data
    processing_results = process_cost_data(
        data, value_column="cost", date_column="date", clean_data=True
    )

    # Show quality report
    print("Data Quality Report:")
    print(processing_results["quality_report"].summary())
    print()

    # Show outlier detection
    print("Outlier Detection:")
    print(processing_results["outlier_result"].summary())
    print()

    # Show seasonality analysis
    print("Seasonality Analysis:")
    print(processing_results["seasonality_result"].summary())
    print()

    # Processing summary
    print("Processing Summary:")
    print(f"Original samples: {processing_results['n_original']}")
    print(f"Processed samples: {processing_results['n_processed']}")
    print("Notes:")
    for note in processing_results["processing_notes"]:
        print(f"  - {note}")

    return processing_results


def demonstrate_outlier_detection():
    """Demonstrate outlier detection methods."""
    print("Outlier Detection Methods")
    print("=" * 25)

    # Generate data with known outliers
    np.random.seed(42)
    normal_data = np.random.normal(50000, 10000, 100)
    outliers = [120000, 150000, 200000]  # Clear outliers
    data_with_outliers = np.concatenate([normal_data, outliers])

    methods = ["iqr", "zscore", "modified_zscore"]

    for method in methods:
        print(f"\n{method.upper()} Method:")
        print("-" * 15)

        outlier_result = detect_outliers(data_with_outliers, method=method)
        print(outlier_result.summary())

        print(f"Detected outliers: {outlier_result.outlier_values}")


def demonstrate_seasonality_analysis():
    """Demonstrate seasonality analysis."""
    print("Seasonality Analysis")
    print("=" * 20)

    # Generate data with strong seasonal pattern
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=104, freq="W")  # 2 years weekly

    # Base trend
    trend = 50000 + 1000 * np.arange(104)

    # Strong seasonal pattern (annual cycle)
    seasonal = 5000 * np.sin(2 * np.pi * np.arange(104) / 52)

    # Random noise
    noise = np.random.normal(0, 2000, 104)

    # Combined series
    ts_data = trend + seasonal + noise

    # Analyze seasonality
    seasonality_result = analyze_seasonality(ts_data, dates)

    print("Seasonality Analysis Results:")
    print(seasonality_result.summary())

    if seasonality_result.has_seasonality:
        print("\nSeasonal factors by period:")
        if seasonality_result.seasonal_factors is not None:
            for i, factor in enumerate(
                seasonality_result.seasonal_factors[:12]
            ):  # Show first 12
                print(f"  Period {i+1}: {factor:6.0f}")

    return seasonality_result


def create_visualization():
    """Create visualizations of the fitting results."""
    print("Creating Visualizations")
    print("=" * 20)

    try:
        import matplotlib.pyplot as plt

        # Generate sample data
        data = generate_sample_cost_data()
        costs = data["cost"].values

        # Fit multiple distributions
        distributions = ["normal", "lognormal", "gamma"]
        results = compare_distributions(costs, distributions)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Distribution Fitting Results", fontsize=14)

        # Histogram of data
        axes[0, 0].hist(
            costs,
            bins=30,
            density=True,
            alpha=0.7,
            color="lightblue",
            edgecolor="black",
        )
        axes[0, 0].set_title("Histogram of Cost Data")
        axes[0, 0].set_xlabel("Cost ($)")
        axes[0, 0].set_ylabel("Density")

        # Q-Q plots for best distributions
        x = np.linspace(np.min(costs), np.max(costs), 100)

        colors = ["red", "green", "blue"]
        for i, (result, color) in enumerate(zip(results[:3], colors)):
            # Plot fitted PDF
            try:
                from scipy import stats

                dist_name = result.distribution_type
                if dist_name == "normal":
                    y = stats.norm.pdf(
                        x,
                        loc=result.parameters["loc"],
                        scale=result.parameters["scale"],
                    )
                elif dist_name == "lognormal":
                    y = stats.lognorm.pdf(
                        x, s=result.parameters["s"], scale=result.parameters["scale"]
                    )
                elif dist_name == "gamma":
                    y = stats.gamma.pdf(
                        x, a=result.parameters["a"], scale=result.parameters["scale"]
                    )

                axes[0, 1].plot(
                    x,
                    y,
                    color=color,
                    label=f"{dist_name} (AIC: {result.aic:.1f})",
                    linewidth=2,
                )
            except:
                pass

        axes[0, 1].hist(costs, bins=30, density=True, alpha=0.3, color="lightgray")
        axes[0, 1].set_title("Fitted Distributions")
        axes[0, 1].set_xlabel("Cost ($)")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].legend()

        # AIC comparison
        dist_names = [r.distribution_type for r in results]
        aics = [r.aic for r in results]
        axes[1, 0].bar(dist_names, aics, color="lightcoral")
        axes[1, 0].set_title("AIC Comparison (Lower is Better)")
        axes[1, 0].set_ylabel("AIC")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Time series plot
        axes[1, 1].plot(data["date"], data["cost"], alpha=0.7, linewidth=1)
        axes[1, 1].set_title("Cost Time Series")
        axes[1, 1].set_xlabel("Date")
        axes[1, 1].set_ylabel("Cost ($)")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig("calibration_results.png", dpi=300, bbox_inches="tight")
        print("Visualization saved as 'calibration_results.png'")

    except ImportError:
        print("Matplotlib not available for visualization")
    except Exception as e:
        print(f"Visualization failed: {e}")


def main():
    """Run all calibration examples."""
    print("Distribution Calibration and Fitting Examples")
    print("=" * 45)
    print()

    # MLE fitting
    mle_results = demonstrate_mle_fitting()
    print("\n" + "=" * 50 + "\n")

    # Bayesian fitting (if available)
    bayesian_result = demonstrate_bayesian_fitting()
    print("\n" + "=" * 50 + "\n")

    # Model comparison
    comparison_results = demonstrate_model_comparison()
    print("\n" + "=" * 50 + "\n")

    # Automatic selection
    selection_result = demonstrate_automatic_selection()
    print("\n" + "=" * 50 + "\n")

    # Data processing
    processing_results = demonstrate_data_processing()
    print("\n" + "=" * 50 + "\n")

    # Outlier detection
    demonstrate_outlier_detection()
    print("\n" + "=" * 50 + "\n")

    # Seasonality analysis
    seasonality_result = demonstrate_seasonality_analysis()
    print("\n" + "=" * 50 + "\n")

    # Visualization
    create_visualization()

    print("\nCalibration examples completed!")
    print("\nKey Takeaways:")
    print("- Lognormal distribution typically fits cost data well")
    print("- AIC/BIC help select the best model objectively")
    print("- Bayesian methods provide parameter uncertainty")
    print("- Data quality assessment is crucial before fitting")
    print("- Outlier detection helps identify data issues")
    print("- Seasonality analysis reveals temporal patterns")


if __name__ == "__main__":
    main()
