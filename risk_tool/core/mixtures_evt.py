"""Advanced distributions: mixtures, extreme value theory, and empirical distributions.

Implements sophisticated distribution types for tail risk modeling and
empirical data fitting.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings


class MixtureDistribution:
    """Mixture distribution implementation."""

    def __init__(self, components: List[Dict[str, Any]], weights: List[float]):
        """Initialize mixture distribution.

        Args:
            components: List of component distribution configurations
            weights: Mixing weights (must sum to 1)
        """
        if len(components) != len(weights):
            raise ValueError("Number of components must match number of weights")

        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

        self.components = components
        self.weights = np.array(weights)
        self.n_components = len(components)

        # Initialize component distributions
        from .distributions import DistributionSampler

        self.component_samplers = []
        for comp in components:
            self.component_samplers.append(comp)

    def sample(self, size: int, random_state: np.random.RandomState) -> np.ndarray:
        """Sample from mixture distribution.

        Args:
            size: Number of samples
            random_state: Random state

        Returns:
            Array of samples
        """
        from .distributions import DistributionSampler

        # Choose components for each sample
        component_choices = random_state.choice(
            self.n_components, size=size, p=self.weights
        )

        samples = np.zeros(size)

        # Sample from each chosen component
        for i in range(size):
            comp_idx = component_choices[i]
            comp_sample = DistributionSampler.sample(
                self.component_samplers[comp_idx], 1, random_state
            )
            samples[i] = comp_sample[0]

        return samples

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function.

        Args:
            x: Values to evaluate

        Returns:
            PDF values
        """
        from .distributions import get_distribution_pdf

        pdf_vals = np.zeros_like(x)

        for i, (comp, weight) in enumerate(zip(self.components, self.weights)):
            comp_pdf = get_distribution_pdf(comp, x)
            pdf_vals += weight * comp_pdf

        return pdf_vals

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function.

        Args:
            x: Values to evaluate

        Returns:
            CDF values
        """
        from .distributions import get_distribution_cdf

        cdf_vals = np.zeros_like(x)

        for i, (comp, weight) in enumerate(zip(self.components, self.weights)):
            comp_cdf = get_distribution_cdf(comp, x)
            cdf_vals += weight * comp_cdf

        return cdf_vals


class GeneralizedParetoDistribution:
    """Generalized Pareto Distribution for extreme value modeling."""

    def __init__(self, shape: float, scale: float, threshold: float):
        """Initialize GPD.

        Args:
            shape: Shape parameter (xi)
            scale: Scale parameter (sigma)
            threshold: Threshold parameter (u)
        """
        self.shape = shape  # xi
        self.scale = scale  # sigma
        self.threshold = threshold  # u

        if scale <= 0:
            raise ValueError("Scale parameter must be positive")

    def sample(self, size: int, random_state: np.random.RandomState) -> np.ndarray:
        """Sample from GPD.

        Args:
            size: Number of samples
            random_state: Random state

        Returns:
            Array of samples
        """
        u = random_state.uniform(0, 1, size)

        if abs(self.shape) < 1e-8:  # Shape ≈ 0 (exponential case)
            samples = -self.scale * np.log(1 - u)
        else:
            samples = (self.scale / self.shape) * (np.power(1 - u, -self.shape) - 1)

        return samples + self.threshold

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """PDF of GPD.

        Args:
            x: Values to evaluate

        Returns:
            PDF values
        """
        x = np.atleast_1d(x)
        pdf_vals = np.zeros_like(x)

        # Only defined for x > threshold
        valid = x >= self.threshold
        z = (x[valid] - self.threshold) / self.scale

        if abs(self.shape) < 1e-8:  # Exponential case
            pdf_vals[valid] = (1 / self.scale) * np.exp(-z)
        else:
            if self.shape > 0:  # Finite upper bound
                valid_z = z < (1 / self.shape)
                z = z[valid_z]
                pdf_vals[valid][valid_z] = (1 / self.scale) * np.power(
                    1 + self.shape * z, -(1 / self.shape + 1)
                )
            else:  # No upper bound
                pdf_vals[valid] = (1 / self.scale) * np.power(
                    1 + self.shape * z, -(1 / self.shape + 1)
                )

        return pdf_vals

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """CDF of GPD.

        Args:
            x: Values to evaluate

        Returns:
            CDF values
        """
        x = np.atleast_1d(x)
        cdf_vals = np.zeros_like(x)

        # CDF is 0 for x < threshold
        valid = x >= self.threshold
        z = (x[valid] - self.threshold) / self.scale

        if abs(self.shape) < 1e-8:  # Exponential case
            cdf_vals[valid] = 1 - np.exp(-z)
        else:
            if self.shape > 0:  # Finite upper bound
                valid_z = z < (1 / self.shape)
                z = z[valid_z]
                cdf_vals[valid][valid_z] = 1 - np.power(
                    1 + self.shape * z, -1 / self.shape
                )
                cdf_vals[valid][~valid_z] = 1  # Above upper bound
            else:  # No upper bound
                cdf_vals[valid] = 1 - np.power(1 + self.shape * z, -1 / self.shape)

        return cdf_vals

    @classmethod
    def fit_peaks_over_threshold(
        cls, data: np.ndarray, threshold: float, method: str = "mle"
    ) -> "GeneralizedParetoDistribution":
        """Fit GPD to exceedances over threshold.

        Args:
            data: Data array
            threshold: Threshold value
            method: Fitting method ('mle' or 'pwm')

        Returns:
            Fitted GPD instance
        """
        exceedances = data[data > threshold] - threshold

        if len(exceedances) < 10:
            warnings.warn("Insufficient exceedances for reliable GPD fitting")

        if method == "mle":
            return cls._fit_mle(exceedances, threshold)
        elif method == "pwm":
            return cls._fit_pwm(exceedances, threshold)
        else:
            raise ValueError(f"Unknown fitting method: {method}")

    @classmethod
    def _fit_mle(
        cls, exceedances: np.ndarray, threshold: float
    ) -> "GeneralizedParetoDistribution":
        """Fit GPD using maximum likelihood estimation."""

        def neg_log_likelihood(params):
            shape, scale = params
            if scale <= 0:
                return np.inf

            try:
                gpd = cls(shape, scale, 0)  # Threshold 0 for exceedances
                pdf_vals = gpd.pdf(exceedances)
                if np.any(pdf_vals <= 0):
                    return np.inf
                return -np.sum(np.log(pdf_vals))
            except:
                return np.inf

        # Initial guess
        x0 = [0.1, np.std(exceedances)]

        # Optimize
        result = minimize(neg_log_likelihood, x0, method="Nelder-Mead")

        if not result.success:
            warnings.warn("GPD MLE fitting did not converge")

        shape, scale = result.x
        return cls(shape, scale, threshold)

    @classmethod
    def _fit_pwm(
        cls, exceedances: np.ndarray, threshold: float
    ) -> "GeneralizedParetoDistribution":
        """Fit GPD using probability weighted moments."""
        n = len(exceedances)
        exceedances_sorted = np.sort(exceedances)

        # Calculate probability weighted moments
        b0 = np.mean(exceedances_sorted)
        b1 = np.sum(exceedances_sorted * np.arange(n)) / (n * (n - 1))

        # Estimate parameters
        shape = 2 * b1 / (b0 - 2 * b1) - 1
        scale = b0 * (b0 - 2 * b1) / (b0 - b1)

        if scale <= 0:
            # Fallback to method of moments
            scale = np.std(exceedances)
            shape = 0.1

        return cls(shape, scale, threshold)


class KDEDistribution:
    """Kernel Density Estimation distribution."""

    def __init__(
        self,
        data: np.ndarray,
        bandwidth: Optional[float] = None,
        kernel: str = "gaussian",
    ):
        """Initialize KDE distribution.

        Args:
            data: Historical data
            bandwidth: Bandwidth parameter (auto-selected if None)
            kernel: Kernel type
        """
        self.data = np.atleast_1d(data)
        self.n_samples = len(self.data)

        if self.n_samples < 5:
            raise ValueError("Need at least 5 data points for KDE")

        # Select bandwidth if not provided
        if bandwidth is None:
            # Silverman's rule of thumb
            std_data = np.std(self.data)
            iqr_data = np.percentile(self.data, 75) - np.percentile(self.data, 25)
            bandwidth = (
                0.9 * min(std_data, iqr_data / 1.34) * np.power(self.n_samples, -0.2)
            )

        self.bandwidth = bandwidth
        self.kernel = kernel

        # Fit KDE
        self.kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        self.kde.fit(self.data.reshape(-1, 1))

        # Store data range for sampling
        self.data_min = np.min(self.data)
        self.data_max = np.max(self.data)
        self.data_range = self.data_max - self.data_min

    def sample(self, size: int, random_state: np.random.RandomState) -> np.ndarray:
        """Sample from KDE distribution.

        Args:
            size: Number of samples
            random_state: Random state

        Returns:
            Array of samples
        """
        # Use KDE sampling
        samples = self.kde.sample(size, random_state=random_state.randint(0, 2**31))
        return samples.flatten()

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """PDF of KDE distribution.

        Args:
            x: Values to evaluate

        Returns:
            PDF values
        """
        x = np.atleast_1d(x)
        log_density = self.kde.score_samples(x.reshape(-1, 1))
        return np.exp(log_density)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """CDF approximation using empirical CDF of original data.

        Args:
            x: Values to evaluate

        Returns:
            CDF values
        """
        x = np.atleast_1d(x)
        cdf_vals = np.zeros_like(x)

        for i, xi in enumerate(x):
            cdf_vals[i] = np.mean(self.data <= xi)

        return cdf_vals


class EmpiricalDistribution:
    """Empirical distribution from historical data."""

    def __init__(self, data: np.ndarray, interpolation: str = "linear"):
        """Initialize empirical distribution.

        Args:
            data: Historical data
            interpolation: Interpolation method for CDF
        """
        self.data = np.atleast_1d(data)
        self.data_sorted = np.sort(self.data)
        self.n_samples = len(self.data)
        self.interpolation = interpolation

        if self.n_samples < 2:
            raise ValueError("Need at least 2 data points for empirical distribution")

        # Create empirical CDF
        self.empirical_cdf = np.arange(1, self.n_samples + 1) / self.n_samples

    def sample(self, size: int, random_state: np.random.RandomState) -> np.ndarray:
        """Sample from empirical distribution using inverse CDF.

        Args:
            size: Number of samples
            random_state: Random state

        Returns:
            Array of samples
        """
        # Generate uniform random values
        u = random_state.uniform(0, 1, size)

        # Use inverse CDF (quantile function)
        samples = np.interp(u, self.empirical_cdf, self.data_sorted)

        return samples

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """PDF approximation using histogram.

        Args:
            x: Values to evaluate

        Returns:
            PDF values (approximation)
        """
        # Use histogram to approximate PDF
        n_bins = max(10, int(np.sqrt(self.n_samples)))
        hist, bin_edges = np.histogram(self.data, bins=n_bins, density=True)

        # Find which bin each x falls into
        x = np.atleast_1d(x)
        pdf_vals = np.zeros_like(x)

        for i, xi in enumerate(x):
            bin_idx = np.digitize(xi, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, len(hist) - 1)
            pdf_vals[i] = hist[bin_idx]

        return pdf_vals

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Empirical CDF.

        Args:
            x: Values to evaluate

        Returns:
            CDF values
        """
        x = np.atleast_1d(x)
        cdf_vals = np.zeros_like(x)

        for i, xi in enumerate(x):
            if xi < self.data_sorted[0]:
                cdf_vals[i] = 0
            elif xi >= self.data_sorted[-1]:
                cdf_vals[i] = 1
            else:
                # Interpolate
                cdf_vals[i] = np.interp(xi, self.data_sorted, self.empirical_cdf)

        return cdf_vals

    def percentile(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate percentiles.

        Args:
            q: Percentile(s) to calculate (0-100)

        Returns:
            Percentile value(s)
        """
        return np.percentile(self.data, q)


def create_mixture_distribution(config: Dict[str, Any]) -> MixtureDistribution:
    """Create mixture distribution from configuration.

    Args:
        config: Mixture distribution configuration

    Returns:
        MixtureDistribution instance
    """
    components = config.get("components", [])

    if not components:
        raise ValueError("Mixture distribution requires components")

    component_configs = []
    weights = []

    for comp in components:
        if "weight" not in comp or "distribution" not in comp:
            raise ValueError("Each component must have 'weight' and 'distribution'")

        weights.append(comp["weight"])
        component_configs.append(comp["distribution"])

    return MixtureDistribution(component_configs, weights)


def create_evt_distribution(config: Dict[str, Any]) -> GeneralizedParetoDistribution:
    """Create EVT distribution from configuration.

    Args:
        config: EVT distribution configuration

    Returns:
        GeneralizedParetoDistribution instance
    """
    required_params = ["shape", "scale", "threshold"]
    for param in required_params:
        if param not in config:
            raise ValueError(f"EVT distribution requires '{param}' parameter")

    return GeneralizedParetoDistribution(
        shape=config["shape"], scale=config["scale"], threshold=config["threshold"]
    )


def create_kde_distribution(
    data: np.ndarray, config: Dict[str, Any]
) -> KDEDistribution:
    """Create KDE distribution from data and configuration.

    Args:
        data: Historical data
        config: KDE configuration

    Returns:
        KDEDistribution instance
    """
    bandwidth = config.get("bandwidth", None)
    kernel = config.get("kernel", "gaussian")

    return KDEDistribution(data, bandwidth, kernel)


def create_empirical_distribution(
    data: np.ndarray, config: Dict[str, Any]
) -> EmpiricalDistribution:
    """Create empirical distribution from data and configuration.

    Args:
        data: Historical data
        config: Empirical distribution configuration

    Returns:
        EmpiricalDistribution instance
    """
    interpolation = config.get("interpolation", "linear")
    return EmpiricalDistribution(data, interpolation)
