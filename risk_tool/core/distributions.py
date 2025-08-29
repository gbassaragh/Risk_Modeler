"""Probability distributions for Monte Carlo simulation.

Implements all supported distributions with proper truncation and validation.
Optimized with NumPy vectorization and efficient memory management.
"""

import numpy as np
from scipy import stats
from scipy.special import gamma, beta as beta_func
from typing import Dict, Any, Union, Optional, Tuple
from pydantic import BaseModel, field_validator
from pydantic.v1 import validator  # Backwards compatibility
import warnings

class DistributionConfig(BaseModel):
    """Configuration for probability distributions."""
    type: str
    
    class Config:
        extra = "allow"


class Triangular(DistributionConfig):
    """Triangular distribution."""
    type: str = "triangular"
    low: float
    mode: float
    high: float
    
    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v, info):
        if info.data and 'low' in info.data and 'high' in info.data:
            low, high = info.data['low'], info.data['high']
            if not (low <= v <= high):
                raise ValueError(f"Mode {v} must be between low {low} and high {high}")
        return v


class PERT(DistributionConfig):
    """PERT distribution (Beta with shape parameters derived from min/mode/max)."""
    type: str = "pert"
    min: float
    most_likely: float
    max: float
    lambda_: float = 4.0  # Shape parameter
    
    @field_validator('most_likely')
    @classmethod
    def validate_most_likely(cls, v, info):
        if info.data and 'min' in info.data and 'max' in info.data:
            min_val, max_val = info.data['min'], info.data['max']
            if not (min_val <= v <= max_val):
                raise ValueError(f"Most likely {v} must be between min {min_val} and max {max_val}")
        return v


class Normal(DistributionConfig):
    """Normal distribution with optional truncation."""
    type: str = "normal"
    mean: float
    stdev: float
    truncate_low: Optional[float] = None
    truncate_high: Optional[float] = None
    
    @validator('stdev')
    def validate_stdev(cls, v):
        if v <= 0:
            raise ValueError("Standard deviation must be positive")
        return v


class LogNormal(DistributionConfig):
    """Log-normal distribution."""
    type: str = "lognormal"
    mean: float  # Mean of underlying normal
    sigma: float  # Std dev of underlying normal
    
    @validator('mean')
    def validate_mean(cls, v):
        if v <= 0:
            raise ValueError("Mean must be positive for lognormal distribution")
        return v
    
    @validator('sigma')
    def validate_sigma(cls, v):
        if v <= 0:
            raise ValueError("Sigma must be positive")
        return v


class Discrete(DistributionConfig):
    """Discrete distribution with probability mass function."""
    type: str = "discrete"
    pmf: list  # List of [value, probability] pairs
    
    @validator('pmf')
    def validate_pmf(cls, v):
        if not v:
            raise ValueError("PMF cannot be empty")
        
        total_prob = 0.0
        for item in v:
            if len(item) != 2:
                raise ValueError("Each PMF item must be [value, probability]")
            prob = item[1]
            if prob < 0 or prob > 1:
                raise ValueError("Probabilities must be between 0 and 1")
            total_prob += prob
        
        if abs(total_prob - 1.0) > 1e-6:
            raise ValueError(f"Probabilities must sum to 1.0, got {total_prob}")
        
        return v


class Uniform(DistributionConfig):
    """Uniform distribution."""
    type: str = "uniform"
    low: float
    high: float
    
    @validator('high')
    def validate_bounds(cls, v, values):
        if 'low' in values and v <= values['low']:
            raise ValueError(f"High {v} must be greater than low {values['low']}")
        return v


class DistributionSampler:
    """Samples from probability distributions with proper truncation."""
    
    @staticmethod
    def sample(dist_config: Dict[str, Any], size: int, random_state: np.random.RandomState) -> np.ndarray:
        """Sample from distribution configuration.
        
        Args:
            dist_config: Distribution parameters
            size: Number of samples
            random_state: Random number generator
            
        Returns:
            Array of samples
        """
        dist_type = dist_config.get('type', '').lower()
        
        if dist_type == 'triangular':
            return DistributionSampler._sample_triangular(dist_config, size, random_state)
        elif dist_type == 'pert':
            return DistributionSampler._sample_pert(dist_config, size, random_state)
        elif dist_type == 'normal':
            return DistributionSampler._sample_normal(dist_config, size, random_state)
        elif dist_type == 'lognormal':
            return DistributionSampler._sample_lognormal(dist_config, size, random_state)
        elif dist_type == 'discrete':
            return DistributionSampler._sample_discrete(dist_config, size, random_state)
        elif dist_type == 'mixture':
            return DistributionSampler._sample_mixture(dist_config, size, random_state)
        elif dist_type == 'evt':
            return DistributionSampler._sample_evt(dist_config, size, random_state)
        elif dist_type == 'kde':
            return DistributionSampler._sample_kde(dist_config, size, random_state)
        elif dist_type == 'empirical':
            return DistributionSampler._sample_empirical(dist_config, size, random_state)
        elif dist_type == 'uniform':
            return DistributionSampler._sample_uniform(dist_config, size, random_state)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
    
    @staticmethod
    def _sample_triangular(config: Dict[str, Any], size: int, rs: np.random.RandomState) -> np.ndarray:
        """Sample from triangular distribution."""
        low = config['low']
        mode = config['mode']
        high = config['high']
        
        # Scipy triangular uses (high - low) scale and (mode - low) / (high - low) shape
        c = (mode - low) / (high - low)
        return stats.triang.rvs(c, loc=low, scale=high - low, size=size, random_state=rs)
    
    @staticmethod
    def _sample_pert(config: Dict[str, Any], size: int, rs: np.random.RandomState) -> np.ndarray:
        """Sample from PERT distribution (Beta distribution)."""
        min_val = config['min']
        mode = config['most_likely']
        max_val = config['max']
        lambda_ = config.get('lambda_', 4.0)
        
        # PERT to Beta conversion
        mu = (min_val + lambda_ * mode + max_val) / (lambda_ + 2)
        
        if mu == min_val or mu == max_val:
            # Degenerate case
            return np.full(size, mu)
        
        # Beta parameters
        alpha = ((mu - min_val) * (2 * mode - min_val - max_val)) / ((mode - mu) * (max_val - min_val))
        beta = alpha * (max_val - mu) / (mu - min_val)
        
        # Ensure valid beta parameters
        alpha = max(alpha, 0.1)
        beta = max(beta, 0.1)
        
        # Sample from beta and scale
        beta_samples = stats.beta.rvs(alpha, beta, size=size, random_state=rs)
        return min_val + beta_samples * (max_val - min_val)
    
    @staticmethod
    def _sample_normal(config: Dict[str, Any], size: int, rs: np.random.RandomState) -> np.ndarray:
        """Sample from normal distribution with optional truncation."""
        mean = config['mean']
        stdev = config['stdev']
        truncate_low = config.get('truncate_low')
        truncate_high = config.get('truncate_high')
        
        if truncate_low is not None or truncate_high is not None:
            # Truncated normal
            a = (truncate_low - mean) / stdev if truncate_low is not None else -np.inf
            b = (truncate_high - mean) / stdev if truncate_high is not None else np.inf
            return stats.truncnorm.rvs(a, b, loc=mean, scale=stdev, size=size, random_state=rs)
        else:
            return rs.normal(mean, stdev, size)
    
    @staticmethod
    def _sample_lognormal(config: Dict[str, Any], size: int, rs: np.random.RandomState) -> np.ndarray:
        """Sample from log-normal distribution."""
        mean = config['mean']
        sigma = config['sigma']
        
        # Convert to log-normal parameters
        # If mean is given as the mean of the lognormal (not underlying normal)
        if 'geometric_mean' in config and config['geometric_mean']:
            # Interpret mean as geometric mean
            mu = np.log(mean)
        else:
            # Convert arithmetic mean to log-normal parameters
            variance = (sigma * mean) ** 2  # Assuming sigma is coefficient of variation
            mu = np.log(mean / np.sqrt(1 + variance / mean**2))
            sigma_log = np.sqrt(np.log(1 + variance / mean**2))
            
            return rs.lognormal(mu, sigma_log, size)
        
        return rs.lognormal(mu, sigma, size)
    
    @staticmethod
    def _sample_discrete(config: Dict[str, Any], size: int, rs: np.random.RandomState) -> np.ndarray:
        """Sample from discrete distribution."""
        pmf = config['pmf']
        
        values = []
        probs = []
        for value, prob in pmf:
            values.append(float(value))
            probs.append(float(prob))
        
        return rs.choice(values, size=size, p=probs)
    
    @staticmethod
    def _sample_uniform(config: Dict[str, Any], size: int, rs: np.random.RandomState) -> np.ndarray:
        """Sample from uniform distribution."""
        low = config['low']
        high = config['high']
        return rs.uniform(low, high, size)
    
    @staticmethod
    def _sample_mixture(config: Dict[str, Any], size: int, rs: np.random.RandomState) -> np.ndarray:
        """Sample from mixture distribution."""
        from .mixtures_evt import create_mixture_distribution
        
        mixture = create_mixture_distribution(config)
        return mixture.sample(size, rs)
    
    @staticmethod
    def _sample_evt(config: Dict[str, Any], size: int, rs: np.random.RandomState) -> np.ndarray:
        """Sample from EVT/GPD distribution."""
        from .mixtures_evt import create_evt_distribution
        
        evt_dist = create_evt_distribution(config)
        return evt_dist.sample(size, rs)
    
    @staticmethod
    def _sample_kde(config: Dict[str, Any], size: int, rs: np.random.RandomState) -> np.ndarray:
        """Sample from KDE distribution."""
        from .mixtures_evt import create_kde_distribution
        
        if 'data' not in config:
            raise ValueError("KDE distribution requires 'data' parameter")
        
        data = np.array(config['data'])
        kde_dist = create_kde_distribution(data, config)
        return kde_dist.sample(size, rs)
    
    @staticmethod
    def _sample_empirical(config: Dict[str, Any], size: int, rs: np.random.RandomState) -> np.ndarray:
        """Sample from empirical distribution."""
        from .mixtures_evt import create_empirical_distribution
        
        if 'data' not in config:
            raise ValueError("Empirical distribution requires 'data' parameter")
        
        data = np.array(config['data'])
        empirical_dist = create_empirical_distribution(data, config)
        return empirical_dist.sample(size, rs)


def get_distribution_stats(dist_config: Dict[str, Any]) -> Tuple[float, float]:
    """Get analytical mean and variance for a distribution.
    
    Args:
        dist_config: Distribution parameters
        
    Returns:
        Tuple of (mean, variance)
    """
    dist_type = dist_config.get('type', '').lower()
    
    if dist_type == 'triangular':
        a, b, c = dist_config['low'], dist_config['high'], dist_config['mode']
        mean = (a + b + c) / 3
        var = (a**2 + b**2 + c**2 - a*b - a*c - b*c) / 18
        return mean, var
    
    elif dist_type == 'pert':
        min_val, mode, max_val = dist_config['min'], dist_config['most_likely'], dist_config['max']
        lambda_ = dist_config.get('lambda_', 4.0)
        mean = (min_val + lambda_ * mode + max_val) / (lambda_ + 2)
        
        # Approximate variance
        var = ((max_val - min_val) / 6) ** 2
        return mean, var
    
    elif dist_type == 'normal':
        mean = dist_config['mean']
        var = dist_config['stdev'] ** 2
        return mean, var
    
    elif dist_type == 'lognormal':
        mu = dist_config['mean']
        sigma = dist_config['sigma']
        
        # Assuming mu is the mean of the underlying normal
        mean = np.exp(mu + sigma**2 / 2)
        var = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
        return mean, var
    
    elif dist_type == 'uniform':
        a, b = dist_config['low'], dist_config['high']
        mean = (a + b) / 2
        var = (b - a) ** 2 / 12
        return mean, var
    
    elif dist_type == 'discrete':
        pmf = dist_config['pmf']
        values = [float(item[0]) for item in pmf]
        probs = [float(item[1]) for item in pmf]
        
        mean = sum(v * p for v, p in zip(values, probs))
        var = sum(v**2 * p for v, p in zip(values, probs)) - mean**2
        return mean, var
    
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def validate_distribution_config(config: Dict[str, Any]) -> None:
    """Validate distribution configuration.
    
    Args:
        config: Distribution parameters
        
    Raises:
        ValueError: If configuration is invalid
    """
    dist_type = config.get('type', '').lower()
    
    if dist_type == 'triangular':
        Triangular(**config)
    elif dist_type == 'pert':
        PERT(**config)
    elif dist_type == 'normal':
        Normal(**config)
    elif dist_type == 'lognormal':
        LogNormal(**config)
    elif dist_type == 'discrete':
        Discrete(**config)
    elif dist_type == 'uniform':
        Uniform(**config)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")