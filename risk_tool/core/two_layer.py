"""Two-layer uncertainty model for epistemic and aleatory uncertainty.

Implements nested Monte Carlo simulation where:
- Outer loop: Epistemic (parameter) uncertainty 
- Inner loop: Aleatory (inherent variability)

Used for reporting percentile bands (e.g., 5-95% band around P80).
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import warnings
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from .distributions import DistributionSampler
from .sampler import MonteCarloSampler, ConvergenceDiagnostics
from .latent_factors import LatentFactorModel


@dataclass
class TwoLayerConfig:
    """Configuration for two-layer uncertainty model."""
    n_epistemic: int = 100  # Outer loop iterations
    n_aleatory: int = 1000  # Inner loop iterations per epistemic sample
    target_percentiles: List[float] = None  # Target percentiles to track
    confidence_band: Tuple[float, float] = (5, 95)  # Confidence band around percentiles
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    
    def __post_init__(self):
        if self.target_percentiles is None:
            self.target_percentiles = [50, 80, 90, 95]


@dataclass
class TwoLayerResults:
    """Results from two-layer uncertainty analysis."""
    # Primary results
    epistemic_samples: np.ndarray  # Shape: (n_epistemic, n_percentiles)
    aleatory_samples: List[np.ndarray]  # List of inner loop samples
    target_percentiles: List[float]
    
    # Summary statistics
    percentile_bands: Dict[str, Dict[str, float]]  # P50: {mean: x, lower: y, upper: z}
    epistemic_stats: Dict[str, float]  # Overall epistemic uncertainty
    aleatory_stats: Dict[str, float]  # Average aleatory uncertainty
    
    # Diagnostics
    convergence_info: Dict[str, Any]
    total_samples: int
    
    def get_percentile_band(self, percentile: float) -> Tuple[float, float, float]:
        """Get confidence band around a target percentile.
        
        Args:
            percentile: Target percentile (e.g., 80 for P80)
            
        Returns:
            Tuple of (lower_bound, mean, upper_bound) for the percentile
        """
        key = f"P{int(percentile)}"
        if key not in self.percentile_bands:
            raise ValueError(f"Percentile {percentile} not in results")
        
        band = self.percentile_bands[key]
        return band['lower'], band['mean'], band['upper']
    
    def summary_table(self) -> str:
        """Generate summary table of results."""
        lines = ["Two-Layer Uncertainty Results"]
        lines.append("=" * 35)
        lines.append(f"Epistemic samples: {len(self.epistemic_samples)}")
        lines.append(f"Aleatory samples per epistemic: {len(self.aleatory_samples[0]) if self.aleatory_samples else 0}")
        lines.append(f"Total samples: {self.total_samples:,}")
        lines.append("")
        
        # Percentile bands
        lines.append("Percentile Confidence Bands:")
        lines.append("-" * 30)
        for p in self.target_percentiles:
            key = f"P{int(p)}"
            if key in self.percentile_bands:
                band = self.percentile_bands[key]
                lines.append(f"{key:>3}: {band['lower']:>10.2f} | {band['mean']:>10.2f} | {band['upper']:>10.2f}")
        
        lines.append("")
        lines.append("Uncertainty Decomposition:")
        lines.append("-" * 25)
        lines.append(f"Epistemic CV: {self.epistemic_stats.get('cv', 0):.3f}")
        lines.append(f"Aleatory CV:  {self.aleatory_stats.get('cv', 0):.3f}")
        
        return "\n".join(lines)


class EpistemicParameter:
    """Represents a parameter with epistemic uncertainty."""
    
    def __init__(self, name: str, distribution_config: Dict[str, Any]):
        """Initialize epistemic parameter.
        
        Args:
            name: Parameter name
            distribution_config: Distribution configuration for parameter uncertainty
        """
        self.name = name
        self.distribution_config = distribution_config
    
    def sample(self, n_samples: int, random_state: np.random.RandomState) -> np.ndarray:
        """Sample parameter values for epistemic uncertainty.
        
        Args:
            n_samples: Number of epistemic samples
            random_state: Random state
            
        Returns:
            Array of parameter samples
        """
        return DistributionSampler.sample(
            self.distribution_config, 
            n_samples, 
            random_state
        )


class TwoLayerMonteCarlo:
    """Two-layer Monte Carlo simulation for epistemic and aleatory uncertainty."""
    
    def __init__(self, 
                 config: TwoLayerConfig,
                 epistemic_parameters: List[EpistemicParameter],
                 random_seed: Optional[int] = None):
        """Initialize two-layer Monte Carlo.
        
        Args:
            config: Two-layer configuration
            epistemic_parameters: List of parameters with epistemic uncertainty
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.epistemic_parameters = epistemic_parameters
        self.random_seed = random_seed
        
        # Initialize random state
        self.random_state = np.random.RandomState(random_seed)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.n_epistemic <= 0:
            raise ValueError("Number of epistemic samples must be positive")
        
        if self.config.n_aleatory <= 0:
            raise ValueError("Number of aleatory samples must be positive")
        
        if not self.config.target_percentiles:
            raise ValueError("Must specify target percentiles")
        
        for p in self.config.target_percentiles:
            if not (0 < p < 100):
                raise ValueError(f"Percentiles must be between 0 and 100, got {p}")
        
        lower, upper = self.config.confidence_band
        if not (0 < lower < upper < 100):
            raise ValueError(f"Confidence band must be (lower, upper) with 0 < lower < upper < 100")
    
    def run_simulation(self, 
                      aleatory_model: Callable,
                      model_args: Dict[str, Any] = None) -> TwoLayerResults:
        """Run two-layer Monte Carlo simulation.
        
        Args:
            aleatory_model: Function that runs aleatory simulation
                           Should accept (epistemic_params, n_samples, random_state, **kwargs)
                           and return array of simulation results
            model_args: Additional arguments for aleatory model
            
        Returns:
            TwoLayerResults containing all analysis results
        """
        if model_args is None:
            model_args = {}
        
        # Sample epistemic parameters
        epistemic_samples = self._sample_epistemic_parameters()
        
        # Run nested simulation
        if self.config.parallel_processing:
            results = self._run_parallel_simulation(epistemic_samples, aleatory_model, model_args)
        else:
            results = self._run_sequential_simulation(epistemic_samples, aleatory_model, model_args)
        
        # Process results
        return self._process_results(results, epistemic_samples)
    
    def _sample_epistemic_parameters(self) -> List[Dict[str, float]]:
        """Sample epistemic parameter values.
        
        Returns:
            List of parameter dictionaries for each epistemic iteration
        """
        n_epistemic = self.config.n_epistemic
        epistemic_samples = []
        
        for i in range(n_epistemic):
            # Use different random state for each epistemic sample
            ep_random_state = np.random.RandomState(
                self.random_state.randint(0, 2**31) if self.random_seed else None
            )
            
            sample = {}
            for param in self.epistemic_parameters:
                param_sample = param.sample(1, ep_random_state)
                sample[param.name] = param_sample[0]
            
            epistemic_samples.append(sample)
        
        return epistemic_samples
    
    def _run_sequential_simulation(self,
                                 epistemic_samples: List[Dict[str, float]],
                                 aleatory_model: Callable,
                                 model_args: Dict[str, Any]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Run simulation sequentially.
        
        Args:
            epistemic_samples: List of epistemic parameter samples
            aleatory_model: Aleatory simulation function
            model_args: Model arguments
            
        Returns:
            Tuple of (percentile_results, aleatory_samples)
        """
        n_epistemic = len(epistemic_samples)
        n_percentiles = len(self.config.target_percentiles)
        
        percentile_results = np.zeros((n_epistemic, n_percentiles))
        aleatory_samples = []
        
        for i, ep_params in enumerate(epistemic_samples):
            # Create random state for this epistemic iteration
            al_random_state = np.random.RandomState(
                self.random_state.randint(0, 2**31) if self.random_seed else None
            )
            
            # Run aleatory simulation with these epistemic parameters
            aleatory_result = aleatory_model(
                ep_params, 
                self.config.n_aleatory, 
                al_random_state,
                **model_args
            )
            
            aleatory_samples.append(aleatory_result)
            
            # Calculate target percentiles for this epistemic iteration
            percentiles = np.percentile(aleatory_result, self.config.target_percentiles)
            percentile_results[i, :] = percentiles
        
        return percentile_results, aleatory_samples
    
    def _run_parallel_simulation(self,
                               epistemic_samples: List[Dict[str, float]],
                               aleatory_model: Callable,
                               model_args: Dict[str, Any]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Run simulation in parallel.
        
        Args:
            epistemic_samples: List of epistemic parameter samples
            aleatory_model: Aleatory simulation function
            model_args: Model arguments
            
        Returns:
            Tuple of (percentile_results, aleatory_samples)
        """
        n_epistemic = len(epistemic_samples)
        n_percentiles = len(self.config.target_percentiles)
        
        # Determine number of workers
        max_workers = self.config.max_workers or min(
            multiprocessing.cpu_count(), 
            n_epistemic
        )
        
        percentile_results = np.zeros((n_epistemic, n_percentiles))
        aleatory_samples = [None] * n_epistemic
        
        # Create worker function
        def worker_function(args):
            i, ep_params = args
            
            # Create random state for this worker
            worker_seed = (self.random_seed + i) if self.random_seed else None
            al_random_state = np.random.RandomState(worker_seed)
            
            # Run aleatory simulation
            aleatory_result = aleatory_model(
                ep_params,
                self.config.n_aleatory,
                al_random_state,
                **model_args
            )
            
            # Calculate percentiles
            percentiles = np.percentile(aleatory_result, self.config.target_percentiles)
            
            return i, aleatory_result, percentiles
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [
                executor.submit(worker_function, (i, ep_params))
                for i, ep_params in enumerate(epistemic_samples)
            ]
            
            # Collect results
            for future in as_completed(futures):
                i, aleatory_result, percentiles = future.result()
                percentile_results[i, :] = percentiles
                aleatory_samples[i] = aleatory_result
        
        return percentile_results, aleatory_samples
    
    def _process_results(self,
                        results: Tuple[np.ndarray, List[np.ndarray]],
                        epistemic_samples: List[Dict[str, float]]) -> TwoLayerResults:
        """Process simulation results into TwoLayerResults.
        
        Args:
            results: Tuple of (percentile_results, aleatory_samples)
            epistemic_samples: Original epistemic parameter samples
            
        Returns:
            Processed TwoLayerResults
        """
        percentile_results, aleatory_samples = results
        
        # Calculate percentile bands
        percentile_bands = {}
        lower_conf, upper_conf = self.config.confidence_band
        
        for i, target_p in enumerate(self.config.target_percentiles):
            p_values = percentile_results[:, i]
            
            # Calculate confidence band around this percentile
            lower_band = np.percentile(p_values, lower_conf)
            upper_band = np.percentile(p_values, upper_conf)
            mean_p = np.mean(p_values)
            
            percentile_bands[f"P{int(target_p)}"] = {
                'mean': mean_p,
                'lower': lower_band,
                'upper': upper_band,
                'std': np.std(p_values)
            }
        
        # Calculate epistemic statistics (from percentile variation)
        epistemic_means = np.mean(percentile_results, axis=1)  # Mean across percentiles for each epistemic
        epistemic_stats = {
            'mean': np.mean(epistemic_means),
            'std': np.std(epistemic_means),
            'cv': np.std(epistemic_means) / np.mean(epistemic_means) if np.mean(epistemic_means) != 0 else 0
        }
        
        # Calculate aleatory statistics (average within-simulation variation)
        aleatory_means = [np.mean(samples) for samples in aleatory_samples]
        aleatory_stds = [np.std(samples) for samples in aleatory_samples]
        
        aleatory_stats = {
            'mean_of_means': np.mean(aleatory_means),
            'mean_of_stds': np.mean(aleatory_stds),
            'cv': np.mean(aleatory_stds) / np.mean(aleatory_means) if np.mean(aleatory_means) != 0 else 0
        }
        
        # Convergence diagnostics
        convergence_info = {
            'epistemic_samples': self.config.n_epistemic,
            'aleatory_samples_per_epistemic': self.config.n_aleatory,
            'total_samples': self.config.n_epistemic * self.config.n_aleatory,
            'confidence_band': self.config.confidence_band
        }
        
        return TwoLayerResults(
            epistemic_samples=percentile_results,
            aleatory_samples=aleatory_samples,
            target_percentiles=self.config.target_percentiles,
            percentile_bands=percentile_bands,
            epistemic_stats=epistemic_stats,
            aleatory_stats=aleatory_stats,
            convergence_info=convergence_info,
            total_samples=convergence_info['total_samples']
        )


def create_default_two_layer_config(n_epistemic: int = 100,
                                   n_aleatory: int = 1000,
                                   target_percentiles: List[float] = None) -> TwoLayerConfig:
    """Create default two-layer configuration.
    
    Args:
        n_epistemic: Number of outer loop (epistemic) iterations
        n_aleatory: Number of inner loop (aleatory) iterations
        target_percentiles: Target percentiles to analyze
        
    Returns:
        TwoLayerConfig with reasonable defaults
    """
    if target_percentiles is None:
        target_percentiles = [50, 80, 90, 95]
    
    return TwoLayerConfig(
        n_epistemic=n_epistemic,
        n_aleatory=n_aleatory,
        target_percentiles=target_percentiles,
        confidence_band=(5, 95),  # 90% confidence band
        parallel_processing=True,
        max_workers=None  # Auto-detect
    )


def create_epistemic_from_distribution(name: str, 
                                     base_dist: Dict[str, Any],
                                     uncertainty_factor: float = 0.1) -> EpistemicParameter:
    """Create epistemic parameter from base distribution with uncertainty.
    
    Args:
        name: Parameter name
        base_dist: Base distribution configuration
        uncertainty_factor: Relative uncertainty in the parameter (e.g., 0.1 = 10%)
        
    Returns:
        EpistemicParameter representing uncertainty in the distribution parameter
    """
    dist_type = base_dist.get('type', '').lower()
    
    if dist_type in ['normal', 'triangular']:
        # For normal/triangular, add uncertainty to the mean/mode
        if 'mean' in base_dist:
            base_value = base_dist['mean']
        elif 'mode' in base_dist:
            base_value = base_dist['mode']
        else:
            raise ValueError(f"Cannot create epistemic parameter for {dist_type} without mean/mode")
        
        # Create normal distribution around the base parameter
        epistemic_config = {
            'type': 'normal',
            'mean': base_value,
            'stdev': abs(base_value * uncertainty_factor)
        }
    
    elif dist_type == 'lognormal':
        # For lognormal, add uncertainty to the scale parameter
        base_mean = base_dist['mean']
        epistemic_config = {
            'type': 'lognormal',
            'mean': base_mean,
            'sigma': base_dist.get('sigma', 0.5) + uncertainty_factor
        }
    
    else:
        raise ValueError(f"Epistemic parameter creation not supported for {dist_type}")
    
    return EpistemicParameter(name, epistemic_config)


# Convenience functions for common use cases

def analyze_parameter_uncertainty(distributions: Dict[str, Dict[str, Any]],
                                model_function: Callable,
                                uncertainty_factors: Dict[str, float] = None,
                                n_epistemic: int = 100,
                                n_aleatory: int = 1000) -> TwoLayerResults:
    """Analyze parameter uncertainty for a set of distributions.
    
    Args:
        distributions: Dictionary of {param_name: distribution_config}
        model_function: Function that runs simulation given parameters
        uncertainty_factors: Dictionary of {param_name: uncertainty_factor}
        n_epistemic: Number of epistemic samples
        n_aleatory: Number of aleatory samples
        
    Returns:
        TwoLayerResults from analysis
    """
    if uncertainty_factors is None:
        uncertainty_factors = {}
    
    # Create epistemic parameters
    epistemic_params = []
    for name, dist_config in distributions.items():
        factor = uncertainty_factors.get(name, 0.1)  # Default 10% uncertainty
        ep_param = create_epistemic_from_distribution(name, dist_config, factor)
        epistemic_params.append(ep_param)
    
    # Create configuration
    config = create_default_two_layer_config(n_epistemic, n_aleatory)
    
    # Run analysis
    two_layer = TwoLayerMonteCarlo(config, epistemic_params)
    return two_layer.run_simulation(model_function)