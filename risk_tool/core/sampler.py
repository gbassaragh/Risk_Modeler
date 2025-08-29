"""Monte Carlo sampling with LHS, Sobol, and convergence diagnostics.

Implements LHS, Sobol quasi-Monte Carlo, standard MC, and antithetic variates
with convergence gates and variance reduction techniques.

Optimized for high-performance numerical computing with NumPy vectorization.
"""

import numpy as np
from scipy.stats import qmc
from typing import Optional, Dict, List, Tuple, Union, Callable
import warnings
import time
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class ConvergenceResult:
    """Result of convergence checking."""
    converged: bool
    iterations: int
    p50_error: float
    p80_error: float
    message: str


class MonteCarloSampler:
    """Advanced Monte Carlo sampler with multiple methods and variance reduction."""
    
    def __init__(self, 
                 n_samples: int,
                 n_dimensions: int,
                 method: str = "LHS",
                 random_seed: Optional[int] = None,
                 antithetic_variates: bool = False,
                 skip_first: int = 0):
        """Initialize sampler.
        
        Args:
            n_samples: Number of samples to generate
            n_dimensions: Number of dimensions (variables)
            method: Sampling method ("LHS", "Sobol", or "MC")
            random_seed: Random seed for reproducibility
            antithetic_variates: Use antithetic variates for variance reduction
            skip_first: Skip first N Sobol points (for better uniformity)
        """
        self.n_samples = n_samples
        self.n_dimensions = n_dimensions
        self.method = method.upper()
        self.random_seed = random_seed
        self.antithetic_variates = antithetic_variates
        self.skip_first = skip_first
        
        # Initialize random state
        self.random_state = np.random.RandomState(random_seed)
        
        # Generate base samples
        self.base_samples = self._generate_base_samples()
    
    def _generate_base_samples(self) -> np.ndarray:
        """Generate base uniform samples.
        
        Returns:
            Array of shape (n_samples, n_dimensions) with values in [0, 1]
        """
        if self.method == "LHS":
            samples = self._generate_lhs_samples()
        elif self.method == "SOBOL":
            samples = self._generate_sobol_samples()
        elif self.method == "MC":
            samples = self._generate_mc_samples()
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")
        
        # Apply antithetic variates if requested
        if self.antithetic_variates:
            samples = self._apply_antithetic_variates(samples)
        
        return samples
    
    def _generate_lhs_samples(self) -> np.ndarray:
        """Generate Latin Hypercube Samples.
        
        Returns:
            LHS samples in [0, 1]^n_dimensions
        """
        # Use scipy's qmc for high-quality LHS
        sampler = qmc.LatinHypercube(d=self.n_dimensions, seed=self.random_seed)
        return sampler.random(n=self.n_samples)
    
    def _generate_sobol_samples(self) -> np.ndarray:
        """Generate Sobol quasi-Monte Carlo samples.
        
        Returns:
            Sobol samples in [0, 1]^n_dimensions
        """
        # Generate Sobol samples with optional skip
        total_needed = self.n_samples + self.skip_first
        
        # Calculate required bits (Sobol requires power of 2)
        m_bits = int(np.ceil(np.log2(total_needed)))
        total_generated = 2 ** m_bits
        
        # Generate Sobol sequence
        sampler = qmc.Sobol(d=self.n_dimensions, seed=self.random_seed)
        sobol_samples = sampler.random(total_generated)
        
        # Skip first points and take required number
        return sobol_samples[self.skip_first:self.skip_first + self.n_samples]
    
    def _generate_mc_samples(self) -> np.ndarray:
        """Generate standard Monte Carlo samples.
        
        Returns:
            Random samples in [0, 1]^n_dimensions
        """
        return self.random_state.uniform(0, 1, size=(self.n_samples, self.n_dimensions))
    
    def _apply_antithetic_variates(self, samples: np.ndarray) -> np.ndarray:
        """Apply antithetic variates for variance reduction.
        
        Args:
            samples: Original samples
            
        Returns:
            Extended sample set with antithetic variates
        """
        # Create antithetic pairs: if U ~ Uniform(0,1), then 1-U is also
        # Using in-place operations for better memory efficiency
        antithetic = 1.0 - samples
        
        # Combine original and antithetic samples using concatenate for efficiency
        combined = np.concatenate([samples, antithetic], axis=0)
        
        # Update sample count
        self.n_samples = len(combined)
        
        return combined
    
    def get_samples(self) -> np.ndarray:
        """Get the generated samples.
        
        Returns:
            Base uniform samples
        """
        return self.base_samples.copy()


class ConvergenceDiagnostics:
    """Advanced convergence diagnostics with automatic gates."""
    
    def __init__(self, 
                 p50_tolerance: float = 0.005,
                 p80_tolerance: float = 0.005,
                 min_iterations: int = 1000,
                 check_interval: int = 1000):
        """Initialize convergence diagnostics.
        
        Args:
            p50_tolerance: P50 convergence tolerance (relative error)
            p80_tolerance: P80 convergence tolerance (relative error) 
            min_iterations: Minimum iterations before checking convergence
            check_interval: Check convergence every N iterations
        """
        self.p50_tolerance = p50_tolerance
        self.p80_tolerance = p80_tolerance
        self.min_iterations = min_iterations
        self.check_interval = check_interval
        
        # Tracking variables
        self.p50_history = []
        self.p80_history = []
        self.iteration_history = []
    
    def check_convergence(self, 
                         total_cost_samples: np.ndarray,
                         current_iteration: int) -> ConvergenceResult:
        """Check if simulation has converged.
        
        Args:
            total_cost_samples: Current total cost samples
            current_iteration: Current iteration number
            
        Returns:
            Convergence result
        """
        if current_iteration < self.min_iterations:
            return ConvergenceResult(
                converged=False,
                iterations=current_iteration,
                p50_error=float('inf'),
                p80_error=float('inf'),
                message=f"Need minimum {self.min_iterations} iterations"
            )
        
        # Calculate current percentiles
        current_p50 = np.percentile(total_cost_samples, 50)
        current_p80 = np.percentile(total_cost_samples, 80)
        
        # Store history
        self.p50_history.append(current_p50)
        self.p80_history.append(current_p80)
        self.iteration_history.append(current_iteration)
        
        # Need at least 2 points for convergence check
        if len(self.p50_history) < 2:
            return ConvergenceResult(
                converged=False,
                iterations=current_iteration,
                p50_error=float('inf'),
                p80_error=float('inf'),
                message="Need at least 2 convergence check points"
            )
        
        # Calculate relative errors
        prev_p50 = self.p50_history[-2]
        prev_p80 = self.p80_history[-2]
        
        p50_error = abs(current_p50 - prev_p50) / abs(prev_p50) if prev_p50 != 0 else 0
        p80_error = abs(current_p80 - prev_p80) / abs(prev_p80) if prev_p80 != 0 else 0
        
        # Check convergence
        p50_converged = p50_error < self.p50_tolerance
        p80_converged = p80_error < self.p80_tolerance
        converged = p50_converged and p80_converged
        
        if converged:
            message = f"Converged: ΔP50={p50_error:.4f}, ΔP80={p80_error:.4f}"
        else:
            message = f"Not converged: ΔP50={p50_error:.4f}, ΔP80={p80_error:.4f}"
        
        return ConvergenceResult(
            converged=converged,
            iterations=current_iteration,
            p50_error=p50_error,
            p80_error=p80_error,
            message=message
        )
    
    def get_convergence_history(self) -> Dict[str, List[float]]:
        """Get convergence history for plotting.
        
        Returns:
            Dictionary with convergence history
        """
        return {
            'iterations': self.iteration_history.copy(),
            'p50_values': self.p50_history.copy(),
            'p80_values': self.p80_history.copy(),
            'p50_errors': self._calculate_relative_errors(self.p50_history),
            'p80_errors': self._calculate_relative_errors(self.p80_history)
        }
    
    def _calculate_relative_errors(self, values: List[float]) -> List[float]:
        """Calculate relative errors between consecutive values.
        
        Args:
            values: List of values
            
        Returns:
            List of relative errors
        """
        if len(values) < 2:
            return []
        
        errors = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                error = abs(values[i] - values[i-1]) / abs(values[i-1])
            else:
                error = 0.0
            errors.append(error)
        
        return errors
    
    @staticmethod
    def running_mean(samples: np.ndarray, window: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate running mean for convergence analysis.
        
        Args:
            samples: Array of samples
            window: Window size for smoothing
            
        Returns:
            Tuple of (sample_indices, running_means)
        """
        n_samples = len(samples)
        indices = np.arange(window, n_samples + 1, window)
        means = []
        
        for i in indices:
            means.append(np.mean(samples[:i]))
        
        return indices, np.array(means)
    
    @staticmethod
    def effective_sample_size(samples: np.ndarray, max_lag: int = 100) -> float:
        """Estimate effective sample size accounting for autocorrelation.
        
        Args:
            samples: Sample array
            max_lag: Maximum lag for autocorrelation calculation
            
        Returns:
            Effective sample size
        """
        n = len(samples)
        if n < max_lag * 2:
            return float(n)
        
        # Calculate autocorrelation function
        autocorr = np.correlate(samples - np.mean(samples), 
                               samples - np.mean(samples), 
                               mode='full')
        autocorr = autocorr[n-1:n-1+max_lag]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find integrated autocorrelation time
        tau_int = 1.0
        for i in range(1, max_lag):
            if autocorr[i] <= 0:
                break
            tau_int += 2 * autocorr[i]
        
        # Effective sample size
        return n / (2 * tau_int + 1) if tau_int > 0 else float(n)
    
    @staticmethod
    def running_percentile(samples: np.ndarray, percentile: float = 80, window: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate running percentile for convergence analysis.
        
        Args:
            samples: Array of samples
            percentile: Percentile to track (0-100)
            window: Window size for smoothing
            
        Returns:
            Tuple of (sample_indices, running_percentiles)
        """
        n_samples = len(samples)
        indices = np.arange(window, n_samples + 1, window)
        percentiles = []
        
        for i in indices:
            percentiles.append(np.percentile(samples[:i], percentile))
        
        return indices, np.array(percentiles)
    
    @staticmethod
    def convergence_test(samples: np.ndarray, 
                        confidence: float = 0.95,
                        tolerance: float = 0.05) -> Dict[str, Union[bool, float]]:
        """Test for convergence based on confidence intervals.
        
        Args:
            samples: Array of samples
            confidence: Confidence level for test
            tolerance: Relative tolerance for convergence
            
        Returns:
            Dictionary with convergence results
        """
        n = len(samples)
        mean = np.mean(samples)
        std = np.std(samples, ddof=1)
        
        # Standard error
        se = std / np.sqrt(n)
        
        # Confidence interval half-width
        z_score = 1.96 if confidence == 0.95 else 2.576  # For 99%
        ci_half_width = z_score * se
        
        # Relative error
        relative_error = ci_half_width / abs(mean) if mean != 0 else float('inf')
        
        # Convergence check
        converged = relative_error < tolerance
        
        return {
            'converged': converged,
            'mean': mean,
            'std': std,
            'standard_error': se,
            'confidence_interval_half_width': ci_half_width,
            'relative_error': relative_error,
            'required_tolerance': tolerance,
            'n_samples': n
        }
    
    @staticmethod
    def effective_sample_size(samples: np.ndarray, max_lag: int = 100) -> float:
        """Estimate effective sample size for autocorrelated samples.
        
        Args:
            samples: Array of samples
            max_lag: Maximum lag for autocorrelation
            
        Returns:
            Effective sample size
        """
        n = len(samples)
        
        # Calculate autocorrelation
        autocorr = np.correlate(samples - np.mean(samples), 
                               samples - np.mean(samples), 
                               mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        
        # Find first negative autocorrelation or max_lag
        cutoff = min(max_lag, len(autocorr))
        for i in range(1, cutoff):
            if autocorr[i] <= 0:
                cutoff = i
                break
        
        # Integrated autocorrelation time
        tau_int = 1 + 2 * np.sum(autocorr[1:cutoff])
        
        # Effective sample size
        n_eff = n / (2 * tau_int) if tau_int > 0 else n
        
        return max(1.0, n_eff)


def validate_sampling_parameters(n_samples: int, 
                               n_dimensions: int, 
                               method: str) -> None:
    """Validate sampling parameters.
    
    Args:
        n_samples: Number of samples
        n_dimensions: Number of dimensions
        method: Sampling method
        
    Raises:
        ValueError: If parameters are invalid
    """
    if n_samples <= 0:
        raise ValueError("Number of samples must be positive")
    
    if n_dimensions <= 0:
        raise ValueError("Number of dimensions must be positive")
    
    if method.upper() not in ["LHS", "MC"]:
        raise ValueError("Method must be 'LHS' or 'MC'")
    
    if method.upper() == "LHS" and n_samples < n_dimensions:
        warnings.warn(f"LHS with {n_samples} samples and {n_dimensions} dimensions "
                     "may not provide good space-filling properties")


def optimize_sample_size(target_precision: float = 0.05,
                        confidence: float = 0.95,
                        estimated_cv: float = 0.3) -> int:
    """Estimate required sample size for target precision.
    
    Args:
        target_precision: Target relative precision (e.g., 0.05 for 5%)
        confidence: Confidence level
        estimated_cv: Estimated coefficient of variation
        
    Returns:
        Recommended number of samples
    """
    # Z-score for confidence level
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)
    
    # Sample size formula for mean estimation
    n = ((z * estimated_cv) / target_precision) ** 2
    
    # Round up and ensure minimum
    return max(1000, int(np.ceil(n)))