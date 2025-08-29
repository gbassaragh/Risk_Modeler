"""Performance optimization utilities for Risk Modeling Tool.

Provides performance monitoring, profiling, and optimization utilities
for Monte Carlo simulations and numerical computations.
"""

import functools
import time
import cProfile
import pstats
from typing import Callable, Dict, Any, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
from numba import jit, njit
import warnings


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    duration: float
    memory_usage: Optional[float] = None
    iterations: Optional[int] = None
    throughput: Optional[float] = None
    
    @property
    def iterations_per_second(self) -> Optional[float]:
        """Calculate iterations per second."""
        if self.iterations and self.duration > 0:
            return self.iterations / self.duration
        return None


class PerformanceTimer:
    """Context manager for performance timing."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        
    @property
    def duration(self) -> float:
        """Get operation duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


def performance_monitor(name: Optional[str] = None):
    """Decorator for monitoring function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name or f"{func.__module__}.{func.__name__}"
            
            with PerformanceTimer(func_name) as timer:
                result = func(*args, **kwargs)
            
            # Store metrics (could be extended to log to external system)
            metrics = PerformanceMetrics(duration=timer.duration)
            
            # Optionally log performance
            if timer.duration > 1.0:  # Log slow operations
                print(f"Performance: {func_name} took {timer.duration:.3f}s")
            
            return result
        return wrapper
    return decorator


@contextmanager
def profiler(output_file: Optional[str] = None, sort_by: str = 'cumulative'):
    """Context manager for profiling code sections."""
    pr = cProfile.Profile()
    pr.enable()
    
    try:
        yield pr
    finally:
        pr.disable()
        
        if output_file:
            pr.dump_stats(output_file)
        else:
            stats = pstats.Stats(pr)
            stats.sort_stats(sort_by)
            stats.print_stats(20)  # Top 20 functions


@jit(nopython=True, cache=True)
def fast_correlation_matrix(samples: np.ndarray) -> np.ndarray:
    """Numba-optimized correlation matrix calculation.
    
    Args:
        samples: 2D array of samples (n_samples, n_variables)
        
    Returns:
        Correlation matrix (n_variables, n_variables)
    """
    n_samples, n_vars = samples.shape
    corr_matrix = np.zeros((n_vars, n_vars))
    
    # Calculate means
    means = np.mean(samples, axis=0)
    
    # Calculate correlation coefficients
    for i in range(n_vars):
        for j in range(i, n_vars):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # Calculate covariance
                cov = 0.0
                var_i = 0.0
                var_j = 0.0
                
                for k in range(n_samples):
                    diff_i = samples[k, i] - means[i]
                    diff_j = samples[k, j] - means[j]
                    cov += diff_i * diff_j
                    var_i += diff_i * diff_i
                    var_j += diff_j * diff_j
                
                # Calculate correlation
                if var_i > 0 and var_j > 0:
                    corr = cov / np.sqrt(var_i * var_j)
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                
    return corr_matrix


@njit(cache=True)
def fast_percentiles(data: np.ndarray, percentiles: np.ndarray) -> np.ndarray:
    """Fast percentile calculation using Numba.
    
    Args:
        data: 1D array of data points
        percentiles: Array of percentile values (0-100)
        
    Returns:
        Array of percentile values
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    results = np.zeros(len(percentiles))
    
    for i, p in enumerate(percentiles):
        if p <= 0:
            results[i] = sorted_data[0]
        elif p >= 100:
            results[i] = sorted_data[-1]
        else:
            # Linear interpolation
            index = (p / 100.0) * (n - 1)
            lower_idx = int(np.floor(index))
            upper_idx = int(np.ceil(index))
            
            if lower_idx == upper_idx:
                results[i] = sorted_data[lower_idx]
            else:
                weight = index - lower_idx
                results[i] = (1 - weight) * sorted_data[lower_idx] + weight * sorted_data[upper_idx]
    
    return results


class MemoryProfiler:
    """Memory usage profiler for large simulations."""
    
    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
        
    def __enter__(self):
        try:
            import psutil
            import os
            self.process = psutil.Process(os.getpid())
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            return self
        except ImportError:
            warnings.warn("psutil not available for memory profiling")
            return self
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = max(self.start_memory, end_memory)
            print(f"Memory usage: {self.peak_memory:.1f} MB (peak)")
        except AttributeError:
            pass


def optimize_numpy_settings():
    """Optimize NumPy settings for performance."""
    # Set optimal BLAS thread count
    import os
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '4'
    if 'MKL_NUM_THREADS' not in os.environ:
        os.environ['MKL_NUM_THREADS'] = '4'
    
    # Set NumPy random seed for reproducibility
    np.random.seed(42)
    
    # Enable NumPy error handling
    np.seterr(all='warn')


def benchmark_simulation_performance(n_iterations_list: list = None) -> Dict[str, Any]:
    """Benchmark Monte Carlo simulation performance.
    
    Args:
        n_iterations_list: List of iteration counts to benchmark
        
    Returns:
        Performance benchmark results
    """
    if n_iterations_list is None:
        n_iterations_list = [1000, 5000, 10000, 50000]
    
    results = {}
    
    for n_iter in n_iterations_list:
        with PerformanceTimer(f"MC_{n_iter}") as timer:
            # Simple Monte Carlo benchmark
            samples = np.random.random((n_iter, 10))
            result = np.mean(samples, axis=1)
            percentiles = fast_percentiles(result, np.array([10, 50, 90]))
        
        results[n_iter] = {
            'duration': timer.duration,
            'iterations_per_second': n_iter / timer.duration,
            'percentiles': percentiles
        }
    
    return results


# Utility functions for performance optimization
def enable_numba_parallel():
    """Enable Numba parallel execution."""
    import numba
    numba.config.THREADING_LAYER = 'threadsafe'


def cache_computation(func: Callable) -> Callable:
    """Simple caching decorator for expensive computations."""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from arguments
        key = str(args) + str(sorted(kwargs.items()))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    return wrapper