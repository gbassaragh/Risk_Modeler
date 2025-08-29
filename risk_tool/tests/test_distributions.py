"""Tests for probability distributions module.

Validates distribution sampling against analytical results.
"""

import numpy as np
import pytest
from scipy import stats
from risk_tool.engine.distributions import (
    DistributionSampler, 
    get_distribution_stats, 
    validate_distribution_config
)


class TestDistributionSampler:
    """Test distribution sampling functionality."""
    
    @pytest.fixture
    def random_state(self):
        """Provide consistent random state for testing."""
        return np.random.RandomState(42)
    
    def test_triangular_distribution(self, random_state):
        """Test triangular distribution sampling."""
        config = {
            'type': 'triangular',
            'low': 1.0,
            'mode': 2.0,
            'high': 4.0
        }
        
        samples = DistributionSampler.sample(config, 10000, random_state)
        
        # Check basic properties
        assert len(samples) == 10000
        assert np.all(samples >= 1.0)
        assert np.all(samples <= 4.0)
        
        # Check statistical properties (with tolerance)
        expected_mean = (1.0 + 2.0 + 4.0) / 3
        assert abs(np.mean(samples) - expected_mean) < 0.1
    
    def test_pert_distribution(self, random_state):
        """Test PERT distribution sampling."""
        config = {
            'type': 'pert',
            'min': 10.0,
            'most_likely': 20.0,
            'max': 40.0
        }
        
        samples = DistributionSampler.sample(config, 10000, random_state)
        
        # Check bounds
        assert len(samples) == 10000
        assert np.all(samples >= 10.0)
        assert np.all(samples <= 40.0)
        
        # Mean should be close to PERT expected value
        # E[X] = (min + 4*mode + max) / 6 for lambda=4
        expected_mean = (10.0 + 4*20.0 + 40.0) / 6
        assert abs(np.mean(samples) - expected_mean) < 1.0
    
    def test_normal_distribution(self, random_state):
        """Test normal distribution sampling."""
        config = {
            'type': 'normal',
            'mean': 100.0,
            'stdev': 15.0
        }
        
        samples = DistributionSampler.sample(config, 10000, random_state)
        
        # Check statistical properties
        assert abs(np.mean(samples) - 100.0) < 1.0
        assert abs(np.std(samples) - 15.0) < 1.0
    
    def test_truncated_normal(self, random_state):
        """Test truncated normal distribution."""
        config = {
            'type': 'normal',
            'mean': 100.0,
            'stdev': 15.0,
            'truncate_low': 80.0,
            'truncate_high': 120.0
        }
        
        samples = DistributionSampler.sample(config, 10000, random_state)
        
        # Check truncation bounds
        assert np.all(samples >= 80.0)
        assert np.all(samples <= 120.0)
    
    def test_lognormal_distribution(self, random_state):
        """Test log-normal distribution sampling."""
        config = {
            'type': 'lognormal',
            'mean': 1.0,  # Mean of underlying normal
            'sigma': 0.5
        }
        
        samples = DistributionSampler.sample(config, 10000, random_state)
        
        # All samples should be positive
        assert np.all(samples > 0)
        
        # Check that log of samples follows normal distribution
        log_samples = np.log(samples)
        assert abs(np.mean(log_samples) - 1.0) < 0.1
        assert abs(np.std(log_samples) - 0.5) < 0.1
    
    def test_discrete_distribution(self, random_state):
        """Test discrete distribution sampling."""
        config = {
            'type': 'discrete',
            'pmf': [['10', 0.3], ['20', 0.5], ['30', 0.2]]
        }
        
        samples = DistributionSampler.sample(config, 10000, random_state)
        
        # Check that only valid values are sampled
        unique_values = np.unique(samples)
        assert set(unique_values) == {10.0, 20.0, 30.0}
        
        # Check approximate probabilities
        value_counts = np.bincount(samples.astype(int))
        prob_10 = value_counts[10] / 10000
        prob_20 = value_counts[20] / 10000
        prob_30 = value_counts[30] / 10000
        
        assert abs(prob_10 - 0.3) < 0.05
        assert abs(prob_20 - 0.5) < 0.05
        assert abs(prob_30 - 0.2) < 0.05
    
    def test_uniform_distribution(self, random_state):
        """Test uniform distribution sampling."""
        config = {
            'type': 'uniform',
            'low': 5.0,
            'high': 15.0
        }
        
        samples = DistributionSampler.sample(config, 10000, random_state)
        
        # Check bounds
        assert np.all(samples >= 5.0)
        assert np.all(samples <= 15.0)
        
        # Check uniform distribution properties
        expected_mean = (5.0 + 15.0) / 2
        expected_std = (15.0 - 5.0) / np.sqrt(12)
        
        assert abs(np.mean(samples) - expected_mean) < 0.1
        assert abs(np.std(samples) - expected_std) < 0.1
    
    def test_invalid_distribution_type(self, random_state):
        """Test error handling for invalid distribution type."""
        config = {
            'type': 'invalid_distribution',
            'param1': 1.0
        }
        
        with pytest.raises(ValueError, match="Unknown distribution type"):
            DistributionSampler.sample(config, 100, random_state)


class TestDistributionStatistics:
    """Test analytical distribution statistics."""
    
    def test_triangular_stats(self):
        """Test triangular distribution statistics."""
        config = {
            'type': 'triangular',
            'low': 1.0,
            'mode': 2.0,
            'high': 4.0
        }
        
        mean, var = get_distribution_stats(config)
        
        expected_mean = (1.0 + 2.0 + 4.0) / 3
        expected_var = (1**2 + 2**2 + 4**2 - 1*2 - 1*4 - 2*4) / 18
        
        assert abs(mean - expected_mean) < 1e-10
        assert abs(var - expected_var) < 1e-10
    
    def test_normal_stats(self):
        """Test normal distribution statistics."""
        config = {
            'type': 'normal',
            'mean': 100.0,
            'stdev': 15.0
        }
        
        mean, var = get_distribution_stats(config)
        
        assert abs(mean - 100.0) < 1e-10
        assert abs(var - 15.0**2) < 1e-10
    
    def test_uniform_stats(self):
        """Test uniform distribution statistics."""
        config = {
            'type': 'uniform',
            'low': 5.0,
            'high': 15.0
        }
        
        mean, var = get_distribution_stats(config)
        
        expected_mean = (5.0 + 15.0) / 2
        expected_var = (15.0 - 5.0)**2 / 12
        
        assert abs(mean - expected_mean) < 1e-10
        assert abs(var - expected_var) < 1e-10
    
    def test_discrete_stats(self):
        """Test discrete distribution statistics."""
        config = {
            'type': 'discrete',
            'pmf': [['10', 0.3], ['20', 0.5], ['30', 0.2]]
        }
        
        mean, var = get_distribution_stats(config)
        
        # Expected value: 10*0.3 + 20*0.5 + 30*0.2 = 19
        # Variance: E[X^2] - E[X]^2
        # E[X^2] = 100*0.3 + 400*0.5 + 900*0.2 = 410
        # Var = 410 - 19^2 = 49
        
        expected_mean = 19.0
        expected_var = 49.0
        
        assert abs(mean - expected_mean) < 1e-10
        assert abs(var - expected_var) < 1e-10


class TestDistributionValidation:
    """Test distribution configuration validation."""
    
    def test_valid_triangular(self):
        """Test valid triangular distribution."""
        config = {
            'type': 'triangular',
            'low': 1.0,
            'mode': 2.0,
            'high': 4.0
        }
        
        # Should not raise exception
        validate_distribution_config(config)
    
    def test_invalid_triangular_mode(self):
        """Test invalid triangular distribution (mode out of bounds)."""
        config = {
            'type': 'triangular',
            'low': 1.0,
            'mode': 5.0,  # Mode > high
            'high': 4.0
        }
        
        with pytest.raises(ValueError):
            validate_distribution_config(config)
    
    def test_valid_pert(self):
        """Test valid PERT distribution."""
        config = {
            'type': 'pert',
            'min': 10.0,
            'most_likely': 20.0,
            'max': 40.0
        }
        
        # Should not raise exception
        validate_distribution_config(config)
    
    def test_invalid_normal_stdev(self):
        """Test invalid normal distribution (negative stdev)."""
        config = {
            'type': 'normal',
            'mean': 100.0,
            'stdev': -15.0  # Negative standard deviation
        }
        
        with pytest.raises(ValueError):
            validate_distribution_config(config)
    
    def test_valid_discrete_pmf(self):
        """Test valid discrete distribution PMF."""
        config = {
            'type': 'discrete',
            'pmf': [['1', 0.3], ['2', 0.4], ['3', 0.3]]
        }
        
        # Should not raise exception
        validate_distribution_config(config)
    
    def test_invalid_discrete_pmf_sum(self):
        """Test invalid discrete distribution (PMF doesn't sum to 1)."""
        config = {
            'type': 'discrete',
            'pmf': [['1', 0.3], ['2', 0.4], ['3', 0.4]]  # Sum = 1.1
        }
        
        with pytest.raises(ValueError):
            validate_distribution_config(config)
    
    def test_invalid_distribution_type(self):
        """Test unknown distribution type."""
        config = {
            'type': 'unknown_distribution'
        }
        
        with pytest.raises(ValueError):
            validate_distribution_config(config)
    
    def test_missing_required_parameters(self):
        """Test missing required parameters."""
        config = {
            'type': 'normal',
            'mean': 100.0
            # Missing 'stdev'
        }
        
        with pytest.raises(ValueError):
            validate_distribution_config(config)


class TestDistributionEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.fixture
    def random_state(self):
        return np.random.RandomState(12345)
    
    def test_degenerate_triangular(self, random_state):
        """Test degenerate triangular distribution (all parameters equal)."""
        config = {
            'type': 'triangular',
            'low': 5.0,
            'mode': 5.0,
            'high': 5.0
        }
        
        samples = DistributionSampler.sample(config, 100, random_state)
        
        # All samples should be exactly 5.0
        assert np.all(np.abs(samples - 5.0) < 1e-10)
    
    def test_zero_variance_normal(self, random_state):
        """Test normal distribution with zero variance."""
        config = {
            'type': 'normal',
            'mean': 10.0,
            'stdev': 0.0
        }
        
        with pytest.raises(ValueError):
            validate_distribution_config(config)
    
    def test_large_sample_size(self, random_state):
        """Test with large sample size for performance."""
        config = {
            'type': 'normal',
            'mean': 0.0,
            'stdev': 1.0
        }
        
        samples = DistributionSampler.sample(config, 100000, random_state)
        
        assert len(samples) == 100000
        assert abs(np.mean(samples)) < 0.01  # Should be very close to 0
        assert abs(np.std(samples) - 1.0) < 0.01  # Should be very close to 1


if __name__ == "__main__":
    pytest.main([__file__])