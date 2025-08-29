"""Core risk modeling engine."""

# Core simulation components
from .distributions import DistributionSampler, get_distribution_stats, validate_distribution_config
from .sampler import MonteCarloSampler, ConvergenceDiagnostics
from .correlation import apply_correlations, ImanConovierTransform, CholekyTransform
from .aggregation import SimulationEngine, run_simulation

# Cost and risk modeling
from .cost_models import Project, WBSItem, CostSimulator, SimulationConfig
from .risk_driver import RiskItem, RiskRegister, RiskSimulator, RiskAggregator

# Advanced modeling features
from .escalation import EscalationEngine, MarketIndexEngine, ProductivityFactors
from .schedule_link import Schedule, ScheduleSimulator, ScheduleCostCalculator
from .sensitivity import SensitivityAnalyzer, ShapleyAnalyzer, RiskContributionAnalyzer

# Validation and audit
from .validation import ValidationEngine, validate_simulation_inputs
from .audit import AuditLogger, DeterminismVerifier, ComplianceReporter

# Enhanced data models
from .data_models import (
    ProjectInfo, WBSItem as EnhancedWBSItem, RiskItem as EnhancedRiskItem, 
    Distribution, LatentFactor, TwoLayerConfig
)

# Advanced distributions
from .mixtures_evt import (
    MixtureDistribution, GeneralizedParetoDistribution, 
    KDEDistribution, EmpiricalDistribution,
    create_mixture_distribution, create_evt_distribution,
    create_kde_distribution, create_empirical_distribution
)

# Factor modeling
from .latent_factors import (
    LatentFactorModel, FactorModelBuilder,
    create_commodity_factor_model, create_weather_factor_model,
    create_labor_market_factor_model, combine_factor_models
)

# Two-layer analysis
from .two_layer import (
    TwoLayerMonteCarlo, TwoLayerConfig, TwoLayerResults,
    EpistemicParameter, create_default_two_layer_config,
    create_epistemic_from_distribution, analyze_parameter_uncertainty
)

# Exception handling and logging
from .exceptions import (
    RiskModelingError, ValidationError, ComputationError, 
    IOError, PerformanceError
)
from .logging_config import setup_logging, get_logger

# Performance optimizations
from .performance import (
    fast_correlation_matrix, fast_percentiles, 
    fast_variance_reduction, monte_carlo_numba
)

__all__ = [
    # Core simulation components
    'DistributionSampler', 'get_distribution_stats', 'validate_distribution_config',
    'MonteCarloSampler', 'ConvergenceDiagnostics',
    'apply_correlations', 'ImanConovierTransform', 'CholekyTransform',
    'SimulationEngine', 'run_simulation',
    
    # Cost and risk modeling
    'Project', 'WBSItem', 'CostSimulator', 'SimulationConfig',
    'RiskItem', 'RiskRegister', 'RiskSimulator', 'RiskAggregator',
    
    # Advanced modeling features
    'EscalationEngine', 'MarketIndexEngine', 'ProductivityFactors',
    'Schedule', 'ScheduleSimulator', 'ScheduleCostCalculator',
    'SensitivityAnalyzer', 'ShapleyAnalyzer', 'RiskContributionAnalyzer',
    
    # Validation and audit
    'ValidationEngine', 'validate_simulation_inputs',
    'AuditLogger', 'DeterminismVerifier', 'ComplianceReporter',
    
    # Enhanced data models
    'ProjectInfo', 'EnhancedWBSItem', 'EnhancedRiskItem', 'Distribution', 'LatentFactor', 'TwoLayerConfig',
    
    # Advanced distributions
    'MixtureDistribution', 'GeneralizedParetoDistribution', 'KDEDistribution', 'EmpiricalDistribution',
    'create_mixture_distribution', 'create_evt_distribution', 'create_kde_distribution', 'create_empirical_distribution',
    
    # Factor modeling
    'LatentFactorModel', 'FactorModelBuilder',
    'create_commodity_factor_model', 'create_weather_factor_model', 'create_labor_market_factor_model', 'combine_factor_models',
    
    # Two-layer analysis
    'TwoLayerMonteCarlo', 'TwoLayerResults', 'EpistemicParameter',
    'create_default_two_layer_config', 'create_epistemic_from_distribution', 'analyze_parameter_uncertainty',
    
    # Exception handling and logging
    'RiskModelingError', 'ValidationError', 'ComputationError', 'IOError', 'PerformanceError',
    'setup_logging', 'get_logger',
    
    # Performance optimizations
    'fast_correlation_matrix', 'fast_percentiles', 'fast_variance_reduction', 'monte_carlo_numba'
]