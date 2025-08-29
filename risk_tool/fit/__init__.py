"""Calibration and fitting module for risk modeling.

Provides MLE and Bayesian parameter fitting for distributions,
model calibration from historical data, and goodness-of-fit testing.
"""

from .calibration import (
    DistributionFitter,
    MLEFitter, 
    BayesianFitter,
    GoodnessOfFitTester,
    CalibrationResults,
    fit_distribution_mle,
    fit_distribution_bayesian,
    compare_distributions
)

from .historical_data import (
    HistoricalDataProcessor,
    DataQualityChecker,
    OutlierDetector,
    SeasonalityAnalyzer,
    process_cost_data,
    detect_outliers,
    analyze_seasonality
)

from .model_selection import (
    ModelSelector,
    CrossValidator,
    InformationCriteria,
    select_best_distribution,
    validate_model_performance,
    calculate_aic_bic
)

__all__ = [
    # Calibration
    'DistributionFitter', 'MLEFitter', 'BayesianFitter', 'GoodnessOfFitTester',
    'CalibrationResults', 'fit_distribution_mle', 'fit_distribution_bayesian', 'compare_distributions',
    
    # Historical data processing
    'HistoricalDataProcessor', 'DataQualityChecker', 'OutlierDetector', 'SeasonalityAnalyzer',
    'process_cost_data', 'detect_outliers', 'analyze_seasonality',
    
    # Model selection
    'ModelSelector', 'CrossValidator', 'InformationCriteria',
    'select_best_distribution', 'validate_model_performance', 'calculate_aic_bic',
]