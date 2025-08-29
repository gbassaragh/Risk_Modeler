"""Model selection and validation for distribution fitting.

Provides tools for comparing distributions, cross-validation,
and model selection using information criteria.
"""

import numpy as np
from scipy import stats
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import warnings
from dataclasses import dataclass, field
import logging

from .calibration import CalibrationResults, MLEFitter, BayesianFitter


@dataclass
class ModelComparisonResult:
    """Result from comparing multiple models."""
    models: List[CalibrationResults]
    best_model: CalibrationResults
    ranking_criterion: str
    ranking_scores: Dict[str, float]
    
    # Comparison metrics
    likelihood_ratios: Optional[Dict[str, float]] = None
    delta_aic: Optional[Dict[str, float]] = None
    delta_bic: Optional[Dict[str, float]] = None
    
    def summary(self) -> str:
        """Generate comparison summary."""
        lines = [f"Model Comparison Results (ranked by {self.ranking_criterion})"]
        lines.append("=" * 50)
        
        for i, model in enumerate(self.models, 1):
            score = self.ranking_scores.get(model.distribution_type, 0)
            lines.append(f"{i}. {model.distribution_type}: {score:.2f}")
        
        lines.append("")
        lines.append(f"Best model: {self.best_model.distribution_type}")
        lines.append(f"Best score: {self.ranking_scores[self.best_model.distribution_type]:.2f}")
        
        return "\n".join(lines)


@dataclass
class CrossValidationResult:
    """Result from cross-validation."""
    distribution_type: str
    cv_method: str
    n_splits: int
    
    # Performance metrics
    mean_log_likelihood: float
    std_log_likelihood: float
    mean_aic: float
    std_aic: float
    mean_bic: float
    std_bic: float
    
    # Fold-wise results
    fold_results: List[CalibrationResults] = field(default_factory=list)
    
    # Predictive performance
    mean_pred_error: Optional[float] = None
    std_pred_error: Optional[float] = None
    
    def summary(self) -> str:
        """Generate CV summary."""
        lines = [f"Cross-Validation Results: {self.distribution_type}"]
        lines.append(f"Method: {self.cv_method}, Splits: {self.n_splits}")
        lines.append(f"Mean Log-Likelihood: {self.mean_log_likelihood:.2f} ± {self.std_log_likelihood:.2f}")
        lines.append(f"Mean AIC: {self.mean_aic:.2f} ± {self.std_aic:.2f}")
        lines.append(f"Mean BIC: {self.mean_bic:.2f} ± {self.std_bic:.2f}")
        
        if self.mean_pred_error is not None:
            lines.append(f"Mean Prediction Error: {self.mean_pred_error:.4f} ± {self.std_pred_error:.4f}")
        
        return "\n".join(lines)


@dataclass
class InformationCriteriaResult:
    """Information criteria calculation results."""
    distribution_type: str
    n_params: int
    n_samples: int
    
    log_likelihood: float
    aic: float
    bic: float
    aicc: float  # Corrected AIC
    
    # Relative measures
    delta_aic: Optional[float] = None
    delta_bic: Optional[float] = None
    akaike_weight: Optional[float] = None
    
    def summary(self) -> str:
        """Generate summary."""
        lines = [f"Information Criteria: {self.distribution_type}"]
        lines.append(f"Parameters: {self.n_params}, Samples: {self.n_samples}")
        lines.append(f"Log-Likelihood: {self.log_likelihood:.2f}")
        lines.append(f"AIC: {self.aic:.2f}")
        lines.append(f"BIC: {self.bic:.2f}")
        lines.append(f"AICc: {self.aicc:.2f}")
        
        if self.delta_aic is not None:
            lines.append(f"ΔΑΙc: {self.delta_aic:.2f}")
        
        if self.akaike_weight is not None:
            lines.append(f"Akaike Weight: {self.akaike_weight:.3f}")
        
        return "\n".join(lines)


class InformationCriteria:
    """Calculate and compare information criteria."""
    
    @staticmethod
    def calculate(log_likelihood: float, 
                 n_params: int, 
                 n_samples: int,
                 distribution_type: str = "") -> InformationCriteriaResult:
        """Calculate information criteria.
        
        Args:
            log_likelihood: Log-likelihood value
            n_params: Number of parameters
            n_samples: Number of samples
            distribution_type: Distribution name
            
        Returns:
            InformationCriteriaResult
        """
        # AIC = 2k - 2ln(L)
        aic = 2 * n_params - 2 * log_likelihood
        
        # BIC = k*ln(n) - 2ln(L)
        bic = n_params * np.log(n_samples) - 2 * log_likelihood
        
        # Corrected AIC for small samples
        if n_samples - n_params - 1 > 0:
            aicc = aic + (2 * n_params * (n_params + 1)) / (n_samples - n_params - 1)
        else:
            aicc = np.inf  # Undefined for very small samples
        
        return InformationCriteriaResult(
            distribution_type=distribution_type,
            n_params=n_params,
            n_samples=n_samples,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            aicc=aicc
        )
    
    @staticmethod
    def compare_models(results: List[InformationCriteriaResult]) -> List[InformationCriteriaResult]:
        """Compare models using information criteria.
        
        Args:
            results: List of InformationCriteriaResult objects
            
        Returns:
            List sorted by AIC with relative measures calculated
        """
        if not results:
            return []
        
        # Sort by AIC
        sorted_results = sorted(results, key=lambda r: r.aic)
        best_aic = sorted_results[0].aic
        best_bic = min(r.bic for r in results)
        
        # Calculate relative measures
        delta_aics = []
        for result in sorted_results:
            result.delta_aic = result.aic - best_aic
            result.delta_bic = result.bic - best_bic
            delta_aics.append(result.delta_aic)
        
        # Calculate Akaike weights
        exp_values = np.exp(-0.5 * np.array(delta_aics))
        sum_exp = np.sum(exp_values)
        
        for i, result in enumerate(sorted_results):
            result.akaike_weight = exp_values[i] / sum_exp if sum_exp > 0 else 0
        
        return sorted_results


class CrossValidator:
    """Cross-validation for distribution fitting."""
    
    def __init__(self, 
                 cv_method: str = 'kfold',
                 n_splits: int = 5,
                 random_state: int = 42):
        """Initialize cross-validator.
        
        Args:
            cv_method: Cross-validation method ('kfold', 'stratified', 'timeseries')
            n_splits: Number of splits
            random_state: Random state for reproducibility
        """
        self.cv_method = cv_method
        self.n_splits = n_splits
        self.random_state = random_state
        
        # Create CV splitter
        if cv_method == 'kfold':
            self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        elif cv_method == 'timeseries':
            self.cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            raise ValueError(f"Unknown CV method: {cv_method}")
    
    def validate_distribution(self, 
                            data: np.ndarray,
                            distribution: str,
                            method: str = 'mle') -> CrossValidationResult:
        """Cross-validate distribution fitting.
        
        Args:
            data: Input data
            distribution: Distribution type
            method: Fitting method ('mle' or 'bayesian')
            
        Returns:
            CrossValidationResult
        """
        data = np.array(data)
        data = data[~np.isnan(data)]  # Remove NaN
        
        if len(data) < self.n_splits * 5:
            warnings.warn(f"Small dataset for {self.n_splits}-fold CV. Results may be unreliable.")
        
        fold_results = []
        log_likelihoods = []
        aics = []
        bics = []
        pred_errors = []
        
        for fold, (train_idx, test_idx) in enumerate(self.cv.split(data)):
            train_data = data[train_idx]
            test_data = data[test_idx]
            
            if len(train_data) < 5 or len(test_data) < 2:
                warnings.warn(f"Fold {fold} has insufficient data")
                continue
            
            try:
                # Fit on training data
                if method == 'mle':
                    fitter = MLEFitter(train_data)
                    result = fitter.fit(distribution)
                elif method == 'bayesian':
                    fitter = BayesianFitter(train_data)
                    result = fitter.fit(distribution, n_samples=1000)  # Fewer samples for CV
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                fold_results.append(result)
                
                # Evaluate on test data
                test_ll = self._evaluate_on_test_data(test_data, distribution, result.parameters)
                log_likelihoods.append(test_ll)
                
                # Calculate AIC/BIC for test data
                n_params = len(result.parameters)
                test_aic = 2 * n_params - 2 * test_ll
                test_bic = n_params * np.log(len(test_data)) - 2 * test_ll
                
                aics.append(test_aic)
                bics.append(test_bic)
                
                # Predictive error (if applicable)
                pred_error = self._calculate_predictive_error(test_data, distribution, result.parameters)
                if pred_error is not None:
                    pred_errors.append(pred_error)
            
            except Exception as e:
                logging.warning(f"Fold {fold} failed: {e}")
                continue
        
        if not log_likelihoods:
            raise RuntimeError("All CV folds failed")
        
        # Calculate summary statistics
        mean_ll = np.mean(log_likelihoods)
        std_ll = np.std(log_likelihoods)
        mean_aic = np.mean(aics)
        std_aic = np.std(aics)
        mean_bic = np.mean(bics)
        std_bic = np.std(bics)
        
        mean_pred_error = np.mean(pred_errors) if pred_errors else None
        std_pred_error = np.std(pred_errors) if pred_errors else None
        
        return CrossValidationResult(
            distribution_type=distribution,
            cv_method=self.cv_method,
            n_splits=len(fold_results),
            mean_log_likelihood=mean_ll,
            std_log_likelihood=std_ll,
            mean_aic=mean_aic,
            std_aic=std_aic,
            mean_bic=mean_bic,
            std_bic=std_bic,
            fold_results=fold_results,
            mean_pred_error=mean_pred_error,
            std_pred_error=std_pred_error
        )
    
    def _evaluate_on_test_data(self, 
                              test_data: np.ndarray,
                              distribution: str,
                              parameters: Dict[str, float]) -> float:
        """Evaluate fitted distribution on test data."""
        try:
            dist_info = MLEFitter.SUPPORTED_DISTRIBUTIONS[distribution]
            scipy_dist = dist_info['scipy']
            param_names = dist_info['params']
            
            # Extract parameters in scipy order
            params = []
            for param_name in param_names:
                if param_name in parameters:
                    params.append(parameters[param_name])
                else:
                    warnings.warn(f"Parameter {param_name} not found")
                    params.append(1.0)  # Default value
            
            # Calculate log-likelihood
            log_ll = np.sum(scipy_dist.logpdf(test_data, *params))
            
            # Check for invalid log-likelihood
            if np.isnan(log_ll) or np.isinf(log_ll):
                return -1e6  # Very poor fit
            
            return log_ll
        
        except Exception as e:
            logging.warning(f"Failed to evaluate on test data: {e}")
            return -1e6
    
    def _calculate_predictive_error(self,
                                   test_data: np.ndarray,
                                   distribution: str,
                                   parameters: Dict[str, float]) -> Optional[float]:
        """Calculate predictive error (if meaningful for the distribution)."""
        try:
            # For continuous distributions, calculate mean absolute error
            # between empirical and theoretical quantiles
            
            dist_info = MLEFitter.SUPPORTED_DISTRIBUTIONS[distribution]
            scipy_dist = dist_info['scipy']
            param_names = dist_info['params']
            
            params = [parameters.get(name, 1.0) for name in param_names]
            
            # Calculate theoretical quantiles
            p_values = np.linspace(0.01, 0.99, len(test_data))
            theoretical_quantiles = scipy_dist.ppf(p_values, *params)
            
            # Calculate empirical quantiles
            empirical_quantiles = np.quantile(test_data, p_values)
            
            # Mean absolute error
            mae = np.mean(np.abs(theoretical_quantiles - empirical_quantiles))
            
            return mae
        
        except Exception:
            return None


class ModelSelector:
    """Model selection using multiple criteria."""
    
    def __init__(self, 
                 selection_criteria: List[str] = None,
                 cv_enabled: bool = True):
        """Initialize model selector.
        
        Args:
            selection_criteria: List of criteria to use ('aic', 'bic', 'cv_likelihood')
            cv_enabled: Whether to use cross-validation
        """
        if selection_criteria is None:
            selection_criteria = ['aic', 'bic']
        
        self.selection_criteria = selection_criteria
        self.cv_enabled = cv_enabled
        
        if cv_enabled:
            self.cross_validator = CrossValidator()
    
    def select_best_distribution(self, 
                                data: np.ndarray,
                                candidate_distributions: List[str],
                                method: str = 'mle') -> ModelComparisonResult:
        """Select best distribution from candidates.
        
        Args:
            data: Input data
            candidate_distributions: List of distributions to compare
            method: Fitting method ('mle' or 'bayesian')
            
        Returns:
            ModelComparisonResult with best distribution
        """
        if not candidate_distributions:
            raise ValueError("No candidate distributions provided")
        
        # Fit all candidate distributions
        fitted_models = []
        cv_results = {}
        
        for dist in candidate_distributions:
            try:
                # Fit distribution
                if method == 'mle':
                    fitter = MLEFitter(data)
                    result = fitter.fit(dist)
                elif method == 'bayesian':
                    fitter = BayesianFitter(data)
                    result = fitter.fit(dist)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                fitted_models.append(result)
                
                # Cross-validation if enabled
                if self.cv_enabled and 'cv_likelihood' in self.selection_criteria:
                    cv_result = self.cross_validator.validate_distribution(data, dist, method)
                    cv_results[dist] = cv_result
                
            except Exception as e:
                logging.warning(f"Failed to fit {dist}: {e}")
                continue
        
        if not fitted_models:
            raise RuntimeError("No distributions could be fitted")
        
        # Calculate ranking scores
        ranking_scores = {}
        
        for criterion in self.selection_criteria:
            scores = self._calculate_ranking_scores(fitted_models, cv_results, criterion)
            
            # Combine scores (simple average for now)
            for dist, score in scores.items():
                if dist not in ranking_scores:
                    ranking_scores[dist] = 0
                ranking_scores[dist] += score / len(self.selection_criteria)
        
        # Find best model
        best_dist = min(ranking_scores.keys(), key=lambda k: ranking_scores[k])
        best_model = next(m for m in fitted_models if m.distribution_type == best_dist)
        
        # Sort models by ranking score
        sorted_models = sorted(fitted_models, key=lambda m: ranking_scores[m.distribution_type])
        
        # Calculate comparison metrics
        comparison_result = ModelComparisonResult(
            models=sorted_models,
            best_model=best_model,
            ranking_criterion=', '.join(self.selection_criteria),
            ranking_scores=ranking_scores
        )
        
        # Add detailed comparison metrics
        comparison_result = self._add_comparison_metrics(comparison_result)
        
        return comparison_result
    
    def _calculate_ranking_scores(self, 
                                 fitted_models: List[CalibrationResults],
                                 cv_results: Dict[str, CrossValidationResult],
                                 criterion: str) -> Dict[str, float]:
        """Calculate ranking scores for a specific criterion."""
        scores = {}
        
        if criterion == 'aic':
            for model in fitted_models:
                scores[model.distribution_type] = model.aic
        
        elif criterion == 'bic':
            for model in fitted_models:
                scores[model.distribution_type] = model.bic
        
        elif criterion == 'cv_likelihood' and cv_results:
            for model in fitted_models:
                if model.distribution_type in cv_results:
                    # Negative because we want to minimize (higher likelihood is better)
                    scores[model.distribution_type] = -cv_results[model.distribution_type].mean_log_likelihood
                else:
                    scores[model.distribution_type] = model.aic  # Fallback
        
        else:
            # Default to AIC
            for model in fitted_models:
                scores[model.distribution_type] = model.aic
        
        return scores
    
    def _add_comparison_metrics(self, result: ModelComparisonResult) -> ModelComparisonResult:
        """Add detailed comparison metrics."""
        if len(result.models) < 2:
            return result
        
        # Likelihood ratios
        best_ll = result.best_model.log_likelihood
        likelihood_ratios = {}
        
        for model in result.models:
            if model.distribution_type != result.best_model.distribution_type:
                lr = 2 * (best_ll - model.log_likelihood)
                likelihood_ratios[model.distribution_type] = lr
        
        result.likelihood_ratios = likelihood_ratios
        
        # Delta AIC and BIC
        best_aic = result.best_model.aic
        best_bic = result.best_model.bic
        
        delta_aic = {}
        delta_bic = {}
        
        for model in result.models:
            delta_aic[model.distribution_type] = model.aic - best_aic
            delta_bic[model.distribution_type] = model.bic - best_bic
        
        result.delta_aic = delta_aic
        result.delta_bic = delta_bic
        
        return result


# Convenience functions

def select_best_distribution(data: np.ndarray,
                           candidate_distributions: List[str] = None,
                           method: str = 'mle') -> ModelComparisonResult:
    """Select best distribution using default criteria.
    
    Args:
        data: Input data
        candidate_distributions: List of distributions to try
        method: Fitting method
        
    Returns:
        ModelComparisonResult with best distribution
    """
    if candidate_distributions is None:
        candidate_distributions = ['normal', 'lognormal', 'gamma', 'weibull']
    
    selector = ModelSelector()
    return selector.select_best_distribution(data, candidate_distributions, method)


def validate_model_performance(data: np.ndarray,
                              distribution: str,
                              method: str = 'mle',
                              n_splits: int = 5) -> CrossValidationResult:
    """Validate model performance using cross-validation.
    
    Args:
        data: Input data
        distribution: Distribution type
        method: Fitting method
        n_splits: Number of CV splits
        
    Returns:
        CrossValidationResult
    """
    validator = CrossValidator(n_splits=n_splits)
    return validator.validate_distribution(data, distribution, method)


def calculate_aic_bic(log_likelihood: float,
                     n_params: int,
                     n_samples: int) -> Tuple[float, float]:
    """Calculate AIC and BIC.
    
    Args:
        log_likelihood: Log-likelihood value
        n_params: Number of parameters
        n_samples: Number of samples
        
    Returns:
        Tuple of (AIC, BIC)
    """
    result = InformationCriteria.calculate(log_likelihood, n_params, n_samples)
    return result.aic, result.bic