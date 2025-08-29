"""Distribution calibration with MLE and Bayesian fitting.

Provides comprehensive parameter estimation capabilities for fitting
distributions to historical data with uncertainty quantification.
"""

import numpy as np
from scipy import stats, optimize
from scipy.special import gamma, beta as beta_func, digamma
from sklearn.model_selection import KFold
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import warnings
from dataclasses import dataclass, field
import logging

# Bayesian inference
try:
    import pymc as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    warnings.warn("PyMC not available. Bayesian fitting will be disabled.")


@dataclass
class CalibrationResults:
    """Results from distribution calibration."""
    distribution_type: str
    method: str  # 'MLE' or 'Bayesian'
    
    # Parameter estimates
    parameters: Dict[str, float]
    parameter_uncertainties: Optional[Dict[str, float]] = None
    parameter_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    
    # Goodness of fit
    log_likelihood: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    
    # Statistical tests
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    ad_statistic: Optional[float] = None
    ad_pvalue: Optional[float] = None
    
    # Model diagnostics
    convergence_info: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [f"Distribution: {self.distribution_type}"]
        lines.append(f"Method: {self.method}")
        lines.append(f"Parameters:")
        
        for param, value in self.parameters.items():
            if self.parameter_uncertainties and param in self.parameter_uncertainties:
                uncertainty = self.parameter_uncertainties[param]
                lines.append(f"  {param}: {value:.4f} ± {uncertainty:.4f}")
            else:
                lines.append(f"  {param}: {value:.4f}")
        
        lines.append(f"Log-likelihood: {self.log_likelihood:.2f}")
        lines.append(f"AIC: {self.aic:.2f}")
        lines.append(f"BIC: {self.bic:.2f}")
        
        if self.ks_pvalue is not None:
            lines.append(f"KS test p-value: {self.ks_pvalue:.4f}")
        
        return "\n".join(lines)


class DistributionFitter:
    """Base class for distribution fitting."""
    
    SUPPORTED_DISTRIBUTIONS = {
        'normal': {'params': ['loc', 'scale'], 'scipy': stats.norm},
        'lognormal': {'params': ['s', 'scale'], 'scipy': stats.lognorm},
        'triangular': {'params': ['c', 'loc', 'scale'], 'scipy': stats.triang},
        'beta': {'params': ['a', 'b'], 'scipy': stats.beta},
        'gamma': {'params': ['a', 'scale'], 'scipy': stats.gamma},
        'weibull': {'params': ['c', 'scale'], 'scipy': stats.weibull_min},
        'uniform': {'params': ['loc', 'scale'], 'scipy': stats.uniform},
        'exponential': {'params': ['scale'], 'scipy': stats.expon},
        'pareto': {'params': ['b', 'scale'], 'scipy': stats.pareto}
    }
    
    def __init__(self, data: np.ndarray):
        """Initialize fitter with data.
        
        Args:
            data: Observed data points
        """
        self.data = np.array(data)
        self.n_samples = len(self.data)
        
        if self.n_samples < 5:
            raise ValueError("Need at least 5 data points for fitting")
        
        # Data statistics
        self.data_mean = np.mean(self.data)
        self.data_std = np.std(self.data, ddof=1)
        self.data_min = np.min(self.data)
        self.data_max = np.max(self.data)
    
    def _validate_distribution(self, distribution: str) -> None:
        """Validate distribution type."""
        if distribution not in self.SUPPORTED_DISTRIBUTIONS:
            supported = list(self.SUPPORTED_DISTRIBUTIONS.keys())
            raise ValueError(f"Unsupported distribution: {distribution}. "
                           f"Supported: {supported}")
    
    def _calculate_information_criteria(self, 
                                       log_likelihood: float, 
                                       n_params: int) -> Tuple[float, float]:
        """Calculate AIC and BIC."""
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(self.n_samples) * n_params - 2 * log_likelihood
        return aic, bic


class MLEFitter(DistributionFitter):
    """Maximum likelihood estimation for distribution parameters."""
    
    def fit(self, distribution: str, **kwargs) -> CalibrationResults:
        """Fit distribution using MLE.
        
        Args:
            distribution: Distribution type
            **kwargs: Additional fitting options
            
        Returns:
            CalibrationResults with MLE estimates
        """
        self._validate_distribution(distribution)
        
        try:
            if distribution == 'normal':
                return self._fit_normal()
            elif distribution == 'lognormal':
                return self._fit_lognormal()
            elif distribution == 'triangular':
                return self._fit_triangular()
            elif distribution == 'beta':
                return self._fit_beta()
            elif distribution == 'gamma':
                return self._fit_gamma()
            elif distribution == 'weibull':
                return self._fit_weibull()
            elif distribution == 'uniform':
                return self._fit_uniform()
            elif distribution == 'exponential':
                return self._fit_exponential()
            elif distribution == 'pareto':
                return self._fit_pareto()
            else:
                return self._fit_scipy_generic(distribution)
        
        except Exception as e:
            logging.error(f"MLE fitting failed for {distribution}: {e}")
            raise
    
    def _fit_normal(self) -> CalibrationResults:
        """Fit normal distribution."""
        # MLE estimates
        loc = self.data_mean
        scale = self.data_std
        
        # Calculate log-likelihood
        log_likelihood = np.sum(stats.norm.logpdf(self.data, loc=loc, scale=scale))
        
        # Parameter uncertainties (Fisher information)
        loc_se = scale / np.sqrt(self.n_samples)
        scale_se = scale / np.sqrt(2 * self.n_samples)
        
        parameters = {'loc': loc, 'scale': scale}
        uncertainties = {'loc': loc_se, 'scale': scale_se}
        
        aic, bic = self._calculate_information_criteria(log_likelihood, 2)
        
        # Goodness of fit
        ks_stat, ks_p = stats.kstest(self.data, lambda x: stats.norm.cdf(x, loc=loc, scale=scale))
        
        return CalibrationResults(
            distribution_type='normal',
            method='MLE',
            parameters=parameters,
            parameter_uncertainties=uncertainties,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            ks_statistic=ks_stat,
            ks_pvalue=ks_p
        )
    
    def _fit_lognormal(self) -> CalibrationResults:
        """Fit lognormal distribution."""
        # Transform to log space
        log_data = np.log(self.data)
        
        # MLE estimates in log space
        mu = np.mean(log_data)
        sigma = np.sqrt(np.mean((log_data - mu)**2))  # MLE estimate (biased)
        
        # Convert to scipy parameterization: lognorm(s, scale=exp(mu))
        s = sigma
        scale = np.exp(mu)
        
        # Log-likelihood
        log_likelihood = np.sum(stats.lognorm.logpdf(self.data, s=s, scale=scale))
        
        # Parameter uncertainties
        mu_se = sigma / np.sqrt(self.n_samples)
        sigma_se = sigma / np.sqrt(2 * self.n_samples)
        
        parameters = {'s': s, 'scale': scale}
        uncertainties = {'s': sigma_se, 'scale': scale * mu_se}  # Delta method
        
        aic, bic = self._calculate_information_criteria(log_likelihood, 2)
        
        # Goodness of fit
        ks_stat, ks_p = stats.kstest(self.data, lambda x: stats.lognorm.cdf(x, s=s, scale=scale))
        
        return CalibrationResults(
            distribution_type='lognormal',
            method='MLE',
            parameters=parameters,
            parameter_uncertainties=uncertainties,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            ks_statistic=ks_stat,
            ks_pvalue=ks_p
        )
    
    def _fit_triangular(self) -> CalibrationResults:
        """Fit triangular distribution using method of moments."""
        # Use method of moments for triangular (MLE is complex)
        warnings.warn("Using method of moments for triangular distribution")
        
        # Estimate parameters from data range and mean
        a = self.data_min
        b = self.data_max
        
        # Solve for mode using mean formula: mean = (a + b + c) / 3
        c = 3 * self.data_mean - a - b
        
        # Ensure mode is within bounds
        c = max(a, min(b, c))
        
        # Convert to scipy parameterization
        loc = a
        scale = b - a
        c_param = (c - a) / (b - a) if (b - a) > 0 else 0.5
        
        parameters = {'c': c_param, 'loc': loc, 'scale': scale}
        
        # Log-likelihood
        log_likelihood = np.sum(stats.triang.logpdf(self.data, c=c_param, loc=loc, scale=scale))
        
        aic, bic = self._calculate_information_criteria(log_likelihood, 3)
        
        # Goodness of fit
        ks_stat, ks_p = stats.kstest(
            self.data, 
            lambda x: stats.triang.cdf(x, c=c_param, loc=loc, scale=scale)
        )
        
        return CalibrationResults(
            distribution_type='triangular',
            method='MLE',
            parameters=parameters,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            ks_statistic=ks_stat,
            ks_pvalue=ks_p,
            warnings=["Used method of moments instead of MLE"]
        )
    
    def _fit_beta(self) -> CalibrationResults:
        """Fit beta distribution (assuming data in [0, 1])."""
        # Check if data is in [0, 1] range
        if np.any(self.data < 0) or np.any(self.data > 1):
            # Scale data to [0, 1]
            data_scaled = (self.data - self.data_min) / (self.data_max - self.data_min)
            warnings.warn("Data scaled to [0, 1] for beta distribution")
        else:
            data_scaled = self.data
        
        # Method of moments initial estimates
        mean = np.mean(data_scaled)
        var = np.var(data_scaled)
        
        if var >= mean * (1 - mean):
            # Variance too large for beta distribution
            raise ValueError("Data variance too large for beta distribution")
        
        # Method of moments
        alpha = mean * (mean * (1 - mean) / var - 1)
        beta = (1 - mean) * (mean * (1 - mean) / var - 1)
        
        # Ensure positive parameters
        alpha = max(0.1, alpha)
        beta = max(0.1, beta)
        
        parameters = {'a': alpha, 'b': beta}
        
        # Log-likelihood
        log_likelihood = np.sum(stats.beta.logpdf(data_scaled, a=alpha, b=beta))
        
        aic, bic = self._calculate_information_criteria(log_likelihood, 2)
        
        # Goodness of fit
        ks_stat, ks_p = stats.kstest(data_scaled, lambda x: stats.beta.cdf(x, a=alpha, b=beta))
        
        warnings_list = []
        if np.any(self.data < 0) or np.any(self.data > 1):
            warnings_list.append("Data was scaled to [0, 1]")
        
        return CalibrationResults(
            distribution_type='beta',
            method='MLE',
            parameters=parameters,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            ks_statistic=ks_stat,
            ks_pvalue=ks_p,
            warnings=warnings_list
        )
    
    def _fit_scipy_generic(self, distribution: str) -> CalibrationResults:
        """Generic fitting using scipy's fit method."""
        dist_info = self.SUPPORTED_DISTRIBUTIONS[distribution]
        scipy_dist = dist_info['scipy']
        
        try:
            # Use scipy's MLE fitting
            fitted_params = scipy_dist.fit(self.data)
            
            # Create parameter dictionary
            param_names = dist_info['params']
            if len(fitted_params) != len(param_names):
                # Handle distributions with fixed parameters
                param_names = [f'param_{i}' for i in range(len(fitted_params))]
            
            parameters = dict(zip(param_names, fitted_params))
            
            # Log-likelihood
            log_likelihood = np.sum(scipy_dist.logpdf(self.data, *fitted_params))
            
            aic, bic = self._calculate_information_criteria(log_likelihood, len(fitted_params))
            
            # Goodness of fit
            ks_stat, ks_p = stats.kstest(
                self.data, 
                lambda x: scipy_dist.cdf(x, *fitted_params)
            )
            
            return CalibrationResults(
                distribution_type=distribution,
                method='MLE',
                parameters=parameters,
                log_likelihood=log_likelihood,
                aic=aic,
                bic=bic,
                ks_statistic=ks_stat,
                ks_pvalue=ks_p
            )
        
        except Exception as e:
            raise RuntimeError(f"Scipy fitting failed for {distribution}: {e}")
    
    def _fit_gamma(self) -> CalibrationResults:
        """Fit gamma distribution."""
        return self._fit_scipy_generic('gamma')
    
    def _fit_weibull(self) -> CalibrationResults:
        """Fit Weibull distribution."""
        return self._fit_scipy_generic('weibull')
    
    def _fit_uniform(self) -> CalibrationResults:
        """Fit uniform distribution."""
        loc = self.data_min
        scale = self.data_max - self.data_min
        
        parameters = {'loc': loc, 'scale': scale}
        
        # Log-likelihood
        log_likelihood = np.sum(stats.uniform.logpdf(self.data, loc=loc, scale=scale))
        
        aic, bic = self._calculate_information_criteria(log_likelihood, 2)
        
        # Goodness of fit
        ks_stat, ks_p = stats.kstest(self.data, lambda x: stats.uniform.cdf(x, loc=loc, scale=scale))
        
        return CalibrationResults(
            distribution_type='uniform',
            method='MLE',
            parameters=parameters,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            ks_statistic=ks_stat,
            ks_pvalue=ks_p
        )
    
    def _fit_exponential(self) -> CalibrationResults:
        """Fit exponential distribution."""
        scale = self.data_mean
        
        parameters = {'scale': scale}
        
        # Log-likelihood
        log_likelihood = np.sum(stats.expon.logpdf(self.data, scale=scale))
        
        aic, bic = self._calculate_information_criteria(log_likelihood, 1)
        
        # Goodness of fit
        ks_stat, ks_p = stats.kstest(self.data, lambda x: stats.expon.cdf(x, scale=scale))
        
        return CalibrationResults(
            distribution_type='exponential',
            method='MLE',
            parameters=parameters,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            ks_statistic=ks_stat,
            ks_pvalue=ks_p
        )
    
    def _fit_pareto(self) -> CalibrationResults:
        """Fit Pareto distribution."""
        return self._fit_scipy_generic('pareto')


class BayesianFitter(DistributionFitter):
    """Bayesian parameter estimation for distributions."""
    
    def __init__(self, data: np.ndarray):
        """Initialize Bayesian fitter."""
        super().__init__(data)
        
        if not BAYESIAN_AVAILABLE:
            raise RuntimeError("PyMC not available. Install with: pip install pymc")
    
    def fit(self, 
            distribution: str, 
            n_samples: int = 2000,
            n_tune: int = 1000,
            chains: int = 2,
            **kwargs) -> CalibrationResults:
        """Fit distribution using Bayesian inference.
        
        Args:
            distribution: Distribution type
            n_samples: Number of posterior samples
            n_tune: Number of tuning samples
            chains: Number of MCMC chains
            **kwargs: Additional PyMC options
            
        Returns:
            CalibrationResults with Bayesian estimates
        """
        self._validate_distribution(distribution)
        
        if not BAYESIAN_AVAILABLE:
            raise RuntimeError("PyMC not available")
        
        try:
            if distribution == 'normal':
                return self._fit_normal_bayesian(n_samples, n_tune, chains, **kwargs)
            elif distribution == 'lognormal':
                return self._fit_lognormal_bayesian(n_samples, n_tune, chains, **kwargs)
            elif distribution == 'gamma':
                return self._fit_gamma_bayesian(n_samples, n_tune, chains, **kwargs)
            else:
                raise ValueError(f"Bayesian fitting not implemented for {distribution}")
        
        except Exception as e:
            logging.error(f"Bayesian fitting failed for {distribution}: {e}")
            raise
    
    def _fit_normal_bayesian(self, n_samples, n_tune, chains, **kwargs) -> CalibrationResults:
        """Bayesian fitting for normal distribution."""
        with pm.Model() as model:
            # Priors
            mu = pm.Normal('mu', mu=self.data_mean, sigma=self.data_std * 2)
            sigma = pm.HalfNormal('sigma', sigma=self.data_std * 2)
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=self.data)
            
            # Sample posterior
            trace = pm.sample(n_samples, tune=n_tune, chains=chains, **kwargs)
        
        # Extract results
        posterior = az.convert_to_inference_data(trace)
        summary = az.summary(posterior)
        
        # Parameter estimates (posterior means)
        parameters = {
            'loc': float(summary.loc['mu', 'mean']),
            'scale': float(summary.loc['sigma', 'mean'])
        }
        
        # Parameter uncertainties (posterior standard deviations)
        uncertainties = {
            'loc': float(summary.loc['mu', 'sd']),
            'scale': float(summary.loc['sigma', 'sd'])
        }
        
        # Credible intervals
        intervals = {
            'loc': (float(summary.loc['mu', 'hdi_5%']), float(summary.loc['mu', 'hdi_95%'])),
            'scale': (float(summary.loc['sigma', 'hdi_5%']), float(summary.loc['sigma', 'hdi_95%']))
        }
        
        # Log-likelihood at posterior mean
        log_likelihood = np.sum(
            stats.norm.logpdf(self.data, loc=parameters['loc'], scale=parameters['scale'])
        )
        
        aic, bic = self._calculate_information_criteria(log_likelihood, 2)
        
        # Convergence diagnostics
        convergence_info = {
            'r_hat': {var: float(summary.loc[var, 'r_hat']) for var in ['mu', 'sigma']},
            'eff_sample_size': {var: float(summary.loc[var, 'ess_bulk']) for var in ['mu', 'sigma']},
            'n_divergences': int(posterior.sample_stats.diverging.sum())
        }
        
        return CalibrationResults(
            distribution_type='normal',
            method='Bayesian',
            parameters=parameters,
            parameter_uncertainties=uncertainties,
            parameter_intervals=intervals,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            convergence_info=convergence_info
        )
    
    def _fit_lognormal_bayesian(self, n_samples, n_tune, chains, **kwargs) -> CalibrationResults:
        """Bayesian fitting for lognormal distribution."""
        log_data = np.log(self.data)
        
        with pm.Model() as model:
            # Priors for log-normal parameters
            mu = pm.Normal('mu', mu=np.mean(log_data), sigma=np.std(log_data) * 2)
            sigma = pm.HalfNormal('sigma', sigma=np.std(log_data) * 2)
            
            # Likelihood
            y_obs = pm.Lognormal('y_obs', mu=mu, sigma=sigma, observed=self.data)
            
            # Sample posterior
            trace = pm.sample(n_samples, tune=n_tune, chains=chains, **kwargs)
        
        # Extract results
        posterior = az.convert_to_inference_data(trace)
        summary = az.summary(posterior)
        
        # Convert to scipy parameterization
        mu_mean = float(summary.loc['mu', 'mean'])
        sigma_mean = float(summary.loc['sigma', 'mean'])
        
        parameters = {
            's': sigma_mean,
            'scale': np.exp(mu_mean)
        }
        
        uncertainties = {
            's': float(summary.loc['sigma', 'sd']),
            'scale': np.exp(mu_mean) * float(summary.loc['mu', 'sd'])  # Delta method
        }
        
        # Log-likelihood
        log_likelihood = np.sum(
            stats.lognorm.logpdf(self.data, s=parameters['s'], scale=parameters['scale'])
        )
        
        aic, bic = self._calculate_information_criteria(log_likelihood, 2)
        
        # Convergence diagnostics
        convergence_info = {
            'r_hat': {var: float(summary.loc[var, 'r_hat']) for var in ['mu', 'sigma']},
            'eff_sample_size': {var: float(summary.loc[var, 'ess_bulk']) for var in ['mu', 'sigma']},
            'n_divergences': int(posterior.sample_stats.diverging.sum())
        }
        
        return CalibrationResults(
            distribution_type='lognormal',
            method='Bayesian',
            parameters=parameters,
            parameter_uncertainties=uncertainties,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            convergence_info=convergence_info
        )
    
    def _fit_gamma_bayesian(self, n_samples, n_tune, chains, **kwargs) -> CalibrationResults:
        """Bayesian fitting for gamma distribution."""
        with pm.Model() as model:
            # Priors
            alpha = pm.Gamma('alpha', alpha=2, beta=1)  # Shape parameter
            beta = pm.Gamma('beta', alpha=2, beta=1)    # Rate parameter
            
            # Likelihood (PyMC uses rate parameterization)
            y_obs = pm.Gamma('y_obs', alpha=alpha, beta=beta, observed=self.data)
            
            # Sample posterior
            trace = pm.sample(n_samples, tune=n_tune, chains=chains, **kwargs)
        
        # Extract results
        posterior = az.convert_to_inference_data(trace)
        summary = az.summary(posterior)
        
        # Convert to scipy parameterization (shape, scale)
        alpha_mean = float(summary.loc['alpha', 'mean'])
        beta_mean = float(summary.loc['beta', 'mean'])
        
        parameters = {
            'a': alpha_mean,  # Shape
            'scale': 1.0 / beta_mean  # Scale = 1/rate
        }
        
        uncertainties = {
            'a': float(summary.loc['alpha', 'sd']),
            'scale': float(summary.loc['beta', 'sd']) / (beta_mean ** 2)  # Delta method
        }
        
        # Log-likelihood
        log_likelihood = np.sum(
            stats.gamma.logpdf(self.data, a=parameters['a'], scale=parameters['scale'])
        )
        
        aic, bic = self._calculate_information_criteria(log_likelihood, 2)
        
        # Convergence diagnostics
        convergence_info = {
            'r_hat': {var: float(summary.loc[var, 'r_hat']) for var in ['alpha', 'beta']},
            'eff_sample_size': {var: float(summary.loc[var, 'ess_bulk']) for var in ['alpha', 'beta']},
            'n_divergences': int(posterior.sample_stats.diverging.sum())
        }
        
        return CalibrationResults(
            distribution_type='gamma',
            method='Bayesian',
            parameters=parameters,
            parameter_uncertainties=uncertainties,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            convergence_info=convergence_info
        )


class GoodnessOfFitTester:
    """Statistical tests for goodness of fit."""
    
    @staticmethod
    def kolmogorov_smirnov_test(data: np.ndarray, 
                               distribution: str, 
                               parameters: Dict[str, float]) -> Tuple[float, float]:
        """Kolmogorov-Smirnov test for goodness of fit."""
        fitter = MLEFitter(data)
        fitter._validate_distribution(distribution)
        
        dist_info = MLEFitter.SUPPORTED_DISTRIBUTIONS[distribution]
        scipy_dist = dist_info['scipy']
        
        # Extract parameters in scipy order
        param_names = dist_info['params']
        params = [parameters[name] for name in param_names if name in parameters]
        
        # KS test
        ks_stat, ks_p = stats.kstest(data, lambda x: scipy_dist.cdf(x, *params))
        
        return ks_stat, ks_p
    
    @staticmethod
    def anderson_darling_test(data: np.ndarray) -> Tuple[float, float]:
        """Anderson-Darling test for normality."""
        # Only implemented for normal distribution in scipy
        ad_stat, critical_values, significance_level = stats.anderson(data, dist='norm')
        
        # Estimate p-value based on critical values
        if ad_stat < critical_values[0]:
            p_value = 0.25
        elif ad_stat < critical_values[1]:
            p_value = 0.10
        elif ad_stat < critical_values[2]:
            p_value = 0.05
        elif ad_stat < critical_values[3]:
            p_value = 0.025
        elif ad_stat < critical_values[4]:
            p_value = 0.01
        else:
            p_value = 0.001
        
        return ad_stat, p_value


# Convenience functions

def fit_distribution_mle(data: np.ndarray, distribution: str) -> CalibrationResults:
    """Convenience function for MLE fitting.
    
    Args:
        data: Observed data
        distribution: Distribution type
        
    Returns:
        CalibrationResults from MLE fitting
    """
    fitter = MLEFitter(data)
    return fitter.fit(distribution)


def fit_distribution_bayesian(data: np.ndarray, 
                             distribution: str,
                             n_samples: int = 2000) -> CalibrationResults:
    """Convenience function for Bayesian fitting.
    
    Args:
        data: Observed data
        distribution: Distribution type
        n_samples: Number of posterior samples
        
    Returns:
        CalibrationResults from Bayesian fitting
    """
    if not BAYESIAN_AVAILABLE:
        raise RuntimeError("PyMC not available for Bayesian fitting")
    
    fitter = BayesianFitter(data)
    return fitter.fit(distribution, n_samples=n_samples)


def compare_distributions(data: np.ndarray, 
                         distributions: List[str]) -> List[CalibrationResults]:
    """Compare multiple distributions using MLE.
    
    Args:
        data: Observed data
        distributions: List of distribution types
        
    Returns:
        List of CalibrationResults sorted by AIC
    """
    results = []
    fitter = MLEFitter(data)
    
    for dist in distributions:
        try:
            result = fitter.fit(dist)
            results.append(result)
        except Exception as e:
            logging.warning(f"Failed to fit {dist}: {e}")
    
    # Sort by AIC (lower is better)
    results.sort(key=lambda r: r.aic)
    
    return results