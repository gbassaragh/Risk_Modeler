"""Sensitivity analysis and tornado diagrams.

Implements regression-based sensitivity analysis and Shapley value attribution.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
import warnings


class SensitivityAnalyzer:
    """Performs sensitivity analysis on simulation results."""
    
    @staticmethod
    def calculate_tornado_diagram(input_variables: Dict[str, np.ndarray],
                                output_variable: np.ndarray,
                                method: str = "spearman",
                                top_n: int = 15) -> Dict[str, Any]:
        """Calculate tornado diagram data for sensitivity analysis.
        
        Args:
            input_variables: Dictionary of input variable arrays
            output_variable: Output variable array
            method: Correlation method ("spearman", "pearson", or "regression")
            top_n: Number of top variables to include
            
        Returns:
            Dictionary with tornado diagram data
        """
        if not input_variables:
            return {'variables': [], 'sensitivities': [], 'method': method}
        
        sensitivities = []
        variable_names = []
        
        for name, values in input_variables.items():
            if len(values) != len(output_variable):
                warnings.warn(f"Variable {name} length mismatch, skipping")
                continue
            
            # Calculate sensitivity measure
            if method == "spearman":
                if np.std(values) > 0 and np.std(output_variable) > 0:
                    sensitivity, _ = stats.spearmanr(values, output_variable)
                    if np.isnan(sensitivity):
                        sensitivity = 0.0
                else:
                    sensitivity = 0.0
                    
            elif method == "pearson":
                if np.std(values) > 0 and np.std(output_variable) > 0:
                    sensitivity, _ = stats.pearsonr(values, output_variable)
                    if np.isnan(sensitivity):
                        sensitivity = 0.0
                else:
                    sensitivity = 0.0
                    
            elif method == "regression":
                sensitivity = SensitivityAnalyzer._calculate_standardized_regression_coefficient(
                    values, output_variable
                )
            else:
                raise ValueError(f"Unknown sensitivity method: {method}")
            
            sensitivities.append(sensitivity)
            variable_names.append(name)
        
        # Sort by absolute sensitivity
        sorted_indices = np.argsort(np.abs(sensitivities))[::-1][:top_n]
        
        tornado_data = {
            'variables': [variable_names[i] for i in sorted_indices],
            'sensitivities': [sensitivities[i] for i in sorted_indices],
            'abs_sensitivities': [abs(sensitivities[i]) for i in sorted_indices],
            'method': method,
            'n_variables': len(input_variables),
            'top_n': min(top_n, len(input_variables))
        }
        
        return tornado_data
    
    @staticmethod
    def _calculate_standardized_regression_coefficient(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate standardized regression coefficient (beta).
        
        Args:
            x: Input variable
            y: Output variable
            
        Returns:
            Standardized regression coefficient
        """
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        
        # Standardize variables
        x_std = (x - np.mean(x)) / np.std(x)
        y_std = (y - np.mean(y)) / np.std(y)
        
        # Simple linear regression coefficient
        covariance = np.mean(x_std * y_std)
        return covariance
    
    @staticmethod
    def calculate_variance_decomposition(input_variables: Dict[str, np.ndarray],
                                       output_variable: np.ndarray) -> Dict[str, float]:
        """Calculate variance decomposition using correlation-based method.
        
        Args:
            input_variables: Dictionary of input variable arrays
            output_variable: Output variable array
            
        Returns:
            Dictionary mapping variable names to variance contributions
        """
        total_variance = np.var(output_variable)
        
        if total_variance == 0:
            return {name: 0.0 for name in input_variables.keys()}
        
        contributions = {}
        
        for name, values in input_variables.items():
            if len(values) != len(output_variable):
                contributions[name] = 0.0
                continue
            
            # Calculate correlation
            if np.std(values) > 0 and np.std(output_variable) > 0:
                correlation, _ = stats.pearsonr(values, output_variable)
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
            
            # Contribution = (correlation * std_input * std_output)^2 / var_output
            contribution = ((correlation * np.std(values) * np.std(output_variable)) ** 2) / total_variance
            contributions[name] = contribution
        
        # Normalize contributions to sum to 1
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            contributions = {name: contrib / total_contribution 
                           for name, contrib in contributions.items()}
        
        return contributions
    
    @staticmethod
    def calculate_sobol_indices(input_variables: Dict[str, np.ndarray],
                              output_variable: np.ndarray,
                              confidence_level: float = 0.95) -> Dict[str, Dict[str, float]]:
        """Calculate first-order Sobol sensitivity indices.
        
        This is a simplified implementation. For precise Sobol indices,
        specialized sampling methods would be required.
        
        Args:
            input_variables: Dictionary of input variable arrays
            output_variable: Output variable array
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with Sobol indices and confidence intervals
        """
        n_samples = len(output_variable)
        total_variance = np.var(output_variable)
        
        if total_variance == 0:
            return {name: {'S1': 0.0, 'confidence_low': 0.0, 'confidence_high': 0.0} 
                   for name in input_variables.keys()}
        
        sobol_indices = {}
        
        for name, values in input_variables.items():
            if len(values) != n_samples:
                sobol_indices[name] = {'S1': 0.0, 'confidence_low': 0.0, 'confidence_high': 0.0}
                continue
            
            # Simplified first-order Sobol index using correlation
            # This is an approximation - true Sobol indices require specific sampling
            if np.std(values) > 0:
                correlation, _ = stats.pearsonr(values, output_variable)
                if np.isnan(correlation):
                    correlation = 0.0
                
                # Approximate S1 as squared correlation
                s1 = correlation ** 2
                
                # Bootstrap confidence interval
                s1_bootstrap = []
                n_bootstrap = 100
                
                for _ in range(n_bootstrap):
                    # Bootstrap sample
                    indices = np.random.choice(n_samples, n_samples, replace=True)
                    boot_x = values[indices]
                    boot_y = output_variable[indices]
                    
                    if np.std(boot_x) > 0 and np.std(boot_y) > 0:
                        boot_corr, _ = stats.pearsonr(boot_x, boot_y)
                        if not np.isnan(boot_corr):
                            s1_bootstrap.append(boot_corr ** 2)
                
                if s1_bootstrap:
                    alpha = 1 - confidence_level
                    confidence_low = np.percentile(s1_bootstrap, 100 * alpha / 2)
                    confidence_high = np.percentile(s1_bootstrap, 100 * (1 - alpha / 2))
                else:
                    confidence_low = s1
                    confidence_high = s1
                
            else:
                s1 = 0.0
                confidence_low = 0.0
                confidence_high = 0.0
            
            sobol_indices[name] = {
                'S1': s1,
                'confidence_low': confidence_low,
                'confidence_high': confidence_high
            }
        
        return sobol_indices


class ShapleyAnalyzer:
    """Implements Shapley value-based attribution for risk contributions."""
    
    @staticmethod
    def calculate_shapley_values(input_variables: Dict[str, np.ndarray],
                               output_variable: np.ndarray,
                               max_coalitions: int = 1000) -> Dict[str, float]:
        """Calculate approximate Shapley values for variable importance.
        
        This is a sampling-based approximation for computational efficiency.
        
        Args:
            input_variables: Dictionary of input variable arrays
            output_variable: Output variable array
            max_coalitions: Maximum number of coalitions to sample
            
        Returns:
            Dictionary mapping variable names to Shapley values
        """
        variable_names = list(input_variables.keys())
        n_variables = len(variable_names)
        
        if n_variables == 0:
            return {}
        
        if n_variables == 1:
            return {variable_names[0]: 1.0}
        
        # Initialize Shapley values
        shapley_values = {name: 0.0 for name in variable_names}
        
        # Sample coalitions
        n_samples = min(max_coalitions, 2 ** n_variables)
        
        for _ in range(n_samples):
            # Random permutation of variables
            permutation = np.random.permutation(variable_names)
            
            # Calculate marginal contributions
            for i, var_name in enumerate(permutation):
                # Coalition without current variable
                coalition_without = permutation[:i]
                # Coalition with current variable
                coalition_with = permutation[:i+1]
                
                # Calculate contribution
                contribution = ShapleyAnalyzer._calculate_coalition_value(
                    coalition_with, input_variables, output_variable
                ) - ShapleyAnalyzer._calculate_coalition_value(
                    coalition_without, input_variables, output_variable
                )
                
                shapley_values[var_name] += contribution
        
        # Average over samples
        for name in shapley_values:
            shapley_values[name] /= n_samples
        
        return shapley_values
    
    @staticmethod
    def _calculate_coalition_value(coalition: List[str],
                                 input_variables: Dict[str, np.ndarray],
                                 output_variable: np.ndarray) -> float:
        """Calculate value of a coalition using R-squared.
        
        Args:
            coalition: List of variable names in coalition
            input_variables: Dictionary of input variable arrays
            output_variable: Output variable array
            
        Returns:
            Coalition value (R-squared)
        """
        if not coalition:
            return 0.0
        
        # Prepare design matrix
        X = np.column_stack([input_variables[name] for name in coalition])
        y = output_variable
        
        if X.shape[1] == 0 or np.var(y) == 0:
            return 0.0
        
        try:
            # Multiple linear regression
            # Using normal equations: beta = (X'X)^(-1) X'y
            XtX = X.T @ X
            
            # Add small regularization for numerical stability
            regularization = 1e-8 * np.eye(XtX.shape[0])
            beta = np.linalg.solve(XtX + regularization, X.T @ y)
            
            # Predictions
            y_pred = X @ beta
            
            # R-squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            if ss_tot == 0:
                return 0.0
            
            r_squared = 1 - (ss_res / ss_tot)
            return max(0.0, r_squared)  # Ensure non-negative
            
        except (np.linalg.LinAlgError, np.linalg.linalg.LinAlgError):
            # Fallback to correlation-based measure
            if len(coalition) == 1:
                var_values = input_variables[coalition[0]]
                if np.std(var_values) > 0 and np.std(y) > 0:
                    corr, _ = stats.pearsonr(var_values, y)
                    if not np.isnan(corr):
                        return corr ** 2
            return 0.0


class RiskContributionAnalyzer:
    """Analyzes individual risk contributions to total uncertainty."""
    
    @staticmethod
    def analyze_risk_contributions(base_costs: np.ndarray,
                                 risk_adjusted_costs: np.ndarray,
                                 individual_risk_impacts: Dict[str, np.ndarray],
                                 risk_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze contribution of each risk to total cost uncertainty.
        
        Args:
            base_costs: Base case costs (before risks)
            risk_adjusted_costs: Final costs after all risks applied
            individual_risk_impacts: Impact arrays for each risk
            risk_metadata: Metadata for each risk
            
        Returns:
            Dictionary of risk contribution metrics
        """
        total_variance = np.var(risk_adjusted_costs)
        
        contributions = {}
        
        for risk_id, impact_values in individual_risk_impacts.items():
            if risk_id not in risk_metadata:
                continue
            
            metadata = risk_metadata[risk_id]
            
            # Calculate various contribution metrics
            contribution_metrics = {
                'variance_contribution': 0.0,
                'correlation_with_total': 0.0,
                'mean_impact': np.mean(impact_values),
                'std_impact': np.std(impact_values),
                'occurrence_rate': 0.0,
                'conditional_mean_impact': 0.0,
                'risk_category': metadata.get('category', 'Unknown'),
                'probability': metadata.get('probability', 0.0),
                'impact_mode': metadata.get('impact_mode', 'multiplicative')
            }
            
            # Correlation with total
            if np.std(impact_values) > 0 and np.std(risk_adjusted_costs) > 0:
                correlation, _ = stats.pearsonr(impact_values, risk_adjusted_costs)
                if not np.isnan(correlation):
                    contribution_metrics['correlation_with_total'] = correlation
            
            # Variance contribution
            if total_variance > 0:
                covariance = np.cov(impact_values, risk_adjusted_costs)[0, 1]
                contribution = (correlation * np.std(impact_values) * np.std(risk_adjusted_costs)) / total_variance
                contribution_metrics['variance_contribution'] = contribution
            
            # Occurrence rate and conditional impact
            if metadata.get('impact_mode') == 'multiplicative':
                # For multiplicative risks, non-occurrence means impact = 1.0
                occurred = impact_values != 1.0
                contribution_metrics['occurrence_rate'] = np.mean(occurred)
                
                if np.any(occurred):
                    contribution_metrics['conditional_mean_impact'] = np.mean(impact_values[occurred])
                
            else:
                # For additive risks, non-occurrence means impact = 0.0
                occurred = impact_values != 0.0
                contribution_metrics['occurrence_rate'] = np.mean(occurred)
                
                if np.any(occurred):
                    contribution_metrics['conditional_mean_impact'] = np.mean(impact_values[occurred])
            
            contributions[risk_id] = contribution_metrics
        
        return contributions
    
    @staticmethod
    def rank_risks_by_contribution(risk_contributions: Dict[str, Dict[str, float]],
                                 ranking_method: str = "variance_contribution",
                                 top_n: int = 10) -> List[Tuple[str, float]]:
        """Rank risks by their contribution to total uncertainty.
        
        Args:
            risk_contributions: Risk contribution metrics
            ranking_method: Method to use for ranking
            top_n: Number of top risks to return
            
        Returns:
            List of (risk_id, contribution_value) tuples, sorted by contribution
        """
        valid_methods = ['variance_contribution', 'correlation_with_total', 'occurrence_rate']
        
        if ranking_method not in valid_methods:
            raise ValueError(f"Ranking method must be one of {valid_methods}")
        
        # Extract values for ranking
        risk_values = []
        for risk_id, metrics in risk_contributions.items():
            value = abs(metrics.get(ranking_method, 0.0))  # Use absolute value for correlation
            risk_values.append((risk_id, value))
        
        # Sort by value (descending)
        risk_values.sort(key=lambda x: x[1], reverse=True)
        
        return risk_values[:top_n]


def create_sensitivity_summary(tornado_data: Dict[str, Any],
                             risk_contributions: Dict[str, Dict[str, float]],
                             top_n: int = 10) -> Dict[str, Any]:
    """Create comprehensive sensitivity analysis summary.
    
    Args:
        tornado_data: Tornado diagram data
        risk_contributions: Risk contribution analysis
        top_n: Number of top items to include
        
    Returns:
        Sensitivity analysis summary
    """
    summary = {
        'tornado_analysis': {
            'top_drivers': list(zip(
                tornado_data.get('variables', [])[:top_n],
                tornado_data.get('sensitivities', [])[:top_n]
            )),
            'method': tornado_data.get('method', 'spearman'),
            'n_variables_analyzed': tornado_data.get('n_variables', 0)
        },
        'risk_ranking': [],
        'key_insights': []
    }
    
    # Risk ranking by variance contribution
    if risk_contributions:
        risk_ranking = RiskContributionAnalyzer.rank_risks_by_contribution(
            risk_contributions, 'variance_contribution', top_n
        )
        
        summary['risk_ranking'] = [
            {
                'risk_id': risk_id,
                'variance_contribution': contribution,
                'metadata': risk_contributions[risk_id]
            }
            for risk_id, contribution in risk_ranking
        ]
    
    # Generate key insights
    if tornado_data.get('variables'):
        top_driver = tornado_data['variables'][0]
        top_sensitivity = tornado_data['sensitivities'][0]
        summary['key_insights'].append(
            f"Top driver: {top_driver} (sensitivity: {top_sensitivity:.3f})"
        )
    
    if summary['risk_ranking']:
        top_risk = summary['risk_ranking'][0]
        summary['key_insights'].append(
            f"Highest risk contributor: {top_risk['risk_id']} "
            f"({top_risk['variance_contribution']:.1%} of variance)"
        )
    
    return summary