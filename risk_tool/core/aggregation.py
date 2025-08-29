"""Monte Carlo simulation orchestration and aggregation.

Main simulation engine that coordinates all components.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import time
import warnings

from .sampler import MonteCarloSampler, ConvergenceDiagnostics
from .distributions import DistributionSampler, validate_distribution_config
from .correlation import apply_correlations
from .cost_models import Project, CostSimulator, CostBreakdown, SimulationConfig, calculate_contingency_recommendation
from .risk_driver import RiskRegister, RiskSimulator, RiskAggregator, RiskAnalyzer, RiskItem
from .escalation import EscalationEngine, MarketIndexEngine, ScheduleEscalation, ProductivityFactors
from .schedule_link import ScheduleSimulator, ScheduleCostCalculator, Schedule
from .sensitivity import SensitivityAnalyzer, ShapleyAnalyzer, RiskContributionAnalyzer
from ..reporting.reporting import SimulationResults, ResultsCalculator
from .validation import ValidationEngine


class SimulationEngine:
    """Main Monte Carlo simulation engine."""
    
    def __init__(self, config: SimulationConfig) -> None:
        """Initialize simulation engine.
        
        Args:
            config: Simulation configuration
        """
        self.config: SimulationConfig = config
        self.random_state: np.random.RandomState = np.random.RandomState(config.random_seed)
        
        # Initialize components
        self.sampler: Optional[MonteCarloSampler] = None
        self.cost_simulator: Optional[CostSimulator] = None
        self.risk_simulator: Optional[RiskSimulator] = None
        self.escalation_engine: EscalationEngine = EscalationEngine(self.random_state)
        
        # Results storage
        self.results: Dict[str, Any] = {}
        
    def run_simulation(self, 
                      project: Project, 
                      risks: Optional[List[RiskItem]] = None,
                      correlations: Optional[List[Dict[str, Any]]] = None,
                      schedule: Optional[Schedule] = None) -> SimulationResults:
        """Run complete Monte Carlo simulation.
        
        Args:
            project: Project model
            risks: Optional list of risks
            correlations: Optional correlations
            schedule: Optional schedule model
            
        Returns:
            Simulation results
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_inputs(project, risks, correlations)
            
            # Setup simulation
            self._setup_simulation(project, risks)
            
            # Generate base samples
            base_samples = self._generate_base_samples(project, risks)
            
            # Apply correlations if specified
            if correlations:
                base_samples = self._apply_correlations(base_samples, correlations, project, risks)
            
            # Run cost simulation
            wbs_costs = self._simulate_costs(project, base_samples)
            
            # Run risk simulation
            risk_adjusted_costs = self._simulate_risks(wbs_costs, risks, base_samples)
            
            # Apply escalation
            escalated_costs = self._apply_escalation(risk_adjusted_costs, project)
            
            # Apply schedule effects
            if schedule:
                final_costs = self._apply_schedule_effects(escalated_costs, schedule, project)
            else:
                final_costs = escalated_costs
            
            # Calculate total costs
            total_costs = self._calculate_total_costs(final_costs)
            
            # Analyze results
            results = self._analyze_results(project, total_costs, final_costs, risks)
            
            # Add performance metrics
            results['simulation_time'] = time.time() - start_time
            results['iterations_per_second'] = self.config.iterations / results['simulation_time']
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Simulation failed: {e}")
    
    def _validate_inputs(self, 
                        project: Project, 
                        risks: Optional[List[RiskItem]], 
                        correlations: Optional[List[Dict[str, Any]]]):
        """Validate all input data."""
        # Validate project
        for item in project.wbs:
            if item.dist_quantity:
                validate_distribution_config(item.dist_quantity)
            if item.dist_unit_cost:
                validate_distribution_config(item.dist_unit_cost)
        
        # Validate risks
        if risks:
            for risk in risks:
                validate_distribution_config(risk.impact_dist)
                if risk.schedule_days_dist:
                    validate_distribution_config(risk.schedule_days_dist)
    
    def _setup_simulation(self, project: Project, risks: Optional[List[RiskItem]]):
        """Setup simulation components."""
        # Count variables for sampling
        n_variables = 0
        
        # WBS variables (quantity and unit cost distributions)
        for item in project.wbs:
            if item.dist_quantity:
                n_variables += 1
            if item.dist_unit_cost:
                n_variables += 1
        
        # Risk variables
        if risks:
            for risk in risks:
                n_variables += 1  # Impact
                if risk.schedule_days_dist:
                    n_variables += 1  # Schedule
        
        # Escalation variables
        if project.escalation and project.escalation.annual_rate_dist:
            n_variables += 1
        
        # Initialize sampler
        self.sampler = MonteCarloSampler(
            n_samples=self.config.iterations,
            n_dimensions=n_variables,
            method=self.config.sampling,
            random_seed=self.config.random_seed
        )
        
        # Initialize component simulators
        self.cost_simulator = CostSimulator(project, self.random_state)
        
        if risks:
            risk_register = RiskRegister(risks)
            self.risk_simulator = RiskSimulator(risk_register, self.random_state)
    
    def _generate_base_samples(self, 
                              project: Project, 
                              risks: Optional[List[RiskItem]]) -> Dict[str, np.ndarray]:
        """Generate base uniform samples for all variables."""
        base_uniform = self.sampler.get_samples()
        variable_samples = {}
        sample_index = 0
        
        # Sample WBS variables
        for item in project.wbs:
            if item.dist_quantity:
                uniform_sample = base_uniform[:, sample_index]
                # Convert uniform to distribution samples
                dist_samples = self._convert_uniform_to_distribution(
                    uniform_sample, item.dist_quantity
                )
                variable_samples[f'qty_{item.code}'] = dist_samples
                sample_index += 1
            
            if item.dist_unit_cost:
                uniform_sample = base_uniform[:, sample_index]
                dist_samples = self._convert_uniform_to_distribution(
                    uniform_sample, item.dist_unit_cost
                )
                variable_samples[f'cost_{item.code}'] = dist_samples
                sample_index += 1
        
        # Sample risk variables
        if risks:
            for risk in risks:
                # Risk impact
                uniform_sample = base_uniform[:, sample_index]
                impact_samples = self._convert_uniform_to_distribution(
                    uniform_sample, risk.impact_dist
                )
                variable_samples[f'risk_impact_{risk.id}'] = impact_samples
                sample_index += 1
                
                # Risk schedule impact
                if risk.schedule_days_dist:
                    uniform_sample = base_uniform[:, sample_index]
                    schedule_samples = self._convert_uniform_to_distribution(
                        uniform_sample, risk.schedule_days_dist
                    )
                    variable_samples[f'risk_schedule_{risk.id}'] = schedule_samples
                    sample_index += 1
        
        # Sample escalation
        if project.escalation and project.escalation.annual_rate_dist:
            uniform_sample = base_uniform[:, sample_index]
            escalation_samples = self._convert_uniform_to_distribution(
                uniform_sample, project.escalation.annual_rate_dist
            )
            variable_samples['escalation_rate'] = escalation_samples
            sample_index += 1
        
        return variable_samples
    
    def _convert_uniform_to_distribution(self, 
                                       uniform_samples: np.ndarray, 
                                       dist_config: Dict[str, Any]) -> np.ndarray:
        """Convert uniform samples to distribution samples."""
        # Use inverse CDF method for better correlation preservation
        from scipy import stats
        
        dist_type = dist_config.get('type', '').lower()
        
        if dist_type == 'triangular':
            # Triangular distribution
            low, mode, high = dist_config['low'], dist_config['mode'], dist_config['high']
            c = (mode - low) / (high - low)
            return stats.triang.ppf(uniform_samples, c, loc=low, scale=high - low)
        
        elif dist_type == 'normal':
            mean, std = dist_config['mean'], dist_config['stdev']
            samples = stats.norm.ppf(uniform_samples, loc=mean, scale=std)
            
            # Apply truncation if specified
            if 'truncate_low' in dist_config:
                samples = np.maximum(samples, dist_config['truncate_low'])
            if 'truncate_high' in dist_config:
                samples = np.minimum(samples, dist_config['truncate_high'])
            
            return samples
        
        elif dist_type == 'lognormal':
            mean, sigma = dist_config['mean'], dist_config['sigma']
            return stats.lognorm.ppf(uniform_samples, s=sigma, scale=mean)
        
        else:
            # Fallback to direct sampling
            return DistributionSampler.sample(dist_config, len(uniform_samples), self.random_state)
    
    def _apply_correlations(self, 
                           base_samples: Dict[str, np.ndarray],
                           correlations: List[Dict[str, Any]],
                           project: Project,
                           risks: Optional[List[RiskItem]]) -> Dict[str, np.ndarray]:
        """Apply correlations to samples."""
        if not correlations:
            return base_samples
        
        # Convert samples to array format
        variable_names = list(base_samples.keys())
        sample_array = np.column_stack([base_samples[name] for name in variable_names])
        
        # Apply correlations
        correlated_samples = apply_correlations(
            sample_array, correlations, variable_names, method="iman_conover"
        )
        
        # Convert back to dictionary
        correlated_dict = {}
        for i, name in enumerate(variable_names):
            correlated_dict[name] = correlated_samples[:, i]
        
        return correlated_dict
    
    def _simulate_costs(self, 
                       project: Project, 
                       samples: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Simulate WBS-level costs."""
        wbs_costs = {}
        
        for item in project.wbs:
            # Get quantity samples
            if item.dist_quantity and f'qty_{item.code}' in samples:
                quantities = samples[f'qty_{item.code}']
            else:
                quantities = np.full(self.config.iterations, item.quantity)
            
            # Get unit cost samples
            if item.dist_unit_cost and f'cost_{item.code}' in samples:
                unit_costs = samples[f'cost_{item.code}']
            else:
                unit_costs = np.full(self.config.iterations, item.unit_cost)
            
            # Ensure non-negative values
            quantities = np.maximum(quantities, 0)
            unit_costs = np.maximum(unit_costs, 0)
            
            # Calculate direct costs
            direct_costs = quantities * unit_costs
            
            # Apply indirect factors
            total_costs = direct_costs * (1 + item.indirect_factor)
            
            wbs_costs[item.code] = total_costs
        
        return wbs_costs
    
    def _simulate_risks(self, 
                       wbs_costs: Dict[str, np.ndarray], 
                       risks: Optional[List[RiskItem]],
                       samples: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply risk impacts to costs."""
        if not risks:
            return wbs_costs
        
        risk_adjusted_costs = {code: costs.copy() for code, costs in wbs_costs.items()}
        project_level_adjustments = np.zeros(self.config.iterations)
        
        # Process each risk
        for risk in risks:
            # Get risk occurrence
            risk_occurs = self.random_state.binomial(1, risk.probability, self.config.iterations).astype(bool)
            
            # Get impact samples
            if f'risk_impact_{risk.id}' in samples:
                all_impacts = samples[f'risk_impact_{risk.id}']
            else:
                all_impacts = DistributionSampler.sample(
                    risk.impact_dist, self.config.iterations, self.random_state
                )
            
            # Apply impacts only where risk occurs
            if risk.impact_mode == "multiplicative":
                impacts = np.where(risk_occurs, all_impacts, 1.0)
            else:  # additive
                impacts = np.where(risk_occurs, all_impacts, 0.0)
            
            # Apply to relevant WBS items
            if risk.applies_to:
                for wbs_code in risk.applies_to:
                    if wbs_code in risk_adjusted_costs:
                        if risk.impact_mode == "multiplicative":
                            risk_adjusted_costs[wbs_code] *= impacts
                        else:
                            risk_adjusted_costs[wbs_code] += impacts
            
            # Apply by tags
            if risk.applies_by_tag:
                # Find WBS items with matching tags (simplified - would need project reference)
                # For now, apply to project level
                if risk.impact_mode == "multiplicative":
                    # Apply to all WBS items (simplified)
                    for code in risk_adjusted_costs:
                        risk_adjusted_costs[code] *= impacts
                else:
                    project_level_adjustments += impacts
        
        # Add project-level additive adjustments
        if np.any(project_level_adjustments != 0):
            # Distribute across WBS items proportionally
            for code in risk_adjusted_costs:
                base_proportion = np.mean(wbs_costs[code]) / sum(np.mean(costs) for costs in wbs_costs.values())
                risk_adjusted_costs[code] += project_level_adjustments * base_proportion
        
        return risk_adjusted_costs
    
    def _apply_escalation(self, 
                         costs: Dict[str, np.ndarray], 
                         project: Project) -> Dict[str, np.ndarray]:
        """Apply escalation to costs."""
        if not project.escalation or not project.escalation.annual_rate_dist:
            return costs
        
        # Get escalation factors
        if project.escalation.target_year:
            escalation_factors = self.escalation_engine.calculate_escalation_factors(
                project.escalation.base_year,
                project.escalation.target_year,
                project.escalation.annual_rate_dist,
                self.config.iterations
            )
        else:
            escalation_factors = np.ones(self.config.iterations)
        
        # Apply to all costs
        escalated_costs = {}
        for code, cost_array in costs.items():
            escalated_costs[code] = cost_array * escalation_factors
        
        return escalated_costs
    
    def _apply_schedule_effects(self, 
                               costs: Dict[str, np.ndarray],
                               schedule: Schedule,
                               project: Project) -> Dict[str, np.ndarray]:
        """Apply schedule-driven cost effects."""
        # Simulate schedule
        schedule_simulator = ScheduleSimulator(schedule, self.random_state)
        activity_durations = schedule_simulator.simulate_durations(self.config.iterations)
        project_durations, _ = schedule_simulator.calculate_project_duration(activity_durations)
        
        # Calculate schedule-driven costs
        schedule_costs = ScheduleCostCalculator.calculate_indirect_costs(
            project_durations, project.indirects_per_day
        )
        
        # Add to total costs
        schedule_adjusted_costs = costs.copy()
        
        # Add as new WBS item or distribute
        if 'schedule_indirects' not in schedule_adjusted_costs:
            schedule_adjusted_costs['schedule_indirects'] = schedule_costs
        else:
            schedule_adjusted_costs['schedule_indirects'] += schedule_costs
        
        return schedule_adjusted_costs
    
    def _calculate_total_costs(self, costs: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate total project costs."""
        total = np.zeros(self.config.iterations)
        
        for cost_array in costs.values():
            total += cost_array
        
        return total
    
    def _analyze_results(self, 
                        project: Project, 
                        total_costs: np.ndarray,
                        wbs_costs: Dict[str, np.ndarray],
                        risks: Optional[List[RiskItem]]) -> SimulationResults:
        """Analyze simulation results and create results object."""
        # Basic statistics
        percentiles = ResultsCalculator.calculate_percentiles(
            total_costs, self.config.outputs.get('percentiles', [10, 50, 80, 90, 95])
        )
        
        # WBS breakdown
        wbs_statistics = CostBreakdown.calculate_wbs_statistics(wbs_costs)
        
        # Contingency recommendations
        base_cost = project.calculate_base_total()
        contingency = calculate_contingency_recommendation(total_costs, base_cost, 0.8)
        
        # Risk analysis (if risks provided)
        risk_contributions = None
        if risks:
            # This would require storing individual risk impacts - simplified for now
            risk_contributions = {}
        
        # Sensitivity analysis
        sensitivity_analysis = None
        # This would require variable tracking - simplified for now
        
        # Create results object
        results = SimulationResults(
            project_id=project.id,
            currency=project.currency,
            n_samples=self.config.iterations,
            random_seed=self.config.random_seed,
            base_cost=base_cost,
            total_costs=total_costs.tolist(),
            percentiles=percentiles,
            wbs_statistics=wbs_statistics,
            risk_contributions=risk_contributions,
            sensitivity_analysis=sensitivity_analysis,
            contingency_recommendations=contingency
        )
        
        return results


def run_simulation(project_data: Dict[str, Any],
                  risks_data: Optional[List[Dict[str, Any]]] = None,
                  config_data: Optional[Dict[str, Any]] = None) -> SimulationResults:
    """Main entry point for running simulations.
    
    Args:
        project_data: Project dictionary
        risks_data: Optional risks list
        config_data: Optional configuration dictionary
        
    Returns:
        Simulation results
    """
    # Parse inputs
    project = Project(**project_data)
    
    risks = None
    if risks_data:
        risks = [RiskItem(**risk) for risk in risks_data]
    
    config = SimulationConfig(**(config_data or {}))
    
    # Run simulation
    engine = SimulationEngine(config)
    return engine.run_simulation(project, risks)