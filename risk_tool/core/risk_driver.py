"""Risk modeling components for Monte Carlo simulation.

Implements risk-driver method with occurrence and impact modeling.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from pydantic import BaseModel, validator
from .distributions import DistributionSampler


class RiskItem(BaseModel):
    """Individual risk item specification."""
    id: str
    title: str
    category: str
    probability: float
    impact_mode: str  # "multiplicative" or "additive"
    impact_dist: Dict[str, Any]
    applies_to: Optional[List[str]] = None  # WBS codes
    applies_by_tag: Optional[List[str]] = None  # Tags
    schedule_days_dist: Optional[Dict[str, Any]] = None
    
    @validator('probability')
    def validate_probability(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Probability must be between 0 and 1")
        return v
    
    @validator('impact_mode')
    def validate_impact_mode(cls, v):
        if v not in ["multiplicative", "additive"]:
            raise ValueError("Impact mode must be 'multiplicative' or 'additive'")
        return v


class RiskRegister:
    """Collection of risks with application logic."""
    
    def __init__(self, risks: List[RiskItem]):
        """Initialize risk register.
        
        Args:
            risks: List of risk items
        """
        self.risks = risks
        self.risk_by_id = {risk.id: risk for risk in risks}
    
    def get_applicable_risks(self, 
                           wbs_code: str, 
                           tags: List[str]) -> List[RiskItem]:
        """Get risks applicable to a WBS item.
        
        Args:
            wbs_code: WBS code
            tags: List of tags for the WBS item
            
        Returns:
            List of applicable risks
        """
        applicable = []
        
        for risk in self.risks:
            # Check direct WBS code application
            if risk.applies_to and wbs_code in risk.applies_to:
                applicable.append(risk)
                continue
            
            # Check tag-based application
            if risk.applies_by_tag:
                if any(tag in tags for tag in risk.applies_by_tag):
                    applicable.append(risk)
        
        return applicable
    
    def get_project_level_risks(self) -> List[RiskItem]:
        """Get risks that apply to the entire project.
        
        Returns:
            List of project-level risks
        """
        project_risks = []
        
        for risk in self.risks:
            # Check for project-level indicators
            if risk.applies_to:
                if any(code in ["project", "indirects", "construction_management"] 
                      for code in risk.applies_to):
                    project_risks.append(risk)
        
        return project_risks


class RiskSimulator:
    """Simulates risk occurrence and impact."""
    
    def __init__(self, risk_register: RiskRegister, random_state: np.random.RandomState):
        """Initialize risk simulator.
        
        Args:
            risk_register: Risk register
            random_state: Random number generator
        """
        self.risk_register = risk_register
        self.random_state = random_state
    
    def simulate_risk_occurrences(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Simulate which risks occur in each sample.
        
        Args:
            n_samples: Number of simulation samples
            
        Returns:
            Dictionary mapping risk ID to boolean occurrence array
        """
        occurrences = {}
        
        for risk in self.risk_register.risks:
            # Bernoulli trials for occurrence
            occurrences[risk.id] = self.random_state.binomial(
                1, risk.probability, size=n_samples
            ).astype(bool)
        
        return occurrences
    
    def simulate_risk_impacts(self, 
                            n_samples: int,
                            occurrences: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Simulate impact magnitudes for occurring risks.
        
        Args:
            n_samples: Number of simulation samples
            occurrences: Risk occurrence indicators
            
        Returns:
            Dictionary mapping risk ID to impact arrays
        """
        impacts = {}
        
        for risk in self.risk_register.risks:
            # Sample impact distribution
            all_impacts = DistributionSampler.sample(
                risk.impact_dist, n_samples, self.random_state
            )
            
            # Apply only where risk occurs
            risk_occurs = occurrences[risk.id]
            impacts[risk.id] = np.where(
                risk_occurs,
                all_impacts,
                1.0 if risk.impact_mode == "multiplicative" else 0.0
            )
        
        return impacts
    
    def simulate_schedule_impacts(self, 
                                n_samples: int,
                                occurrences: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Simulate schedule delay impacts for occurring risks.
        
        Args:
            n_samples: Number of simulation samples
            occurrences: Risk occurrence indicators
            
        Returns:
            Dictionary mapping risk ID to schedule delay arrays (in days)
        """
        schedule_impacts = {}
        
        for risk in self.risk_register.risks:
            if risk.schedule_days_dist is None:
                schedule_impacts[risk.id] = np.zeros(n_samples)
                continue
            
            # Sample schedule distribution
            all_delays = DistributionSampler.sample(
                risk.schedule_days_dist, n_samples, self.random_state
            )
            
            # Apply only where risk occurs
            risk_occurs = occurrences[risk.id]
            schedule_impacts[risk.id] = np.where(risk_occurs, all_delays, 0.0)
        
        return schedule_impacts


class RiskAggregator:
    """Aggregates risk impacts across WBS and project levels."""
    
    @staticmethod
    def apply_wbs_risks(base_costs: np.ndarray,
                       applicable_risks: List[RiskItem],
                       risk_impacts: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply risks to WBS-level costs.
        
        Args:
            base_costs: Base cost array for WBS item
            applicable_risks: Risks applicable to this WBS item
            risk_impacts: Simulated risk impacts
            
        Returns:
            Risk-adjusted costs
        """
        adjusted_costs = base_costs.copy()
        
        for risk in applicable_risks:
            impact = risk_impacts[risk.id]
            
            if risk.impact_mode == "multiplicative":
                adjusted_costs *= impact
            elif risk.impact_mode == "additive":
                # For WBS-level additive risks, distribute across base cost
                # This is a modeling choice - could be implemented differently
                adjusted_costs += impact * (base_costs / np.sum(base_costs))
        
        return adjusted_costs
    
    @staticmethod
    def apply_project_risks(total_costs: np.ndarray,
                           project_risks: List[RiskItem],
                           risk_impacts: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply project-level risks to total costs.
        
        Args:
            total_costs: Total project cost array
            project_risks: Project-level risks
            risk_impacts: Simulated risk impacts
            
        Returns:
            Risk-adjusted total costs
        """
        adjusted_costs = total_costs.copy()
        
        for risk in project_risks:
            impact = risk_impacts[risk.id]
            
            if risk.impact_mode == "multiplicative":
                adjusted_costs *= impact
            elif risk.impact_mode == "additive":
                adjusted_costs += impact
        
        return adjusted_costs
    
    @staticmethod
    def aggregate_schedule_impacts(schedule_impacts: Dict[str, np.ndarray]) -> np.ndarray:
        """Aggregate schedule impacts across all risks.
        
        Args:
            schedule_impacts: Dictionary of schedule delays by risk
            
        Returns:
            Total schedule delay array
        """
        if not schedule_impacts:
            return np.array([0.0])
        
        # Sum all schedule delays
        total_delays = np.zeros_like(next(iter(schedule_impacts.values())))
        
        for delay in schedule_impacts.values():
            total_delays += delay
        
        return total_delays


class RiskAnalyzer:
    """Analyzes risk contributions and sensitivities."""
    
    @staticmethod
    def calculate_risk_contributions(base_total: float,
                                   risk_adjusted_total: np.ndarray,
                                   risk_impacts: Dict[str, np.ndarray],
                                   risk_register: RiskRegister) -> Dict[str, Dict[str, float]]:
        """Calculate contribution of each risk to total cost variance.
        
        Args:
            base_total: Base case total cost
            risk_adjusted_total: Risk-adjusted total costs
            risk_impacts: Risk impact simulations
            risk_register: Risk register
            
        Returns:
            Dictionary of risk contributions
        """
        contributions = {}
        
        total_variance = np.var(risk_adjusted_total)
        
        for risk in risk_register.risks:
            impact = risk_impacts[risk.id]
            
            # Calculate correlation with total
            if np.std(impact) > 0 and np.std(risk_adjusted_total) > 0:
                correlation = np.corrcoef(impact, risk_adjusted_total)[0, 1]
            else:
                correlation = 0.0
            
            # Contribution to variance (simplified approach)
            if risk.impact_mode == "multiplicative":
                # For multiplicative risks, impact on relative change
                impact_std = np.std(impact)
                contribution = correlation * impact_std * np.std(risk_adjusted_total)
            else:
                # For additive risks, direct impact
                contribution = correlation * np.std(impact) * np.std(risk_adjusted_total)
            
            # Normalize by total variance
            variance_contribution = (contribution ** 2) / total_variance if total_variance > 0 else 0
            
            contributions[risk.id] = {
                'title': risk.title,
                'category': risk.category,
                'probability': risk.probability,
                'impact_mode': risk.impact_mode,
                'correlation': correlation,
                'variance_contribution': variance_contribution,
                'occurrence_rate': np.mean(impact != (1.0 if risk.impact_mode == "multiplicative" else 0.0)),
                'mean_impact_when_occurs': np.mean(impact[impact != (1.0 if risk.impact_mode == "multiplicative" else 0.0)])
                if np.any(impact != (1.0 if risk.impact_mode == "multiplicative" else 0.0)) else 0.0
            }
        
        return contributions


def validate_risk_register(risks: List[Dict[str, Any]]) -> List[str]:
    """Validate risk register data.
    
    Args:
        risks: List of risk dictionaries
        
    Returns:
        List of validation errors
    """
    errors = []
    
    for i, risk_data in enumerate(risks):
        try:
            risk = RiskItem(**risk_data)
            
            # Additional validations
            if not risk.applies_to and not risk.applies_by_tag:
                errors.append(f"Risk {risk.id}: Must specify either applies_to or applies_by_tag")
            
            # Validate impact distribution
            from .distributions import validate_distribution_config
            try:
                validate_distribution_config(risk.impact_dist)
            except Exception as e:
                errors.append(f"Risk {risk.id}: Invalid impact distribution - {e}")
            
            # Validate schedule distribution if present
            if risk.schedule_days_dist:
                try:
                    validate_distribution_config(risk.schedule_days_dist)
                except Exception as e:
                    errors.append(f"Risk {risk.id}: Invalid schedule distribution - {e}")
        
        except Exception as e:
            errors.append(f"Risk {i}: {e}")
    
    return errors