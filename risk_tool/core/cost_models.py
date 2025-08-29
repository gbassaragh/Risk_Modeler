"""Cost modeling with WBS structure and line-item uncertainty.

Implements project cost models with work breakdown structure.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, validator
from datetime import datetime, date
from .distributions import DistributionSampler


class WBSItem(BaseModel):
    """Work Breakdown Structure item."""
    code: str
    name: str
    quantity: float
    uom: str  # Unit of measure
    unit_cost: float
    dist_quantity: Optional[Dict[str, Any]] = None
    dist_unit_cost: Optional[Dict[str, Any]] = None
    tags: List[str] = []
    indirect_factor: float = 0.0
    
    @validator('quantity')
    def validate_quantity(cls, v):
        if v < 0:
            raise ValueError("Quantity must be non-negative")
        return v
    
    @validator('unit_cost')
    def validate_unit_cost(cls, v):
        if v < 0:
            raise ValueError("Unit cost must be non-negative")
        return v
    
    @validator('indirect_factor')
    def validate_indirect_factor(cls, v):
        if v < 0:
            raise ValueError("Indirect factor must be non-negative")
        return v
    
    def calculate_base_cost(self) -> float:
        """Calculate base cost for this WBS item."""
        direct_cost = self.quantity * self.unit_cost
        total_cost = direct_cost * (1 + self.indirect_factor)
        return total_cost


class EscalationConfig(BaseModel):
    """Escalation configuration."""
    annual_rate_dist: Optional[Dict[str, Any]] = None
    base_year: str = "2025"
    target_year: Optional[str] = None
    
    def get_escalation_years(self) -> float:
        """Calculate years between base and target."""
        if not self.target_year:
            return 0.0
        
        base = datetime.strptime(self.base_year, "%Y")
        target = datetime.strptime(self.target_year, "%Y")
        return (target - base).days / 365.25


class Project(BaseModel):
    """Project model with WBS and configuration."""
    id: str
    type: str  # "TransmissionLine", "Substation", "Hybrid"
    currency: str = "USD"
    base_date: str
    region: str
    wbs: List[WBSItem]
    indirects_per_day: float = 0.0
    escalation: Optional[EscalationConfig] = None
    
    @validator('type')
    def validate_type(cls, v):
        valid_types = ["TransmissionLine", "Substation", "Hybrid"]
        if v not in valid_types:
            raise ValueError(f"Type must be one of {valid_types}")
        return v
    
    @validator('wbs')
    def validate_wbs(cls, v):
        if not v:
            raise ValueError("WBS cannot be empty")
        
        # Check for duplicate codes
        codes = [item.code for item in v]
        if len(codes) != len(set(codes)):
            raise ValueError("WBS codes must be unique")
        
        return v
    
    def calculate_base_total(self) -> float:
        """Calculate total base cost."""
        return sum(item.calculate_base_cost() for item in self.wbs)
    
    def get_wbs_by_code(self, code: str) -> Optional[WBSItem]:
        """Get WBS item by code."""
        for item in self.wbs:
            if item.code == code:
                return item
        return None
    
    def get_wbs_by_tags(self, tags: List[str]) -> List[WBSItem]:
        """Get WBS items that have any of the specified tags."""
        result = []
        for item in self.wbs:
            if any(tag in item.tags for tag in tags):
                result.append(item)
        return result


class CostSimulator:
    """Simulates costs with line-item uncertainty."""
    
    def __init__(self, project: Project, random_state: np.random.RandomState):
        """Initialize cost simulator.
        
        Args:
            project: Project model
            random_state: Random number generator
        """
        self.project = project
        self.random_state = random_state
    
    def simulate_wbs_costs(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Simulate costs for all WBS items.
        
        Args:
            n_samples: Number of simulation samples
            
        Returns:
            Dictionary mapping WBS code to cost arrays
        """
        wbs_costs = {}
        
        for item in self.project.wbs:
            # Sample quantities
            if item.dist_quantity:
                quantities = DistributionSampler.sample(
                    item.dist_quantity, n_samples, self.random_state
                )
                # Ensure non-negative
                quantities = np.maximum(quantities, 0)
            else:
                quantities = np.full(n_samples, item.quantity)
            
            # Sample unit costs
            if item.dist_unit_cost:
                unit_costs = DistributionSampler.sample(
                    item.dist_unit_cost, n_samples, self.random_state
                )
                # Ensure non-negative
                unit_costs = np.maximum(unit_costs, 0)
            else:
                unit_costs = np.full(n_samples, item.unit_cost)
            
            # Calculate direct costs
            direct_costs = quantities * unit_costs
            
            # Apply indirect factors
            total_costs = direct_costs * (1 + item.indirect_factor)
            
            wbs_costs[item.code] = total_costs
        
        return wbs_costs
    
    def calculate_total_costs(self, wbs_costs: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate total project costs.
        
        Args:
            wbs_costs: WBS-level costs
            
        Returns:
            Total cost array
        """
        total = np.zeros(len(next(iter(wbs_costs.values()))))
        
        for costs in wbs_costs.values():
            total += costs
        
        return total


class CostBreakdown:
    """Cost breakdown analysis."""
    
    @staticmethod
    def calculate_wbs_statistics(wbs_costs: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each WBS item.
        
        Args:
            wbs_costs: WBS-level costs
            
        Returns:
            Dictionary of statistics by WBS code
        """
        stats = {}
        
        for code, costs in wbs_costs.items():
            stats[code] = {
                'mean': np.mean(costs),
                'std': np.std(costs),
                'min': np.min(costs),
                'max': np.max(costs),
                'p10': np.percentile(costs, 10),
                'p50': np.percentile(costs, 50),
                'p80': np.percentile(costs, 80),
                'p90': np.percentile(costs, 90),
                'p95': np.percentile(costs, 95),
                'cv': np.std(costs) / np.mean(costs) if np.mean(costs) > 0 else 0
            }
        
        return stats
    
    @staticmethod
    def calculate_category_breakdown(wbs_costs: Dict[str, np.ndarray],
                                   project: Project) -> Dict[str, np.ndarray]:
        """Break down costs by category (based on tags).
        
        Args:
            wbs_costs: WBS-level costs
            project: Project model
            
        Returns:
            Dictionary of category costs
        """
        # Collect all unique tags
        all_tags = set()
        for item in project.wbs:
            all_tags.update(item.tags)
        
        category_costs = {}
        
        for tag in all_tags:
            tag_total = None
            
            for item in project.wbs:
                if tag in item.tags:
                    costs = wbs_costs[item.code]
                    if tag_total is None:
                        tag_total = costs.copy()
                    else:
                        tag_total += costs
            
            if tag_total is not None:
                category_costs[tag] = tag_total
        
        return category_costs
    
    @staticmethod
    def calculate_contribution_to_variance(wbs_costs: Dict[str, np.ndarray],
                                         total_costs: np.ndarray) -> Dict[str, float]:
        """Calculate each WBS item's contribution to total variance.
        
        Args:
            wbs_costs: WBS-level costs
            total_costs: Total cost array
            
        Returns:
            Dictionary of variance contributions by WBS code
        """
        total_var = np.var(total_costs)
        
        if total_var == 0:
            return {code: 0.0 for code in wbs_costs.keys()}
        
        contributions = {}
        
        for code, costs in wbs_costs.items():
            # Correlation with total
            if np.std(costs) > 0:
                corr = np.corrcoef(costs, total_costs)[0, 1]
                # Contribution = correlation * std_individual * std_total / var_total
                contribution = (corr * np.std(costs) * np.std(total_costs)) / total_var
            else:
                contribution = 0.0
            
            contributions[code] = contribution
        
        return contributions


class SimulationConfig(BaseModel):
    """Simulation configuration."""
    iterations: int = 50000
    sampling: str = "LHS"  # "LHS" or "MC"
    random_seed: Optional[int] = 20250829
    currency: str = "USD"
    outputs: Dict[str, Any] = {
        "percentiles": [10, 50, 80, 90, 95],
        "charts": ["histogram", "cdf_curve", "tornado", "contribution_pareto"],
        "export_formats": ["csv", "xlsx", "json", "pdf"]
    }
    validation: Dict[str, bool] = {
        "fail_on_missing_wbs": True,
        "warn_on_unmapped_risk": True
    }
    performance: Dict[str, Any] = {
        "parallel": "auto",
        "num_threads": -1
    }
    
    @validator('iterations')
    def validate_iterations(cls, v):
        if v <= 0:
            raise ValueError("Iterations must be positive")
        if v > 1000000:
            raise ValueError("Iterations cannot exceed 1,000,000")
        return v
    
    @validator('sampling')
    def validate_sampling(cls, v):
        if v not in ["LHS", "MC"]:
            raise ValueError("Sampling must be 'LHS' or 'MC'")
        return v


def validate_project_data(project_data: Dict[str, Any]) -> List[str]:
    """Validate project data structure.
    
    Args:
        project_data: Project dictionary
        
    Returns:
        List of validation errors
    """
    errors = []
    
    try:
        project = Project(**project_data)
        
        # Additional validations
        for item in project.wbs:
            # Validate distributions if present
            if item.dist_quantity:
                try:
                    from .distributions import validate_distribution_config
                    validate_distribution_config(item.dist_quantity)
                except Exception as e:
                    errors.append(f"WBS {item.code}: Invalid quantity distribution - {e}")
            
            if item.dist_unit_cost:
                try:
                    from .distributions import validate_distribution_config
                    validate_distribution_config(item.dist_unit_cost)
                except Exception as e:
                    errors.append(f"WBS {item.code}: Invalid unit cost distribution - {e}")
        
        # Validate escalation if present
        if project.escalation and project.escalation.annual_rate_dist:
            try:
                from .distributions import validate_distribution_config
                validate_distribution_config(project.escalation.annual_rate_dist)
            except Exception as e:
                errors.append(f"Escalation: Invalid rate distribution - {e}")
        
    except Exception as e:
        errors.append(f"Project validation: {e}")
    
    return errors


def calculate_contingency_recommendation(total_costs: np.ndarray, 
                                       base_cost: float,
                                       confidence_level: float = 0.8) -> Dict[str, float]:
    """Calculate contingency recommendations.
    
    Args:
        total_costs: Simulated total costs
        base_cost: Base case cost
        confidence_level: Confidence level for contingency
        
    Returns:
        Dictionary with contingency recommendations
    """
    percentile = confidence_level * 100
    target_cost = np.percentile(total_costs, percentile)
    
    # Management reserve (additional buffer)
    p95_cost = np.percentile(total_costs, 95)
    
    contingency = target_cost - base_cost
    management_reserve = p95_cost - target_cost
    
    return {
        'base_cost': base_cost,
        'target_cost': target_cost,
        'p95_cost': p95_cost,
        'contingency': max(0, contingency),
        'contingency_percent': (contingency / base_cost * 100) if base_cost > 0 else 0,
        'management_reserve': max(0, management_reserve),
        'management_reserve_percent': (management_reserve / base_cost * 100) if base_cost > 0 else 0,
        'total_buffer': max(0, contingency + management_reserve),
        'total_buffer_percent': ((contingency + management_reserve) / base_cost * 100) if base_cost > 0 else 0,
        'confidence_level': confidence_level
    }