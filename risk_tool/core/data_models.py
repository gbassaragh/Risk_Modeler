"""Pydantic data models for risk modeling tool.

Enhanced models supporting new specification requirements:
- Tags and UoM for WBS items
- Conditional logic for risks
- Latent factors and advanced correlation
- Multi-currency support
- Two-layer uncertainty

This module provides comprehensive type-safe data models for risk analysis,
cost estimation, and Monte Carlo simulation configuration.
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Dict, List, Optional, Union, Any, Literal, Tuple
from enum import Enum
from datetime import date
import numpy as np


class ProjectType(str, Enum):
    """Project type enumeration."""

    TRANSMISSION_LINE = "TransmissionLine"
    SUBSTATION = "Substation"
    HYBRID = "Hybrid"


class AACEClass(str, Enum):
    """AACE classification for estimate accuracy."""

    CLASS_1 = "Class 1"  # -5% to +15%
    CLASS_2 = "Class 2"  # -10% to +25%
    CLASS_3 = "Class 3"  # -20% to +30%
    CLASS_4 = "Class 4"  # -30% to +50%
    CLASS_5 = "Class 5"  # -50% to +100%


class Currency(str, Enum):
    """Supported currencies."""

    USD = "USD"
    CAD = "CAD"
    EUR = "EUR"


class UnitOfMeasure(str, Enum):
    """Standard units of measure."""

    MILE = "mile"
    KILOMETER = "km"
    MILE_CONDUCTOR = "mile-conductor"
    KM_CONDUCTOR = "km-conductor"
    MVA = "mva"
    LUMP_SUM = "lump_sum"
    EACH = "each"
    ACRE = "acre"
    CUBIC_YARD = "cubic_yard"
    TON = "ton"


class DistributionType(str, Enum):
    """Supported distribution types."""

    TRIANGULAR = "triangular"
    PERT = "pert"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    DISCRETE = "discrete"
    MIXTURE = "mixture"
    EVT = "evt"
    KDE = "kde"
    EMPIRICAL = "empirical"


# Distribution Configurations
class BaseDistribution(BaseModel):
    """Base distribution configuration."""

    type: DistributionType


class TriangularDistribution(BaseDistribution):
    """Triangular distribution."""

    type: Literal[DistributionType.TRIANGULAR] = DistributionType.TRIANGULAR
    low: float = Field(..., description="Minimum value")
    mode: float = Field(..., description="Most likely value")
    high: float = Field(..., description="Maximum value")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v, info):
        if info.data and "low" in info.data and "high" in info.data:
            low, high = info.data["low"], info.data["high"]
            if not (low <= v <= high):
                raise ValueError(f"Mode {v} must be between low {low} and high {high}")
        return v


class PERTDistribution(BaseDistribution):
    """PERT (Beta) distribution."""

    type: Literal[DistributionType.PERT] = DistributionType.PERT
    min: float = Field(..., description="Minimum value")
    most_likely: float = Field(..., description="Most likely value")
    max: float = Field(..., description="Maximum value")
    lambda_: float = Field(4.0, description="Shape parameter", alias="lambda")

    @field_validator("most_likely")
    @classmethod
    def validate_most_likely(cls, v, info):
        if info.data and "min" in info.data and "max" in info.data:
            min_val, max_val = info.data["min"], info.data["max"]
            if not (min_val <= v <= max_val):
                raise ValueError(
                    f"Most likely {v} must be between min {min_val} and max {max_val}"
                )
        return v


class NormalDistribution(BaseDistribution):
    """Normal distribution with optional truncation."""

    type: Literal[DistributionType.NORMAL] = DistributionType.NORMAL
    mean: float = Field(..., description="Mean value")
    stdev: float = Field(..., gt=0, description="Standard deviation")
    truncate_low: Optional[float] = Field(None, description="Lower truncation")
    truncate_high: Optional[float] = Field(None, description="Upper truncation")


class LogNormalDistribution(BaseDistribution):
    """Log-normal distribution."""

    type: Literal[DistributionType.LOGNORMAL] = DistributionType.LOGNORMAL
    mean: float = Field(..., description="Mean of ln(X)")
    sigma: float = Field(..., gt=0, description="Std dev of ln(X)")


class UniformDistribution(BaseDistribution):
    """Uniform distribution."""

    type: Literal[DistributionType.UNIFORM] = DistributionType.UNIFORM
    low: float = Field(..., description="Lower bound")
    high: float = Field(..., description="Upper bound")

    @field_validator("high")
    @classmethod
    def validate_bounds(cls, v, info):
        if info.data and "low" in info.data and v <= info.data["low"]:
            raise ValueError(f"High {v} must be greater than low {info.data['low']}")
        return v


class DiscreteDistribution(BaseDistribution):
    """Discrete distribution with PMF."""

    type: Literal[DistributionType.DISCRETE] = DistributionType.DISCRETE
    pmf: List[List[Union[float, str]]] = Field(
        ..., description="[value, probability] pairs"
    )

    @field_validator("pmf")
    @classmethod
    def validate_pmf(cls, v):
        if not v:
            raise ValueError("PMF cannot be empty")

        total_prob = 0.0
        for item in v:
            if len(item) != 2:
                raise ValueError("Each PMF item must be [value, probability]")
            prob = float(item[1])
            if prob < 0 or prob > 1:
                raise ValueError("Probabilities must be between 0 and 1")
            total_prob += prob

        if abs(total_prob - 1.0) > 1e-6:
            raise ValueError(f"Probabilities must sum to 1.0, got {total_prob}")

        return v


class MixtureComponent(BaseModel):
    """Component of a mixture distribution."""

    weight: float = Field(..., ge=0, le=1, description="Mixture weight")
    distribution: Union[
        TriangularDistribution,
        PERTDistribution,
        NormalDistribution,
        LogNormalDistribution,
        UniformDistribution,
        DiscreteDistribution,
    ] = Field(..., description="Component distribution")


class MixtureDistribution(BaseDistribution):
    """Mixture distribution."""

    type: Literal[DistributionType.MIXTURE] = DistributionType.MIXTURE
    components: List[MixtureComponent] = Field(..., description="Mixture components")

    @field_validator("components")
    @classmethod
    def validate_weights(cls, v):
        total_weight = sum(comp.weight for comp in v)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Component weights must sum to 1.0, got {total_weight}")
        return v


# Union type for all distributions
Distribution = Union[
    TriangularDistribution,
    PERTDistribution,
    NormalDistribution,
    LogNormalDistribution,
    UniformDistribution,
    DiscreteDistribution,
    MixtureDistribution,
]


class WBSItem(BaseModel):
    """Enhanced WBS item with tags and advanced features."""

    code: str = Field(..., description="WBS code (e.g., '1.1')")
    name: str = Field(..., description="Descriptive name")
    quantity: float = Field(..., gt=0, description="Quantity")
    uom: UnitOfMeasure = Field(..., description="Unit of measure")
    unit_cost: float = Field(..., gt=0, description="Unit cost")
    tags: List[str] = Field(default_factory=list, description="Classification tags")
    indirect_factor: float = Field(0.0, ge=0, description="Indirect cost factor")
    dist_quantity: Optional[Distribution] = Field(
        None, description="Quantity uncertainty"
    )
    dist_unit_cost: Optional[Distribution] = Field(
        None, description="Unit cost uncertainty"
    )

    @property
    def base_cost(self) -> float:
        """Calculate base cost."""
        return self.quantity * self.unit_cost * (1 + self.indirect_factor)


class EscalationMethod(str, Enum):
    """Escalation calculation methods."""

    STOCHASTIC_RATE = "stochastic_rate"
    INDEX_SERIES = "index_series"
    FIXED_RATE = "fixed_rate"


class EscalationConfig(BaseModel):
    """Escalation configuration."""

    method: EscalationMethod = EscalationMethod.FIXED_RATE
    annual_rate_dist: Optional[Distribution] = None
    index_map: Dict[str, str] = Field(
        default_factory=dict, description="Tag to index file mapping"
    )
    base_year: int = Field(2025, description="Base year for escalation")


class ProjectInfo(BaseModel):
    """Enhanced project information."""

    id: str = Field(..., description="Unique project identifier")
    type: ProjectType = Field(..., description="Project type")
    currency: Currency = Field(Currency.USD, description="Project currency")
    base_date: date = Field(..., description="Base date for costs")
    region: str = Field(..., description="Geographic region")
    aace_class: AACEClass = Field(AACEClass.CLASS_3, description="AACE estimate class")
    indirects_per_day: float = Field(0.0, ge=0, description="Daily indirect costs")

    # Project-specific fields
    name: Optional[str] = None
    voltage: Optional[str] = None
    length: Optional[float] = Field(None, gt=0, description="Length in miles/km")
    capacity: Optional[float] = Field(None, gt=0, description="Capacity in MVA")
    terrain: Optional[str] = None
    voltage_levels: Optional[int] = Field(
        None, gt=0, description="Number of voltage levels"
    )


class Project(BaseModel):
    """Enhanced project model."""

    project: ProjectInfo = Field(..., description="Project information")
    wbs: List[WBSItem] = Field(..., description="WBS items")
    escalation: EscalationConfig = Field(
        default_factory=EscalationConfig, description="Escalation config"
    )


class RiskCategory(str, Enum):
    """Risk categories."""

    SCHEDULE = "schedule"
    TECHNICAL = "technical"
    REGULATORY = "regulatory"
    MARKET = "market"
    ENVIRONMENTAL = "environmental"
    CONSTRUCTION = "construction"
    SUPPLY_CHAIN = "supply_chain"
    OPERATIONAL = "operational"


class ImpactMode(str, Enum):
    """Risk impact modes."""

    MULTIPLICATIVE = "multiplicative"  # cost *= impact
    ADDITIVE = "additive"  # cost += impact


class RiskItem(BaseModel):
    """Enhanced risk item with conditional logic."""

    id: str = Field(..., description="Unique risk identifier")
    title: str = Field(..., description="Risk title")
    category: RiskCategory = Field(..., description="Risk category")
    probability: float = Field(..., ge=0, le=1, description="Occurrence probability")
    impact_mode: ImpactMode = Field(ImpactMode.ADDITIVE, description="Impact mode")
    impact_dist: Distribution = Field(..., description="Impact distribution")
    applies_to: List[str] = Field(
        default_factory=list, description="WBS codes affected"
    )
    applies_by_tag: List[str] = Field(default_factory=list, description="Tags affected")
    schedule_days_dist: Optional[Distribution] = Field(
        None, description="Schedule delay distribution"
    )
    description: Optional[str] = None
    mitigation: Optional[str] = None
    owner: Optional[str] = None


class CorrelationMethod(str, Enum):
    """Correlation methods."""

    SPEARMAN = "spearman"
    PEARSON = "pearson"


class CorrelationPair(BaseModel):
    """Correlation between two variables."""

    pair: List[str] = Field(..., min_items=2, max_items=2, description="Variable pair")
    rho: float = Field(..., ge=-1, le=1, description="Correlation coefficient")
    method: CorrelationMethod = Field(
        CorrelationMethod.SPEARMAN, description="Correlation method"
    )
    rationale: Optional[str] = Field(None, description="Justification")


class LatentFactor(BaseModel):
    """Latent factor driving correlations."""

    name: str = Field(..., description="Factor name (e.g., 'weather', 'commodity')")
    loadings: Dict[str, float] = Field(..., description="Variable loadings")
    distribution: Distribution = Field(..., description="Factor distribution")


class ConditionalLogic(BaseModel):
    """Conditional risk logic."""

    condition: str = Field(..., description="If condition (risk ID)")
    action: str = Field(..., description="Then action description")
    probability_delta: Optional[float] = Field(None, description="Probability change")
    impact_multiplier: Optional[float] = Field(None, description="Impact multiplier")


class RiskRegister(BaseModel):
    """Risk register with advanced features."""

    risks: List[RiskItem] = Field(..., description="Risk items")
    correlations: List[CorrelationPair] = Field(
        default_factory=list, description="Risk correlations"
    )
    latent_factors: List[LatentFactor] = Field(
        default_factory=list, description="Latent factors"
    )
    conditional_logic: List[ConditionalLogic] = Field(
        default_factory=list, description="Conditional logic"
    )


class SamplingMethod(str, Enum):
    """Monte Carlo sampling methods."""

    LHS = "LHS"  # Latin Hypercube
    SOBOL = "Sobol"  # Sobol sequences
    MC = "MC"  # Standard Monte Carlo


class VarianceReduction(BaseModel):
    """Variance reduction techniques."""

    antithetic_variates: bool = Field(False, description="Use antithetic variates")
    control_variates: bool = Field(False, description="Use control variates")
    importance_sampling: bool = Field(False, description="Use importance sampling")


class ConvergenceConfig(BaseModel):
    """Convergence criteria configuration."""

    enabled: bool = Field(True, description="Enable convergence checking")
    p50_tolerance: float = Field(0.005, description="P50 convergence tolerance")
    p80_tolerance: float = Field(0.005, description="P80 convergence tolerance")
    min_iterations: int = Field(1000, description="Minimum iterations")
    check_interval: int = Field(1000, description="Check interval")


class TwoLayerConfig(BaseModel):
    """Two-layer uncertainty configuration."""

    enabled: bool = Field(False, description="Enable two-layer uncertainty")
    epistemic_iterations: int = Field(100, description="Outer loop iterations")
    aleatory_iterations: int = Field(1000, description="Inner loop iterations")
    parameter_uncertainty: Dict[str, Distribution] = Field(
        default_factory=dict, description="Parameter uncertainty distributions"
    )


class SimulationConfig(BaseModel):
    """Enhanced simulation configuration."""

    n_iterations: int = Field(10000, gt=0, description="Number of iterations")
    random_seed: Optional[int] = Field(None, description="Random seed")
    sampling_method: SamplingMethod = Field(
        SamplingMethod.LHS, description="Sampling method"
    )
    variance_reduction: VarianceReduction = Field(default_factory=VarianceReduction)
    convergence: ConvergenceConfig = Field(default_factory=ConvergenceConfig)
    two_layer: TwoLayerConfig = Field(default_factory=TwoLayerConfig)
    performance_threads: Optional[int] = Field(None, description="Number of threads")


class ReportingConfig(BaseModel):
    """Reporting configuration."""

    percentiles: List[float] = Field(
        [5, 10, 25, 50, 75, 80, 90, 95], description="Percentiles to report"
    )
    enable_charts: bool = Field(True, description="Generate charts")
    sensitivity_analysis: bool = Field(True, description="Enable sensitivity analysis")
    confidence_level: float = Field(0.95, ge=0, le=1, description="Confidence level")
    export_samples: bool = Field(False, description="Export raw samples")


class ValidationConfig(BaseModel):
    """Validation configuration."""

    strict_mode: bool = Field(False, description="Enable strict validation")
    warning_threshold: float = Field(0.1, description="Warning threshold")
    max_correlation: float = Field(0.95, description="Maximum correlation")
    check_positive_definite: bool = Field(
        True, description="Check matrix positive definiteness"
    )


class FullConfig(BaseModel):
    """Complete configuration."""

    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


# Results Models
class StatisticalSummary(BaseModel):
    """Statistical summary of results."""

    count: int = Field(..., description="Number of samples")
    mean: float = Field(..., description="Mean value")
    stdev: float = Field(..., description="Standard deviation")
    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")
    percentiles: Dict[str, float] = Field(..., description="Percentile values")
    cv: float = Field(..., description="Coefficient of variation")


class TwoLayerSummary(BaseModel):
    """Two-layer uncertainty summary."""

    percentile_bands: Dict[str, Dict[str, float]] = Field(
        ..., description="Percentile bands"
    )
    epistemic_uncertainty: float = Field(
        ..., description="Epistemic uncertainty measure"
    )
    aleatory_uncertainty: float = Field(..., description="Aleatory uncertainty measure")


class SimulationResults(BaseModel):
    """Enhanced simulation results."""

    # Metadata
    simulation_info: Dict[str, Any] = Field(..., description="Simulation metadata")
    config: FullConfig = Field(..., description="Configuration used")

    # Core results
    total_cost_statistics: StatisticalSummary = Field(
        ..., description="Total cost statistics"
    )
    wbs_cost_statistics: Dict[str, StatisticalSummary] = Field(
        ..., description="WBS cost statistics"
    )
    risk_statistics: Dict[str, StatisticalSummary] = Field(
        ..., description="Risk statistics"
    )

    # Advanced results
    two_layer_results: Optional[TwoLayerSummary] = Field(
        None, description="Two-layer uncertainty results"
    )
    sensitivity_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Sensitivity analysis"
    )
    risk_contributions: Optional[Dict[str, Any]] = Field(
        None, description="Risk contributions"
    )

    # Raw data (optional)
    total_cost_samples: Optional[List[float]] = Field(
        None, description="Total cost samples"
    )
    wbs_cost_samples: Optional[Dict[str, List[float]]] = Field(
        None, description="WBS cost samples"
    )
    risk_cost_samples: Optional[Dict[str, List[float]]] = Field(
        None, description="Risk cost samples"
    )

    # Performance metrics
    convergence_achieved: bool = Field(
        False, description="Whether convergence was achieved"
    )
    actual_iterations: int = Field(..., description="Actual iterations performed")
    runtime_seconds: float = Field(..., description="Runtime in seconds")

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
