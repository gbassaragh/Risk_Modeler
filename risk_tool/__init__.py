"""Risk Modeling Tool for Utility T&D Projects.

A production-grade Monte Carlo simulation engine for probabilistic cost analysis
of transmission lines and substations.
"""

__version__ = "1.0.0"
__author__ = "T&D Risk Modeling Team"

from .core.aggregation import run_simulation
from .core.cost_models import Project, WBSItem, SimulationConfig
from .core.risk_driver import RiskItem
from .reporting.reporting import SimulationResults

__all__ = [
    "run_simulation",
    "Project", 
    "WBSItem",
    "RiskItem", 
    "SimulationConfig",
    "SimulationResults",
]