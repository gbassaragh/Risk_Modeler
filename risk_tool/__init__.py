"""Risk Modeling Tool for Utility T&D Projects.

A production-grade Monte Carlo simulation engine for probabilistic cost analysis
of transmission lines and substations.
"""

__version__ = "1.0.0"
__author__ = "T&D Risk Modeling Team"

# Core functionality
from .core.aggregation import run_simulation
from .core.cost_models import Project, WBSItem, SimulationConfig
from .core.risk_driver import RiskItem

# Results and reporting
from .reporting.reporting import SimulationResults

# Exception handling
from .core.exceptions import (
    RiskModelingError,
    ValidationError,
    ComputationError,
    IOError,
    PerformanceError,
)

# Logging configuration
from .core.logging_config import setup_logging, get_logger

__all__ = [
    # Core functionality
    "run_simulation",
    "Project",
    "WBSItem",
    "RiskItem",
    "SimulationConfig",
    # Results and reporting
    "SimulationResults",
    # Exception handling
    "RiskModelingError",
    "ValidationError",
    "ComputationError",
    "IOError",
    "PerformanceError",
    # Logging
    "setup_logging",
    "get_logger",
]
