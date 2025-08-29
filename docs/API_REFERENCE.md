# Risk_Modeler API Reference

Complete API reference for programmatic access to Risk_Modeler functionality.

## Table of Contents

1. [Overview](#overview)
2. [Python API](#python-api)
3. [Command Line Interface](#command-line-interface)
4. [Web API](#web-api)
5. [Core Modules](#core-modules)
6. [Data Models](#data-models)
7. [Examples](#examples)
8. [Error Handling](#error-handling)

## Overview

Risk_Modeler provides multiple API interfaces for different use cases:

- **Python API**: Direct programmatic access to all functionality
- **Command Line Interface**: Shell-based automation and scripting
- **Web API**: RESTful HTTP interface for web applications
- **Streamlit UI**: Interactive web interface

## Python API

### Core Functions

#### `run_simulation(project, risks=None, correlations=None, config=None)`

Execute Monte Carlo simulation for a project.

**Parameters:**
- `project` (Project): Project configuration with WBS items
- `risks` (List[RiskItem], optional): Risk events to include
- `correlations` (List[Dict], optional): Correlation definitions
- `config` (SimulationConfig, optional): Simulation parameters

**Returns:**
- `SimulationResults`: Complete simulation results with statistics

**Example:**
```python
from risk_tool import run_simulation, Project, WBSItem, SimulationConfig

# Define project
project = Project(
    name="Test Project",
    project_type="TransmissionLine",
    wbs_items=[
        WBSItem(
            code="01-TEST",
            name="Test Item",
            quantity=1.0,
            uom="project",
            unit_cost=100000,
            dist_unit_cost={
                "type": "triangular",
                "low": 80000,
                "mode": 100000,
                "high": 120000
            }
        )
    ]
)

# Configure simulation
config = SimulationConfig(
    iterations=10000,
    random_seed=12345,
    sampling_method="LHS"
)

# Run simulation
results = run_simulation(project, config=config)

# Access results
print(f"Mean cost: ${results.mean_cost:,.0f}")
print(f"P80 cost: ${results.percentiles['p80']:,.0f}")
```

### Data Models

#### Project Class

```python
class Project(BaseModel):
    name: str
    description: Optional[str] = None
    project_type: Literal["TransmissionLine", "Substation", "Hybrid"]
    voltage_class: str
    region: Optional[str] = None
    base_year: str = "2025"
    currency: str = "USD"
    wbs_items: List[WBSItem] = []
    
    # Transmission line specific
    length_miles: Optional[float] = None
    circuit_count: Optional[int] = None
    terrain_type: Optional[str] = None
    
    # Substation specific  
    capacity_mva: Optional[float] = None
    bay_count: Optional[int] = None
    substation_type: Optional[str] = None
```

#### WBSItem Class

```python
class WBSItem(BaseModel):
    code: str
    name: str
    quantity: float
    uom: str
    unit_cost: float
    dist_quantity: Optional[Dict[str, Any]] = None
    dist_unit_cost: Optional[Dict[str, Any]] = None
    tags: List[str] = []
    indirect_factor: float = 0.0
```

#### SimulationConfig Class

```python
class SimulationConfig(BaseModel):
    iterations: int = 10000
    random_seed: Optional[int] = None
    sampling_method: str = "LHS"
    convergence_threshold: float = 0.01
    max_iterations: int = 100000
    confidence_levels: List[float] = [0.1, 0.5, 0.8, 0.9, 0.95]
```

#### SimulationResults Class

```python
class SimulationResults(BaseModel):
    project_id: str
    timestamp: str
    currency: str
    n_samples: int
    base_cost: float
    mean_cost: float
    std_cost: float
    percentiles: Dict[str, float]
    wbs_statistics: Dict[str, Dict[str, float]]
    convergence_achieved: bool
    iterations_completed: int
```

### Distribution Functions

#### `create_triangular_distribution(low, mode, high)`

Create a triangular distribution specification.

```python
from risk_tool.core.distributions import create_triangular_distribution

dist = create_triangular_distribution(80000, 100000, 120000)
```

#### `create_pert_distribution(min_val, most_likely, max_val, lambda_=4.0)`

Create a PERT distribution specification.

```python
from risk_tool.core.distributions import create_pert_distribution

dist = create_pert_distribution(90000, 100000, 130000)
```

#### `create_normal_distribution(mean, stdev, truncate_low=None, truncate_high=None)`

Create a normal distribution specification.

```python
from risk_tool.core.distributions import create_normal_distribution

dist = create_normal_distribution(100000, 15000, truncate_low=70000)
```

### Validation Functions

#### `validate_project(project)`

Validate project configuration.

**Parameters:**
- `project` (Project): Project to validate

**Returns:**
- `ValidationResult`: Validation results with errors and warnings

```python
from risk_tool.core.validation import validate_project

validation = validate_project(project)
if validation.is_valid:
    print("Project is valid!")
else:
    for error in validation.errors:
        print(f"Error: {error.message}")
```

### I/O Functions

#### `load_project_from_json(file_path)`

Load project from JSON file.

```python
from risk_tool.io.io_json import load_project_from_json

project = load_project_from_json("my_project.json")
```

#### `load_project_from_excel(file_path)`

Load project from Excel file.

```python
from risk_tool.io.io_excel import load_project_from_excel

project = load_project_from_excel("my_project.xlsx")
```

#### `export_results(results, output_path, format="json")`

Export simulation results.

```python
from risk_tool.reporting.reporting import export_results

export_results(results, "output/", format="json")
export_results(results, "output/", format="excel")
```

### Template Functions

#### `create_transmission_line_template()`

Generate transmission line project template.

```python
from risk_tool.templates.template_generator import create_transmission_line_template

template = create_transmission_line_template()
```

#### `create_substation_template()`

Generate substation project template.

```python
from risk_tool.templates.template_generator import create_substation_template

template = create_substation_template()
```

## Command Line Interface

### Main Commands

#### `risk-tool run`

Execute Monte Carlo simulation.

```bash
risk-tool run PROJECT_FILE [OPTIONS]

Options:
  --risks FILE              Risk events file
  --correlations FILE       Correlations file  
  --config FILE            Simulation configuration
  --output DIR             Output directory [default: ./results]
  --format FORMAT          Output format [json|excel|csv]
  --iterations INTEGER     Number of iterations [default: 10000]
  --seed INTEGER           Random seed
  --method [LHS|MCS]       Sampling method [default: LHS]
  --help                   Show help message
```

**Examples:**
```bash
# Basic simulation
risk-tool run project.json

# Custom parameters
risk-tool run project.json --iterations 25000 --seed 42 --output results/

# Multiple formats
risk-tool run project.json --format excel --format json
```

#### `risk-tool template`

Generate project templates.

```bash
risk-tool template CATEGORY FORMAT OUTPUT_FILE

Arguments:
  CATEGORY    Project category [transmission_line|substation]
  FORMAT      Output format [json|excel|csv]  
  OUTPUT_FILE Output file path

Options:
  --help      Show help message
```

**Examples:**
```bash
# JSON template
risk-tool template transmission_line json my_project.json

# Excel template  
risk-tool template substation excel substation_template.xlsx
```

#### `risk-tool validate`

Validate project configuration.

```bash
risk-tool validate PROJECT_FILE [OPTIONS]

Options:
  --strict     Enable strict validation
  --output DIR Output validation report
  --help       Show help message
```

**Examples:**
```bash
# Basic validation
risk-tool validate project.json

# Strict validation with report
risk-tool validate project.json --strict --output validation_report/
```

#### `risk-tool report`

Generate reports from results.

```bash
risk-tool report RESULTS_FILE [OPTIONS]

Options:
  --format [pdf|html|excel]  Report format [default: pdf]
  --output DIR              Output directory
  --template FILE           Custom report template
  --help                    Show help message
```

#### `risk-tool verify`

Verify simulation determinism.

```bash
risk-tool verify RESULTS_FILE [OPTIONS]

Options:
  --tolerance FLOAT  Tolerance for differences [default: 1e-6]
  --help            Show help message
```

### Global Options

Available for all commands:

```bash
Options:
  --log-level [DEBUG|INFO|WARNING|ERROR]  Set logging level
  --config FILE                          Global configuration file
  --version                              Show version
  --help                                 Show help message
```

## Web API

### Overview

RESTful HTTP API for web applications and remote access.

**Base URL**: `http://localhost:8000/api/v1/`

**Authentication**: API key required for production deployments

### Endpoints

#### POST `/simulations`

Execute Monte Carlo simulation.

**Request:**
```json
{
  "project_data": {
    "project_info": { ... },
    "wbs_items": [ ... ]
  },
  "simulation_config": {
    "iterations": 10000,
    "sampling_method": "LHS"
  },
  "output_format": "json"
}
```

**Response:**
```json
{
  "simulation_id": "uuid",
  "status": "completed",
  "results": { ... }
}
```

#### GET `/simulations/{simulation_id}`

Get simulation results.

**Response:**
```json
{
  "simulation_id": "uuid",
  "status": "completed", 
  "created_at": "2025-01-01T00:00:00Z",
  "results": { ... }
}
```

#### POST `/validate`

Validate project configuration.

**Request:**
```json
{
  "project_data": { ... },
  "validation_level": "strict"
}
```

**Response:**
```json
{
  "valid": true,
  "errors": [],
  "warnings": [ ... ]
}
```

#### GET `/templates/{category}/{format}`

Download project templates.

**Parameters:**
- `category`: `transmission_line` or `substation`
- `format`: `json`, `excel`, or `csv`

**Response:** File download

### Error Responses

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Project validation failed",
    "details": [ ... ]
  }
}
```

## Core Modules

### risk_tool.core.aggregation

Main simulation orchestration and execution.

**Classes:**
- `SimulationEngine`: Main Monte Carlo engine
- `ResultsCalculator`: Statistical analysis utilities

**Functions:**
- `run_simulation()`: Primary simulation interface

### risk_tool.core.distributions

Probability distribution handling and sampling.

**Classes:**
- `DistributionSampler`: Generate samples from distributions
- `Triangular`, `PERT`, `Normal`, etc.: Distribution specifications

**Functions:**
- `validate_distribution_config()`: Validate distribution parameters
- `get_distribution_stats()`: Calculate distribution statistics

### risk_tool.core.sampler

Monte Carlo sampling methods.

**Classes:**
- `MonteCarloSampler`: Main sampling engine
- `ConvergenceDiagnostics`: Convergence monitoring

### risk_tool.core.correlation

Correlation modeling and application.

**Classes:**
- `ImanConovierTransform`: Rank correlation transformation
- `CholekyTransform`: Pearson correlation transformation

**Functions:**
- `apply_correlations()`: Apply correlation structure to samples

### risk_tool.core.validation

Input validation and verification.

**Classes:**
- `ValidationEngine`: Comprehensive validation
- `ValidationResult`: Validation results container

### risk_tool.reporting

Results analysis and reporting.

**Classes:**
- `SimulationResults`: Results container
- `ReportGenerator`: Professional report generation

**Functions:**
- `create_summary_report()`: Generate summary reports
- `export_results()`: Export results in various formats

## Data Models

### Pydantic Models

All data models use Pydantic for validation and serialization:

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any

class ProjectInfo(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    project_type: str = Field(..., regex="^(TransmissionLine|Substation|Hybrid)$")
    voltage_class: str
    base_year: str = Field(..., regex="^\\d{4}$")
    currency: str = Field(..., regex="^(USD|CAD|EUR)$")
    
    @validator('base_year')
    def validate_base_year(cls, v):
        year = int(v)
        if not 2020 <= year <= 2030:
            raise ValueError("Base year must be between 2020 and 2030")
        return v
```

## Examples

### Complete Python Example

```python
from risk_tool import (
    Project, WBSItem, SimulationConfig, RiskItem,
    run_simulation, setup_logging
)
from risk_tool.core.logging_config import get_logger
import json

# Setup logging
logger = setup_logging(log_level="INFO")

# Create project
project = Project(
    name="Sample Transmission Line",
    description="25-mile 138kV line with uncertainty analysis",
    project_type="TransmissionLine", 
    voltage_class="138kV",
    length_miles=25.0,
    region="Northeast",
    base_year="2025",
    currency="USD",
    wbs_items=[
        WBSItem(
            code="01-ROW",
            name="Right of Way",
            quantity=25.0,
            uom="miles",
            unit_cost=50000,
            dist_unit_cost={
                "type": "triangular",
                "low": 40000,
                "mode": 50000, 
                "high": 75000
            },
            tags=["land", "legal"],
            indirect_factor=0.15
        ),
        WBSItem(
            code="03-STRUCT",
            name="Transmission Structures", 
            quantity=125.0,
            uom="structures",
            unit_cost=8000,
            dist_unit_cost={
                "type": "pert",
                "min": 7000,
                "most_likely": 8000,
                "max": 12000
            },
            tags=["steel", "foundation"],
            indirect_factor=0.12
        )
    ]
)

# Define correlations
correlations = [
    {
        "items": ["01-ROW", "03-STRUCT"],
        "correlation": 0.3,
        "method": "spearman"
    }
]

# Configure simulation
config = SimulationConfig(
    iterations=25000,
    random_seed=12345,
    sampling_method="LHS",
    confidence_levels=[0.1, 0.5, 0.8, 0.9, 0.95]
)

# Run simulation
logger.info("Starting Monte Carlo simulation...")
results = run_simulation(
    project=project,
    correlations=correlations,
    config=config
)

# Process results
logger.info("Simulation completed successfully!")
print(f"Project: {project.name}")
print(f"Base Cost: ${results.base_cost:,.0f}")
print(f"Mean Cost: ${results.mean_cost:,.0f}")
print(f"P50 Cost: ${results.percentiles['p50']:,.0f}")
print(f"P80 Cost: ${results.percentiles['p80']:,.0f}")

# Calculate contingency
contingency = (results.percentiles['p80'] - results.base_cost) / results.base_cost * 100
print(f"P80 Contingency: {contingency:.1f}%")

# Export results
from risk_tool.reporting.reporting import export_results
export_results(results, "output/", format="json")
export_results(results, "output/", format="excel") 

logger.info("Results exported successfully!")
```

### Batch Processing Example

```python
from pathlib import Path
import json
from risk_tool import run_simulation
from risk_tool.io.io_json import load_project_from_json

def batch_process_projects(projects_dir: str, output_dir: str):
    """Process multiple projects in batch."""
    projects_path = Path(projects_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for project_file in projects_path.glob("*.json"):
        print(f"Processing {project_file.name}...")
        
        try:
            # Load project
            project = load_project_from_json(project_file)
            
            # Run simulation
            results = run_simulation(project)
            
            # Save results
            output_file = output_path / f"{project_file.stem}_results.json"
            with open(output_file, 'w') as f:
                json.dump(results.dict(), f, indent=2)
                
            print(f"  ✅ Completed: P80 = ${results.percentiles['p80']:,.0f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")

# Usage
batch_process_projects("projects/", "results/")
```

## Error Handling

### Exception Hierarchy

```python
from risk_tool.core.exceptions import (
    RiskModelingError,          # Base exception
    ValidationError,            # Input validation errors
    ComputationError,           # Numerical computation errors  
    ConvergenceError,           # Monte Carlo convergence issues
    IOError,                    # File I/O problems
    PerformanceError            # Performance/resource issues
)

try:
    results = run_simulation(project)
except ValidationError as e:
    print(f"Validation error: {e.message}")
    for suggestion in e.recovery_suggestions:
        print(f"  - {suggestion}")
except ConvergenceError as e:
    print(f"Convergence error: {e.message}")
    print(f"Completed {e.iterations_completed} iterations")
except RiskModelingError as e:
    print(f"General error: {e.message}")
    logger.error("Error details", extra=e.to_dict())
```

### Error Codes

Common error codes and their meanings:

| Code | Description | Resolution |
|------|-------------|------------|
| `RM_VALIDATION_ERROR` | Input validation failed | Check input data format and values |
| `RM_CONVERGENCE_ERROR` | Simulation didn't converge | Increase iterations or check distributions |
| `RM_CORRELATION_ERROR` | Correlation matrix issues | Verify correlation values and matrix properties |
| `RM_DISTRIBUTION_ERROR` | Invalid distribution parameters | Check parameter constraints |
| `RM_IO_ERROR` | File access problems | Verify file paths and permissions |

---

*Risk_Modeler API Reference v1.0.0 - Complete programmatic interface documentation*