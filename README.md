# Risk_Modeler - Monte Carlo Risk Analysis for T&D Projects

A production-grade Monte Carlo simulation engine for probabilistic cost analysis of transmission lines and substations. Built with modern Python architecture, comprehensive error handling, and professional DevOps practices.

## Features

### Core Capabilities
- **Monte Carlo Simulation**: 10,000-200,000 iterations with Latin Hypercube Sampling (LHS)
- **Dual Cost Modeling Approaches**:
  - Line-item uncertainty modeling for WBS cost elements
  - Risk-driver methodology for discrete risk events
- **Advanced Correlation Modeling**: Iman-Conover rank correlation transformation
- **Comprehensive Risk Analysis**: Bernoulli occurrence with impact distributions
- **Multiple Distribution Types**: Triangular, PERT, Normal, Log-Normal, Uniform, Discrete
- **Sensitivity Analysis**: Tornado diagrams and Spearman correlation analysis
- **Performance Optimized**: 50,000 iterations in ≤10 seconds

### Input/Output Support
- **Excel**: Full import/export with multi-sheet workbooks
- **CSV**: Structured import/export with multiple files
- **JSON**: Native format for programmatic integration
- **Template Generation**: Pre-configured templates for transmission lines and substations

### Professional Features
- **Audit Trail**: Complete simulation logging and determinism verification
- **Statistical Validation**: Convergence diagnostics and distribution testing
- **Reporting**: P10/P50/P80/P90 percentiles, confidence intervals, risk contributions
- **CLI Interface**: Command-line tools for automation and batch processing

## Quick Start

### Installation

```bash
git clone <repository-url>
cd Risk_Modeler
pip install -e .
```

### Basic Usage

1. **Generate a template**:
```bash
risk-tool template transmission_line excel my_project.xlsx
```

2. **Edit the template** with your project data

3. **Run simulation**:
```bash
risk-tool run my_project.xlsx --output ./results --format json
```

4. **View results** in the generated output files

## Project Structure

```
Risk_Modeler/
├── risk_tool/
│   ├── engine/              # Core simulation engine
│   │   ├── distributions.py # Probability distributions
│   │   ├── sampler.py      # Monte Carlo sampling (LHS)
│   │   ├── correlation.py  # Iman-Conover correlation
│   │   ├── risk_models.py  # Risk occurrence and impact
│   │   ├── cost_models.py  # WBS cost modeling
│   │   ├── aggregation.py  # Main simulation orchestrator
│   │   ├── sensitivity.py  # Tornado and correlation analysis
│   │   ├── reporting.py    # Results formatting
│   │   ├── validation.py   # Input validation
│   │   └── audit.py        # Audit trail and determinism
│   ├── cli.py              # Command-line interface
│   └── tests/              # Comprehensive test suite
├── setup.py                # Package configuration
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Documentation

- [USER_GUIDE.md](USER_GUIDE.md) - Comprehensive user guide
- [DATA_DICT.md](DATA_DICT.md) - Data format specifications
- [Examples](examples/) - Sample projects and use cases

## Command Line Interface

The tool provides several CLI commands:

### `risk-tool run`
Run Monte Carlo simulation on project data.

```bash
risk-tool run PROJECT_FILE [OPTIONS]
```

**Options:**
- `--risks FILE`: Risk data file
- `--correlations FILE`: Correlation data file  
- `--config FILE`: Simulation configuration
- `--output DIR`: Output directory (default: ./results)
- `--format FORMAT`: Output format (json/excel/csv)
- `--iterations N`: Number of iterations (default: 10000)
- `--seed N`: Random seed for reproducibility
- `--method METHOD`: Sampling method (LHS/MCS)

### `risk-tool template`
Generate project template files.

```bash
risk-tool template CATEGORY FORMAT OUTPUT_FILE
```

**Categories:** `transmission_line`, `substation`  
**Formats:** `excel`, `csv`, `json`

### `risk-tool validate`
Validate project data before simulation.

```bash
risk-tool validate PROJECT_FILE [OPTIONS]
```

### `risk-tool report`
Generate reports from existing results.

```bash
risk-tool report RESULTS_FILE [OPTIONS]
```

### `risk-tool verify`
Verify simulation determinism and audit trail.

```bash
risk-tool verify RESULTS_FILE
```

## Data Formats

### Project Data Structure

**Project Information:**
```json
{
  "project_info": {
    "name": "Project Name",
    "category": "transmission_line|substation",
    "voltage": "138kV",
    "length": 25.5,
    "capacity": 100,
    "terrain": "flat|hilly|mixed",
    "region": "Northeast"
  }
}
```

**WBS Items:**
```json
{
  "wbs_items": [
    {
      "id": "structures",
      "name": "Transmission Structures", 
      "base_cost": 1000000,
      "distribution": {
        "type": "triangular",
        "low": 900000,
        "mode": 1000000,
        "high": 1200000
      }
    }
  ]
}
```

### Risk Data Structure

```json
[
  {
    "id": "weather_delays",
    "name": "Weather-Related Delays",
    "category": "schedule|technical|regulatory|market",
    "probability": 0.3,
    "impact_distribution": {
      "type": "pert",
      "min": 100000,
      "most_likely": 250000,
      "max": 500000
    },
    "affected_wbs": ["structures", "conductor"]
  }
]
```

### Correlation Data Structure

```json
[
  {
    "pair": ["structures", "conductor"],
    "rho": 0.6,
    "method": "spearman|pearson"
  }
]
```

## Supported Distributions

| Distribution | Parameters | Use Case |
|--------------|------------|----------|
| **Triangular** | `low`, `mode`, `high` | Expert estimates with min/most-likely/max |
| **PERT** | `min`, `most_likely`, `max` | Three-point estimates with beta shape |
| **Normal** | `mean`, `stdev`, `truncate_low`, `truncate_high` | Symmetric uncertainty |
| **Log-Normal** | `mean`, `sigma` | Multiplicative processes |
| **Uniform** | `low`, `high` | Equal probability over range |
| **Discrete** | `pmf`: [[value, probability]] | Specific scenarios |

## Performance Characteristics

- **Speed**: 50,000 iterations in ≤10 seconds (typical hardware)
- **Memory**: Efficient sample storage and processing
- **Scalability**: Handles projects with 100+ WBS items and 50+ risks
- **Accuracy**: LHS provides superior convergence vs. simple Monte Carlo
- **Repeatability**: Deterministic results with seed control

## Validation Features

### Input Validation
- Distribution parameter validation
- WBS item consistency checks
- Risk data completeness verification
- Correlation matrix positive definiteness
- Business logic validation for T&D projects

### Statistical Validation
- Sample size adequacy assessment
- Convergence diagnostics
- Distribution fitting validation
- Correlation achievement verification

### Audit & Compliance
- Complete parameter logging
- Seed and method documentation
- Determinism verification
- Regulatory reporting support

## Example Use Cases

### 1. Transmission Line Project
- 25-mile 138kV transmission line
- 8 WBS categories (ROW, structures, conductor, etc.)
- 7 major risk categories
- Realistic correlations between construction activities
- **Result**: P80 contingency of 25-35% above base estimate

### 2. Distribution Substation
- 50 MVA distribution substation
- 7 WBS categories (transformers, switchgear, civil, etc.)
- 4 major risks (equipment delays, site issues, etc.)
- **Result**: P80 contingency of 15-25% above base estimate

## API Usage

For programmatic access:

```python
from risk_tool.engine.aggregation import run_simulation

# Load your project data
project_data = {...}
risks_data = [...]
config_data = {...}

# Run simulation
results = run_simulation(
    project_data, 
    risks_data, 
    config_data=config_data
)

# Access results
print(f"Mean cost: ${results.total_cost_statistics['mean']:,.0f}")
print(f"P80 cost: ${results.total_cost_statistics['p80']:,.0f}")
```

## Testing

The tool includes a comprehensive test suite:

```bash
# Run all tests
pytest risk_tool/tests/

# Run specific test categories
pytest risk_tool/tests/test_distributions.py     # Distribution tests
pytest risk_tool/tests/test_correlation.py      # Correlation tests  
pytest risk_tool/tests/test_engine_smoke.py     # Integration tests
pytest risk_tool/tests/test_io.py               # Import/export tests
pytest risk_tool/tests/test_examples_end_to_end.py  # Realistic examples
```

## Dependencies

Core dependencies:
- **numpy**: Numerical computing
- **scipy**: Statistical functions and LHS sampling
- **pandas**: Data manipulation
- **matplotlib/plotly**: Visualization
- **openpyxl**: Excel file handling
- **pydantic**: Data validation
- **typer**: CLI interface
- **pytest**: Testing framework

## Support

For questions, issues, or feature requests:
1. Check the [USER_GUIDE.md](USER_GUIDE.md) for detailed documentation
2. Review the [examples/](examples/) directory for sample projects
3. Run `risk-tool --help` for command-line help
4. Check test files for usage examples

## Contributing

This tool is designed for utility T&D project risk modeling. When contributing:
1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure backward compatibility with existing project files

## License

[Add your license information here]