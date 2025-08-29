# Risk_Modeler User Guide

Comprehensive guide for using the Risk_Modeler Monte Carlo simulation tool for T&D utility project risk analysis.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Project Setup](#project-setup)
5. [Data Models](#data-models)
6. [Running Simulations](#running-simulations)
7. [Understanding Results](#understanding-results)
8. [Web Interface](#web-interface)
9. [Command Line Interface](#command-line-interface)
10. [Advanced Features](#advanced-features)
11. [Troubleshooting](#troubleshooting)
12. [Best Practices](#best-practices)

## Introduction

Risk_Modeler is a production-grade Monte Carlo simulation engine designed specifically for probabilistic cost analysis of transmission and distribution (T&D) utility projects. It provides comprehensive risk modeling capabilities with professional-grade statistical analysis and reporting.

### Key Features

- **Monte Carlo Simulation**: 10,000-200,000 iterations with Latin Hypercube Sampling (LHS)
- **Dual Cost Modeling**: Line-item uncertainty and risk-driver methodologies
- **Professional Analysis**: P10/P50/P80/P90 percentiles, sensitivity analysis, tornado diagrams
- **T&D Project Focus**: Pre-configured templates for transmission lines and substations
- **Multiple Interfaces**: Web UI, Command Line, and API access
- **Performance Optimized**: 50,000 iterations in ≤10 seconds

## Installation

### Requirements

- Python 3.11 or higher
- 4GB+ RAM recommended
- 1GB+ available disk space

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/gbassaragh/Risk_Modeler.git
cd Risk_Modeler

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -e .
```

### Docker Installation

```bash
# Build and run with Docker
docker build -t risk-modeler .
docker run -p 8501:8501 -p 8000:8000 risk-modeler
```

### Verification

```bash
# Test CLI
risk-tool --help

# Test Web UI
risk-tool-web
```

## Quick Start

### 1. Launch Web Interface

The easiest way to get started is with the web interface:

```bash
risk-tool-web
```

Open your browser to `http://localhost:8501`

### 2. Create Your First Project

1. **Choose a template**: Select "Transmission Line" or "Substation"
2. **Configure project**: Enter project name, voltage class, and basic parameters
3. **Define WBS items**: Add cost elements with uncertainty distributions
4. **Run simulation**: Execute 10,000 Monte Carlo iterations
5. **Analyze results**: Review percentiles, charts, and recommendations

### 3. Command Line Quick Start

```bash
# Generate a template
risk-tool template transmission_line json my_project.json

# Edit the template with your data (in any text editor)
# Then run simulation
risk-tool run my_project.json --output results/ --format json

# View results
ls results/
```

## Project Setup

### Project Information

Every project requires basic information:

```json
{
  "project_info": {
    "name": "138kV Transmission Line Project",
    "description": "25-mile transmission line construction",
    "project_type": "TransmissionLine",
    "voltage_class": "138kV",
    "length_miles": 25.0,
    "region": "Northeast",
    "base_year": "2025",
    "currency": "USD",
    "aace_class": "Class 3"
  }
}
```

### Work Breakdown Structure (WBS)

WBS items represent cost elements with uncertainty:

```json
{
  "wbs_items": [
    {
      "code": "01-ROW",
      "name": "Right of Way & Land Acquisition",
      "quantity": 25.0,
      "uom": "miles",
      "unit_cost": 50000,
      "dist_unit_cost": {
        "type": "triangular",
        "low": 40000,
        "mode": 50000,
        "high": 75000
      },
      "tags": ["land", "legal", "environmental"],
      "indirect_factor": 0.15
    }
  ]
}
```

### Supported Distributions

| Distribution | Parameters | Use Case |
|--------------|------------|----------|
| **Triangular** | low, mode, high | Expert estimates with min/most-likely/max |
| **PERT** | min, most_likely, max | Three-point estimates with beta shape |
| **Normal** | mean, stdev, truncate_low, truncate_high | Symmetric uncertainty |
| **Log-Normal** | mean, sigma | Multiplicative processes |
| **Uniform** | low, high | Equal probability over range |
| **Discrete** | pmf: [[value, probability]] | Specific scenarios |

### Correlations

Define relationships between cost elements:

```json
{
  "correlations": [
    {
      "items": ["03-STRUCT", "04-FOUND"],
      "correlation": 0.8,
      "method": "spearman"
    }
  ]
}
```

## Data Models

### Project Types

- **TransmissionLine**: Overhead transmission lines (38kV-765kV)
- **Substation**: Power substations (distribution and transmission)
- **Hybrid**: Combined transmission and substation projects

### Cost Modeling Approaches

1. **Line-item Uncertainty**: Uncertainty distributions on individual WBS items
2. **Risk-driver Methodology**: Discrete risk events with probability and impact

### Validation Rules

- All costs must be non-negative
- Distribution parameters must be mathematically valid
- Correlation values must be between -1 and 1
- Project type must match WBS structure

## Running Simulations

### Simulation Configuration

```json
{
  "simulation_config": {
    "iterations": 10000,
    "random_seed": 12345,
    "sampling_method": "LHS",
    "convergence_threshold": 0.01,
    "max_iterations": 50000,
    "confidence_levels": [0.1, 0.5, 0.8, 0.9, 0.95]
  }
}
```

### Performance Guidelines

| Iterations | Accuracy | Time | Use Case |
|------------|----------|------|-----------|
| 1,000 | ±5% | <1s | Quick estimates |
| 10,000 | ±2% | 2-5s | Standard analysis |
| 50,000 | ±1% | 5-10s | High precision |
| 100,000+ | ±0.5% | 10s+ | Critical decisions |

### Sampling Methods

- **LHS (Latin Hypercube)**: Better convergence, stratified sampling
- **MCS (Monte Carlo)**: Traditional random sampling

## Understanding Results

### Key Metrics

- **P10**: 10% probability of exceeding (optimistic scenario)
- **P50**: Median/most likely outcome
- **P80**: 20% probability of exceeding (typical contingency level)
- **P90**: 10% probability of exceeding (conservative scenario)

### Contingency Recommendations

Based on P80 results:

- **Transmission Lines**: Typically 20-35% contingency
- **Substations**: Typically 15-25% contingency
- **AACE Class 3**: 20-30% expected range

### Statistical Validation

- **Convergence**: Results stabilize with sufficient iterations
- **Distribution Fitting**: Generated samples match input distributions
- **Correlation Achievement**: Output correlations match inputs

## Web Interface

### Navigation

1. **🏠 Home**: Overview and quick start guide
2. **📁 Project Setup**: Configure project information and WBS
3. **⚙️ Configuration**: Simulation parameters and risk events
4. **🎯 Run Simulation**: Execute Monte Carlo analysis
5. **📊 Results**: Interactive charts and analysis
6. **📋 Templates**: Download pre-configured templates

### Features

- **Interactive Forms**: User-friendly data entry
- **Real-time Validation**: Immediate feedback on inputs
- **Progress Tracking**: Visual progress during simulation
- **Export Options**: Download results in multiple formats
- **Template Integration**: Direct template download

## Command Line Interface

### Main Commands

```bash
# Run simulation
risk-tool run PROJECT_FILE [OPTIONS]

# Generate templates
risk-tool template CATEGORY FORMAT OUTPUT_FILE

# Validate inputs
risk-tool validate PROJECT_FILE [OPTIONS]

# Generate reports
risk-tool report RESULTS_FILE [OPTIONS]

# Verify determinism
risk-tool verify RESULTS_FILE
```

### Common Options

- `--output DIR`: Output directory for results
- `--format FORMAT`: Output format (json/excel/csv)
- `--iterations N`: Number of Monte Carlo iterations
- `--seed N`: Random seed for reproducibility
- `--method METHOD`: Sampling method (LHS/MCS)

### Examples

```bash
# Create transmission line project
risk-tool template transmission_line excel my_line.xlsx

# Run with custom parameters
risk-tool run my_line.xlsx --iterations 25000 --seed 42 --output results/

# Generate comprehensive report
risk-tool report results/simulation_results.json --format pdf
```

## Advanced Features

### Risk Events

Define discrete risk events with probability and impact:

```json
{
  "risk_events": [
    {
      "id": "WEATHER-001",
      "name": "Weather-Related Delays",
      "category": "schedule",
      "probability": 0.4,
      "impact_distribution": {
        "type": "pert",
        "min": 50000,
        "most_likely": 150000,
        "max": 400000
      },
      "affected_wbs": ["02-CLEAR", "03-STRUCT"]
    }
  ]
}
```

### Sensitivity Analysis

Identify cost drivers with:

- **Tornado Diagrams**: Rank variables by impact on output variance
- **Spearman Correlation**: Measure monotonic relationships
- **Contribution Analysis**: Quantify each element's risk contribution

### Custom Distributions

Extend with custom distributions:

```python
from risk_tool.core.distributions import DistributionConfig

class CustomDistribution(DistributionConfig):
    type: str = "custom"
    param1: float
    param2: float
```

### Batch Processing

Process multiple projects:

```bash
# Process all projects in directory
for file in projects/*.json; do
    risk-tool run "$file" --output "results/$(basename "$file" .json)/"
done
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Check Python version
python --version  # Should be 3.11+

# Reinstall dependencies
pip install -r requirements.txt
```

#### Memory Issues
```bash
# Reduce iterations for large projects
risk-tool run project.json --iterations 5000

# Use batch processing
risk-tool run project.json --batch-size 1000
```

#### Convergence Problems
```bash
# Increase maximum iterations
risk-tool run project.json --max-iterations 100000

# Check for extreme distributions
risk-tool validate project.json --verbose
```

### Debugging

Enable detailed logging:

```bash
export RISK_TOOL_LOG_LEVEL=DEBUG
risk-tool run project.json
```

### Performance Optimization

```python
# Enable performance monitoring
from risk_tool.core.performance import optimize_numpy_settings
optimize_numpy_settings()
```

## Best Practices

### Project Setup

1. **Start with Templates**: Use provided templates as starting points
2. **Validate Early**: Check inputs before running large simulations
3. **Document Assumptions**: Record rationale for distributions and correlations
4. **Use Realistic Data**: Base estimates on historical project data

### Distribution Selection

1. **Triangular**: Good for expert estimates with clear min/mode/max
2. **PERT**: Better than triangular for emphasizing most likely value
3. **Normal**: Use for well-understood, symmetric uncertainties
4. **Log-Normal**: For cost elements with natural lower bounds

### Correlation Modeling

1. **Be Conservative**: High correlations can significantly impact results
2. **Use Domain Knowledge**: Base correlations on physical relationships
3. **Validate Results**: Check that output correlations match inputs
4. **Start Simple**: Add correlations incrementally

### Simulation Parameters

1. **Iterations**: Start with 10,000, increase for important decisions
2. **Seeds**: Use fixed seeds for reproducible results
3. **Convergence**: Monitor convergence for stable results
4. **Validation**: Always validate inputs before simulation

### Results Analysis

1. **Focus on P80**: Standard contingency level for most projects
2. **Consider Range**: Look at full distribution, not just point estimates
3. **Sensitivity Analysis**: Identify key cost drivers for management focus
4. **Document Results**: Capture assumptions and rationale in reports

### Quality Assurance

1. **Peer Review**: Have experienced colleagues review models
2. **Benchmark**: Compare results to similar historical projects
3. **Sensitivity Testing**: Vary key assumptions to test robustness
4. **Documentation**: Maintain clear audit trail

## Support and Resources

### Documentation

- **API Reference**: Complete function and class documentation
- **Examples**: Sample projects and use cases
- **Video Tutorials**: Step-by-step guidance

### Community

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Contributing**: Guidelines for code contributions

### Professional Support

For enterprise support, consulting, and custom development:
- Contact: [risk-modeler-support@utility.com]
- Training: Available for teams and organizations
- Consulting: Model development and validation services

---

*Risk_Modeler v1.0.0 - Production-grade Monte Carlo risk analysis for T&D utility projects*