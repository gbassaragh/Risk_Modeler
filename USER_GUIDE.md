# User Guide: Risk Modeling Tool for T&D Projects

## Table of Contents
1. [Getting Started](#getting-started)
2. [Understanding the Tool](#understanding-the-tool)
3. [Data Preparation](#data-preparation)
4. [Running Simulations](#running-simulations)
5. [Interpreting Results](#interpreting-results)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation and Setup

1. **Install the tool**:
   ```bash
   cd Risk_Modeler
   pip install -e .
   ```

2. **Verify installation**:
   ```bash
   risk-tool --help
   ```

3. **Generate your first template**:
   ```bash
   risk-tool template transmission_line excel my_first_project.xlsx
   ```

### Your First Simulation

1. **Open the generated template** in Excel
2. **Edit the Project_Info sheet** with your project details
3. **Customize the WBS_Items sheet** with your cost elements
4. **Add risks** in the Risks sheet (optional for first run)
5. **Run the simulation**:
   ```bash
   risk-tool run my_first_project.xlsx --output ./my_results
   ```
6. **Review results** in the `my_results` directory

## Understanding the Tool

### Core Concepts

**Monte Carlo Simulation**: The tool runs thousands of iterations, sampling from probability distributions to model uncertainty in costs and risks.

**Two Modeling Approaches**:

1. **Line-Item Uncertainty**: Models uncertainty in WBS cost elements directly
   - Each WBS item has a base cost and probability distribution
   - Accounts for estimation uncertainty, material price variation, labor productivity

2. **Risk-Driver Method**: Models discrete risk events separately  
   - Each risk has occurrence probability and impact distribution
   - Risks affect specific WBS items when they occur
   - Better for modeling discrete threats vs. general uncertainty

**Latin Hypercube Sampling (LHS)**: Advanced sampling method that ensures better coverage of the probability space than simple random sampling, leading to faster convergence.

**Correlation Modeling**: Uses Iman-Conover method to maintain specified rank correlations between variables while preserving their marginal distributions.

### Project Categories

**Transmission Lines**:
- Linear infrastructure (cost per mile)
- Common WBS: ROW acquisition, surveying, foundations, structures, conductor, grounding, protection
- Typical risks: weather delays, permitting, geotechnical issues, material escalation

**Substations**:  
- Point infrastructure (cost per facility)
- Common WBS: site prep, transformers, switchgear, protection systems, civil structures
- Typical risks: equipment delays, site contamination, utility conflicts

## Data Preparation

### Project Information

Required fields vary by project category:

**All Projects**:
- `name`: Descriptive project name
- `category`: "transmission_line" or "substation"
- `voltage`: Operating voltage (e.g., "138kV", "69kV")

**Transmission Lines**:
- `length`: Length in miles
- `terrain`: "flat", "hilly", or "mixed"

**Substations**:
- `capacity`: Capacity in MVA
- `voltage_levels`: Number of voltage levels

### WBS Items Structure

Each WBS item requires:

```json
{
  "id": "unique_identifier",           // Used for correlations and risk targeting
  "name": "Descriptive Name",
  "base_cost": 1000000,              // Base estimate in dollars
  "distribution": {                   // Probability distribution
    "type": "triangular",             // See distribution guide below
    "low": 900000,
    "mode": 1000000, 
    "high": 1200000
  }
}
```

**Base Cost Guidelines**:
- Use your current best estimate (P50 level)
- Include direct costs, labor, materials, equipment
- Exclude contingency (that's what we're calculating!)

### Distribution Selection Guide

| Distribution | When to Use | Parameters |
|--------------|-------------|------------|
| **Triangular** | Three-point estimates (min/most-likely/max) | `low`, `mode`, `high` |
| **PERT** | Three-point with more weight on most-likely | `min`, `most_likely`, `max` |
| **Normal** | Symmetric uncertainty, large sample size | `mean`, `stdev` + optional truncation |
| **Log-Normal** | Multiplicative processes, right-skewed | `mean`, `sigma` (of underlying normal) |
| **Uniform** | Equal probability across range | `low`, `high` |
| **Discrete** | Specific scenarios with known probabilities | `pmf`: [[value1, prob1], [value2, prob2]] |

**Distribution Tips**:
- **Triangular** is most common for expert estimates
- Use **truncation** with Normal distributions to prevent negative costs
- **PERT** gives less weight to extreme values than Triangular
- **Log-Normal** is good for material price escalation

### Risk Data Structure

Each risk requires:

```json
{
  "id": "weather_delays",
  "name": "Weather-Related Construction Delays", 
  "category": "schedule",              // schedule|technical|regulatory|market|environmental
  "probability": 0.3,                  // 0.0 to 1.0
  "impact_distribution": {             // Cost impact if risk occurs
    "type": "triangular",
    "low": 100000,
    "mode": 250000, 
    "high": 500000
  },
  "affected_wbs": ["structures", "conductor"],  // Which WBS items are impacted
  "description": "Optional description"
}
```

**Risk Categories**:
- **Schedule**: Weather, resource availability, productivity issues
- **Technical**: Design changes, constructability issues, performance requirements
- **Regulatory**: Permitting delays, compliance requirements, policy changes
- **Market**: Material price escalation, labor cost increases, economic factors
- **Environmental**: Site contamination, protected species, wetlands

**Probability Guidelines**:
- 0.1 = 10% chance (rare but possible)
- 0.3 = 30% chance (possible, monitor closely)
- 0.5 = 50% chance (likely to occur)
- 0.8 = 80% chance (almost certain)

### Correlation Data

Correlations capture relationships between cost elements:

```json
{
  "pair": ["foundations", "structures"],
  "rho": 0.7,                         // -1.0 to +1.0
  "method": "spearman",               // "spearman" or "pearson" 
  "rationale": "Same crews and weather dependency"
}
```

**When to Use Correlations**:
- Same work crews or contractors
- Same weather sensitivity
- Shared material suppliers
- Sequential work dependencies
- Common external factors

**Correlation Strength Guidelines**:
- 0.1-0.3: Weak correlation
- 0.3-0.7: Moderate correlation  
- 0.7-0.9: Strong correlation
- 0.9-1.0: Very strong correlation (use carefully)

## Running Simulations

### Basic Simulation

```bash
risk-tool run project.xlsx --output ./results
```

### Advanced Options

```bash
risk-tool run project.xlsx \
  --risks risks.xlsx \
  --correlations correlations.json \
  --config simulation_config.json \
  --output ./results \
  --format json \
  --iterations 50000 \
  --seed 12345
```

### Configuration Options

Create a `config.json` file:

```json
{
  "simulation": {
    "n_iterations": 10000,           // Number of Monte Carlo iterations
    "random_seed": 42,               // For reproducible results
    "sampling_method": "LHS"         // LHS or MCS
  },
  "reporting": {
    "percentiles": [5, 10, 25, 50, 75, 80, 90, 95],
    "enable_charts": true,           // Generate charts (requires display)
    "sensitivity_analysis": true,    // Enable tornado diagrams
    "confidence_level": 0.95
  },
  "validation": {
    "strict_mode": false,            // Strict validation rules
    "warning_threshold": 0.1         // Threshold for warnings
  }
}
```

### Iteration Count Guidelines

| Project Complexity | Recommended Iterations | Run Time |
|-------------------|----------------------|----------|
| Simple (1-5 WBS items) | 5,000 | <5 seconds |
| Moderate (5-15 WBS items) | 10,000 | 5-10 seconds |
| Complex (15+ WBS items) | 25,000 | 10-20 seconds |
| Very Complex (many correlations/risks) | 50,000+ | 20+ seconds |

## Interpreting Results

### Key Statistics

**Percentiles**:
- **P10**: 10% probability of being below this value (optimistic)
- **P50**: 50% probability (median, most likely outcome)  
- **P80**: 80% probability (commonly used for contingency)
- **P90**: 90% probability (conservative planning)

**Contingency Calculation**:
```
Contingency = P80 Cost - Base Cost
Contingency % = (P80 Cost - Base Cost) / Base Cost × 100%
```

### Typical Contingency Ranges

| Project Type | Typical P80 Contingency |
|--------------|----------------------|
| Transmission Line (simple terrain) | 15-25% |
| Transmission Line (difficult terrain) | 25-40% |
| Distribution Substation | 10-20% |
| Transmission Substation | 15-30% |

### Result Files

**JSON Output** (`simulation_results.json`):
```json
{
  "simulation_info": {
    "project_name": "Green Valley 138kV Line",
    "n_iterations": 10000,
    "total_runtime": 8.5
  },
  "total_cost_statistics": {
    "mean": 23500000,
    "stdev": 2800000,
    "p10": 20200000,
    "p50": 23300000,
    "p80": 26100000,
    "p90": 27800000
  },
  "wbs_cost_statistics": { ... },
  "risk_statistics": { ... },
  "sensitivity_analysis": { ... }
}
```

**Excel Output**:
- **Summary**: Key project statistics
- **Total_Cost_Statistics**: Detailed cost statistics
- **WBS_Cost_Statistics**: Statistics by WBS item  
- **Risk_Statistics**: Risk contribution analysis
- **Charts** (if enabled): Histograms and tornado diagrams

### Sensitivity Analysis

**Tornado Diagram**: Shows which variables have the most impact on total cost variation:
- Variables are ranked by correlation with total cost
- Longer bars = more influential variables
- Focus risk management on top contributors

**Correlation Analysis**: Shows statistical relationships between inputs and total cost:
- Spearman correlation coefficients
- Helps identify key cost drivers
- Values close to ±1.0 indicate strong relationships

## Advanced Features

### Custom Distributions

Add custom distribution parameters:

```json
{
  "type": "normal",
  "mean": 1000000,
  "stdev": 150000,
  "truncate_low": 700000,      // Prevent negative values
  "truncate_high": 1500000     // Cap maximum values
}
```

### Schedule Integration

Link costs to schedule duration:

```json
{
  "schedule": {
    "duration_months": 18,
    "indirect_cost_rate": 50000,    // Monthly indirect costs
    "schedule_risk_factor": 0.1     // Additional risk from delays
  }
}
```

### Escalation Modeling

Model cost escalation over time:

```json
{
  "escalation": {
    "base_year": 2024,
    "project_midpoint": "2025-06",
    "escalation_rates": {
      "labor": 0.035,           // 3.5% annual
      "materials": 0.025,       // 2.5% annual
      "equipment": 0.020        // 2.0% annual
    }
  }
}
```

### Multiple Scenario Analysis

Run scenarios with different assumptions:

```bash
# Base case
risk-tool run project.xlsx --config base_config.json --output ./base_case

# High risk scenario  
risk-tool run project.xlsx --config high_risk_config.json --output ./high_risk

# Compare results
risk-tool compare ./base_case ./high_risk --output ./comparison
```

## Best Practices

### Data Quality

1. **Use Multiple Sources**: Don't rely on single estimates
2. **Historical Calibration**: Validate distributions against past projects
3. **Expert Review**: Have distributions reviewed by subject matter experts
4. **Documentation**: Document assumptions and rationale

### Risk Identification

1. **Systematic Approach**: Use risk registers and checklists
2. **Multi-Disciplinary Input**: Include engineering, construction, regulatory, environmental
3. **Historical Analysis**: Review risks from similar projects
4. **Regular Updates**: Update risk assessments as project progresses

### Model Validation

1. **Sanity Checks**: Verify results make intuitive sense
2. **Sensitivity Testing**: Test with extreme values
3. **Historical Comparison**: Compare to actual outcomes from similar projects
4. **Peer Review**: Have models reviewed by independent experts

### Documentation Standards

1. **Version Control**: Track model versions and changes
2. **Assumption Log**: Document all key assumptions
3. **Source Documentation**: Reference data sources and expert input
4. **Results Summary**: Create executive summaries for stakeholders

### Common Pitfalls to Avoid

1. **Correlation Overuse**: Don't add correlations without justification
2. **Distribution Mismatch**: Match distribution type to data characteristics
3. **Base Cost Errors**: Ensure base costs exclude contingency
4. **Risk Double-Counting**: Avoid overlapping risk categories
5. **Insufficient Iterations**: Use adequate sample sizes for stable results

## Troubleshooting

### Common Issues

**"Validation Failed" Errors**:
- Check that all required fields are present
- Verify distribution parameters (e.g., low < mode < high)
- Ensure probability values are between 0 and 1
- Check that WBS IDs referenced in risks exist

**"Matrix Not Positive Definite" Warnings**:
- Review correlation values for consistency
- Reduce extreme correlations (>0.9)
- Check for circular correlation dependencies

**Unexpected Results**:
- Verify base cost calculations
- Check risk probability and impact values
- Review correlation specifications
- Compare to historical project outcomes

**Performance Issues**:
- Reduce number of iterations for testing
- Disable chart generation for batch runs
- Use JSON format for faster I/O
- Check available system memory

### Getting Help

1. **Check Logs**: Review console output for detailed error messages
2. **Validate Input**: Run `risk-tool validate` to check data quality
3. **Test with Templates**: Start with generated templates to isolate issues
4. **Review Examples**: Check example projects for proper data format
5. **Debug Mode**: Use `--verbose` flag for detailed execution information

### Data Recovery

**Corrupted Files**:
- Check file permissions and disk space
- Try importing with different formats (Excel → CSV → JSON)
- Use backup copies if available

**Lost Configuration**:
- Regenerate with `risk-tool template`
- Review version control for previous configurations
- Check default configuration values

## Conclusion

This tool provides sophisticated Monte Carlo analysis capabilities for T&D project risk modeling. Start with simple models and gradually add complexity as you gain experience. Focus on data quality and proper validation of assumptions. Remember that the model is only as good as the input data and expert judgment that goes into it.

For additional support, review the example projects and test cases included with the tool.