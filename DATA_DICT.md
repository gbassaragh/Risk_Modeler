# Data Dictionary: Risk Modeling Tool

## Overview

This document provides detailed specifications for all data formats supported by the Risk Modeling Tool. The tool supports Excel, CSV, and JSON formats with consistent data structures.

## Table of Contents

1. [Project Data Structure](#project-data-structure)
2. [Risk Data Structure](#risk-data-structure)
3. [Correlation Data Structure](#correlation-data-structure)
4. [Configuration Data Structure](#configuration-data-structure)
5. [Results Data Structure](#results-data-structure)
6. [Distribution Specifications](#distribution-specifications)
7. [Excel File Format](#excel-file-format)
8. [CSV File Format](#csv-file-format)
9. [JSON File Format](#json-file-format)

## Project Data Structure

### Project Information

| Field | Type | Required | Description | Valid Values |
|-------|------|----------|-------------|--------------|
| `name` | string | Yes | Project name | Any descriptive string |
| `category` | string | Yes | Project type | "transmission_line", "substation" |
| `voltage` | string | Yes | Operating voltage | e.g., "138kV", "69kV", "230kV" |
| `length` | number | TL only | Length in miles | Positive number |
| `capacity` | number | Sub only | Capacity in MVA | Positive number |
| `terrain` | string | Optional | Terrain type | "flat", "hilly", "mixed" |
| `region` | string | Optional | Geographic region | Any string |
| `description` | string | Optional | Project description | Any string |
| `voltage_levels` | number | Sub only | Number of voltage levels | Integer ≥ 1 |

**Notes:**
- TL = Transmission Line projects only
- Sub = Substation projects only

### WBS Items

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `id` | string | Yes | Unique identifier | Alphanumeric, no spaces |
| `name` | string | Yes | Descriptive name | Any string |
| `base_cost` | number | Yes | Base cost estimate ($) | Positive number |
| `distribution` | object | Yes | Probability distribution | See Distribution Specifications |
| `category` | string | Optional | WBS category | Any string |
| `unit` | string | Optional | Cost unit | "lump_sum", "per_mile", "per_mva" |
| `quantity` | number | Optional | Quantity multiplier | Positive number |

## Risk Data Structure

### Risk Items

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `id` | string | Yes | Unique identifier | Alphanumeric, no spaces |
| `name` | string | Yes | Risk name | Any string |
| `category` | string | Yes | Risk category | "schedule", "technical", "regulatory", "market", "environmental", "construction", "supply_chain", "operational" |
| `probability` | number | Yes | Occurrence probability | 0.0 to 1.0 |
| `impact_distribution` | object | Yes | Impact if occurs | See Distribution Specifications |
| `affected_wbs` | array | Yes | Affected WBS items | Array of WBS IDs |
| `description` | string | Optional | Risk description | Any string |
| `mitigation` | string | Optional | Mitigation strategy | Any string |
| `owner` | string | Optional | Risk owner | Any string |

## Correlation Data Structure

### Correlation Pairs

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `pair` | array | Yes | Variable pair | Array of 2 WBS IDs |
| `rho` | number | Yes | Correlation coefficient | -1.0 to +1.0 |
| `method` | string | Yes | Correlation type | "spearman", "pearson" |
| `rationale` | string | Optional | Justification | Any string |

**Alternative Format** (for Excel/CSV):

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `var1` | string | Yes | First variable | WBS ID |
| `var2` | string | Yes | Second variable | WBS ID |
| `rho` | number | Yes | Correlation coefficient | -1.0 to +1.0 |
| `method` | string | Yes | Correlation type | "spearman", "pearson" |

## Configuration Data Structure

### Simulation Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `n_iterations` | integer | No | 10000 | Number of Monte Carlo iterations |
| `random_seed` | integer | No | null | Random seed for reproducibility |
| `sampling_method` | string | No | "LHS" | Sampling method |

**Valid Sampling Methods:** "LHS", "MCS"

### Reporting Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `percentiles` | array | No | [10, 50, 80, 90] | Percentiles to calculate |
| `enable_charts` | boolean | No | true | Generate charts |
| `sensitivity_analysis` | boolean | No | true | Enable sensitivity analysis |
| `confidence_level` | number | No | 0.95 | Confidence level for intervals |

### Validation Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `strict_mode` | boolean | No | false | Enable strict validation |
| `warning_threshold` | number | No | 0.1 | Threshold for warnings |
| `max_correlation` | number | No | 0.95 | Maximum allowed correlation |

## Results Data Structure

### Simulation Information

| Field | Type | Description |
|-------|------|-------------|
| `project_name` | string | Project name |
| `simulation_date` | string | ISO date when run |
| `n_iterations` | integer | Number of iterations |
| `random_seed` | integer | Random seed used |
| `sampling_method` | string | Sampling method used |
| `total_runtime` | number | Runtime in seconds |

### Cost Statistics

Statistics are provided for total cost, each WBS item, and each risk.

| Field | Type | Description |
|-------|------|-------------|
| `mean` | number | Arithmetic mean |
| `stdev` | number | Standard deviation |
| `min` | number | Minimum value |
| `max` | number | Maximum value |
| `p05`, `p10`, `p25`, `p50`, `p75`, `p80`, `p90`, `p95` | number | Percentile values |
| `cv` | number | Coefficient of variation (stdev/mean) |

### Sensitivity Analysis

| Field | Type | Description |
|-------|------|-------------|
| `tornado_data` | array | Variables ranked by impact |
| `correlation_analysis` | object | Correlation with total cost |
| `variable_contributions` | object | Contribution to total variance |

## Distribution Specifications

### Triangular Distribution

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|-------------|
| `type` | string | Yes | "triangular" | Literal value |
| `low` | number | Yes | Minimum value | low ≤ mode ≤ high |
| `mode` | number | Yes | Most likely value | low ≤ mode ≤ high |
| `high` | number | Yes | Maximum value | low ≤ mode ≤ high |

### PERT Distribution

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|-------------|
| `type` | string | Yes | "pert" | Literal value |
| `min` | number | Yes | Minimum value | min ≤ most_likely ≤ max |
| `most_likely` | number | Yes | Most likely value | min ≤ most_likely ≤ max |
| `max` | number | Yes | Maximum value | min ≤ most_likely ≤ max |

### Normal Distribution

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|-------------|
| `type` | string | Yes | "normal" | Literal value |
| `mean` | number | Yes | Mean value | Any number |
| `stdev` | number | Yes | Standard deviation | Positive number |
| `truncate_low` | number | No | Lower truncation | Any number |
| `truncate_high` | number | No | Upper truncation | > truncate_low |

### Log-Normal Distribution

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|-------------|
| `type` | string | Yes | "lognormal" | Literal value |
| `mean` | number | Yes | Mean of ln(X) | Any number |
| `sigma` | number | Yes | Std dev of ln(X) | Positive number |

### Uniform Distribution

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|-------------|
| `type` | string | Yes | "uniform" | Literal value |
| `low` | number | Yes | Lower bound | low < high |
| `high` | number | Yes | Upper bound | high > low |

### Discrete Distribution

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|-------------|
| `type` | string | Yes | "discrete" | Literal value |
| `pmf` | array | Yes | Probability mass function | [[value1, prob1], [value2, prob2], ...] |

**PMF Constraints:**
- Each entry is [value, probability]
- All probabilities must sum to 1.0
- All probabilities must be ≥ 0

## Excel File Format

### Sheet Structure

| Sheet Name | Required | Description |
|------------|----------|-------------|
| `Instructions` | No | User guide and instructions |
| `Project_Info` | Yes | Project information (single row) |
| `WBS_Items` | Yes | WBS cost items (multiple rows) |
| `Risks` | No | Risk data (multiple rows) |
| `Correlations` | No | Correlation pairs (multiple rows) |
| `Config` | No | Simulation configuration |

### Project_Info Sheet

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `name` | string | Yes | Project name |
| `category` | string | Yes | Project category |
| `voltage` | string | Yes | Operating voltage |
| `length` | number | TL only | Project length |
| `capacity` | number | Sub only | Project capacity |
| `terrain` | string | No | Terrain type |
| `region` | string | No | Geographic region |
| `description` | string | No | Description |

### WBS_Items Sheet

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `id` | string | Yes | Unique WBS identifier |
| `name` | string | Yes | WBS item name |
| `base_cost` | number | Yes | Base cost estimate |
| `dist_type` | string | Yes | Distribution type |
| `low` | number | Conditional | For triangular |
| `mode` | number | Conditional | For triangular |
| `high` | number | Conditional | For triangular |
| `min` | number | Conditional | For PERT |
| `most_likely` | number | Conditional | For PERT |
| `max` | number | Conditional | For PERT |
| `mean` | number | Conditional | For normal/lognormal |
| `stdev` | number | Conditional | For normal |
| `sigma` | number | Conditional | For lognormal |
| `truncate_low` | number | No | Lower truncation |
| `truncate_high` | number | No | Upper truncation |

### Risks Sheet

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `id` | string | Yes | Risk identifier |
| `name` | string | Yes | Risk name |
| `category` | string | Yes | Risk category |
| `probability` | number | Yes | Occurrence probability |
| `impact_type` | string | Yes | Impact distribution type |
| `impact_low` | number | Conditional | For triangular impact |
| `impact_mode` | number | Conditional | For triangular impact |
| `impact_high` | number | Conditional | For triangular impact |
| `impact_min` | number | Conditional | For PERT impact |
| `impact_most_likely` | number | Conditional | For PERT impact |
| `impact_max` | number | Conditional | For PERT impact |
| `impact_mean` | number | Conditional | For normal impact |
| `impact_stdev` | number | Conditional | For normal impact |
| `affected_wbs` | string | Yes | Comma-separated WBS IDs |
| `description` | string | No | Risk description |

### Correlations Sheet

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `var1` | string | Yes | First variable ID |
| `var2` | string | Yes | Second variable ID |
| `rho` | number | Yes | Correlation coefficient |
| `method` | string | Yes | Correlation method |
| `rationale` | string | No | Justification |

## CSV File Format

### Multiple Files Required

| File Name | Description |
|-----------|-------------|
| `project.csv` | Project information (single row) |
| `wbs_items.csv` | WBS cost items (multiple rows) |
| `risks.csv` | Risk data (multiple rows) |
| `correlations.csv` | Correlation pairs (multiple rows) |

Column specifications are identical to Excel format.

## JSON File Format

### Single File Structure

```json
{
  "project_info": {
    "name": "Project Name",
    "category": "transmission_line",
    // ... other project fields
  },
  "wbs_items": [
    {
      "id": "item1",
      "name": "Item 1",
      "base_cost": 1000000,
      "distribution": {
        "type": "triangular",
        "low": 900000,
        "mode": 1000000,
        "high": 1200000
      }
    }
    // ... more WBS items
  ],
  "risks": [
    {
      "id": "risk1",
      "name": "Risk 1",
      "category": "technical",
      "probability": 0.3,
      "impact_distribution": {
        "type": "pert",
        "min": 50000,
        "most_likely": 100000,
        "max": 200000
      },
      "affected_wbs": ["item1"]
    }
    // ... more risks
  ],
  "correlations": [
    {
      "pair": ["item1", "item2"],
      "rho": 0.6,
      "method": "spearman"
    }
    // ... more correlations
  ]
}
```

### Separate Files Option

Alternative JSON format uses separate files:
- `project.json` - Project data only
- `risks.json` - Risk array only
- `correlations.json` - Correlation array only

## Validation Rules

### General Rules

1. **Required Fields**: All required fields must be present and non-null
2. **Data Types**: Values must match specified types
3. **Constraints**: Numeric constraints must be satisfied
4. **References**: WBS IDs referenced in risks and correlations must exist
5. **Uniqueness**: IDs must be unique within their category

### Distribution Validation

1. **Parameter Completeness**: All required parameters for distribution type must be present
2. **Parameter Relationships**: Ordering constraints must be satisfied (e.g., low ≤ mode ≤ high)
3. **Positive Values**: Cost-related values must be positive
4. **Probability Sums**: Discrete distribution probabilities must sum to 1.0

### Correlation Validation

1. **Range**: Correlation coefficients must be between -1.0 and +1.0
2. **Matrix Properties**: Correlation matrix must be positive semi-definite
3. **No Self-Correlation**: Variables cannot be correlated with themselves
4. **Duplicate Prevention**: Each pair should only appear once

### Risk Validation

1. **Probability Range**: Probabilities must be between 0.0 and 1.0
2. **WBS References**: All referenced WBS items must exist
3. **Impact Distributions**: Impact distributions must be valid
4. **Category Values**: Risk categories must be from valid set

## File Size Limits

| Format | Recommended Max | Hard Limit |
|--------|----------------|------------|
| Excel | 100 WBS items, 50 risks | 1000 WBS items, 500 risks |
| CSV | 500 WBS items, 200 risks | 5000 WBS items, 2000 risks |
| JSON | 1000 WBS items, 500 risks | 10000 WBS items, 5000 risks |

## Encoding Requirements

- **Excel**: Native Excel format (.xlsx)
- **CSV**: UTF-8 encoding required
- **JSON**: UTF-8 encoding required

## Backward Compatibility

The tool maintains backward compatibility with previous versions:
- Optional fields can be omitted (defaults applied)
- New fields are ignored in older file formats
- Distribution parameter names are mapped automatically

## Example Files

See the `examples/` directory for complete example files in all formats:
- `transmission_line_example.xlsx`
- `substation_example.json`  
- `csv_example/` directory with multiple CSV files

These examples demonstrate proper data formatting and can be used as templates for new projects.