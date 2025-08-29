# Risk_Modeler Data Dictionary

Comprehensive reference for all data structures, file formats, and API schemas used in Risk_Modeler.

## Table of Contents

1. [Project Configuration](#project-configuration)
2. [WBS Items](#wbs-items)
3. [Distribution Specifications](#distribution-specifications)
4. [Correlation Definitions](#correlation-definitions)
5. [Risk Events](#risk-events)
6. [Simulation Configuration](#simulation-configuration)
7. [File Formats](#file-formats)
8. [API Schemas](#api-schemas)
9. [Validation Rules](#validation-rules)
10. [Examples](#examples)

## Project Configuration

### ProjectInfo Schema

Complete project metadata and configuration parameters.

| Field | Type | Required | Description | Valid Values |
|-------|------|----------|-------------|--------------|
| `name` | string | Yes | Project name or identifier | Any string, 1-200 chars |
| `description` | string | No | Project description | Any string, max 1000 chars |
| `project_type` | string | Yes | Type of T&D project | `TransmissionLine`, `Substation`, `Hybrid` |
| `voltage_class` | string | Yes | Operating voltage level | e.g., `138kV`, `138kV/12.47kV` |
| `region` | string | No | Geographic region | Any string |
| `base_year` | string | Yes | Base cost year | YYYY format, e.g., `2025` |
| `currency` | string | Yes | Cost currency | `USD`, `CAD`, `EUR` |
| `aace_class` | string | No | AACE estimate class | `Class 1`, `Class 2`, `Class 3`, `Class 4`, `Class 5` |
| `contingency_target` | number | No | Target contingency percentage | 0.0-1.0 (e.g., 0.25 = 25%) |

#### Transmission Line Specific Fields

| Field | Type | Required | Description | Valid Values |
|-------|------|----------|-------------|--------------|
| `length_miles` | number | Yes | Line length in miles | > 0.0 |
| `circuit_count` | integer | No | Number of circuits | ≥ 1, default: 1 |
| `terrain_type` | string | No | Terrain characteristics | `flat`, `hilly`, `mixed` |

#### Substation Specific Fields

| Field | Type | Required | Description | Valid Values |
|-------|------|----------|-------------|--------------|
| `capacity_mva` | number | Yes | Transformer capacity in MVA | > 0.0 |
| `bay_count` | integer | No | Number of bays | ≥ 1 |
| `substation_type` | string | No | Substation classification | `transmission`, `distribution` |

### Example

```json
{
  "project_info": {
    "name": "Oakville 138kV Transmission Project",
    "description": "25-mile 138kV transmission line with new switching station",
    "project_type": "TransmissionLine",
    "voltage_class": "138kV",
    "length_miles": 24.8,
    "circuit_count": 1,
    "terrain_type": "mixed",
    "region": "Northeast",
    "base_year": "2025",
    "currency": "USD",
    "aace_class": "Class 3",
    "contingency_target": 0.25
  }
}
```

## WBS Items

### WBSItem Schema

Work Breakdown Structure items represent individual cost elements with uncertainty distributions.

| Field | Type | Required | Description | Valid Values |
|-------|------|----------|-------------|--------------|
| `code` | string | Yes | Unique WBS identifier | Alphanumeric, recommend XX-YYYY format |
| `name` | string | Yes | Descriptive name | 1-200 characters |
| `quantity` | number | Yes | Item quantity | ≥ 0.0 |
| `uom` | string | Yes | Unit of measure | e.g., `miles`, `structures`, `project` |
| `unit_cost` | number | Yes | Base unit cost | ≥ 0.0 |
| `dist_quantity` | object | No | Quantity uncertainty distribution | See Distribution Specifications |
| `dist_unit_cost` | object | No | Unit cost uncertainty distribution | See Distribution Specifications |
| `tags` | array[string] | No | Classification tags | Array of strings |
| `indirect_factor` | number | No | Indirect cost multiplier | 0.0-2.0, default: 0.0 |

### Common WBS Codes

#### Transmission Line Projects

| Code | Category | Description |
|------|----------|-------------|
| `01-ROW` | Land | Right-of-way acquisition and legal |
| `02-CLEAR` | Site Prep | Clearing and access roads |
| `03-STRUCT` | Structures | Transmission structures and hardware |
| `04-FOUND` | Foundations | Structure foundations |
| `05-COND` | Conductor | Conductor and hardware |
| `06-OPGW` | Communications | OPGW fiber cable |
| `07-TERM` | Terminal | Terminal equipment and switchgear |
| `08-MISC` | Indirect | Engineering and project management |

#### Substation Projects  

| Code | Category | Description |
|------|----------|-------------|
| `01-SITE` | Site | Site preparation and civil works |
| `02-XFMR` | Power | Power transformers |
| `03-138SW` | HV Switch | High voltage switchgear |
| `04-12SW` | LV Switch | Low voltage switchgear |
| `05-PROT` | Control | Protection and control systems |
| `06-STRUCT` | Structures | Steel structures and bus |
| `07-CABLE` | Cables | Power and control cables |
| `08-MISC` | Indirect | Engineering and project management |

### Example

```json
{
  "wbs_items": [
    {
      "code": "03-STRUCT",
      "name": "Transmission Structures",
      "quantity": 125.0,
      "uom": "structures",
      "unit_cost": 8000,
      "dist_unit_cost": {
        "type": "triangular",
        "low": 7000,
        "mode": 8000,
        "high": 12000
      },
      "tags": ["steel", "foundation", "erection"],
      "indirect_factor": 0.12
    }
  ]
}
```

## Distribution Specifications

### Supported Distribution Types

#### Triangular Distribution

Asymmetric distribution defined by three points: minimum, mode (most likely), and maximum.

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|-------------|
| `type` | string | Yes | Distribution type | Must be `"triangular"` |
| `low` | number | Yes | Minimum value | < mode |
| `mode` | number | Yes | Most likely value | low < mode < high |
| `high` | number | Yes | Maximum value | > mode |

```json
{
  "type": "triangular",
  "low": 7000,
  "mode": 8000, 
  "high": 12000
}
```

#### PERT Distribution

Beta distribution based on three-point estimates with emphasis on the most likely value.

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|-------------|
| `type` | string | Yes | Distribution type | Must be `"pert"` |
| `min` | number | Yes | Minimum value | < most_likely |
| `most_likely` | number | Yes | Most likely value | min < most_likely < max |
| `max` | number | Yes | Maximum value | > most_likely |
| `lambda` | number | No | Shape parameter | > 0, default: 4.0 |

```json
{
  "type": "pert",
  "min": 120000,
  "most_likely": 150000,
  "max": 200000,
  "lambda": 4.0
}
```

#### Normal Distribution

Symmetric bell-curve distribution with optional truncation.

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|-------------|
| `type` | string | Yes | Distribution type | Must be `"normal"` |
| `mean` | number | Yes | Mean (center) value | Any value |
| `stdev` | number | Yes | Standard deviation | > 0 |
| `truncate_low` | number | No | Lower truncation bound | < mean |
| `truncate_high` | number | No | Upper truncation bound | > mean |

```json
{
  "type": "normal",
  "mean": 45000,
  "stdev": 6000,
  "truncate_low": 35000,
  "truncate_high": 60000
}
```

#### Log-Normal Distribution

Right-skewed distribution for multiplicative processes.

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|-------------|
| `type` | string | Yes | Distribution type | Must be `"lognormal"` |
| `mean` | number | Yes | Mean of underlying normal | Any value |
| `sigma` | number | Yes | Standard deviation of underlying normal | > 0 |

```json
{
  "type": "lognormal",
  "mean": 10.5,
  "sigma": 0.3
}
```

#### Uniform Distribution

Constant probability across a range.

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|-------------|
| `type` | string | Yes | Distribution type | Must be `"uniform"` |
| `low` | number | Yes | Minimum value | < high |
| `high` | number | Yes | Maximum value | > low |

```json
{
  "type": "uniform",
  "low": 10000,
  "high": 15000
}
```

#### Discrete Distribution

Custom discrete distribution with specific value-probability pairs.

| Parameter | Type | Required | Description | Constraints |
|-----------|------|----------|-------------|-------------|
| `type` | string | Yes | Distribution type | Must be `"discrete"` |
| `pmf` | array | Yes | Probability mass function | Array of [value, probability] pairs |

```json
{
  "type": "discrete",
  "pmf": [
    [10000, 0.3],
    [15000, 0.5], 
    [20000, 0.2]
  ]
}
```

## Correlation Definitions

### Correlation Schema

Defines statistical relationships between WBS items or risk events.

| Field | Type | Required | Description | Valid Values |
|-------|------|----------|-------------|--------------|
| `items` | array[string] | Yes | WBS codes to correlate | Array of 2 valid WBS codes |
| `correlation` | number | Yes | Correlation coefficient | -1.0 to 1.0 |
| `method` | string | No | Correlation method | `spearman`, `pearson` (default: `spearman`) |

### Correlation Strength Guidelines

| Range | Interpretation | Typical Use Cases |
|-------|----------------|-------------------|
| 0.8-1.0 | Very Strong | Same work crew, identical equipment |
| 0.6-0.8 | Strong | Related construction activities |
| 0.4-0.6 | Moderate | Shared market factors |
| 0.2-0.4 | Weak | Indirect relationships |
| 0.0-0.2 | Very Weak | Independent activities |

### Example

```json
{
  "correlations": [
    {
      "items": ["03-STRUCT", "04-FOUND"],
      "correlation": 0.8,
      "method": "spearman"
    },
    {
      "items": ["05-COND", "06-OPGW"], 
      "correlation": 0.6,
      "method": "spearman"
    }
  ]
}
```

## Risk Events

### RiskEvent Schema

Discrete risk events with probability of occurrence and impact distributions.

| Field | Type | Required | Description | Valid Values |
|-------|------|----------|-------------|--------------|
| `id` | string | Yes | Unique risk identifier | Alphanumeric, recommend PREFIX-XXX |
| `name` | string | Yes | Risk event name | 1-200 characters |
| `description` | string | No | Detailed description | Max 1000 characters |
| `category` | string | Yes | Risk category | `schedule`, `technical`, `regulatory`, `market` |
| `probability` | number | Yes | Occurrence probability | 0.0-1.0 |
| `impact_distribution` | object | Yes | Impact cost distribution | See Distribution Specifications |
| `affected_wbs` | array[string] | No | WBS items affected | Array of valid WBS codes |
| `impact_type` | string | No | Type of impact | `cost_increase`, `schedule_cost`, `cost_decrease` |
| `mitigation_strategy` | string | No | Mitigation approach | Free text |
| `risk_owner` | string | No | Responsible party | Free text |

### Risk Categories

| Category | Description | Typical Examples |
|----------|-------------|------------------|
| `schedule` | Time-related delays | Weather, permitting delays |
| `technical` | Engineering/construction issues | Design changes, equipment failures |
| `regulatory` | Compliance and permitting | Environmental requirements, code changes |
| `market` | Economic and market factors | Material price volatility, labor shortages |

### Example

```json
{
  "risk_events": [
    {
      "id": "WEATHER-001",
      "name": "Weather-Related Construction Delays",
      "description": "Extended periods of severe weather causing construction delays",
      "category": "schedule",
      "probability": 0.4,
      "impact_distribution": {
        "type": "pert",
        "min": 50000,
        "most_likely": 150000,
        "max": 400000
      },
      "affected_wbs": ["02-CLEAR", "03-STRUCT", "05-COND"],
      "impact_type": "cost_increase",
      "mitigation_strategy": "Weather contingency planning, flexible scheduling",
      "risk_owner": "Project Manager"
    }
  ]
}
```

## Simulation Configuration

### SimulationConfig Schema

Monte Carlo simulation parameters and settings.

| Field | Type | Required | Description | Valid Values |
|-------|------|----------|-------------|--------------|
| `iterations` | integer | Yes | Number of Monte Carlo iterations | 1,000-1,000,000 |
| `random_seed` | integer | No | Random number seed | Any integer |
| `sampling_method` | string | No | Sampling technique | `LHS`, `MCS` (default: `LHS`) |
| `convergence_threshold` | number | No | Convergence stopping criteria | 0.001-0.1 (default: 0.01) |
| `max_iterations` | integer | No | Maximum iteration limit | ≥ iterations (default: 100,000) |
| `confidence_levels` | array[number] | No | Percentiles to calculate | Array of 0.0-1.0 values |

### Recommended Settings

| Project Size | Iterations | Method | Expected Runtime |
|--------------|------------|---------|------------------|
| Small (<10 WBS items) | 10,000 | LHS | 2-5 seconds |
| Medium (10-25 WBS items) | 25,000 | LHS | 5-10 seconds |
| Large (25-50 WBS items) | 50,000 | LHS | 10-20 seconds |
| Very Large (50+ WBS items) | 100,000 | LHS | 20-60 seconds |

### Example

```json
{
  "simulation_config": {
    "iterations": 25000,
    "random_seed": 12345,
    "sampling_method": "LHS",
    "convergence_threshold": 0.005,
    "max_iterations": 100000,
    "confidence_levels": [0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
  }
}
```

## File Formats

### JSON Format

Complete project specification in single JSON file:

```json
{
  "project_info": { ... },
  "wbs_items": [ ... ],
  "correlations": [ ... ],
  "risk_events": [ ... ],
  "simulation_config": { ... }
}
```

### Excel Format

Multi-worksheet Excel file with data validation:

| Worksheet | Contents | Required |
|-----------|----------|----------|
| `Project_Info` | Project metadata | Yes |
| `WBS_Items` | Cost breakdown structure | Yes |
| `Correlations` | Correlation definitions | No |
| `Risk_Events` | Risk event definitions | No |
| `Simulation_Config` | Monte Carlo settings | No |
| `Instructions` | User guidance | No |

### CSV Format

Multiple CSV files for modular data management:

| File | Contents | Required |
|------|----------|----------|
| `project_info.csv` | Project metadata (2-column format) | Yes |
| `wbs_items.csv` | WBS items with distribution columns | Yes |
| `correlations.csv` | Correlation pairs | No |
| `risk_events.csv` | Risk event definitions | No |
| `simulation_config.csv` | Configuration parameters | No |

## API Schemas

### Request Schemas

#### RunSimulation Request

```json
{
  "project_data": { ... },
  "simulation_config": { ... },
  "output_format": "json"
}
```

#### Validation Request

```json
{
  "project_data": { ... },
  "validation_level": "strict"
}
```

### Response Schemas

#### Simulation Results

```json
{
  "project_id": "string",
  "timestamp": "ISO8601",
  "currency": "USD",
  "n_samples": 25000,
  "base_cost": 5750000,
  "total_costs": [array of cost samples],
  "percentiles": {
    "p10": 5234000,
    "p50": 5845000,
    "p80": 6456000,
    "p90": 6789000,
    "p95": 7012000
  },
  "wbs_statistics": {
    "01-ROW": {
      "mean": 1234000,
      "stdev": 145000,
      "p80": 1345000
    }
  },
  "convergence_achieved": true,
  "iterations_completed": 25000
}
```

#### Validation Results

```json
{
  "valid": true,
  "errors": [],
  "warnings": [
    {
      "code": "CORRELATION_HIGH",
      "message": "Correlation 0.95 between STRUCT and FOUND is very high",
      "severity": "warning"
    }
  ],
  "summary": {
    "wbs_items": 8,
    "total_base_cost": 5750000,
    "distributions": {
      "triangular": 5,
      "pert": 2,
      "normal": 1
    }
  }
}
```

## Validation Rules

### Project Level

1. **Required Fields**: name, project_type, voltage_class, base_year, currency
2. **Project Type Consistency**: Transmission projects require length_miles; Substations require capacity_mva
3. **Date Format**: base_year must be YYYY format
4. **Currency**: Must be supported currency code

### WBS Level

1. **Unique Codes**: All WBS codes must be unique within project
2. **Non-negative Values**: quantity, unit_cost, indirect_factor ≥ 0
3. **Distribution Validity**: All distribution parameters must be mathematically valid
4. **Tags**: If provided, must be non-empty strings

### Distribution Level

1. **Parameter Constraints**: Each distribution type has specific parameter requirements
2. **Logical Ordering**: For triangular/PERT, low < mode < high
3. **Positive Values**: Standard deviations and scale parameters must be positive
4. **Probability Sums**: Discrete distributions must sum to 1.0

### Correlation Level

1. **Valid Range**: Correlation coefficients must be between -1.0 and 1.0
2. **Existing References**: Items must reference valid WBS codes
3. **Matrix Properties**: Correlation matrix must be positive semi-definite
4. **No Self-Correlation**: Items cannot be correlated with themselves

### Risk Event Level

1. **Probability Range**: Probability must be between 0.0 and 1.0
2. **Impact Distribution**: Must be valid distribution specification
3. **WBS References**: affected_wbs must reference valid WBS codes
4. **Category Values**: Must be from approved category list

## Examples

### Complete Project File

```json
{
  "project_info": {
    "name": "Maple Ridge 138kV Line",
    "description": "25-mile 138kV transmission line construction",
    "project_type": "TransmissionLine",
    "voltage_class": "138kV",
    "length_miles": 24.8,
    "circuit_count": 1,
    "terrain_type": "mixed",
    "region": "Northeast",
    "base_year": "2025",
    "currency": "USD",
    "aace_class": "Class 3",
    "contingency_target": 0.25
  },
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
      "tags": ["land", "legal"],
      "indirect_factor": 0.15
    },
    {
      "code": "03-STRUCT",
      "name": "Transmission Structures",
      "quantity": 125.0,
      "uom": "structures", 
      "unit_cost": 8000,
      "dist_unit_cost": {
        "type": "pert",
        "min": 7000,
        "most_likely": 8000,
        "max": 12000
      },
      "tags": ["steel", "foundation"],
      "indirect_factor": 0.12
    }
  ],
  "correlations": [
    {
      "items": ["01-ROW", "03-STRUCT"],
      "correlation": 0.3,
      "method": "spearman"
    }
  ],
  "risk_events": [
    {
      "id": "WEATHER-001",
      "name": "Weather Delays",
      "category": "schedule",
      "probability": 0.4,
      "impact_distribution": {
        "type": "triangular",
        "low": 50000,
        "mode": 150000,
        "high": 400000
      },
      "affected_wbs": ["01-ROW", "03-STRUCT"]
    }
  ],
  "simulation_config": {
    "iterations": 25000,
    "random_seed": 12345,
    "sampling_method": "LHS",
    "confidence_levels": [0.1, 0.5, 0.8, 0.9, 0.95]
  }
}
```

---

*Risk_Modeler Data Dictionary v1.0.0 - Complete reference for data structures and file formats*