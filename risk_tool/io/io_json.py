"""JSON file I/O operations for configuration and results.

Handles JSON serialization with pydantic models and audit trails.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, date
from pydantic import BaseModel, validator
import hashlib


class ConfigurationSchema(BaseModel):
    """Configuration schema for JSON validation."""

    iterations: int = 50000
    sampling: str = "LHS"
    random_seed: Optional[int] = 20250829
    currency: str = "USD"
    outputs: Dict[str, Any] = {
        "percentiles": [10, 50, 80, 90, 95],
        "charts": ["histogram", "cdf_curve", "tornado", "contribution_pareto"],
        "export_formats": ["csv", "xlsx", "json", "pdf"],
    }
    validation: Dict[str, bool] = {
        "fail_on_missing_wbs": True,
        "warn_on_unmapped_risk": True,
    }
    performance: Dict[str, Any] = {"parallel": "auto", "num_threads": -1}


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays and other types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif hasattr(obj, "dict") and callable(obj.dict):
            # Pydantic model
            return obj.dict()
        return super().default(obj)


class JSONImporter:
    """Imports configuration and data from JSON files."""

    def import_configuration(self, file_path: str) -> Dict[str, Any]:
        """Import simulation configuration from JSON file.

        Args:
            file_path: Path to JSON configuration file

        Returns:
            Configuration dictionary
        """
        try:
            with open(file_path, "r") as f:
                config_data = json.load(f)

            # Validate against schema
            config = ConfigurationSchema(**config_data)

            return config.dict()

        except Exception as e:
            raise ValueError(f"Error importing configuration from {file_path}: {e}")

    def import_project(self, file_path: str) -> Dict[str, Any]:
        """Import project data from JSON file.

        Args:
            file_path: Path to JSON project file

        Returns:
            Project dictionary
        """
        try:
            with open(file_path, "r") as f:
                project_data = json.load(f)

            # Basic validation
            required_fields = ["id", "type", "wbs"]
            missing = [field for field in required_fields if field not in project_data]
            if missing:
                raise ValueError(f"Missing required project fields: {missing}")

            return project_data

        except Exception as e:
            raise ValueError(f"Error importing project from {file_path}: {e}")

    def import_risks(self, file_path: str) -> List[Dict[str, Any]]:
        """Import risk data from JSON file.

        Args:
            file_path: Path to JSON risk file

        Returns:
            List of risk dictionaries
        """
        try:
            with open(file_path, "r") as f:
                risk_data = json.load(f)

            # Handle both single risk object and list of risks
            if isinstance(risk_data, dict):
                if "risks" in risk_data:
                    risks = risk_data["risks"]
                else:
                    risks = [risk_data]
            elif isinstance(risk_data, list):
                risks = risk_data
            else:
                raise ValueError("Risk data must be a dictionary or list")

            # Basic validation
            for i, risk in enumerate(risks):
                required_fields = ["id", "probability", "impact_mode", "impact_dist"]
                missing = [field for field in required_fields if field not in risk]
                if missing:
                    raise ValueError(f"Risk {i}: Missing required fields: {missing}")

            return risks

        except Exception as e:
            raise ValueError(f"Error importing risks from {file_path}: {e}")

    def import_correlations(self, file_path: str) -> List[Dict[str, Any]]:
        """Import correlation data from JSON file.

        Args:
            file_path: Path to JSON correlation file

        Returns:
            List of correlation dictionaries
        """
        try:
            with open(file_path, "r") as f:
                corr_data = json.load(f)

            # Handle both single correlation and list
            if isinstance(corr_data, dict):
                if "correlations" in corr_data:
                    correlations = corr_data["correlations"]
                else:
                    correlations = [corr_data]
            elif isinstance(corr_data, list):
                correlations = corr_data
            else:
                raise ValueError("Correlation data must be a dictionary or list")

            # Validate correlations
            for i, corr in enumerate(correlations):
                if "pair" not in corr or "rho" not in corr:
                    raise ValueError(
                        f"Correlation {i}: Must have 'pair' and 'rho' fields"
                    )

                if not isinstance(corr["pair"], list) or len(corr["pair"]) != 2:
                    raise ValueError(
                        f"Correlation {i}: 'pair' must be list of 2 elements"
                    )

                if abs(corr["rho"]) > 1:
                    raise ValueError(f"Correlation {i}: 'rho' must be between -1 and 1")

            return correlations

        except Exception as e:
            raise ValueError(f"Error importing correlations from {file_path}: {e}")


class JSONExporter:
    """Exports results and configuration to JSON files."""

    def export_results(
        self,
        results_dict: Dict[str, Any],
        file_path: str,
        include_raw_data: bool = False,
    ) -> None:
        """Export simulation results to JSON file.

        Args:
            results_dict: Dictionary containing all results
            file_path: Output file path
            include_raw_data: Whether to include raw simulation arrays
        """
        try:
            # Create exportable results
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "currency": results_dict.get("currency", "USD"),
                    "iterations": results_dict.get("n_samples", 0),
                }
            }

            # Add summary statistics
            if "summary" in results_dict:
                export_data["summary"] = results_dict["summary"]

            # Add percentiles
            if "percentiles" in results_dict:
                export_data["percentiles"] = results_dict["percentiles"]

            # Add WBS breakdown
            if "wbs_statistics" in results_dict:
                export_data["wbs_breakdown"] = results_dict["wbs_statistics"]

            # Add risk analysis
            if "risk_contributions" in results_dict:
                export_data["risk_analysis"] = results_dict["risk_contributions"]

            # Add sensitivity analysis
            if "sensitivity_analysis" in results_dict:
                export_data["sensitivity_analysis"] = results_dict[
                    "sensitivity_analysis"
                ]

            # Add contingency recommendations
            if "contingency" in results_dict:
                export_data["contingency"] = results_dict["contingency"]

            # Optionally include raw simulation data (limited)
            if include_raw_data and "simulation_data" in results_dict:
                sim_data = results_dict["simulation_data"]

                # Include sample of raw data to limit file size
                if isinstance(sim_data, dict) and "total_costs" in sim_data:
                    sample_size = min(1000, len(sim_data["total_costs"]))
                    indices = np.linspace(
                        0, len(sim_data["total_costs"]) - 1, sample_size, dtype=int
                    )

                    export_data["simulation_sample"] = {
                        "total_costs": sim_data["total_costs"][indices].tolist(),
                        "sample_size": sample_size,
                        "total_iterations": len(sim_data["total_costs"]),
                    }

            # Add audit information
            export_data["audit"] = self._create_audit_info(results_dict)

            # Write to file
            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2, cls=JSONEncoder)

        except Exception as e:
            raise ValueError(f"Error exporting results to {file_path}: {e}")

    def export_configuration(self, config: Dict[str, Any], file_path: str) -> None:
        """Export configuration to JSON file.

        Args:
            config: Configuration dictionary
            file_path: Output file path
        """
        try:
            # Validate configuration
            validated_config = ConfigurationSchema(**config)

            # Add metadata
            export_data = {
                "metadata": {
                    "created_timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                },
                "configuration": validated_config.dict(),
            }

            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2, cls=JSONEncoder)

        except Exception as e:
            raise ValueError(f"Error exporting configuration to {file_path}: {e}")

    def export_project(self, project: Dict[str, Any], file_path: str) -> None:
        """Export project data to JSON file.

        Args:
            project: Project dictionary
            file_path: Output file path
        """
        try:
            # Add metadata
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                },
                "project": project,
            }

            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2, cls=JSONEncoder)

        except Exception as e:
            raise ValueError(f"Error exporting project to {file_path}: {e}")

    def export_risks(
        self,
        risks: List[Dict[str, Any]],
        file_path: str,
        include_correlations: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Export risk data to JSON file.

        Args:
            risks: List of risk dictionaries
            file_path: Output file path
            include_correlations: Optional correlations to include
        """
        try:
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "risk_count": len(risks),
                },
                "risks": risks,
            }

            if include_correlations:
                export_data["correlations"] = include_correlations

            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2, cls=JSONEncoder)

        except Exception as e:
            raise ValueError(f"Error exporting risks to {file_path}: {e}")

    def create_example_configuration(self, file_path: str) -> None:
        """Create example configuration JSON file.

        Args:
            file_path: Output file path
        """
        try:
            example_config = {
                "iterations": 50000,
                "sampling": "LHS",
                "random_seed": 20250829,
                "currency": "USD",
                "outputs": {
                    "percentiles": [10, 50, 80, 90, 95],
                    "charts": [
                        "histogram",
                        "cdf_curve",
                        "tornado",
                        "contribution_pareto",
                    ],
                    "export_formats": ["csv", "xlsx", "json", "pdf"],
                },
                "validation": {
                    "fail_on_missing_wbs": True,
                    "warn_on_unmapped_risk": True,
                },
                "performance": {"parallel": "auto", "num_threads": -1},
            }

            self.export_configuration(example_config, file_path)

        except Exception as e:
            raise ValueError(
                f"Error creating example configuration at {file_path}: {e}"
            )

    def _create_audit_info(self, results_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create audit information for results."""
        audit_info = {"timestamp": datetime.now().isoformat(), "version": "1.0.0"}

        # Add configuration hash if available
        if "config" in results_dict:
            config_str = json.dumps(results_dict["config"], sort_keys=True)
            audit_info["config_hash"] = hashlib.sha256(config_str.encode()).hexdigest()

        # Add random seed if available
        if "random_seed" in results_dict:
            audit_info["random_seed"] = results_dict["random_seed"]

        # Add file hashes if available
        if "input_files" in results_dict:
            audit_info["input_files"] = results_dict["input_files"]

        return audit_info


def validate_json_file(file_path: str, schema_type: str = "auto") -> List[str]:
    """Validate JSON file against expected schema.

    Args:
        file_path: Path to JSON file
        schema_type: Type of schema to validate against

    Returns:
        List of validation errors
    """
    errors = []

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Auto-detect schema type if not specified
        if schema_type == "auto":
            if "configuration" in data or "iterations" in data:
                schema_type = "configuration"
            elif "project" in data or "wbs" in data:
                schema_type = "project"
            elif "risks" in data or (
                isinstance(data, list) and "probability" in data[0]
            ):
                schema_type = "risks"
            else:
                schema_type = "unknown"

        # Validate based on schema type
        if schema_type == "configuration":
            try:
                if "configuration" in data:
                    ConfigurationSchema(**data["configuration"])
                else:
                    ConfigurationSchema(**data)
            except Exception as e:
                errors.append(f"Configuration validation: {e}")

        elif schema_type == "project":
            project_data = data.get("project", data)
            required_fields = ["id", "type", "wbs"]
            missing = [field for field in required_fields if field not in project_data]
            if missing:
                errors.append(f"Missing required project fields: {missing}")

        elif schema_type == "risks":
            risks = data.get("risks", data if isinstance(data, list) else [])
            for i, risk in enumerate(risks):
                required_fields = ["id", "probability", "impact_mode", "impact_dist"]
                missing = [field for field in required_fields if field not in risk]
                if missing:
                    errors.append(f"Risk {i}: Missing required fields: {missing}")

    except json.JSONDecodeError as e:
        errors.append(f"JSON parsing error: {e}")
    except Exception as e:
        errors.append(f"File reading error: {e}")

    return errors


def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file for audit purposes.

    Args:
        file_path: Path to file

    Returns:
        SHA256 hash string
    """
    hash_sha256 = hashlib.sha256()

    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
    except Exception:
        return ""

    return hash_sha256.hexdigest()
