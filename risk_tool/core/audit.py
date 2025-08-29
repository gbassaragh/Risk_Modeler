"""Audit trail and determinism tracking.

Provides comprehensive audit logging and reproducibility verification.
"""

import json
import hashlib
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np


class AuditLogger:
    """Comprehensive audit logging system."""

    def __init__(self):
        """Initialize audit logger."""
        self.session_id = self._generate_session_id()
        self.audit_entries = []
        self.start_time = time.time()

    def log_simulation_start(
        self,
        config: Dict[str, Any],
        project_id: str,
        input_files: Optional[Dict[str, str]] = None,
    ) -> str:
        """Log simulation start with environment and inputs.

        Args:
            config: Simulation configuration
            project_id: Project identifier
            input_files: Optional mapping of file types to paths

        Returns:
            Audit entry ID
        """
        entry_id = self._generate_entry_id()

        # Calculate input file hashes
        file_hashes = {}
        if input_files:
            for file_type, file_path in input_files.items():
                try:
                    file_hashes[file_type] = self._calculate_file_hash(file_path)
                except Exception as e:
                    file_hashes[file_type] = f"ERROR: {e}"

        entry = {
            "entry_id": entry_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "simulation_start",
            "session_id": self.session_id,
            "data": {
                "project_id": project_id,
                "configuration": config,
                "input_files": input_files or {},
                "file_hashes": file_hashes,
                "environment": self._capture_environment(),
                "versions": self._capture_versions(),
            },
        }

        self.audit_entries.append(entry)
        return entry_id

    def log_simulation_end(
        self, results_summary: Dict[str, Any], performance_metrics: Dict[str, float]
    ) -> str:
        """Log simulation completion with results and performance.

        Args:
            results_summary: Summary of simulation results
            performance_metrics: Performance timing and metrics

        Returns:
            Audit entry ID
        """
        entry_id = self._generate_entry_id()

        entry = {
            "entry_id": entry_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "simulation_end",
            "session_id": self.session_id,
            "data": {
                "results_summary": results_summary,
                "performance_metrics": performance_metrics,
                "total_session_time": time.time() - self.start_time,
            },
        }

        self.audit_entries.append(entry)
        return entry_id

    def log_validation_results(
        self,
        validation_type: str,
        is_valid: bool,
        errors: List[str],
        warnings: List[str],
    ) -> str:
        """Log validation results.

        Args:
            validation_type: Type of validation performed
            is_valid: Whether validation passed
            errors: List of validation errors
            warnings: List of validation warnings

        Returns:
            Audit entry ID
        """
        entry_id = self._generate_entry_id()

        entry = {
            "entry_id": entry_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "validation",
            "session_id": self.session_id,
            "data": {
                "validation_type": validation_type,
                "is_valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "error_count": len(errors),
                "warning_count": len(warnings),
            },
        }

        self.audit_entries.append(entry)
        return entry_id

    def log_sampling_info(
        self,
        method: str,
        n_samples: int,
        n_dimensions: int,
        random_seed: Optional[int],
        convergence_metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Log sampling configuration and diagnostics.

        Args:
            method: Sampling method used
            n_samples: Number of samples
            n_dimensions: Number of dimensions
            random_seed: Random seed used
            convergence_metrics: Optional convergence diagnostics

        Returns:
            Audit entry ID
        """
        entry_id = self._generate_entry_id()

        entry = {
            "entry_id": entry_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "sampling",
            "session_id": self.session_id,
            "data": {
                "method": method,
                "n_samples": n_samples,
                "n_dimensions": n_dimensions,
                "random_seed": random_seed,
                "convergence_metrics": convergence_metrics or {},
            },
        }

        self.audit_entries.append(entry)
        return entry_id

    def log_correlation_application(
        self,
        correlations: List[Dict[str, Any]],
        achieved_correlations: Optional[Dict[str, float]] = None,
    ) -> str:
        """Log correlation application and validation.

        Args:
            correlations: Correlation specifications
            achieved_correlations: Actual achieved correlations

        Returns:
            Audit entry ID
        """
        entry_id = self._generate_entry_id()

        entry = {
            "entry_id": entry_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "correlation",
            "session_id": self.session_id,
            "data": {
                "target_correlations": correlations,
                "achieved_correlations": achieved_correlations or {},
                "correlation_count": len(correlations),
            },
        }

        self.audit_entries.append(entry)
        return entry_id

    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log error occurrence.

        Args:
            error_type: Type/category of error
            error_message: Error message
            context: Optional context information

        Returns:
            Audit entry ID
        """
        entry_id = self._generate_entry_id()

        entry = {
            "entry_id": entry_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "error",
            "session_id": self.session_id,
            "data": {
                "error_type": error_type,
                "error_message": error_message,
                "context": context or {},
            },
        }

        self.audit_entries.append(entry)
        return entry_id

    def export_audit_log(self, file_path: str) -> None:
        """Export complete audit log to file.

        Args:
            file_path: Output file path
        """
        audit_data = {
            "metadata": {
                "session_id": self.session_id,
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_entries": len(self.audit_entries),
                "session_duration": time.time() - self.start_time,
                "audit_version": "1.0.0",
            },
            "entries": self.audit_entries,
        }

        with open(file_path, "w") as f:
            json.dump(audit_data, f, indent=2, default=self._json_serializer)

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current audit session.

        Returns:
            Session summary dictionary
        """
        event_counts = {}
        for entry in self.audit_entries:
            event_type = entry.get("event_type", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "duration": time.time() - self.start_time,
            "total_entries": len(self.audit_entries),
            "event_counts": event_counts,
            "has_errors": any(
                entry.get("event_type") == "error" for entry in self.audit_entries
            ),
        }

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        random_component = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
        return f"SIM_{timestamp}_{random_component}"

    def _generate_entry_id(self) -> str:
        """Generate unique entry ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        entry_num = len(self.audit_entries) + 1
        return f"ENTRY_{timestamp}_{entry_num:04d}"

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            return f"ERROR: {e}"

    def _capture_environment(self) -> Dict[str, Any]:
        """Capture environment information."""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        }

    def _capture_versions(self) -> Dict[str, str]:
        """Capture package versions."""
        versions = {}

        # Core packages
        packages = [
            "numpy",
            "pandas",
            "scipy",
            "matplotlib",
            "plotly",
            "openpyxl",
            "pydantic",
            "typer",
            "numba",
        ]

        for package in packages:
            try:
                module = __import__(package)
                versions[package] = getattr(module, "__version__", "unknown")
            except ImportError:
                versions[package] = "not_installed"

        return versions

    def _json_serializer(self, obj):
        """JSON serializer for special types."""
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)


class DeterminismVerifier:
    """Verifies simulation determinism and reproducibility."""

    @staticmethod
    def verify_reproducibility(
        config: Dict[str, Any],
        project_data: Dict[str, Any],
        risks_data: Optional[List[Dict[str, Any]]] = None,
        n_runs: int = 3,
    ) -> Dict[str, Any]:
        """Verify that simulation produces identical results with same seed.

        Args:
            config: Simulation configuration
            project_data: Project data
            risks_data: Risk data
            n_runs: Number of verification runs

        Returns:
            Verification results
        """
        if not config.get("random_seed"):
            return {
                "reproducible": False,
                "reason": "No random seed specified",
                "runs": [],
            }

        results = []

        try:
            from .aggregation import run_simulation

            for run_num in range(n_runs):
                # Run simulation with same inputs
                result = run_simulation(project_data, risks_data, config)

                # Extract key metrics for comparison
                run_summary = {
                    "run_number": run_num + 1,
                    "total_costs_hash": DeterminismVerifier._hash_array(
                        result.total_costs
                    ),
                    "p50": result.percentiles.get("P50", 0),
                    "p80": result.percentiles.get("P80", 0),
                    "base_cost": result.base_cost,
                    "n_samples": result.n_samples,
                }

                results.append(run_summary)

            # Check if all runs produced identical results
            first_hash = results[0]["total_costs_hash"]
            all_identical = all(r["total_costs_hash"] == first_hash for r in results)

            verification_result = {
                "reproducible": all_identical,
                "runs": results,
                "config_seed": config.get("random_seed"),
                "verification_timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if not all_identical:
                verification_result["reason"] = (
                    "Results differ between runs with same seed"
                )

            return verification_result

        except Exception as e:
            return {
                "reproducible": False,
                "reason": f"Verification failed: {e}",
                "runs": results,
            }

    @staticmethod
    def _hash_array(arr: List[float]) -> str:
        """Create hash of numeric array for comparison."""
        # Convert to numpy array and create consistent string representation
        np_arr = np.array(arr)
        # Use fixed precision to avoid floating point differences
        arr_str = np.array2string(np_arr, precision=10, separator=",")
        return hashlib.sha256(arr_str.encode()).hexdigest()[:16]


class ComplianceReporter:
    """Generates compliance reports for audit purposes."""

    @staticmethod
    def generate_compliance_report(
        audit_log: AuditLogger,
        simulation_results: Dict[str, Any],
        requirements: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate compliance report against requirements.

        Args:
            audit_log: Audit logger with session data
            simulation_results: Simulation results
            requirements: Compliance requirements

        Returns:
            Compliance report
        """
        session_summary = audit_log.get_session_summary()

        compliance_checks = {
            "determinism": ComplianceReporter._check_determinism(audit_log),
            "validation": ComplianceReporter._check_validation(audit_log),
            "documentation": ComplianceReporter._check_documentation(
                audit_log, requirements
            ),
            "methodology": ComplianceReporter._check_methodology(
                simulation_results, requirements
            ),
            "quality_assurance": ComplianceReporter._check_quality(
                audit_log, simulation_results
            ),
        }

        # Overall compliance status
        all_checks_passed = all(
            check.get("compliant", False) for check in compliance_checks.values()
        )

        return {
            "compliance_summary": {
                "overall_compliant": all_checks_passed,
                "report_timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_summary["session_id"],
                "total_issues": sum(
                    len(check.get("issues", [])) for check in compliance_checks.values()
                ),
            },
            "detailed_checks": compliance_checks,
            "recommendations": ComplianceReporter._generate_recommendations(
                compliance_checks
            ),
        }

    @staticmethod
    def _check_determinism(audit_log: AuditLogger) -> Dict[str, Any]:
        """Check determinism compliance."""
        # Look for random seed configuration
        sampling_entries = [
            e for e in audit_log.audit_entries if e.get("event_type") == "sampling"
        ]

        if not sampling_entries:
            return {
                "compliant": False,
                "issues": ["No sampling configuration found in audit log"],
                "details": {},
            }

        sampling_data = sampling_entries[0]["data"]
        random_seed = sampling_data.get("random_seed")

        if random_seed is None:
            return {
                "compliant": False,
                "issues": ["No random seed specified - results are not reproducible"],
                "details": {"random_seed": None},
            }

        return {
            "compliant": True,
            "issues": [],
            "details": {
                "random_seed": random_seed,
                "sampling_method": sampling_data.get("method"),
            },
        }

    @staticmethod
    def _check_validation(audit_log: AuditLogger) -> Dict[str, Any]:
        """Check validation compliance."""
        validation_entries = [
            e for e in audit_log.audit_entries if e.get("event_type") == "validation"
        ]

        if not validation_entries:
            return {
                "compliant": False,
                "issues": ["No validation performed"],
                "details": {},
            }

        issues = []
        validation_details = {}

        for entry in validation_entries:
            data = entry["data"]
            validation_type = data["validation_type"]
            validation_details[validation_type] = {
                "is_valid": data["is_valid"],
                "error_count": data["error_count"],
                "warning_count": data["warning_count"],
            }

            if not data["is_valid"]:
                issues.append(
                    f"{validation_type} validation failed with {data['error_count']} errors"
                )

        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "details": validation_details,
        }

    @staticmethod
    def _check_documentation(
        audit_log: AuditLogger, requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check documentation compliance."""
        session_summary = audit_log.get_session_summary()
        required_docs = requirements.get("documentation", [])

        issues = []

        # Check if audit trail is complete
        if session_summary["total_entries"] == 0:
            issues.append("No audit trail recorded")

        # Check for required documentation types
        event_types = session_summary.get("event_counts", {})

        for required_doc in required_docs:
            if required_doc not in event_types:
                issues.append(f"Required documentation type missing: {required_doc}")

        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "details": {
                "total_entries": session_summary["total_entries"],
                "event_types": list(event_types.keys()),
            },
        }

    @staticmethod
    def _check_methodology(
        simulation_results: Dict[str, Any], requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check methodology compliance."""
        issues = []

        # Check minimum iterations
        min_iterations = requirements.get("min_iterations", 10000)
        actual_iterations = simulation_results.get("n_samples", 0)

        if actual_iterations < min_iterations:
            issues.append(
                f"Insufficient iterations: {actual_iterations} < {min_iterations}"
            )

        # Check required percentiles
        required_percentiles = requirements.get(
            "required_percentiles", ["P50", "P80", "P90"]
        )
        actual_percentiles = simulation_results.get("percentiles", {})

        for percentile in required_percentiles:
            if percentile not in actual_percentiles:
                issues.append(f"Required percentile missing: {percentile}")

        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "details": {
                "iterations": actual_iterations,
                "percentiles": list(actual_percentiles.keys()),
            },
        }

    @staticmethod
    def _check_quality(
        audit_log: AuditLogger, simulation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check quality assurance compliance."""
        session_summary = audit_log.get_session_summary()
        issues = []

        # Check for errors during simulation
        if session_summary.get("has_errors", False):
            error_entries = [
                e for e in audit_log.audit_entries if e.get("event_type") == "error"
            ]
            issues.append(f"Simulation had {len(error_entries)} errors")

        # Check convergence (if available)
        convergence_data = None
        sampling_entries = [
            e for e in audit_log.audit_entries if e.get("event_type") == "sampling"
        ]

        for entry in sampling_entries:
            if "convergence_metrics" in entry["data"]:
                convergence_data = entry["data"]["convergence_metrics"]
                break

        if convergence_data:
            converged = convergence_data.get("converged", False)
            if not converged:
                issues.append("Simulation did not converge to required tolerance")

        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "details": {
                "has_errors": session_summary.get("has_errors", False),
                "convergence_data": convergence_data,
            },
        }

    @staticmethod
    def _generate_recommendations(
        compliance_checks: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Generate recommendations based on compliance check results."""
        recommendations = []

        for check_name, check_result in compliance_checks.items():
            if not check_result.get("compliant", True):
                issues = check_result.get("issues", [])

                if check_name == "determinism" and any(
                    "random seed" in issue for issue in issues
                ):
                    recommendations.append(
                        "Set a fixed random seed for reproducible results"
                    )

                if check_name == "validation" and any(
                    "validation failed" in issue for issue in issues
                ):
                    recommendations.append(
                        "Review and fix input data validation errors before running simulation"
                    )

                if check_name == "methodology" and any(
                    "iterations" in issue for issue in issues
                ):
                    recommendations.append(
                        "Increase number of Monte Carlo iterations for more stable results"
                    )

                if check_name == "quality_assurance" and any(
                    "converge" in issue for issue in issues
                ):
                    recommendations.append(
                        "Investigate convergence issues and consider increasing sample size"
                    )

        return recommendations
