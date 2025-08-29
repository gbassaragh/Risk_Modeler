"""Smoke tests for the main simulation engine.

Tests end-to-end integration of all engine components.
"""

import json
import numpy as np
import pytest
from pathlib import Path
from risk_tool.engine.aggregation import run_simulation
from risk_tool.engine.validation import validate_simulation_inputs


class TestSimulationEngine:
    """Test main simulation engine integration."""

    @pytest.fixture
    def minimal_project(self):
        """Minimal project configuration for testing."""
        return {
            "project_info": {
                "name": "Test Transmission Line",
                "category": "transmission_line",
                "voltage": "138kV",
                "length": 10.0,
                "terrain": "flat",
            },
            "wbs_items": [
                {
                    "id": "structure",
                    "name": "Structures",
                    "base_cost": 100000,
                    "distribution": {
                        "type": "triangular",
                        "low": 90000,
                        "mode": 100000,
                        "high": 120000,
                    },
                },
                {
                    "id": "conductor",
                    "name": "Conductor",
                    "base_cost": 50000,
                    "distribution": {
                        "type": "normal",
                        "mean": 50000,
                        "stdev": 5000,
                        "truncate_low": 40000,
                        "truncate_high": 65000,
                    },
                },
            ],
        }

    @pytest.fixture
    def basic_config(self):
        """Basic simulation configuration."""
        return {
            "simulation": {
                "n_iterations": 1000,
                "random_seed": 12345,
                "sampling_method": "LHS",
            },
            "reporting": {
                "percentiles": [10, 50, 80, 90],
                "enable_charts": False,
                "sensitivity_analysis": True,
            },
        }

    @pytest.fixture
    def sample_risks(self):
        """Sample risk data for testing."""
        return [
            {
                "id": "weather_delay",
                "name": "Weather Delays",
                "category": "schedule",
                "probability": 0.3,
                "impact_distribution": {
                    "type": "triangular",
                    "low": 10000,
                    "mode": 25000,
                    "high": 50000,
                },
                "affected_wbs": ["structure", "conductor"],
            },
            {
                "id": "permit_delay",
                "name": "Permit Delays",
                "category": "regulatory",
                "probability": 0.2,
                "impact_distribution": {
                    "type": "normal",
                    "mean": 30000,
                    "stdev": 10000,
                    "truncate_low": 5000,
                    "truncate_high": 75000,
                },
                "affected_wbs": ["structure"],
            },
        ]

    def test_minimal_simulation_runs(self, minimal_project, basic_config):
        """Test that minimal simulation completes successfully."""
        results = run_simulation(minimal_project, config_data=basic_config)

        assert results is not None
        assert results.total_cost_statistics is not None
        assert len(results.total_cost_samples) == 1000
        assert results.total_cost_statistics["mean"] > 0
        assert results.total_cost_statistics["p50"] > 0
        assert (
            results.total_cost_statistics["p80"] > results.total_cost_statistics["p50"]
        )
        assert (
            results.total_cost_statistics["p90"] > results.total_cost_statistics["p80"]
        )

    def test_simulation_with_risks(self, minimal_project, basic_config, sample_risks):
        """Test simulation with risk data."""
        results = run_simulation(
            minimal_project, risks_data=sample_risks, config_data=basic_config
        )

        assert results is not None
        assert len(results.total_cost_samples) == 1000
        assert "weather_delay" in results.risk_cost_samples
        assert "permit_delay" in results.risk_cost_samples

        # With risks, total cost should be higher than base
        base_cost = sum(item["base_cost"] for item in minimal_project["wbs_items"])
        assert results.total_cost_statistics["mean"] >= base_cost

        # Risk contributions should be available
        assert results.risk_contributions is not None
        assert "weather_delay" in results.risk_contributions
        assert "permit_delay" in results.risk_contributions

    def test_simulation_with_correlations(self, minimal_project, basic_config):
        """Test simulation with correlations."""
        correlations = [
            {"pair": ["structure", "conductor"], "rho": 0.5, "method": "spearman"}
        ]

        results = run_simulation(
            minimal_project, config_data=basic_config, correlations_data=correlations
        )

        assert results is not None
        assert len(results.total_cost_samples) == 1000

        # Check that correlation was applied
        structure_samples = results.wbs_cost_samples["structure"]
        conductor_samples = results.wbs_cost_samples["conductor"]

        from scipy import stats

        actual_corr, _ = stats.spearmanr(structure_samples, conductor_samples)
        assert abs(actual_corr - 0.5) < 0.15  # Allow for sampling variation

    def test_simulation_determinism(self, minimal_project, basic_config):
        """Test that simulation is deterministic with same seed."""
        results1 = run_simulation(minimal_project, config_data=basic_config)

        results2 = run_simulation(minimal_project, config_data=basic_config)

        # Results should be identical with same seed
        np.testing.assert_array_equal(
            results1.total_cost_samples, results2.total_cost_samples
        )

        for wbs_id in results1.wbs_cost_samples:
            np.testing.assert_array_equal(
                results1.wbs_cost_samples[wbs_id], results2.wbs_cost_samples[wbs_id]
            )

    def test_simulation_different_seeds(self, minimal_project, basic_config):
        """Test that different seeds produce different results."""
        config1 = basic_config.copy()
        config1["simulation"]["random_seed"] = 12345

        config2 = basic_config.copy()
        config2["simulation"]["random_seed"] = 54321

        results1 = run_simulation(minimal_project, config_data=config1)
        results2 = run_simulation(minimal_project, config_data=config2)

        # Results should be different with different seeds
        assert not np.array_equal(
            results1.total_cost_samples, results2.total_cost_samples
        )

    def test_simulation_sensitivity_analysis(
        self, minimal_project, basic_config, sample_risks
    ):
        """Test that sensitivity analysis is computed."""
        config = basic_config.copy()
        config["reporting"]["sensitivity_analysis"] = True

        results = run_simulation(
            minimal_project, risks_data=sample_risks, config_data=config
        )

        assert results.sensitivity_analysis is not None
        assert "tornado_data" in results.sensitivity_analysis
        assert "correlation_analysis" in results.sensitivity_analysis

        # Should have entries for WBS items and risks
        tornado_data = results.sensitivity_analysis["tornado_data"]
        variable_names = [item["variable"] for item in tornado_data]

        assert "structure" in variable_names
        assert "conductor" in variable_names
        assert "weather_delay" in variable_names or "permit_delay" in variable_names

    def test_large_simulation_performance(self, minimal_project):
        """Test performance with larger simulation."""
        config = {
            "simulation": {
                "n_iterations": 10000,
                "random_seed": 12345,
                "sampling_method": "LHS",
            },
            "reporting": {
                "percentiles": [10, 50, 80, 90],
                "enable_charts": False,
                "sensitivity_analysis": False,  # Disable for speed
            },
        }

        import time

        start_time = time.time()

        results = run_simulation(minimal_project, config_data=config)

        elapsed_time = time.time() - start_time

        assert results is not None
        assert len(results.total_cost_samples) == 10000
        assert elapsed_time < 30  # Should complete within 30 seconds

    def test_simulation_with_all_distributions(self, basic_config):
        """Test simulation with all supported distribution types."""
        project = {
            "project_info": {
                "name": "Distribution Test",
                "category": "substation",
                "voltage": "69kV",
            },
            "wbs_items": [
                {
                    "id": "triangular_item",
                    "name": "Triangular",
                    "base_cost": 100000,
                    "distribution": {
                        "type": "triangular",
                        "low": 80000,
                        "mode": 100000,
                        "high": 130000,
                    },
                },
                {
                    "id": "normal_item",
                    "name": "Normal",
                    "base_cost": 50000,
                    "distribution": {"type": "normal", "mean": 50000, "stdev": 8000},
                },
                {
                    "id": "lognormal_item",
                    "name": "LogNormal",
                    "base_cost": 75000,
                    "distribution": {
                        "type": "lognormal",
                        "mean": 11.2,  # ln(75000)
                        "sigma": 0.2,
                    },
                },
                {
                    "id": "pert_item",
                    "name": "PERT",
                    "base_cost": 60000,
                    "distribution": {
                        "type": "pert",
                        "min": 45000,
                        "most_likely": 60000,
                        "max": 85000,
                    },
                },
                {
                    "id": "uniform_item",
                    "name": "Uniform",
                    "base_cost": 40000,
                    "distribution": {"type": "uniform", "low": 35000, "high": 50000},
                },
                {
                    "id": "discrete_item",
                    "name": "Discrete",
                    "base_cost": 30000,
                    "distribution": {
                        "type": "discrete",
                        "pmf": [["25000", 0.3], ["30000", 0.4], ["40000", 0.3]],
                    },
                },
            ],
        }

        results = run_simulation(project, config_data=basic_config)

        assert results is not None
        assert len(results.wbs_cost_samples) == 6

        # Check that all distributions produced reasonable results
        for wbs_id, samples in results.wbs_cost_samples.items():
            assert len(samples) == 1000
            assert np.all(samples > 0)  # All costs should be positive
            assert np.std(samples) > 0  # Should have some variation


class TestValidationIntegration:
    """Test validation integration with simulation engine."""

    def test_validation_success(self, minimal_project, basic_config, sample_risks):
        """Test successful validation."""
        is_valid, errors, warnings = validate_simulation_inputs(
            minimal_project, sample_risks, config_data=basic_config
        )

        assert is_valid
        assert len(errors) == 0
        # Warnings are acceptable

    def test_validation_with_invalid_project(self, basic_config):
        """Test validation failure with invalid project data."""
        invalid_project = {
            "project_info": {
                "name": "Test Project"
                # Missing required fields
            },
            "wbs_items": [],  # Empty WBS
        }

        is_valid, errors, warnings = validate_simulation_inputs(
            invalid_project, config_data=basic_config
        )

        assert not is_valid
        assert len(errors) > 0

    def test_validation_with_invalid_distribution(self, basic_config):
        """Test validation failure with invalid distribution."""
        invalid_project = {
            "project_info": {"name": "Test Project", "category": "transmission_line"},
            "wbs_items": [
                {
                    "id": "test_item",
                    "name": "Test Item",
                    "base_cost": 100000,
                    "distribution": {
                        "type": "triangular",
                        "low": 100000,
                        "mode": 80000,  # Mode less than low - invalid
                        "high": 120000,
                    },
                }
            ],
        }

        is_valid, errors, warnings = validate_simulation_inputs(
            invalid_project, config_data=basic_config
        )

        assert not is_valid
        assert len(errors) > 0


class TestSimulationEdgeCases:
    """Test edge cases and error conditions."""

    def test_simulation_with_zero_cost_item(self, basic_config):
        """Test simulation with zero cost WBS item."""
        project = {
            "project_info": {"name": "Zero Cost Test", "category": "transmission_line"},
            "wbs_items": [
                {
                    "id": "zero_item",
                    "name": "Zero Cost Item",
                    "base_cost": 0,
                    "distribution": {
                        "type": "triangular",
                        "low": 0,
                        "mode": 0,
                        "high": 1000,
                    },
                },
                {
                    "id": "normal_item",
                    "name": "Normal Item",
                    "base_cost": 50000,
                    "distribution": {"type": "normal", "mean": 50000, "stdev": 5000},
                },
            ],
        }

        results = run_simulation(project, config_data=basic_config)

        assert results is not None
        assert len(results.wbs_cost_samples) == 2

        # Zero item should have some samples at zero
        zero_samples = results.wbs_cost_samples["zero_item"]
        assert np.min(zero_samples) == 0
        assert np.max(zero_samples) <= 1000

    def test_simulation_with_single_wbs_item(self, basic_config):
        """Test simulation with only one WBS item."""
        project = {
            "project_info": {"name": "Single Item Test", "category": "substation"},
            "wbs_items": [
                {
                    "id": "single_item",
                    "name": "Single Item",
                    "base_cost": 100000,
                    "distribution": {"type": "normal", "mean": 100000, "stdev": 10000},
                }
            ],
        }

        results = run_simulation(project, config_data=basic_config)

        assert results is not None
        assert len(results.wbs_cost_samples) == 1
        assert "single_item" in results.wbs_cost_samples

        # Total cost should equal the single item cost
        np.testing.assert_array_equal(
            results.total_cost_samples, results.wbs_cost_samples["single_item"]
        )

    def test_simulation_with_small_iteration_count(self, minimal_project):
        """Test simulation with very small iteration count."""
        config = {
            "simulation": {
                "n_iterations": 10,  # Very small
                "random_seed": 12345,
                "sampling_method": "LHS",
            },
            "reporting": {
                "percentiles": [10, 50, 90],
                "enable_charts": False,
                "sensitivity_analysis": False,
            },
        }

        results = run_simulation(minimal_project, config_data=config)

        assert results is not None
        assert len(results.total_cost_samples) == 10
        assert results.total_cost_statistics is not None


if __name__ == "__main__":
    pytest.main([__file__])
