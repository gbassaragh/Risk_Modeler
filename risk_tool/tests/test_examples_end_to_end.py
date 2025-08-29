"""End-to-end tests with realistic T&D project examples.

Tests complete workflows from data input through simulation to reporting.
"""

import json
import numpy as np
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from risk_tool.engine.aggregation import run_simulation
from risk_tool.engine.validation import validate_simulation_inputs
from risk_tool.engine.io_json import JSONImporter, JSONExporter
from risk_tool.cli import app
from typer.testing import CliRunner


class TestTransmissionLineExample:
    """End-to-end test for transmission line project."""

    @pytest.fixture
    def transmission_line_project(self):
        """Realistic transmission line project data."""
        return {
            "project_info": {
                "name": "Green Valley 138kV Transmission Line",
                "category": "transmission_line",
                "voltage": "138kV",
                "length": 25.5,
                "terrain": "mixed",
                "region": "Southeast",
                "description": "25.5-mile 138kV transmission line connecting Green Valley substation to Mountain View substation",
            },
            "wbs_items": [
                {
                    "id": "ROW_acquisition",
                    "name": "Right-of-Way Acquisition",
                    "base_cost": 2550000,  # $100k per mile
                    "distribution": {
                        "type": "triangular",
                        "low": 1912500,  # 75% of base
                        "mode": 2550000,
                        "high": 3825000,  # 150% of base
                    },
                },
                {
                    "id": "surveying",
                    "name": "Survey and Design",
                    "base_cost": 765000,  # $30k per mile
                    "distribution": {
                        "type": "pert",
                        "min": 637500,
                        "most_likely": 765000,
                        "max": 1020000,
                    },
                },
                {
                    "id": "foundations",
                    "name": "Structure Foundations",
                    "base_cost": 3825000,  # $150k per mile
                    "distribution": {
                        "type": "normal",
                        "mean": 3825000,
                        "stdev": 382500,
                        "truncate_low": 3060000,
                        "truncate_high": 4845000,
                    },
                },
                {
                    "id": "structures",
                    "name": "Transmission Structures",
                    "base_cost": 7650000,  # $300k per mile
                    "distribution": {
                        "type": "triangular",
                        "low": 6885000,
                        "mode": 7650000,
                        "high": 9180000,
                    },
                },
                {
                    "id": "conductor",
                    "name": "Conductor Installation",
                    "base_cost": 5100000,  # $200k per mile
                    "distribution": {
                        "type": "normal",
                        "mean": 5100000,
                        "stdev": 510000,
                        "truncate_low": 4080000,
                        "truncate_high": 6375000,
                    },
                },
                {
                    "id": "grounding",
                    "name": "Grounding System",
                    "base_cost": 1275000,  # $50k per mile
                    "distribution": {
                        "type": "triangular",
                        "low": 1020000,
                        "mode": 1275000,
                        "high": 1657500,
                    },
                },
                {
                    "id": "protection",
                    "name": "Protection & Control",
                    "base_cost": 1530000,  # $60k per mile
                    "distribution": {
                        "type": "pert",
                        "min": 1275000,
                        "most_likely": 1530000,
                        "max": 2040000,
                    },
                },
                {
                    "id": "commissioning",
                    "name": "Testing & Commissioning",
                    "base_cost": 510000,  # $20k per mile
                    "distribution": {
                        "type": "triangular",
                        "low": 459000,
                        "mode": 510000,
                        "high": 612000,
                    },
                },
            ],
        }

    @pytest.fixture
    def transmission_line_risks(self):
        """Realistic risk data for transmission line."""
        return [
            {
                "id": "weather_delays",
                "name": "Weather-Related Construction Delays",
                "category": "schedule",
                "probability": 0.6,
                "impact_distribution": {
                    "type": "triangular",
                    "low": 250000,
                    "mode": 750000,
                    "high": 1500000,
                },
                "affected_wbs": ["foundations", "structures", "conductor"],
                "description": "Extended periods of severe weather preventing construction activities",
            },
            {
                "id": "permitting_delays",
                "name": "Environmental Permitting Delays",
                "category": "regulatory",
                "probability": 0.35,
                "impact_distribution": {
                    "type": "pert",
                    "min": 400000,
                    "most_likely": 1000000,
                    "max": 2500000,
                },
                "affected_wbs": ["ROW_acquisition", "surveying"],
                "description": "Delays in obtaining environmental permits and approvals",
            },
            {
                "id": "geotechnical_issues",
                "name": "Unexpected Geotechnical Conditions",
                "category": "technical",
                "probability": 0.25,
                "impact_distribution": {
                    "type": "triangular",
                    "low": 500000,
                    "mode": 1200000,
                    "high": 3000000,
                },
                "affected_wbs": ["foundations", "structures"],
                "description": "Poor soil conditions requiring foundation design changes",
            },
            {
                "id": "material_escalation",
                "name": "Steel Price Escalation",
                "category": "market",
                "probability": 0.4,
                "impact_distribution": {
                    "type": "normal",
                    "mean": 800000,
                    "stdev": 300000,
                    "truncate_low": 200000,
                    "truncate_high": 2000000,
                },
                "affected_wbs": ["structures"],
                "description": "Significant increases in steel prices above baseline",
            },
            {
                "id": "access_issues",
                "name": "Difficult Access/Terrain Issues",
                "category": "construction",
                "probability": 0.3,
                "impact_distribution": {
                    "type": "triangular",
                    "low": 300000,
                    "mode": 800000,
                    "high": 1800000,
                },
                "affected_wbs": ["foundations", "structures", "conductor"],
                "description": "Challenging terrain requiring specialized equipment",
            },
            {
                "id": "outage_constraints",
                "name": "System Outage Limitations",
                "category": "operational",
                "probability": 0.45,
                "impact_distribution": {
                    "type": "triangular",
                    "low": 200000,
                    "mode": 600000,
                    "high": 1200000,
                },
                "affected_wbs": ["commissioning", "protection"],
                "description": "Limited outage windows extending project duration",
            },
            {
                "id": "equipment_delivery",
                "name": "Long Lead Equipment Delays",
                "category": "supply_chain",
                "probability": 0.25,
                "impact_distribution": {
                    "type": "pert",
                    "min": 600000,
                    "most_likely": 1500000,
                    "max": 3500000,
                },
                "affected_wbs": ["structures", "protection"],
                "description": "Delays in delivery of specialized transmission equipment",
            },
        ]

    @pytest.fixture
    def transmission_line_correlations(self):
        """Realistic correlations for transmission line."""
        return [
            {
                "pair": ["foundations", "structures"],
                "rho": 0.7,
                "method": "spearman",
                "rationale": "Foundation and structure work often use same crews and weather-dependent",
            },
            {
                "pair": ["structures", "conductor"],
                "rho": 0.5,
                "method": "spearman",
                "rationale": "Conductor installation follows structure completion",
            },
            {
                "pair": ["ROW_acquisition", "surveying"],
                "rho": 0.6,
                "method": "spearman",
                "rationale": "Both affected by same regulatory and landowner issues",
            },
            {
                "pair": ["foundations", "conductor"],
                "rho": 0.3,
                "method": "spearman",
                "rationale": "Moderate correlation due to weather and terrain factors",
            },
        ]

    def test_transmission_line_full_simulation(
        self,
        transmission_line_project,
        transmission_line_risks,
        transmission_line_correlations,
    ):
        """Test complete transmission line simulation."""
        config = {
            "simulation": {
                "n_iterations": 5000,
                "random_seed": 12345,
                "sampling_method": "LHS",
            },
            "reporting": {
                "percentiles": [5, 10, 25, 50, 75, 80, 90, 95],
                "enable_charts": False,
                "sensitivity_analysis": True,
            },
        }

        # Run simulation
        results = run_simulation(
            transmission_line_project,
            transmission_line_risks,
            transmission_line_correlations,
            config,
        )

        # Validate results structure
        assert results is not None
        assert results.simulation_info is not None
        assert results.total_cost_statistics is not None
        assert results.wbs_cost_statistics is not None
        assert results.risk_statistics is not None
        assert results.sensitivity_analysis is not None

        # Check sample sizes
        assert len(results.total_cost_samples) == 5000
        assert len(results.wbs_cost_samples) == 8  # Number of WBS items
        assert len(results.risk_cost_samples) == 7  # Number of risks

        # Check statistical reasonableness
        base_cost = sum(
            item["base_cost"] for item in transmission_line_project["wbs_items"]
        )
        stats = results.total_cost_statistics

        assert stats["mean"] > base_cost  # Mean should be higher due to risks
        assert stats["p10"] > base_cost * 0.9  # P10 should be close to base
        assert stats["p90"] > stats["p50"]  # P90 > P50
        assert stats["p95"] > stats["p90"]  # P95 > P90

        # Check that correlations were applied
        assert "correlation_analysis" in results.sensitivity_analysis

        # Check risk contributions
        assert results.risk_contributions is not None
        assert len(results.risk_contributions) == 7

        # Validate that high-probability risks have meaningful contributions
        weather_contrib = results.risk_contributions.get("weather_delays", {})
        if weather_contrib:
            assert (
                weather_contrib.get("mean_impact", 0) > 100000
            )  # Should have significant impact

    def test_transmission_line_validation(
        self,
        transmission_line_project,
        transmission_line_risks,
        transmission_line_correlations,
    ):
        """Test validation of transmission line data."""
        config = {
            "simulation": {"n_iterations": 1000, "random_seed": 42},
            "reporting": {"percentiles": [10, 50, 90]},
        }

        is_valid, errors, warnings = validate_simulation_inputs(
            transmission_line_project,
            transmission_line_risks,
            transmission_line_correlations,
            config,
        )

        assert is_valid, f"Validation failed with errors: {errors}"
        assert len(errors) == 0
        # Warnings are acceptable for realistic data

    def test_transmission_line_contingency_calculation(
        self,
        transmission_line_project,
        transmission_line_risks,
        transmission_line_correlations,
    ):
        """Test contingency calculation for transmission line."""
        config = {
            "simulation": {"n_iterations": 10000, "random_seed": 42},
            "reporting": {"percentiles": [10, 50, 80, 90]},
        }

        results = run_simulation(
            transmission_line_project,
            transmission_line_risks,
            transmission_line_correlations,
            config,
        )

        base_cost = sum(
            item["base_cost"] for item in transmission_line_project["wbs_items"]
        )
        p80_cost = results.total_cost_statistics["p80"]
        contingency = p80_cost - base_cost
        contingency_percent = (contingency / base_cost) * 100

        # For a realistic transmission line with these risks,
        # contingency should be reasonable (typically 15-40%)
        assert (
            10 <= contingency_percent <= 60
        ), f"Contingency {contingency_percent:.1f}% seems unrealistic"

        # P80 contingency should include most risk impacts
        assert contingency > 0
        assert p80_cost > base_cost * 1.1  # At least 10% above base


class TestSubstationExample:
    """End-to-end test for substation project."""

    @pytest.fixture
    def substation_project(self):
        """Realistic substation project data."""
        return {
            "project_info": {
                "name": "Riverside 69/12.47kV Distribution Substation",
                "category": "substation",
                "voltage": "69kV",
                "capacity": 50,  # MVA
                "region": "Northeast",
                "description": "50 MVA distribution substation with two transformers",
            },
            "wbs_items": [
                {
                    "id": "site_prep",
                    "name": "Site Preparation",
                    "base_cost": 750000,
                    "distribution": {
                        "type": "triangular",
                        "low": 600000,
                        "mode": 750000,
                        "high": 1050000,
                    },
                },
                {
                    "id": "transformers",
                    "name": "Power Transformers (2x25MVA)",
                    "base_cost": 2400000,
                    "distribution": {
                        "type": "normal",
                        "mean": 2400000,
                        "stdev": 240000,
                        "truncate_low": 2000000,
                        "truncate_high": 3000000,
                    },
                },
                {
                    "id": "switchgear_69kv",
                    "name": "69kV Switchgear",
                    "base_cost": 1800000,
                    "distribution": {
                        "type": "pert",
                        "min": 1500000,
                        "most_likely": 1800000,
                        "max": 2400000,
                    },
                },
                {
                    "id": "switchgear_12kv",
                    "name": "12.47kV Switchgear",
                    "base_cost": 900000,
                    "distribution": {
                        "type": "triangular",
                        "low": 750000,
                        "mode": 900000,
                        "high": 1200000,
                    },
                },
                {
                    "id": "protection_control",
                    "name": "Protection & Control Systems",
                    "base_cost": 1200000,
                    "distribution": {
                        "type": "normal",
                        "mean": 1200000,
                        "stdev": 180000,
                        "truncate_low": 900000,
                        "truncate_high": 1650000,
                    },
                },
                {
                    "id": "civil_structures",
                    "name": "Civil & Structural",
                    "base_cost": 1500000,
                    "distribution": {
                        "type": "triangular",
                        "low": 1200000,
                        "mode": 1500000,
                        "high": 2100000,
                    },
                },
                {
                    "id": "installation",
                    "name": "Installation & Testing",
                    "base_cost": 600000,
                    "distribution": {
                        "type": "pert",
                        "min": 480000,
                        "most_likely": 600000,
                        "max": 840000,
                    },
                },
            ],
        }

    @pytest.fixture
    def substation_risks(self):
        """Realistic risk data for substation."""
        return [
            {
                "id": "transformer_delay",
                "name": "Transformer Delivery Delay",
                "category": "supply_chain",
                "probability": 0.3,
                "impact_distribution": {
                    "type": "triangular",
                    "low": 500000,
                    "mode": 1200000,
                    "high": 2500000,
                },
                "affected_wbs": ["transformers", "installation"],
            },
            {
                "id": "site_contamination",
                "name": "Site Contamination Discovery",
                "category": "environmental",
                "probability": 0.15,
                "impact_distribution": {
                    "type": "pert",
                    "min": 800000,
                    "most_likely": 2000000,
                    "max": 5000000,
                },
                "affected_wbs": ["site_prep", "civil_structures"],
            },
            {
                "id": "utility_conflicts",
                "name": "Underground Utility Conflicts",
                "category": "construction",
                "probability": 0.4,
                "impact_distribution": {
                    "type": "triangular",
                    "low": 200000,
                    "mode": 600000,
                    "high": 1200000,
                },
                "affected_wbs": ["site_prep", "civil_structures"],
            },
            {
                "id": "protection_complexity",
                "name": "Protection Scheme Complexity",
                "category": "technical",
                "probability": 0.25,
                "impact_distribution": {
                    "type": "normal",
                    "mean": 400000,
                    "stdev": 150000,
                    "truncate_low": 100000,
                    "truncate_high": 800000,
                },
                "affected_wbs": ["protection_control", "installation"],
            },
        ]

    def test_substation_simulation(self, substation_project, substation_risks):
        """Test substation simulation."""
        config = {
            "simulation": {"n_iterations": 3000, "random_seed": 67890},
            "reporting": {"percentiles": [10, 50, 80, 90], "enable_charts": False},
        }

        results = run_simulation(
            substation_project, substation_risks, config_data=config
        )

        assert results is not None
        assert len(results.total_cost_samples) == 3000

        # Check that results are reasonable for a substation
        base_cost = sum(item["base_cost"] for item in substation_project["wbs_items"])
        assert results.total_cost_statistics["mean"] > base_cost

        # Substation projects typically have lower contingencies than transmission lines
        contingency_percent = (
            (results.total_cost_statistics["p80"] - base_cost) / base_cost
        ) * 100
        assert 5 <= contingency_percent <= 40


class TestCLIIntegration:
    """Test CLI integration with realistic examples."""

    @pytest.fixture
    def sample_project_files(self):
        """Create sample project files for CLI testing."""
        project_data = {
            "project_info": {
                "name": "CLI Test Project",
                "category": "transmission_line",
                "voltage": "138kV",
                "length": 10.0,
            },
            "wbs_items": [
                {
                    "id": "structures",
                    "name": "Structures",
                    "base_cost": 1000000,
                    "distribution": {
                        "type": "triangular",
                        "low": 900000,
                        "mode": 1000000,
                        "high": 1200000,
                    },
                }
            ],
        }

        risks_data = [
            {
                "id": "test_risk",
                "name": "Test Risk",
                "category": "technical",
                "probability": 0.3,
                "impact_distribution": {
                    "type": "triangular",
                    "low": 50000,
                    "mode": 100000,
                    "high": 200000,
                },
                "affected_wbs": ["structures"],
            }
        ]

        config_data = {
            "simulation": {"n_iterations": 1000, "random_seed": 42},
            "reporting": {"percentiles": [10, 50, 90], "enable_charts": False},
        }

        # Create temporary files
        temp_dir = Path(tempfile.mkdtemp())

        project_file = temp_dir / "project.json"
        with open(project_file, "w") as f:
            json.dump(project_data, f, indent=2)

        risks_file = temp_dir / "risks.json"
        with open(risks_file, "w") as f:
            json.dump(risks_data, f, indent=2)

        config_file = temp_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        return {
            "project": project_file,
            "risks": risks_file,
            "config": config_file,
            "temp_dir": temp_dir,
        }

    def test_cli_run_command(self, sample_project_files):
        """Test CLI run command."""
        runner = CliRunner()

        output_dir = sample_project_files["temp_dir"] / "output"

        result = runner.invoke(
            app,
            [
                "run",
                str(sample_project_files["project"]),
                "--risks",
                str(sample_project_files["risks"]),
                "--config",
                str(sample_project_files["config"]),
                "--output",
                str(output_dir),
                "--format",
                "json",
            ],
        )

        assert result.exit_code == 0, f"CLI command failed: {result.output}"

        # Check that output files were created
        assert output_dir.exists()
        results_file = output_dir / "simulation_results.json"
        assert results_file.exists()

        # Verify results content
        with open(results_file) as f:
            results_data = json.load(f)

        assert "simulation_info" in results_data
        assert "total_cost_statistics" in results_data

        # Clean up
        import shutil

        shutil.rmtree(sample_project_files["temp_dir"])

    def test_cli_template_command(self):
        """Test CLI template generation."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "template.json"

            result = runner.invoke(
                app, ["template", "transmission_line", "json", str(output_file)]
            )

            assert result.exit_code == 0, f"Template command failed: {result.output}"
            assert output_file.exists()

            # Verify template content
            with open(output_file) as f:
                template_data = json.load(f)

            assert "project_info" in template_data
            assert "wbs_items" in template_data
            assert len(template_data["wbs_items"]) > 0

    def test_cli_validate_command(self, sample_project_files):
        """Test CLI validate command."""
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "validate",
                str(sample_project_files["project"]),
                "--risks",
                str(sample_project_files["risks"]),
                "--config",
                str(sample_project_files["config"]),
            ],
        )

        assert result.exit_code == 0, f"Validate command failed: {result.output}"
        assert "Validation passed" in result.output or "PASSED" in result.output

        # Clean up
        import shutil

        shutil.rmtree(sample_project_files["temp_dir"])


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_large_simulation_performance(self):
        """Test performance with large simulation."""
        # Create a moderately complex project
        project = {
            "project_info": {
                "name": "Performance Test Project",
                "category": "transmission_line",
                "voltage": "230kV",
            },
            "wbs_items": [
                {
                    "id": f"wbs_{i}",
                    "name": f"WBS Item {i}",
                    "base_cost": 100000 * (i + 1),
                    "distribution": {
                        "type": "triangular",
                        "low": 80000 * (i + 1),
                        "mode": 100000 * (i + 1),
                        "high": 130000 * (i + 1),
                    },
                }
                for i in range(10)  # 10 WBS items
            ],
        }

        # Create multiple risks
        risks = [
            {
                "id": f"risk_{i}",
                "name": f"Risk {i}",
                "category": "technical",
                "probability": 0.2 + (i * 0.05),
                "impact_distribution": {
                    "type": "triangular",
                    "low": 50000,
                    "mode": 100000 * (i + 1),
                    "high": 200000 * (i + 1),
                },
                "affected_wbs": [
                    f"wbs_{j}" for j in range(min(3, 10))
                ],  # Affect first 3 WBS items
            }
            for i in range(5)  # 5 risks
        ]

        config = {
            "simulation": {
                "n_iterations": 50000,  # Large simulation
                "random_seed": 12345,
                "sampling_method": "LHS",
            },
            "reporting": {
                "percentiles": [5, 10, 25, 50, 75, 80, 90, 95],
                "enable_charts": False,
                "sensitivity_analysis": True,
            },
        }

        import time

        start_time = time.time()

        results = run_simulation(project, risks, config_data=config)

        elapsed_time = time.time() - start_time

        assert results is not None
        assert len(results.total_cost_samples) == 50000

        # Performance target: 50k iterations in under 30 seconds
        print(f"50,000 iterations completed in {elapsed_time:.2f} seconds")
        assert elapsed_time < 60  # Generous limit for CI environments

        # Verify results quality with large sample
        stats = results.total_cost_statistics
        assert stats["mean"] > 0
        assert stats["stdev"] > 0
        assert stats["p05"] < stats["p50"] < stats["p95"]

    def test_memory_efficiency(self):
        """Test memory efficiency with large simulations."""
        # This test ensures the tool doesn't consume excessive memory
        project = {
            "project_info": {"name": "Memory Test", "category": "transmission_line"},
            "wbs_items": [
                {
                    "id": "test_item",
                    "name": "Test",
                    "base_cost": 1000000,
                    "distribution": {
                        "type": "normal",
                        "mean": 1000000,
                        "stdev": 100000,
                    },
                }
            ],
        }

        config = {
            "simulation": {"n_iterations": 100000, "random_seed": 42},
            "reporting": {
                "percentiles": [50],
                "enable_charts": False,
                "sensitivity_analysis": False,
            },
        }

        # Run simulation and check it completes without memory issues
        results = run_simulation(project, config_data=config)

        assert results is not None
        assert len(results.total_cost_samples) == 100000

        # Basic sanity check on results
        assert results.total_cost_statistics["mean"] > 900000
        assert results.total_cost_statistics["mean"] < 1100000


class TestEdgeCasesEndToEnd:
    """Test edge cases in end-to-end scenarios."""

    def test_project_with_zero_risk_probability(self):
        """Test project with risks that have zero probability."""
        project = {
            "project_info": {"name": "Zero Risk Test", "category": "substation"},
            "wbs_items": [
                {
                    "id": "test_item",
                    "name": "Test Item",
                    "base_cost": 100000,
                    "distribution": {
                        "type": "triangular",
                        "low": 90000,
                        "mode": 100000,
                        "high": 120000,
                    },
                }
            ],
        }

        risks = [
            {
                "id": "zero_risk",
                "name": "Never Happens",
                "category": "technical",
                "probability": 0.0,  # Zero probability
                "impact_distribution": {
                    "type": "triangular",
                    "low": 50000,
                    "mode": 100000,
                    "high": 200000,
                },
                "affected_wbs": ["test_item"],
            }
        ]

        config = {
            "simulation": {"n_iterations": 1000, "random_seed": 42},
            "reporting": {"percentiles": [50, 90]},
        }

        results = run_simulation(project, risks, config_data=config)

        assert results is not None

        # Zero probability risk should have zero impact
        if "zero_risk" in results.risk_cost_samples:
            assert np.all(results.risk_cost_samples["zero_risk"] == 0)

    def test_project_with_perfect_correlations(self):
        """Test project with perfect correlations."""
        project = {
            "project_info": {
                "name": "Perfect Correlation Test",
                "category": "transmission_line",
            },
            "wbs_items": [
                {
                    "id": "item1",
                    "name": "Item 1",
                    "base_cost": 100000,
                    "distribution": {"type": "normal", "mean": 100000, "stdev": 10000},
                },
                {
                    "id": "item2",
                    "name": "Item 2",
                    "base_cost": 200000,
                    "distribution": {"type": "normal", "mean": 200000, "stdev": 20000},
                },
            ],
        }

        correlations = [
            {
                "pair": ["item1", "item2"],
                "rho": 1.0,  # Perfect correlation
                "method": "spearman",
            }
        ]

        config = {
            "simulation": {"n_iterations": 1000, "random_seed": 42},
            "reporting": {"percentiles": [50]},
        }

        # Should handle perfect correlation gracefully (may be regularized)
        results = run_simulation(
            project, correlations_data=correlations, config_data=config
        )

        assert results is not None
        assert len(results.total_cost_samples) == 1000

        # Check that some correlation was achieved (may not be perfect due to regularization)
        from scipy import stats

        item1_samples = results.wbs_cost_samples["item1"]
        item2_samples = results.wbs_cost_samples["item2"]
        actual_corr, _ = stats.spearmanr(item1_samples, item2_samples)

        assert actual_corr > 0.7  # Should be high, even if not perfect


if __name__ == "__main__":
    pytest.main([__file__])
