"""Tests for IO operations (Excel, CSV, JSON).

Validates import/export functionality and template generation.
"""

import json
import pandas as pd
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from risk_tool.engine.io_excel import (
    ExcelImporter,
    ExcelExporter,
    ExcelTemplateGenerator,
)
from risk_tool.engine.io_csv import CSVImporter, CSVExporter, CSVTemplateGenerator
from risk_tool.engine.io_json import JSONImporter, JSONExporter
from risk_tool.engine.aggregation import run_simulation


class TestExcelIO:
    """Test Excel import and export operations."""

    @pytest.fixture
    def sample_excel_data(self):
        """Create sample Excel data for testing."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

            # Create sample workbook with multiple sheets
            with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
                # Project Info sheet
                project_df = pd.DataFrame(
                    [
                        {
                            "name": "Test Project",
                            "category": "transmission_line",
                            "voltage": "138kV",
                            "length": 15.5,
                            "terrain": "hilly",
                        }
                    ]
                )
                project_df.to_excel(writer, sheet_name="Project_Info", index=False)

                # WBS Items sheet
                wbs_df = pd.DataFrame(
                    [
                        {
                            "id": "structures",
                            "name": "Transmission Structures",
                            "base_cost": 500000,
                            "dist_type": "triangular",
                            "low": 450000,
                            "mode": 500000,
                            "high": 600000,
                            "truncate_low": "",
                            "truncate_high": "",
                        },
                        {
                            "id": "conductor",
                            "name": "Conductor Installation",
                            "base_cost": 300000,
                            "dist_type": "normal",
                            "mean": 300000,
                            "stdev": 30000,
                            "truncate_low": 250000,
                            "truncate_high": 400000,
                        },
                    ]
                )
                wbs_df.to_excel(writer, sheet_name="WBS_Items", index=False)

                # Risks sheet
                risks_df = pd.DataFrame(
                    [
                        {
                            "id": "weather_risk",
                            "name": "Weather Delays",
                            "category": "schedule",
                            "probability": 0.25,
                            "impact_type": "pert",
                            "impact_min": 50000,
                            "impact_most_likely": 100000,
                            "impact_max": 200000,
                            "affected_wbs": "structures,conductor",
                        }
                    ]
                )
                risks_df.to_excel(writer, sheet_name="Risks", index=False)

                # Correlations sheet
                corr_df = pd.DataFrame(
                    [
                        {
                            "var1": "structures",
                            "var2": "conductor",
                            "rho": 0.6,
                            "method": "spearman",
                        }
                    ]
                )
                corr_df.to_excel(writer, sheet_name="Correlations", index=False)

        return temp_path

    def test_excel_import_project_only(self, sample_excel_data):
        """Test importing project data only from Excel."""
        importer = ExcelImporter()
        project_data, risks_data, correlations_data = importer.import_project(
            str(sample_excel_data)
        )

        assert project_data is not None
        assert "project_info" in project_data
        assert "wbs_items" in project_data

        # Check project info
        project_info = project_data["project_info"]
        assert project_info["name"] == "Test Project"
        assert project_info["category"] == "transmission_line"
        assert project_info["voltage"] == "138kV"

        # Check WBS items
        wbs_items = project_data["wbs_items"]
        assert len(wbs_items) == 2

        structures = next(item for item in wbs_items if item["id"] == "structures")
        assert structures["base_cost"] == 500000
        assert structures["distribution"]["type"] == "triangular"
        assert structures["distribution"]["low"] == 450000

        conductor = next(item for item in wbs_items if item["id"] == "conductor")
        assert conductor["distribution"]["type"] == "normal"
        assert conductor["distribution"]["mean"] == 300000
        assert conductor["distribution"]["truncate_low"] == 250000

    def test_excel_import_with_risks(self, sample_excel_data):
        """Test importing risks from Excel."""
        importer = ExcelImporter()
        project_data, risks_data, correlations_data = importer.import_project(
            str(sample_excel_data)
        )

        assert risks_data is not None
        assert len(risks_data) == 1

        risk = risks_data[0]
        assert risk["id"] == "weather_risk"
        assert risk["name"] == "Weather Delays"
        assert risk["probability"] == 0.25
        assert risk["impact_distribution"]["type"] == "pert"
        assert risk["affected_wbs"] == ["structures", "conductor"]

    def test_excel_import_with_correlations(self, sample_excel_data):
        """Test importing correlations from Excel."""
        importer = ExcelImporter()
        project_data, risks_data, correlations_data = importer.import_project(
            str(sample_excel_data)
        )

        assert correlations_data is not None
        assert len(correlations_data) == 1

        corr = correlations_data[0]
        assert corr["pair"] == ["structures", "conductor"]
        assert corr["rho"] == 0.6
        assert corr["method"] == "spearman"

    def test_excel_export(self, sample_excel_data):
        """Test exporting results to Excel."""
        # First run a simulation
        importer = ExcelImporter()
        project_data, risks_data, correlations_data = importer.import_project(
            str(sample_excel_data)
        )

        config = {
            "simulation": {"n_iterations": 100, "random_seed": 42},
            "reporting": {"percentiles": [10, 50, 90], "enable_charts": False},
        }

        results = run_simulation(project_data, risks_data, correlations_data, config)

        # Export results
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
            output_path = Path(temp_file.name)

        exporter = ExcelExporter()
        exporter.export_results(results, str(output_path))

        # Verify the exported file exists and has expected sheets
        assert output_path.exists()

        excel_file = pd.ExcelFile(output_path)
        expected_sheets = [
            "Summary",
            "Total_Cost_Statistics",
            "WBS_Cost_Statistics",
            "Risk_Statistics",
        ]

        for sheet in expected_sheets:
            assert sheet in excel_file.sheet_names

        # Verify summary sheet has key statistics
        summary_df = pd.read_excel(output_path, sheet_name="Summary")
        assert len(summary_df) > 0
        assert "Metric" in summary_df.columns
        assert "Value" in summary_df.columns

        # Clean up
        output_path.unlink()
        sample_excel_data.unlink()

    def test_excel_template_generation(self):
        """Test Excel template generation."""
        generator = ExcelTemplateGenerator()

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
            template_path = Path(temp_file.name)

        generator.generate_template(str(template_path), "transmission_line")

        assert template_path.exists()

        # Verify template has expected sheets
        excel_file = pd.ExcelFile(template_path)
        expected_sheets = [
            "Instructions",
            "Project_Info",
            "WBS_Items",
            "Risks",
            "Correlations",
        ]

        for sheet in expected_sheets:
            assert sheet in excel_file.sheet_names

        # Verify WBS_Items sheet has sample data for transmission line
        wbs_df = pd.read_excel(template_path, sheet_name="WBS_Items")
        assert len(wbs_df) > 0
        assert "id" in wbs_df.columns
        assert "base_cost" in wbs_df.columns

        # Clean up
        template_path.unlink()


class TestCSVIO:
    """Test CSV import and export operations."""

    @pytest.fixture
    def sample_csv_files(self):
        """Create sample CSV files for testing."""
        temp_dir = Path(tempfile.mkdtemp())

        # Project info CSV
        project_df = pd.DataFrame(
            [
                {
                    "name": "CSV Test Project",
                    "category": "substation",
                    "voltage": "69kV",
                    "capacity": 100,
                }
            ]
        )
        project_path = temp_dir / "project.csv"
        project_df.to_csv(project_path, index=False)

        # WBS items CSV
        wbs_df = pd.DataFrame(
            [
                {
                    "id": "transformer",
                    "name": "Power Transformer",
                    "base_cost": 750000,
                    "dist_type": "triangular",
                    "low": 650000,
                    "mode": 750000,
                    "high": 900000,
                },
                {
                    "id": "switchgear",
                    "name": "Switchgear",
                    "base_cost": 400000,
                    "dist_type": "normal",
                    "mean": 400000,
                    "stdev": 50000,
                },
            ]
        )
        wbs_path = temp_dir / "wbs.csv"
        wbs_df.to_csv(wbs_path, index=False)

        # Risks CSV
        risks_df = pd.DataFrame(
            [
                {
                    "id": "equipment_delay",
                    "name": "Equipment Delivery Delay",
                    "category": "supply_chain",
                    "probability": 0.15,
                    "impact_type": "triangular",
                    "impact_low": 100000,
                    "impact_mode": 250000,
                    "impact_high": 500000,
                    "affected_wbs": "transformer",
                }
            ]
        )
        risks_path = temp_dir / "risks.csv"
        risks_df.to_csv(risks_path, index=False)

        return {
            "project": project_path,
            "wbs": wbs_path,
            "risks": risks_path,
            "temp_dir": temp_dir,
        }

    def test_csv_import_project(self, sample_csv_files):
        """Test importing project data from CSV files."""
        importer = CSVImporter()
        project_data = importer.import_project(
            str(sample_csv_files["project"]), str(sample_csv_files["wbs"])
        )

        assert project_data is not None
        assert "project_info" in project_data
        assert "wbs_items" in project_data

        # Check project info
        project_info = project_data["project_info"]
        assert project_info["name"] == "CSV Test Project"
        assert project_info["category"] == "substation"
        assert project_info["voltage"] == "69kV"

        # Check WBS items
        wbs_items = project_data["wbs_items"]
        assert len(wbs_items) == 2

        transformer = next(item for item in wbs_items if item["id"] == "transformer")
        assert transformer["base_cost"] == 750000
        assert transformer["distribution"]["type"] == "triangular"

    def test_csv_import_risks(self, sample_csv_files):
        """Test importing risks from CSV."""
        importer = CSVImporter()
        risks_data = importer.import_risks(str(sample_csv_files["risks"]))

        assert risks_data is not None
        assert len(risks_data) == 1

        risk = risks_data[0]
        assert risk["id"] == "equipment_delay"
        assert risk["probability"] == 0.15
        assert risk["impact_distribution"]["type"] == "triangular"
        assert risk["affected_wbs"] == ["transformer"]

    def test_csv_export(self, sample_csv_files):
        """Test exporting results to CSV."""
        # Import and run simulation
        importer = CSVImporter()
        project_data = importer.import_project(
            str(sample_csv_files["project"]), str(sample_csv_files["wbs"])
        )

        config = {
            "simulation": {"n_iterations": 100, "random_seed": 42},
            "reporting": {"percentiles": [10, 50, 90], "enable_charts": False},
        }

        results = run_simulation(project_data, config_data=config)

        # Export results
        output_dir = sample_csv_files["temp_dir"] / "output"
        output_dir.mkdir()

        exporter = CSVExporter()
        exporter.export_results(results, str(output_dir))

        # Verify exported files
        expected_files = ["summary.csv", "total_cost_stats.csv", "wbs_cost_stats.csv"]
        for filename in expected_files:
            file_path = output_dir / filename
            assert file_path.exists()

            # Verify file has content
            df = pd.read_csv(file_path)
            assert len(df) > 0

        # Clean up
        import shutil

        shutil.rmtree(sample_csv_files["temp_dir"])

    def test_csv_template_generation(self):
        """Test CSV template generation."""
        generator = CSVTemplateGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            template_dir = Path(temp_dir)
            generator.generate_template(str(template_dir), "substation")

            # Verify template files exist
            expected_files = [
                "project_template.csv",
                "wbs_template.csv",
                "risks_template.csv",
            ]
            for filename in expected_files:
                file_path = template_dir / filename
                assert file_path.exists()

                # Verify files have headers and sample data
                df = pd.read_csv(file_path)
                assert len(df) > 0
                assert len(df.columns) > 0


class TestJSONIO:
    """Test JSON import and export operations."""

    @pytest.fixture
    def sample_json_data(self):
        """Create sample JSON data for testing."""
        project_data = {
            "project_info": {
                "name": "JSON Test Project",
                "category": "transmission_line",
                "voltage": "230kV",
                "length": 25.0,
                "terrain": "mixed",
            },
            "wbs_items": [
                {
                    "id": "foundations",
                    "name": "Foundation Work",
                    "base_cost": 800000,
                    "distribution": {
                        "type": "pert",
                        "min": 700000,
                        "most_likely": 800000,
                        "max": 1000000,
                    },
                },
                {
                    "id": "towers",
                    "name": "Tower Installation",
                    "base_cost": 1200000,
                    "distribution": {
                        "type": "normal",
                        "mean": 1200000,
                        "stdev": 120000,
                        "truncate_low": 1000000,
                        "truncate_high": 1500000,
                    },
                },
            ],
        }

        risks_data = [
            {
                "id": "geological_risk",
                "name": "Geological Surprises",
                "category": "technical",
                "probability": 0.3,
                "impact_distribution": {
                    "type": "triangular",
                    "low": 100000,
                    "mode": 300000,
                    "high": 600000,
                },
                "affected_wbs": ["foundations"],
            }
        ]

        correlations_data = [
            {"pair": ["foundations", "towers"], "rho": 0.4, "method": "spearman"}
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            project_path = Path(temp_file.name)
            json.dump(project_data, temp_file, indent=2)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            risks_path = Path(temp_file.name)
            json.dump(risks_data, temp_file, indent=2)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            correlations_path = Path(temp_file.name)
            json.dump(correlations_data, temp_file, indent=2)

        return {
            "project": project_path,
            "risks": risks_path,
            "correlations": correlations_path,
        }

    def test_json_import_project(self, sample_json_data):
        """Test importing project data from JSON."""
        importer = JSONImporter()
        project_data = importer.import_project(str(sample_json_data["project"]))

        assert project_data is not None
        assert "project_info" in project_data
        assert "wbs_items" in project_data

        # Check project info
        project_info = project_data["project_info"]
        assert project_info["name"] == "JSON Test Project"
        assert project_info["voltage"] == "230kV"
        assert project_info["length"] == 25.0

        # Check WBS items
        wbs_items = project_data["wbs_items"]
        assert len(wbs_items) == 2

        foundations = next(item for item in wbs_items if item["id"] == "foundations")
        assert foundations["distribution"]["type"] == "pert"
        assert foundations["distribution"]["min"] == 700000

    def test_json_import_risks(self, sample_json_data):
        """Test importing risks from JSON."""
        importer = JSONImporter()
        risks_data = importer.import_risks(str(sample_json_data["risks"]))

        assert risks_data is not None
        assert len(risks_data) == 1

        risk = risks_data[0]
        assert risk["id"] == "geological_risk"
        assert risk["probability"] == 0.3
        assert risk["impact_distribution"]["type"] == "triangular"
        assert risk["affected_wbs"] == ["foundations"]

    def test_json_import_correlations(self, sample_json_data):
        """Test importing correlations from JSON."""
        importer = JSONImporter()
        correlations_data = importer.import_correlations(
            str(sample_json_data["correlations"])
        )

        assert correlations_data is not None
        assert len(correlations_data) == 1

        corr = correlations_data[0]
        assert corr["pair"] == ["foundations", "towers"]
        assert corr["rho"] == 0.4
        assert corr["method"] == "spearman"

    def test_json_export(self, sample_json_data):
        """Test exporting results to JSON."""
        # Import and run simulation
        importer = JSONImporter()
        project_data = importer.import_project(str(sample_json_data["project"]))
        risks_data = importer.import_risks(str(sample_json_data["risks"]))

        config = {
            "simulation": {"n_iterations": 100, "random_seed": 42},
            "reporting": {"percentiles": [10, 50, 90], "enable_charts": False},
        }

        results = run_simulation(project_data, risks_data, config_data=config)

        # Export results
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            output_path = Path(temp_file.name)

        exporter = JSONExporter()
        exporter.export_results(results, str(output_path))

        # Verify exported file
        assert output_path.exists()

        with open(output_path) as f:
            exported_data = json.load(f)

        assert "simulation_info" in exported_data
        assert "total_cost_statistics" in exported_data
        assert "wbs_cost_statistics" in exported_data
        assert "risk_statistics" in exported_data

        # Verify key statistics are present
        total_stats = exported_data["total_cost_statistics"]
        assert "mean" in total_stats
        assert "p50" in total_stats
        assert "p90" in total_stats

        # Clean up
        output_path.unlink()
        for path in sample_json_data.values():
            path.unlink()


class TestIOErrorHandling:
    """Test error handling in IO operations."""

    def test_excel_missing_file(self):
        """Test error handling for missing Excel file."""
        importer = ExcelImporter()

        with pytest.raises(FileNotFoundError):
            importer.import_project("nonexistent_file.xlsx")

    def test_excel_invalid_format(self):
        """Test error handling for invalid Excel format."""
        # Create a text file with .xlsx extension
        with tempfile.NamedTemporaryFile(
            suffix=".xlsx", delete=False, mode="w"
        ) as temp_file:
            temp_file.write("This is not an Excel file")
            temp_path = Path(temp_file.name)

        importer = ExcelImporter()

        with pytest.raises(Exception):  # Should raise some kind of Excel parsing error
            importer.import_project(str(temp_path))

        temp_path.unlink()

    def test_csv_missing_required_columns(self):
        """Test error handling for CSV missing required columns."""
        # Create CSV with missing columns
        df = pd.DataFrame([{"name": "Test Project"}])  # Missing other required fields

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            df.to_csv(temp_path, index=False)

        importer = CSVImporter()

        # Create dummy WBS file
        wbs_df = pd.DataFrame(
            [
                {
                    "id": "test",
                    "name": "Test",
                    "base_cost": 100000,
                    "dist_type": "normal",
                    "mean": 100000,
                    "stdev": 10000,
                }
            ]
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as wbs_temp:
            wbs_path = Path(wbs_temp.name)
            wbs_df.to_csv(wbs_path, index=False)

        with pytest.raises((KeyError, ValueError)):
            importer.import_project(str(temp_path), str(wbs_path))

        temp_path.unlink()
        wbs_path.unlink()

    def test_json_invalid_format(self):
        """Test error handling for invalid JSON format."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            temp_file.write("This is not valid JSON {")
            temp_path = Path(temp_file.name)

        importer = JSONImporter()

        with pytest.raises(json.JSONDecodeError):
            importer.import_project(str(temp_path))

        temp_path.unlink()


class TestIOIntegration:
    """Test integration between different IO formats."""

    def test_excel_to_json_roundtrip(self):
        """Test converting Excel data to JSON and back."""
        # Create Excel template
        generator = ExcelTemplateGenerator()

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
            excel_path = Path(temp_file.name)

        generator.generate_template(str(excel_path), "transmission_line")

        # Import from Excel
        excel_importer = ExcelImporter()
        project_data, risks_data, correlations_data = excel_importer.import_project(
            str(excel_path)
        )

        # Export to JSON
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            json_path = Path(temp_file.name)
            json.dump(project_data, temp_file, indent=2)

        # Import from JSON
        json_importer = JSONImporter()
        imported_project_data = json_importer.import_project(str(json_path))

        # Verify data consistency
        assert (
            imported_project_data["project_info"]["name"]
            == project_data["project_info"]["name"]
        )
        assert len(imported_project_data["wbs_items"]) == len(project_data["wbs_items"])

        # Clean up
        excel_path.unlink()
        json_path.unlink()

    def test_simulation_with_different_formats(self):
        """Test running simulation with data from different formats."""
        # This is a comprehensive test that uses multiple IO formats
        # and verifies they produce consistent simulation results

        # Create sample data
        sample_project = {
            "project_info": {
                "name": "Multi-Format Test",
                "category": "transmission_line",
                "voltage": "138kV",
            },
            "wbs_items": [
                {
                    "id": "test_item",
                    "name": "Test Item",
                    "base_cost": 100000,
                    "distribution": {"type": "normal", "mean": 100000, "stdev": 10000},
                }
            ],
        }

        config = {
            "simulation": {"n_iterations": 100, "random_seed": 42},
            "reporting": {"percentiles": [50, 90], "enable_charts": False},
        }

        # Test with JSON
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            json_path = Path(temp_file.name)
            json.dump(sample_project, temp_file, indent=2)

        json_importer = JSONImporter()
        json_project = json_importer.import_project(str(json_path))
        json_results = run_simulation(json_project, config_data=config)

        # All formats should produce the same results with the same seed
        assert json_results.total_cost_statistics["mean"] > 0
        assert len(json_results.total_cost_samples) == 100

        json_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__])
