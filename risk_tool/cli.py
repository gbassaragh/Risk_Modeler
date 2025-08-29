"""Command-line interface for the Risk Modeling Tool.

Provides comprehensive CLI with commands for running simulations, generating templates,
and managing project data. Built with Typer for modern CLI experience with rich formatting
and comprehensive help documentation.

Features:
- Monte Carlo simulation execution
- Data import/export (Excel, CSV, JSON)
- Template generation and validation
- Audit logging and deterministic verification
- Professional reporting with charts and statistics
"""

import typer
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import json
import sys
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import print as rprint
import time

from .core.aggregation import run_simulation
from .io.io_excel import ExcelImporter, ExcelExporter
from .io.io_csv import CSVImporter, CSVExporter
from .io.io_json import JSONImporter, JSONExporter
from .core.validation import validate_simulation_inputs
from .reporting.reporting import ReportGenerator, create_simple_summary_report
from .core.audit import AuditLogger, DeterminismVerifier

app: typer.Typer = typer.Typer(help="Monte Carlo risk modeling tool for utility T&D projects")
console: Console = Console()

# Global state
current_audit_logger: Optional[AuditLogger] = None


@app.command()
def run(
    project: str = typer.Option(..., "--project", "-p", help="Project file (Excel, CSV, or JSON)"),
    risks: Optional[str] = typer.Option(None, "--risks", "-r", help="Risk file (Excel, CSV, or JSON)"),
    correlations: Optional[str] = typer.Option(None, "--correlations", "-c", help="Correlations file (CSV or JSON)"),
    config: Optional[str] = typer.Option(None, "--config", help="Configuration file (JSON or YAML)"),
    output: str = typer.Option("./results", "--output", "-o", help="Output directory"),
    iterations: Optional[int] = typer.Option(None, "--iterations", "-n", help="Number of iterations"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    sampling: Optional[str] = typer.Option(None, "--sampling", help="Sampling method (LHS or MC)"),
    validate_only: bool = typer.Option(False, "--validate-only", help="Only validate inputs, don't run simulation"),
    export_formats: Optional[List[str]] = typer.Option(None, "--formats", help="Export formats (csv,xlsx,json,pdf)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Run Monte Carlo risk simulation."""
    global current_audit_logger
    
    try:
        # Initialize audit logger
        current_audit_logger = AuditLogger()
        
        # Display header
        if verbose:
            console.print("\n[bold blue]Risk Modeling Tool - Monte Carlo Simulation[/bold blue]")
            console.print("=" * 60)
        
        # Import data
        console.print("[yellow]Loading input data...[/yellow]")
        
        project_data = _load_project_data(project)
        risks_data = _load_risks_data(risks) if risks else None
        correlations_data = _load_correlations_data(correlations) if correlations else None
        config_data = _load_config_data(config)
        
        # Override config with CLI parameters
        if iterations:
            config_data['iterations'] = iterations
        if seed:
            config_data['random_seed'] = seed
        if sampling:
            config_data['sampling'] = sampling
        
        # Validate inputs
        console.print("[yellow]Validating inputs...[/yellow]")
        
        input_files = {
            'project': project,
            'risks': risks,
            'correlations': correlations,
            'config': config
        }
        
        is_valid, errors, warnings = validate_simulation_inputs(
            project_data, risks_data, correlations_data, config_data
        )
        
        # Log validation results
        current_audit_logger.log_validation_results(
            'complete_validation', is_valid, errors, warnings
        )
        
        # Display validation results
        if warnings:
            console.print(f"[orange3]Warnings ({len(warnings)}):[/orange3]")
            for warning in warnings:
                console.print(f"  • {warning}")
        
        if errors:
            console.print(f"[red]Errors ({len(errors)}):[/red]")
            for error in errors:
                console.print(f"  • {error}")
        
        if not is_valid:
            console.print("[red]❌ Validation failed. Please fix errors before running simulation.[/red]")
            raise typer.Exit(1)
        
        if validate_only:
            console.print("[green]✅ Validation passed.[/green]")
            return
        
        # Log simulation start
        current_audit_logger.log_simulation_start(
            config_data, project_data['id'], input_files
        )
        
        # Run simulation
        console.print(f"[yellow]Running simulation with {config_data.get('iterations', 50000):,} iterations...[/yellow]")
        
        start_time = time.time()
        
        # Show progress bar for long-running simulations
        if config_data.get('iterations', 50000) > 10000:
            with console.status("[bold green]Running Monte Carlo simulation..."):
                results = run_simulation(project_data, risks_data, config_data)
        else:
            results = run_simulation(project_data, risks_data, config_data)
        
        simulation_time = time.time() - start_time
        
        # Performance metrics
        performance_metrics = {
            'total_time': simulation_time,
            'iterations_per_second': results.n_samples / simulation_time,
            'time_per_iteration': simulation_time / results.n_samples
        }
        
        # Log simulation completion
        results_summary = {
            'base_cost': results.base_cost,
            'p50': results.percentiles.get('P50', 0),
            'p80': results.percentiles.get('P80', 0),
            'p90': results.percentiles.get('P90', 0)
        }
        
        current_audit_logger.log_simulation_end(results_summary, performance_metrics)
        
        # Display results summary
        _display_results_summary(results, simulation_time, verbose)
        
        # Export results
        console.print(f"[yellow]Exporting results to {output}...[/yellow]")
        
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        export_formats_list = export_formats or ['csv', 'xlsx', 'json']
        _export_results(results, output_dir, export_formats_list)
        
        # Export audit log
        audit_path = output_dir / "audit_log.json"
        current_audit_logger.export_audit_log(str(audit_path))
        
        console.print(f"[green]✅ Simulation completed successfully![/green]")
        console.print(f"[blue]Results saved to: {output_dir.absolute()}[/blue]")
        
    except Exception as e:
        if current_audit_logger:
            current_audit_logger.log_error('simulation_error', str(e))
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def template(
    type: str = typer.Argument(..., help="Template type (wbs, risks, config, correlations)"),
    format: str = typer.Option("excel", "--format", "-f", help="Output format (excel, csv, json)"),
    output: str = typer.Option(".", "--output", "-o", help="Output directory"),
    project_type: str = typer.Option("TransmissionLine", "--project-type", help="Project type for templates")
):
    """Generate input templates."""
    
    console.print(f"[yellow]Generating {type} template in {format} format...[/yellow]")
    
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if type == "wbs":
            if format == "excel":
                file_path = output_dir / "wbs_template.xlsx"
                exporter = ExcelExporter()
                exporter.create_wbs_template(str(file_path))
            elif format == "csv":
                file_path = output_dir / "wbs_template.csv"
                exporter = CSVExporter()
                exporter.create_wbs_template_csv(str(file_path))
            else:
                raise ValueError(f"Format {format} not supported for WBS template")
        
        elif type == "risks":
            if format == "excel":
                file_path = output_dir / "risks_template.xlsx"
                exporter = ExcelExporter()
                exporter.create_risk_template(str(file_path))
            elif format == "csv":
                file_path = output_dir / "risks_template.csv"
                exporter = CSVExporter()
                exporter.create_risk_template_csv(str(file_path))
            else:
                raise ValueError(f"Format {format} not supported for risks template")
        
        elif type == "correlations":
            if format == "csv":
                file_path = output_dir / "correlations_template.csv"
                exporter = CSVExporter()
                exporter.create_correlation_template_csv(str(file_path))
            elif format == "json":
                file_path = output_dir / "correlations_template.json"
                # Create simple correlations template
                template_data = [
                    {
                        "pair": ["commodity_aluminum", "Conductor & OPGW"],
                        "rho": 0.6,
                        "method": "spearman"
                    }
                ]
                with open(file_path, 'w') as f:
                    json.dump({"correlations": template_data}, f, indent=2)
            else:
                raise ValueError(f"Format {format} not supported for correlations template")
        
        elif type == "config":
            file_path = output_dir / "config_template.json"
            exporter = JSONExporter()
            exporter.create_example_configuration(str(file_path))
        
        else:
            raise ValueError(f"Unknown template type: {type}")
        
        console.print(f"[green]✅ Template created: {file_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error creating template: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def validate(
    project: str = typer.Option(..., "--project", "-p", help="Project file"),
    risks: Optional[str] = typer.Option(None, "--risks", "-r", help="Risk file"),
    correlations: Optional[str] = typer.Option(None, "--correlations", "-c", help="Correlations file"),
    config: Optional[str] = typer.Option(None, "--config", help="Configuration file"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed validation results")
):
    """Validate input files without running simulation."""
    
    console.print("[yellow]Validating input files...[/yellow]")
    
    try:
        # Load data
        project_data = _load_project_data(project)
        risks_data = _load_risks_data(risks) if risks else None
        correlations_data = _load_correlations_data(correlations) if correlations else None
        config_data = _load_config_data(config)
        
        # Validate
        is_valid, errors, warnings = validate_simulation_inputs(
            project_data, risks_data, correlations_data, config_data
        )
        
        # Display results
        if detailed:
            _display_detailed_validation_results(errors, warnings)
        else:
            _display_validation_summary(is_valid, len(errors), len(warnings))
        
        if not is_valid:
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[red]❌ Validation error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def report(
    results_dir: str = typer.Argument(..., help="Results directory from previous run"),
    format: str = typer.Option("html", "--format", "-f", help="Report format (html, pdf, json)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path")
):
    """Generate reports from previous simulation results."""
    
    console.print(f"[yellow]Generating {format} report...[/yellow]")
    
    try:
        results_path = Path(results_dir)
        
        # Look for results file
        json_files = list(results_path.glob("*.json"))
        results_file = None
        
        for file in json_files:
            if "results" in file.name or "simulation" in file.name:
                results_file = file
                break
        
        if not results_file:
            console.print(f"[red]❌ No results file found in {results_dir}[/red]")
            raise typer.Exit(1)
        
        # Load results
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        # Generate report based on format
        if output is None:
            output = results_path / f"report.{format}"
        else:
            output = Path(output)
        
        if format == "json":
            # Export structured report
            with open(output, 'w') as f:
                json.dump(results_data, f, indent=2)
        
        elif format == "html":
            # Generate HTML report (simplified)
            _generate_html_report(results_data, output)
        
        else:
            console.print(f"[red]❌ Unsupported report format: {format}[/red]")
            raise typer.Exit(1)
        
        console.print(f"[green]✅ Report generated: {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Report generation error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def verify(
    project: str = typer.Option(..., "--project", "-p", help="Project file"),
    risks: Optional[str] = typer.Option(None, "--risks", "-r", help="Risk file"),
    config: Optional[str] = typer.Option(None, "--config", help="Configuration file"),
    runs: int = typer.Option(3, "--runs", help="Number of verification runs"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed to test")
):
    """Verify simulation reproducibility."""
    
    console.print(f"[yellow]Verifying reproducibility with {runs} runs...[/yellow]")
    
    try:
        # Load data
        project_data = _load_project_data(project)
        risks_data = _load_risks_data(risks) if risks else None
        config_data = _load_config_data(config)
        
        # Set seed if provided
        if seed:
            config_data['random_seed'] = seed
        
        # Run verification
        verification_result = DeterminismVerifier.verify_reproducibility(
            config_data, project_data, risks_data, runs
        )
        
        # Display results
        if verification_result['reproducible']:
            console.print(f"[green]✅ Simulation is reproducible across {runs} runs[/green]")
            console.print(f"Random seed: {verification_result['config_seed']}")
        else:
            console.print(f"[red]❌ Simulation is not reproducible[/red]")
            console.print(f"Reason: {verification_result.get('reason', 'Unknown')}")
        
        # Show detailed results if verbose
        if len(verification_result['runs']) > 0:
            table = Table(title="Verification Runs")
            table.add_column("Run")
            table.add_column("P50")
            table.add_column("P80")
            table.add_column("Hash")
            
            for run in verification_result['runs']:
                table.add_row(
                    str(run['run_number']),
                    f"${run['p50']:,.0f}",
                    f"${run['p80']:,.0f}",
                    run['total_costs_hash'][:8] + "..."
                )
            
            console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Verification error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def info(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project file to analyze")
):
    """Show tool information and project summary."""
    
    console.print("[bold blue]Risk Modeling Tool Information[/bold blue]")
    console.print("=" * 50)
    console.print("Monte Carlo risk modeling for utility T&D projects")
    console.print("Version: 1.0.0")
    console.print("")
    
    if project:
        console.print(f"[yellow]Project Analysis: {project}[/yellow]")
        
        try:
            project_data = _load_project_data(project)
            
            # Display project summary
            table = Table(title="Project Summary")
            table.add_column("Property")
            table.add_column("Value")
            
            table.add_row("Project ID", project_data.get('id', 'N/A'))
            table.add_row("Type", project_data.get('type', 'N/A'))
            table.add_row("Currency", project_data.get('currency', 'USD'))
            table.add_row("Region", project_data.get('region', 'N/A'))
            table.add_row("WBS Items", str(len(project_data.get('wbs', []))))
            
            # Calculate base cost
            base_cost = 0
            for item in project_data.get('wbs', []):
                qty = item.get('quantity', 0)
                rate = item.get('unit_cost', 0)
                indirect = item.get('indirect_factor', 0)
                base_cost += qty * rate * (1 + indirect)
            
            table.add_row("Base Cost", f"${base_cost:,.0f}")
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]❌ Error analyzing project: {e}[/red]")


# Utility functions

def _load_project_data(file_path: str) -> dict:
    """Load project data from file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Project file not found: {file_path}")
    
    if path.suffix.lower() in ['.xlsx', '.xls']:
        importer = ExcelImporter()
        return importer.import_project(file_path)
    elif path.suffix.lower() == '.csv':
        importer = CSVImporter()
        return importer.import_project_from_csv(file_path)
    elif path.suffix.lower() == '.json':
        importer = JSONImporter()
        return importer.import_project(file_path)
    else:
        raise ValueError(f"Unsupported project file format: {path.suffix}")


def _load_risks_data(file_path: str) -> list:
    """Load risks data from file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Risks file not found: {file_path}")
    
    if path.suffix.lower() in ['.xlsx', '.xls']:
        importer = ExcelImporter()
        return importer.import_risks(file_path)
    elif path.suffix.lower() == '.csv':
        importer = CSVImporter()
        return importer.import_risks_from_csv(file_path)
    elif path.suffix.lower() == '.json':
        importer = JSONImporter()
        return importer.import_risks(file_path)
    else:
        raise ValueError(f"Unsupported risks file format: {path.suffix}")


def _load_correlations_data(file_path: str) -> list:
    """Load correlations data from file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Correlations file not found: {file_path}")
    
    if path.suffix.lower() == '.csv':
        importer = CSVImporter()
        return importer.import_correlations_from_csv(file_path)
    elif path.suffix.lower() == '.json':
        importer = JSONImporter()
        return importer.import_correlations(file_path)
    else:
        raise ValueError(f"Unsupported correlations file format: {path.suffix}")


def _load_config_data(file_path: Optional[str]) -> dict:
    """Load configuration data from file or return defaults."""
    if not file_path:
        return {
            'iterations': 50000,
            'sampling': 'LHS',
            'random_seed': None,
            'currency': 'USD'
        }
    
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    if path.suffix.lower() == '.json':
        importer = JSONImporter()
        return importer.import_configuration(file_path)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")


def _display_results_summary(results, simulation_time: float, verbose: bool):
    """Display simulation results summary."""
    
    # Create summary table
    table = Table(title="Simulation Results Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    
    table.add_row("Project ID", results.project_id)
    table.add_row("Base Cost", f"${results.base_cost:,.0f}")
    table.add_row("Iterations", f"{results.n_samples:,}")
    table.add_row("Simulation Time", f"{simulation_time:.1f}s")
    table.add_row("", "")  # Separator
    
    # Add percentiles
    for percentile, value in results.percentiles.items():
        table.add_row(f"Cost {percentile}", f"${value:,.0f}")
    
    # Add contingency recommendations
    if results.contingency_recommendations:
        contingency = results.contingency_recommendations
        table.add_row("", "")  # Separator
        table.add_row("Recommended Contingency", f"${contingency.get('contingency', 0):,.0f}")
        table.add_row("Contingency %", f"{contingency.get('contingency_percent', 0):.1f}%")
    
    console.print(table)
    
    if verbose and results.wbs_statistics:
        console.print("\n[yellow]WBS Breakdown (Top 5 by Mean Cost):[/yellow]")
        
        # Sort by mean cost
        sorted_wbs = sorted(
            results.wbs_statistics.items(), 
            key=lambda x: x[1].get('mean', 0), 
            reverse=True
        )
        
        wbs_table = Table()
        wbs_table.add_column("WBS Code")
        wbs_table.add_column("Mean Cost", justify="right")
        wbs_table.add_column("P80 Cost", justify="right")
        wbs_table.add_column("CV", justify="right")
        
        for wbs_code, stats in sorted_wbs[:5]:
            wbs_table.add_row(
                wbs_code,
                f"${stats.get('mean', 0):,.0f}",
                f"${stats.get('p80', 0):,.0f}",
                f"{stats.get('cv', 0):.2f}"
            )
        
        console.print(wbs_table)


def _display_validation_summary(is_valid: bool, error_count: int, warning_count: int):
    """Display validation summary."""
    
    if is_valid:
        console.print("[green]✅ Validation passed[/green]")
    else:
        console.print("[red]❌ Validation failed[/red]")
    
    if error_count > 0:
        console.print(f"[red]Errors: {error_count}[/red]")
    
    if warning_count > 0:
        console.print(f"[orange3]Warnings: {warning_count}[/orange3]")


def _display_detailed_validation_results(errors: List[str], warnings: List[str]):
    """Display detailed validation results."""
    
    if errors:
        console.print(f"[red]Errors ({len(errors)}):[/red]")
        for i, error in enumerate(errors, 1):
            console.print(f"  {i}. {error}")
        console.print()
    
    if warnings:
        console.print(f"[orange3]Warnings ({len(warnings)}):[/orange3]")
        for i, warning in enumerate(warnings, 1):
            console.print(f"  {i}. {warning}")


def _export_results(results, output_dir: Path, export_formats: List[str]):
    """Export results in specified formats."""
    
    results_dict = {
        'summary': {
            'project_id': results.project_id,
            'base_cost': results.base_cost,
            'currency': results.currency,
            'n_samples': results.n_samples,
            'percentiles': results.percentiles
        },
        'wbs_statistics': results.wbs_statistics,
        'contingency': results.contingency_recommendations
    }
    
    for format_name in export_formats:
        if format_name == 'csv':
            exporter = CSVExporter()
            exporter.export_results_to_csv(results_dict, str(output_dir))
        
        elif format_name == 'xlsx':
            exporter = ExcelExporter()
            excel_path = output_dir / "simulation_results.xlsx"
            exporter.export_results(results_dict, str(excel_path))
        
        elif format_name == 'json':
            exporter = JSONExporter()
            json_path = output_dir / "simulation_results.json"
            exporter.export_results(results_dict, str(json_path))


def _generate_html_report(results_data: dict, output_path: Path):
    """Generate simple HTML report."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Risk Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .header {{ color: #2c3e50; }}
        </style>
    </head>
    <body>
        <h1 class="header">Risk Analysis Report</h1>
        
        <h2>Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Project ID</td><td>{results_data.get('summary', {}).get('project_id', 'N/A')}</td></tr>
            <tr><td>Base Cost</td><td>${results_data.get('summary', {}).get('base_cost', 0):,.0f}</td></tr>
            <tr><td>Iterations</td><td>{results_data.get('summary', {}).get('n_samples', 0):,}</td></tr>
        </table>
        
        <h2>Cost Percentiles</h2>
        <table>
            <tr><th>Percentile</th><th>Cost</th></tr>
    """
    
    percentiles = results_data.get('summary', {}).get('percentiles', {})
    for percentile, value in percentiles.items():
        html_content += f"<tr><td>{percentile}</td><td>${value:,.0f}</td></tr>"
    
    html_content += """
        </table>
        
        <p><em>Generated by Risk Modeling Tool v1.0.0</em></p>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)


if __name__ == "__main__":
    app()