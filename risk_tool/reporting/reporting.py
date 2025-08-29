"""Results reporting and visualization.

Generates comprehensive reports with charts, tables, and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
from pydantic import BaseModel


class SimulationResults(BaseModel):
    """Container for simulation results."""
    project_id: str
    currency: str
    n_samples: int
    random_seed: Optional[int]
    base_cost: float
    total_costs: List[float]
    percentiles: Dict[str, float]
    wbs_statistics: Dict[str, Dict[str, float]]
    risk_contributions: Optional[Dict[str, Dict[str, Any]]] = None
    sensitivity_analysis: Optional[Dict[str, Any]] = None
    contingency_recommendations: Optional[Dict[str, float]] = None
    
    class Config:
        arbitrary_types_allowed = True


class ResultsCalculator:
    """Calculates summary statistics from simulation arrays."""
    
    @staticmethod
    def calculate_percentiles(data: np.ndarray, 
                            percentiles: List[float] = [10, 50, 80, 90, 95]) -> Dict[str, float]:
        """Calculate percentiles from data array.
        
        Args:
            data: Data array
            percentiles: List of percentile values (0-100)
            
        Returns:
            Dictionary mapping percentile names to values
        """
        result = {}
        
        for p in percentiles:
            result[f'P{int(p)}'] = np.percentile(data, p)
        
        return result
    
    @staticmethod
    def calculate_summary_statistics(data: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive summary statistics.
        
        Args:
            data: Data array
            
        Returns:
            Dictionary of statistics
        """
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data),
            'cv': np.std(data) / np.mean(data) if np.mean(data) != 0 else 0,
            'skewness': float(ResultsCalculator._calculate_skewness(data)),
            'kurtosis': float(ResultsCalculator._calculate_kurtosis(data)),
            'count': len(data)
        }
    
    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skew = np.mean(((data - mean) / std) ** 3)
        return skew
    
    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        kurt = np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
        return kurt


class ChartGenerator:
    """Generates various charts for risk analysis results."""
    
    def __init__(self, style: str = 'plotly'):
        """Initialize chart generator.
        
        Args:
            style: Chart style ('plotly' or 'matplotlib')
        """
        self.style = style
        
        # Set default style parameters
        if style == 'matplotlib':
            plt.style.use('seaborn-v0_8')  # Modern seaborn style
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        else:
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def create_histogram(self, data: np.ndarray, 
                        title: str = "Cost Distribution",
                        x_label: str = "Total Cost ($)",
                        percentiles: Optional[Dict[str, float]] = None) -> Any:
        """Create histogram of cost distribution.
        
        Args:
            data: Cost data array
            title: Chart title
            x_label: X-axis label
            percentiles: Optional percentiles to mark
            
        Returns:
            Chart object
        """
        if self.style == 'plotly':
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=50,
                name='Distribution',
                opacity=0.7,
                marker_color=self.colors[0]
            ))
            
            # Add percentile lines
            if percentiles:
                colors_cycle = ['red', 'orange', 'green', 'blue', 'purple']
                for i, (name, value) in enumerate(percentiles.items()):
                    color = colors_cycle[i % len(colors_cycle)]
                    fig.add_vline(
                        x=value, 
                        line_dash="dash", 
                        line_color=color,
                        annotation_text=f"{name}: ${value:,.0f}"
                    )
            
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title="Frequency",
                showlegend=bool(percentiles)
            )
            
            return fig
        
        else:  # matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create histogram
            n, bins, patches = ax.hist(data, bins=50, alpha=0.7, color=self.colors[0])
            
            # Add percentile lines
            if percentiles:
                colors_cycle = ['red', 'orange', 'green', 'blue', 'purple']
                for i, (name, value) in enumerate(percentiles.items()):
                    color = colors_cycle[i % len(colors_cycle)]
                    ax.axvline(value, color=color, linestyle='--', 
                              label=f'{name}: ${value:,.0f}')
                ax.legend()
            
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            return fig
    
    def create_cdf_curve(self, data: np.ndarray,
                        title: str = "Cumulative Distribution",
                        x_label: str = "Total Cost ($)") -> Any:
        """Create cumulative distribution function curve.
        
        Args:
            data: Cost data array
            title: Chart title
            x_label: X-axis label
            
        Returns:
            Chart object
        """
        sorted_data = np.sort(data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        if self.style == 'plotly':
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=sorted_data,
                y=cumulative,
                mode='lines',
                name='CDF',
                line_color=self.colors[0]
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title="Cumulative Probability",
                yaxis=dict(range=[0, 1])
            )
            
            return fig
        
        else:  # matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(sorted_data, cumulative, color=self.colors[0], linewidth=2)
            
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel('Cumulative Probability')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            return fig
    
    def create_tornado_diagram(self, tornado_data: Dict[str, Any],
                              title: str = "Sensitivity Analysis") -> Any:
        """Create tornado diagram for sensitivity analysis.
        
        Args:
            tornado_data: Tornado diagram data
            title: Chart title
            
        Returns:
            Chart object
        """
        variables = tornado_data.get('variables', [])
        sensitivities = tornado_data.get('sensitivities', [])
        
        if not variables or not sensitivities:
            # Return empty figure
            if self.style == 'plotly':
                return go.Figure().add_annotation(text="No sensitivity data available")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No sensitivity data available", 
                       ha='center', va='center', transform=ax.transAxes)
                return fig
        
        if self.style == 'plotly':
            fig = go.Figure()
            
            # Create horizontal bar chart
            fig.add_trace(go.Bar(
                y=variables,
                x=sensitivities,
                orientation='h',
                marker_color=[self.colors[0] if s >= 0 else self.colors[3] 
                             for s in sensitivities]
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Sensitivity Coefficient",
                yaxis_title="Variables",
                yaxis=dict(autorange="reversed")  # Top to bottom
            )
            
            return fig
        
        else:  # matplotlib
            fig, ax = plt.subplots(figsize=(10, 8))
            
            colors = [self.colors[0] if s >= 0 else self.colors[3] for s in sensitivities]
            
            bars = ax.barh(variables, sensitivities, color=colors)
            
            ax.set_title(title)
            ax.set_xlabel('Sensitivity Coefficient')
            ax.set_ylabel('Variables')
            ax.grid(True, axis='x', alpha=0.3)
            
            # Invert y-axis to show most sensitive at top
            ax.invert_yaxis()
            
            return fig
    
    def create_pareto_chart(self, data: Dict[str, float],
                           title: str = "Risk Contribution Pareto",
                           value_label: str = "Contribution") -> Any:
        """Create Pareto chart for risk contributions.
        
        Args:
            data: Dictionary of labels to values
            title: Chart title
            value_label: Y-axis label for values
            
        Returns:
            Chart object
        """
        if not data:
            if self.style == 'plotly':
                return go.Figure().add_annotation(text="No data available")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No data available", 
                       ha='center', va='center', transform=ax.transAxes)
                return fig
        
        # Sort data by value
        sorted_items = sorted(data.items(), key=lambda x: abs(x[1]), reverse=True)
        labels, values = zip(*sorted_items)
        
        # Calculate cumulative percentage
        total = sum(abs(v) for v in values)
        cumulative = np.cumsum([abs(v)/total * 100 for v in values])
        
        if self.style == 'plotly':
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add bars
            fig.add_trace(
                go.Bar(x=labels, y=values, name=value_label, 
                      marker_color=self.colors[0]),
                secondary_y=False,
            )
            
            # Add cumulative line
            fig.add_trace(
                go.Scatter(x=labels, y=cumulative, mode='lines+markers',
                          name='Cumulative %', marker_color=self.colors[3],
                          line=dict(width=3)),
                secondary_y=True,
            )
            
            fig.update_xaxes(title_text="Risk/Variable")
            fig.update_yaxes(title_text=value_label, secondary_y=False)
            fig.update_yaxes(title_text="Cumulative Percentage", secondary_y=True)
            fig.update_layout(title_text=title)
            
            return fig
        
        else:  # matplotlib
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Create bars
            bars = ax1.bar(labels, values, color=self.colors[0], alpha=0.7)
            ax1.set_xlabel('Risk/Variable')
            ax1.set_ylabel(value_label, color=self.colors[0])
            ax1.tick_params(axis='x', rotation=45)
            
            # Create second y-axis for cumulative percentage
            ax2 = ax1.twinx()
            line = ax2.plot(labels, cumulative, color=self.colors[3], 
                          marker='o', linewidth=3, markersize=8)
            ax2.set_ylabel('Cumulative Percentage', color=self.colors[3])
            ax2.set_ylim(0, 100)
            
            plt.title(title)
            plt.tight_layout()
            
            return fig
    
    def create_wbs_breakdown_chart(self, wbs_data: Dict[str, Dict[str, float]],
                                  metric: str = 'mean',
                                  title: str = "WBS Cost Breakdown") -> Any:
        """Create WBS breakdown chart.
        
        Args:
            wbs_data: WBS statistics data
            metric: Metric to display ('mean', 'p80', etc.)
            title: Chart title
            
        Returns:
            Chart object
        """
        if not wbs_data:
            if self.style == 'plotly':
                return go.Figure().add_annotation(text="No WBS data available")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No WBS data available", 
                       ha='center', va='center', transform=ax.transAxes)
                return fig
        
        # Extract data
        wbs_codes = list(wbs_data.keys())
        values = [wbs_data[code].get(metric, 0) for code in wbs_codes]
        
        if self.style == 'plotly':
            fig = go.Figure(data=[
                go.Pie(labels=wbs_codes, values=values, hole=0.3)
            ])
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(title=title)
            
            return fig
        
        else:  # matplotlib
            fig, ax = plt.subplots(figsize=(10, 8))
            
            ax.pie(values, labels=wbs_codes, autopct='%1.1f%%', startangle=90)
            ax.set_title(title)
            
            return fig


class ReportGenerator:
    """Generates comprehensive reports combining data and visualizations."""
    
    def __init__(self, results: SimulationResults):
        """Initialize report generator.
        
        Args:
            results: Simulation results
        """
        self.results = results
        self.chart_generator = ChartGenerator('plotly')
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of results.
        
        Returns:
            Executive summary dictionary
        """
        percentiles = self.results.percentiles
        contingency = self.results.contingency_recommendations or {}
        
        summary = {
            'project_overview': {
                'project_id': self.results.project_id,
                'currency': self.results.currency,
                'base_cost': self.results.base_cost,
                'simulation_iterations': self.results.n_samples
            },
            'cost_outcomes': {
                'expected_cost_p50': percentiles.get('P50', 0),
                'planning_cost_p80': percentiles.get('P80', 0),
                'worst_case_p95': percentiles.get('P95', 0),
                'cost_range_p10_p90': {
                    'low': percentiles.get('P10', 0),
                    'high': percentiles.get('P90', 0)
                }
            },
            'contingency_recommendations': {
                'recommended_contingency': contingency.get('contingency', 0),
                'contingency_percentage': contingency.get('contingency_percent', 0),
                'management_reserve': contingency.get('management_reserve', 0),
                'total_buffer': contingency.get('total_buffer', 0)
            },
            'key_insights': self._generate_key_insights()
        }
        
        return summary
    
    def generate_technical_report(self) -> Dict[str, Any]:
        """Generate detailed technical analysis report.
        
        Returns:
            Technical report dictionary
        """
        # Calculate additional statistics
        total_costs = np.array(self.results.total_costs)
        summary_stats = ResultsCalculator.calculate_summary_statistics(total_costs)
        
        report = {
            'methodology': {
                'simulation_method': 'Monte Carlo with Latin Hypercube Sampling',
                'iterations': self.results.n_samples,
                'random_seed': self.results.random_seed,
                'confidence_level': '80% (P80 recommended for contingency)'
            },
            'statistical_analysis': {
                **summary_stats,
                'percentiles': self.results.percentiles
            },
            'wbs_analysis': self.results.wbs_statistics,
            'risk_analysis': self.results.risk_contributions,
            'sensitivity_analysis': self.results.sensitivity_analysis
        }
        
        return report
    
    def create_charts_package(self, output_dir: str) -> List[str]:
        """Create all standard charts and save to files.
        
        Args:
            output_dir: Output directory for charts
            
        Returns:
            List of created chart file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        created_files = []
        total_costs = np.array(self.results.total_costs)
        
        try:
            # Histogram
            hist_fig = self.chart_generator.create_histogram(
                total_costs, 
                title=f"Cost Distribution - {self.results.project_id}",
                percentiles=self.results.percentiles
            )
            hist_path = output_path / "cost_histogram.html"
            hist_fig.write_html(str(hist_path))
            created_files.append(str(hist_path))
            
            # CDF curve
            cdf_fig = self.chart_generator.create_cdf_curve(
                total_costs,
                title=f"Cumulative Distribution - {self.results.project_id}"
            )
            cdf_path = output_path / "cost_cdf.html"
            cdf_fig.write_html(str(cdf_path))
            created_files.append(str(cdf_path))
            
            # Tornado diagram (if sensitivity analysis available)
            if self.results.sensitivity_analysis:
                tornado_fig = self.chart_generator.create_tornado_diagram(
                    self.results.sensitivity_analysis.get('tornado_analysis', {})
                )
                tornado_path = output_path / "sensitivity_tornado.html"
                tornado_fig.write_html(str(tornado_path))
                created_files.append(str(tornado_path))
            
            # Risk contribution Pareto (if risk analysis available)
            if self.results.risk_contributions:
                risk_data = {risk_id: data.get('variance_contribution', 0) 
                           for risk_id, data in self.results.risk_contributions.items()}
                pareto_fig = self.chart_generator.create_pareto_chart(
                    risk_data, title="Risk Contribution Analysis"
                )
                pareto_path = output_path / "risk_pareto.html"
                pareto_fig.write_html(str(pareto_path))
                created_files.append(str(pareto_path))
            
            # WBS breakdown
            if self.results.wbs_statistics:
                wbs_fig = self.chart_generator.create_wbs_breakdown_chart(
                    self.results.wbs_statistics, metric='mean'
                )
                wbs_path = output_path / "wbs_breakdown.html"
                wbs_fig.write_html(str(wbs_path))
                created_files.append(str(wbs_path))
            
        except Exception as e:
            print(f"Warning: Error creating some charts: {e}")
        
        return created_files
    
    def _generate_key_insights(self) -> List[str]:
        """Generate key insights from results."""
        insights = []
        
        percentiles = self.results.percentiles
        base_cost = self.results.base_cost
        
        # Cost range insight
        if 'P10' in percentiles and 'P90' in percentiles:
            range_pct = ((percentiles['P90'] - percentiles['P10']) / base_cost) * 100
            insights.append(f"Cost uncertainty range: {range_pct:.0f}% of base cost (P10-P90)")
        
        # Expected vs base cost
        if 'P50' in percentiles:
            expected_increase = ((percentiles['P50'] - base_cost) / base_cost) * 100
            insights.append(f"Expected cost increase: {expected_increase:.1f}% above base case")
        
        # Planning contingency
        if 'P80' in percentiles:
            contingency_pct = ((percentiles['P80'] - base_cost) / base_cost) * 100
            insights.append(f"Recommended contingency (P80): {contingency_pct:.1f}% of base cost")
        
        # Risk insights
        if self.results.risk_contributions:
            top_risk = max(self.results.risk_contributions.items(), 
                          key=lambda x: x[1].get('variance_contribution', 0))
            risk_id, risk_data = top_risk
            contribution = risk_data.get('variance_contribution', 0) * 100
            insights.append(f"Top risk driver: {risk_id} ({contribution:.1f}% of variance)")
        
        return insights


def create_simple_summary_report(results: SimulationResults) -> str:
    """Create a simple text summary report.
    
    Args:
        results: Simulation results
        
    Returns:
        Formatted text report
    """
    report_lines = [
        f"Risk Analysis Summary - {results.project_id}",
        "=" * 50,
        "",
        f"Base Cost: ${results.base_cost:,.0f} {results.currency}",
        f"Simulation Iterations: {results.n_samples:,}",
        "",
        "Cost Percentiles:",
    ]
    
    for percentile, value in results.percentiles.items():
        report_lines.append(f"  {percentile}: ${value:,.0f}")
    
    if results.contingency_recommendations:
        contingency = results.contingency_recommendations
        report_lines.extend([
            "",
            "Contingency Recommendations:",
            f"  Recommended Contingency: ${contingency.get('contingency', 0):,.0f} ({contingency.get('contingency_percent', 0):.1f}%)",
            f"  Management Reserve: ${contingency.get('management_reserve', 0):,.0f}",
            f"  Total Buffer: ${contingency.get('total_buffer', 0):,.0f} ({contingency.get('total_buffer_percent', 0):.1f}%)"
        ])
    
    return "\n".join(report_lines)