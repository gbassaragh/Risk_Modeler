"""CSV file I/O operations for project and risk data.

Handles CSV import/export with flexible column mapping.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
from pathlib import Path
import json


class CSVImporter:
    """Imports project and risk data from CSV files."""
    
    def __init__(self):
        """Initialize CSV importer."""
        self.default_wbs_columns = {
            'code': ['code', 'wbs_code', 'item_code'],
            'name': ['name', 'description', 'item_name'],
            'quantity': ['quantity', 'qty'],
            'uom': ['uom', 'unit', 'units'],
            'unit_cost': ['unit_cost', 'rate', 'unit_rate'],
            'tags': ['tags', 'categories'],
            'indirect_factor': ['indirect_factor', 'indirects', 'overhead_factor']
        }
        
        self.default_risk_columns = {
            'id': ['id', 'risk_id', 'code'],
            'title': ['title', 'name', 'description'],
            'category': ['category', 'type'],
            'probability': ['probability', 'p', 'likelihood'],
            'impact_mode': ['impact_mode', 'mode'],
            'applies_to': ['applies_to', 'wbs_codes'],
            'applies_by_tag': ['applies_by_tag', 'tags']
        }
    
    def import_project_from_csv(self, 
                               wbs_file: str, 
                               project_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Import project data from CSV file.
        
        Args:
            wbs_file: Path to CSV file with WBS data
            project_metadata: Optional project metadata dictionary
            
        Returns:
            Project dictionary
        """
        try:
            # Read CSV file
            df = pd.read_csv(wbs_file)
            
            # Clean column names
            df.columns = df.columns.str.strip().str.lower()
            
            # Map columns
            column_mapping = self._map_wbs_columns(df.columns.tolist())
            
            # Validate required columns
            required = ['code', 'name', 'quantity', 'uom', 'unit_cost']
            missing = [col for col in required if col not in column_mapping]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            # Process WBS items
            wbs_items = []
            for _, row in df.iterrows():
                if pd.isna(row.get(column_mapping.get('code'))):
                    continue
                
                wbs_item = self._process_wbs_row(row, column_mapping)
                if wbs_item:
                    wbs_items.append(wbs_item)
            
            # Use provided metadata or defaults
            if not project_metadata:
                project_metadata = {
                    'id': f'PROJECT-{Path(wbs_file).stem}',
                    'type': 'TransmissionLine',
                    'currency': 'USD',
                    'base_date': '2025-01-01',
                    'region': 'US',
                    'indirects_per_day': 25000.0
                }
            
            project = {
                **project_metadata,
                'wbs': wbs_items
            }
            
            return project
            
        except Exception as e:
            raise ValueError(f"Error importing project from {wbs_file}: {e}")
    
    def import_risks_from_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Import risk data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of risk dictionaries
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Clean column names
            df.columns = df.columns.str.strip().str.lower()
            
            # Map columns
            column_mapping = self._map_risk_columns(df.columns.tolist())
            
            # Validate required columns
            required = ['id', 'title', 'probability', 'impact_mode']
            missing = [col for col in required if col not in column_mapping]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            # Process risk items
            risks = []
            for _, row in df.iterrows():
                if pd.isna(row.get(column_mapping.get('id'))):
                    continue
                
                risk_item = self._process_risk_row(row, column_mapping)
                if risk_item:
                    risks.append(risk_item)
            
            return risks
            
        except Exception as e:
            raise ValueError(f"Error importing risks from {file_path}: {e}")
    
    def import_correlations_from_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Import correlation data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of correlation dictionaries
        """
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.lower()
            
            correlations = []
            
            for _, row in df.iterrows():
                if pd.isna(row.get('variable1')) or pd.isna(row.get('variable2')):
                    continue
                
                corr = {
                    'pair': [str(row['variable1']).strip(), str(row['variable2']).strip()],
                    'rho': float(row.get('correlation', 0.5)),
                    'method': str(row.get('method', 'spearman')).strip().lower()
                }
                
                correlations.append(corr)
            
            return correlations
            
        except Exception as e:
            raise ValueError(f"Error importing correlations from {file_path}: {e}")
    
    def _map_wbs_columns(self, columns: List[str]) -> Dict[str, str]:
        """Map WBS columns to standard names."""
        mapping = {}
        
        for standard_name, possible_names in self.default_wbs_columns.items():
            for col in columns:
                if col in possible_names:
                    mapping[standard_name] = col
                    break
        
        return mapping
    
    def _map_risk_columns(self, columns: List[str]) -> Dict[str, str]:
        """Map risk columns to standard names."""
        mapping = {}
        
        for standard_name, possible_names in self.default_risk_columns.items():
            for col in columns:
                if col in possible_names:
                    mapping[standard_name] = col
                    break
        
        return mapping
    
    def _process_wbs_row(self, row: pd.Series, column_mapping: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Process a WBS row into WBS item dictionary."""
        try:
            wbs_item = {
                'code': str(row[column_mapping['code']]).strip(),
                'name': str(row[column_mapping['name']]).strip(),
                'quantity': float(row[column_mapping['quantity']]),
                'uom': str(row[column_mapping['uom']]).strip(),
                'unit_cost': float(row[column_mapping['unit_cost']]),
                'tags': [],
                'indirect_factor': 0.0
            }
            
            # Process optional fields
            if 'tags' in column_mapping and not pd.isna(row[column_mapping['tags']]):
                tags_str = str(row[column_mapping['tags']])
                wbs_item['tags'] = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
            
            if 'indirect_factor' in column_mapping and not pd.isna(row[column_mapping['indirect_factor']]):
                wbs_item['indirect_factor'] = float(row[column_mapping['indirect_factor']])
            
            # Process distributions
            self._add_distributions_to_wbs(wbs_item, row, column_mapping)
            
            return wbs_item
            
        except Exception as e:
            warnings.warn(f"Error processing WBS row {row.get('code', 'unknown')}: {e}")
            return None
    
    def _process_risk_row(self, row: pd.Series, column_mapping: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Process a risk row into risk item dictionary."""
        try:
            risk_item = {
                'id': str(row[column_mapping['id']]).strip(),
                'title': str(row[column_mapping['title']]).strip(),
                'category': str(row[column_mapping.get('category', 'General')]).strip(),
                'probability': float(row[column_mapping['probability']]),
                'impact_mode': str(row[column_mapping['impact_mode']]).strip().lower(),
                'impact_dist': self._create_default_impact_distribution()
            }
            
            # Validate impact mode
            if risk_item['impact_mode'] not in ['multiplicative', 'additive']:
                risk_item['impact_mode'] = 'multiplicative'
            
            # Process applies_to
            if 'applies_to' in column_mapping and not pd.isna(row[column_mapping['applies_to']]):
                applies_str = str(row[column_mapping['applies_to']])
                risk_item['applies_to'] = [item.strip() for item in applies_str.split(',') if item.strip()]
            
            # Process applies_by_tag
            if 'applies_by_tag' in column_mapping and not pd.isna(row[column_mapping['applies_by_tag']]):
                tags_str = str(row[column_mapping['applies_by_tag']])
                risk_item['applies_by_tag'] = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
            
            # Process impact distribution from multiple columns
            self._add_impact_distribution_to_risk(risk_item, row)
            
            # Process schedule distribution if present
            self._add_schedule_distribution_to_risk(risk_item, row)
            
            return risk_item
            
        except Exception as e:
            warnings.warn(f"Error processing risk row {row.get('id', 'unknown')}: {e}")
            return None
    
    def _add_distributions_to_wbs(self, wbs_item: Dict[str, Any], row: pd.Series, column_mapping: Dict[str, str]):
        """Add distribution information to WBS item from row data."""
        # Look for quantity distribution columns
        qty_cols = {col: row[col] for col in row.index 
                   if 'qty' in col.lower() and ('min' in col.lower() or 'max' in col.lower() or 'mode' in col.lower())}
        
        if qty_cols:
            qty_dist = self._create_distribution_from_columns(qty_cols, 'qty')
            if qty_dist:
                wbs_item['dist_quantity'] = qty_dist
        
        # Look for unit cost distribution columns
        cost_cols = {col: row[col] for col in row.index 
                    if ('cost' in col.lower() or 'rate' in col.lower()) and ('min' in col.lower() or 'max' in col.lower() or 'mode' in col.lower())}
        
        if cost_cols:
            cost_dist = self._create_distribution_from_columns(cost_cols, 'cost')
            if cost_dist:
                wbs_item['dist_unit_cost'] = cost_dist
    
    def _add_impact_distribution_to_risk(self, risk_item: Dict[str, Any], row: pd.Series):
        """Add impact distribution to risk item from row data."""
        impact_cols = {col: row[col] for col in row.index 
                      if 'impact' in col.lower() and ('min' in col.lower() or 'max' in col.lower() or 'mode' in col.lower() or 'most_likely' in col.lower())}
        
        if impact_cols:
            impact_dist = self._create_distribution_from_columns(impact_cols, 'impact')
            if impact_dist:
                risk_item['impact_dist'] = impact_dist
    
    def _add_schedule_distribution_to_risk(self, risk_item: Dict[str, Any], row: pd.Series):
        """Add schedule distribution to risk item from row data."""
        schedule_cols = {col: row[col] for col in row.index 
                        if 'schedule' in col.lower() and ('min' in col.lower() or 'max' in col.lower() or 'days' in col.lower())}
        
        if schedule_cols:
            schedule_dist = self._create_distribution_from_columns(schedule_cols, 'schedule')
            if schedule_dist:
                risk_item['schedule_days_dist'] = schedule_dist
    
    def _create_distribution_from_columns(self, cols_dict: Dict[str, Any], prefix: str) -> Optional[Dict[str, Any]]:
        """Create distribution specification from column data."""
        try:
            # Filter out NaN values
            valid_cols = {col: val for col, val in cols_dict.items() if not pd.isna(val)}
            
            if not valid_cols:
                return None
            
            # Look for PERT distribution (min, most_likely, max)
            min_col = next((col for col in valid_cols if 'min' in col.lower()), None)
            max_col = next((col for col in valid_cols if 'max' in col.lower()), None)
            mode_col = next((col for col in valid_cols if ('mode' in col.lower() or 'most_likely' in col.lower())), None)
            
            if min_col and max_col and mode_col:
                return {
                    'type': 'pert',
                    'min': float(valid_cols[min_col]),
                    'most_likely': float(valid_cols[mode_col]),
                    'max': float(valid_cols[max_col])
                }
            
            # Look for triangular distribution (low, mode, high)
            low_col = next((col for col in valid_cols if 'low' in col.lower()), None)
            high_col = next((col for col in valid_cols if 'high' in col.lower()), None)
            
            if low_col and high_col and mode_col:
                return {
                    'type': 'triangular',
                    'low': float(valid_cols[low_col]),
                    'mode': float(valid_cols[mode_col]),
                    'high': float(valid_cols[high_col])
                }
            
            # Look for normal distribution (mean, std)
            mean_col = next((col for col in valid_cols if 'mean' in col.lower()), None)
            std_col = next((col for col in valid_cols if ('std' in col.lower() or 'stdev' in col.lower())), None)
            
            if mean_col and std_col:
                return {
                    'type': 'normal',
                    'mean': float(valid_cols[mean_col]),
                    'stdev': float(valid_cols[std_col])
                }
            
            return None
            
        except Exception:
            return None
    
    def _create_default_impact_distribution(self) -> Dict[str, Any]:
        """Create default impact distribution."""
        return {
            'type': 'pert',
            'min': 1.0,
            'most_likely': 1.1,
            'max': 1.3
        }


class CSVExporter:
    """Exports results and data to CSV files."""
    
    def export_results_to_csv(self, 
                             results_dict: Dict[str, Any], 
                             output_dir: str) -> List[str]:
        """Export simulation results to CSV files.
        
        Args:
            results_dict: Dictionary containing all results
            output_dir: Output directory path
            
        Returns:
            List of created file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        created_files = []
        
        try:
            # Summary statistics
            if 'summary' in results_dict:
                summary_file = output_path / 'summary.csv'
                summary_df = self._create_summary_dataframe(results_dict['summary'])
                summary_df.to_csv(summary_file, index=False)
                created_files.append(str(summary_file))
            
            # WBS breakdown
            if 'wbs_statistics' in results_dict:
                wbs_file = output_path / 'wbs_breakdown.csv'
                wbs_df = pd.DataFrame(results_dict['wbs_statistics']).T
                wbs_df.to_csv(wbs_file)
                created_files.append(str(wbs_file))
            
            # Risk contributions
            if 'risk_contributions' in results_dict:
                risk_file = output_path / 'risk_analysis.csv'
                risk_df = pd.DataFrame(results_dict['risk_contributions']).T
                risk_df.to_csv(risk_file)
                created_files.append(str(risk_file))
            
            # Percentile data
            if 'percentiles' in results_dict:
                percentiles_file = output_path / 'percentiles.csv'
                percentiles_df = pd.DataFrame([results_dict['percentiles']])
                percentiles_df.to_csv(percentiles_file, index=False)
                created_files.append(str(percentiles_file))
            
            # Simulation sample (limited size)
            if 'simulation_data' in results_dict:
                sim_data = results_dict['simulation_data']
                if isinstance(sim_data, dict) and 'total_costs' in sim_data:
                    sample_file = output_path / 'simulation_sample.csv'
                    
                    # Export sample of simulation data to keep file size manageable
                    sample_size = min(10000, len(sim_data['total_costs']))
                    indices = np.linspace(0, len(sim_data['total_costs'])-1, sample_size, dtype=int)
                    
                    sample_df = pd.DataFrame({
                        'iteration': indices + 1,
                        'total_cost': sim_data['total_costs'][indices]
                    })
                    
                    # Add WBS breakdown if available
                    if 'wbs_costs' in sim_data:
                        for wbs_code, costs in sim_data['wbs_costs'].items():
                            sample_df[f'wbs_{wbs_code}'] = costs[indices]
                    
                    sample_df.to_csv(sample_file, index=False)
                    created_files.append(str(sample_file))
            
            return created_files
            
        except Exception as e:
            raise ValueError(f"Error exporting results to CSV: {e}")
    
    def create_wbs_template_csv(self, file_path: str) -> None:
        """Create WBS template CSV file.
        
        Args:
            file_path: Output file path
        """
        try:
            # Template data with T&D specific examples
            template_data = {
                'code': ['1.1', '2.1', '2.2', '2.3', '3.1', '4.1'],
                'name': [
                    'ROW Clearing & Access',
                    'Steel Structures', 
                    'Foundations',
                    'Conductor & OPGW',
                    'Hardware & Fittings',
                    'Construction Management'
                ],
                'quantity': [10.0, 50, 50, 60.0, 1, 1],
                'uom': ['miles', 'each', 'each', 'miles-conductor', 'lot', 'lot'],
                'unit_cost': [120000.0, 25000.0, 15000.0, 85000.0, 150000.0, 500000.0],
                'tags': [
                    'TL,access,weather_sensitive',
                    'structures,steel',
                    'foundations,concrete',
                    'materials,commodity_aluminum',
                    'hardware,steel',
                    'management'
                ],
                'indirect_factor': [0.12, 0.08, 0.10, 0.05, 0.06, 0.0],
                'qty_min': [9.0, '', '', '', '', ''],
                'qty_mode': [10.0, '', '', '', '', ''],
                'qty_max': [12.0, '', '', '', '', ''],
                'cost_min': [100000, 20000, 12000, '', '', ''],
                'cost_most_likely': [120000, 25000, 15000, '', '', ''],
                'cost_max': [160000, 35000, 20000, '', '', '']
            }
            
            df = pd.DataFrame(template_data)
            df.to_csv(file_path, index=False)
            
        except Exception as e:
            raise ValueError(f"Error creating WBS template at {file_path}: {e}")
    
    def create_risk_template_csv(self, file_path: str) -> None:
        """Create risk template CSV file.
        
        Args:
            file_path: Output file path
        """
        try:
            # Template data with T&D specific risks
            template_data = {
                'id': ['R-ENV-001', 'R-PROC-002', 'R-OUT-003', 'R-WX-004', 'R-GEO-005'],
                'title': [
                    'Wetland matting extent > plan',
                    'Steel pole procurement volatility',
                    'Outage window constraints',
                    'Storm season productivity loss',
                    'Foundation rock conditions'
                ],
                'category': [
                    'Environmental/Access',
                    'Commodity/Market',
                    'Operations',
                    'Weather',
                    'Geological'
                ],
                'probability': [0.45, 0.35, 0.25, 0.60, 0.30],
                'impact_mode': ['multiplicative', 'multiplicative', 'additive', 'multiplicative', 'multiplicative'],
                'impact_min': [1.03, 0.90, 200000, 1.10, 1.05],
                'impact_most_likely': [1.12, 1.00, 400000, 1.25, 1.20],
                'impact_max': [1.35, 1.30, 1200000, 1.50, 1.60],
                'applies_to': ['1.1', '', '', '', '2.2'],
                'applies_by_tag': ['', 'steel,structures', 'indirects', 'weather_sensitive', 'foundations'],
                'schedule_days_min': [0, 0, 0, 5, 0],
                'schedule_days_max': [20, 0, 30, 15, 10]
            }
            
            df = pd.DataFrame(template_data)
            df.to_csv(file_path, index=False)
            
        except Exception as e:
            raise ValueError(f"Error creating risk template at {file_path}: {e}")
    
    def create_correlation_template_csv(self, file_path: str) -> None:
        """Create correlation template CSV file.
        
        Args:
            file_path: Output file path
        """
        try:
            template_data = {
                'variable1': [
                    'commodity_aluminum',
                    'weather_sensitive', 
                    'steel',
                    'foundations'
                ],
                'variable2': [
                    'Conductor & OPGW',
                    'indirects_per_day',
                    'Steel Structures',
                    'Foundation rock conditions'
                ],
                'correlation': [0.6, 0.4, 0.5, 0.7],
                'method': ['spearman', 'spearman', 'spearman', 'spearman']
            }
            
            df = pd.DataFrame(template_data)
            df.to_csv(file_path, index=False)
            
        except Exception as e:
            raise ValueError(f"Error creating correlation template at {file_path}: {e}")
    
    def _create_summary_dataframe(self, summary: Dict[str, Any]) -> pd.DataFrame:
        """Create summary DataFrame from results."""
        data = []
        
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                data.append({'Metric': key, 'Value': value})
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        data.append({'Metric': f"{key}_{sub_key}", 'Value': sub_value})
        
        return pd.DataFrame(data)