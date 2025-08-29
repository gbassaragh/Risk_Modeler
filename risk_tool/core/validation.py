"""Validation engine for input data and simulation results.

Provides comprehensive validation with helpful error messages.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
from pathlib import Path

from .distributions import validate_distribution_config
from .cost_models import Project, WBSItem
from .risk_driver import RiskItem


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ValidationWarning(UserWarning):
    """Custom warning for validation issues."""
    pass


class ValidationEngine:
    """Comprehensive validation engine."""
    
    def __init__(self, fail_on_warnings: bool = False):
        """Initialize validation engine.
        
        Args:
            fail_on_warnings: Whether to treat warnings as errors
        """
        self.fail_on_warnings = fail_on_warnings
        self.errors = []
        self.warnings = []
    
    def validate_project(self, project_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Validate project data comprehensively.
        
        Args:
            project_data: Project dictionary
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        try:
            # Basic structure validation
            self._validate_project_structure(project_data)
            
            # Create project object for detailed validation
            project = Project(**project_data)
            
            # Validate WBS items
            self._validate_wbs_items(project.wbs)
            
            # Validate escalation
            if project.escalation:
                self._validate_escalation_config(project.escalation.dict())
            
            # Business logic validation
            self._validate_project_business_logic(project)
            
        except Exception as e:
            self.errors.append(f"Project validation failed: {e}")
        
        is_valid = len(self.errors) == 0 and (not self.fail_on_warnings or len(self.warnings) == 0)
        return is_valid, self.errors.copy(), self.warnings.copy()
    
    def validate_risks(self, risks_data: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
        """Validate risk data comprehensively.
        
        Args:
            risks_data: List of risk dictionaries
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        if not risks_data:
            self.warnings.append("No risks provided - simulation will only include base uncertainty")
            return True, [], self.warnings.copy()
        
        try:
            # Validate each risk
            risk_ids = set()
            
            for i, risk_data in enumerate(risks_data):
                try:
                    risk = RiskItem(**risk_data)
                    
                    # Check for duplicate IDs
                    if risk.id in risk_ids:
                        self.errors.append(f"Risk {i}: Duplicate risk ID '{risk.id}'")
                    risk_ids.add(risk.id)
                    
                    # Validate distributions
                    self._validate_risk_distributions(risk, i)
                    
                    # Validate application logic
                    self._validate_risk_application(risk, i)
                    
                    # Business logic validation
                    self._validate_risk_business_logic(risk, i)
                    
                except Exception as e:
                    self.errors.append(f"Risk {i}: {e}")
            
            # Cross-risk validation
            self._validate_risk_portfolio(risks_data)
            
        except Exception as e:
            self.errors.append(f"Risk validation failed: {e}")
        
        is_valid = len(self.errors) == 0 and (not self.fail_on_warnings or len(self.warnings) == 0)
        return is_valid, self.errors.copy(), self.warnings.copy()
    
    def validate_correlations(self, 
                            correlations_data: List[Dict[str, Any]],
                            project_data: Dict[str, Any],
                            risks_data: Optional[List[Dict[str, Any]]] = None) -> Tuple[bool, List[str], List[str]]:
        """Validate correlation specifications.
        
        Args:
            correlations_data: List of correlation dictionaries
            project_data: Project data for reference
            risks_data: Risk data for reference
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        if not correlations_data:
            return True, [], []
        
        # Build available variable names
        available_vars = set()
        
        # Add WBS codes and names
        if 'wbs' in project_data:
            for wbs_item in project_data['wbs']:
                available_vars.add(wbs_item.get('code', ''))
                available_vars.add(wbs_item.get('name', ''))
                
                # Add tags
                tags = wbs_item.get('tags', [])
                if isinstance(tags, list):
                    available_vars.update(tags)
        
        # Add risk IDs
        if risks_data:
            for risk in risks_data:
                available_vars.add(risk.get('id', ''))
        
        # Add common variables
        available_vars.update([
            'indirects_per_day', 'escalation_rate', 
            'commodity_steel_aluminum', 'weather_sensitive'
        ])
        
        # Validate each correlation
        for i, corr_data in enumerate(correlations_data):
            try:
                # Basic structure
                if 'pair' not in corr_data or 'rho' not in corr_data:
                    self.errors.append(f"Correlation {i}: Missing required fields 'pair' and 'rho'")
                    continue
                
                pair = corr_data['pair']
                rho = corr_data['rho']
                
                # Validate pair
                if not isinstance(pair, list) or len(pair) != 2:
                    self.errors.append(f"Correlation {i}: 'pair' must be list of exactly 2 elements")
                    continue
                
                # Validate correlation coefficient
                if not isinstance(rho, (int, float)) or abs(rho) > 1:
                    self.errors.append(f"Correlation {i}: 'rho' must be number between -1 and 1")
                    continue
                
                # Check if variables exist
                var1, var2 = pair
                if var1 not in available_vars:
                    self.warnings.append(f"Correlation {i}: Variable '{var1}' not found in project/risks")
                
                if var2 not in available_vars:
                    self.warnings.append(f"Correlation {i}: Variable '{var2}' not found in project/risks")
                
                # Validate correlation strength
                if abs(rho) > 0.9:
                    self.warnings.append(f"Correlation {i}: Very strong correlation ({rho:.2f}) - ensure this is realistic")
                
            except Exception as e:
                self.errors.append(f"Correlation {i}: {e}")
        
        is_valid = len(self.errors) == 0 and (not self.fail_on_warnings or len(self.warnings) == 0)
        return is_valid, self.errors.copy(), self.warnings.copy()
    
    def validate_simulation_config(self, config_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Validate simulation configuration.
        
        Args:
            config_data: Configuration dictionary
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        try:
            # Iterations validation
            iterations = config_data.get('iterations', 50000)
            if not isinstance(iterations, int) or iterations <= 0:
                self.errors.append("Iterations must be positive integer")
            elif iterations < 1000:
                self.warnings.append("Less than 1000 iterations may give unstable results")
            elif iterations > 200000:
                self.warnings.append("More than 200,000 iterations may be slow")
            
            # Sampling method validation
            sampling = config_data.get('sampling', 'LHS')
            if sampling not in ['LHS', 'MC']:
                self.errors.append("Sampling method must be 'LHS' or 'MC'")
            
            # Random seed validation
            seed = config_data.get('random_seed')
            if seed is not None and (not isinstance(seed, int) or seed < 0):
                self.errors.append("Random seed must be non-negative integer or None")
            
            # Output validation
            outputs = config_data.get('outputs', {})
            if 'percentiles' in outputs:
                percentiles = outputs['percentiles']
                if not isinstance(percentiles, list) or not all(0 <= p <= 100 for p in percentiles):
                    self.errors.append("Percentiles must be list of numbers between 0 and 100")
            
            # Performance validation
            performance = config_data.get('performance', {})
            if 'num_threads' in performance:
                threads = performance['num_threads']
                if not isinstance(threads, int) or (threads != -1 and threads <= 0):
                    self.errors.append("num_threads must be positive integer or -1 for auto")
            
        except Exception as e:
            self.errors.append(f"Configuration validation failed: {e}")
        
        is_valid = len(self.errors) == 0 and (not self.fail_on_warnings or len(self.warnings) == 0)
        return is_valid, self.errors.copy(), self.warnings.copy()
    
    def validate_file_access(self, file_paths: List[str]) -> Tuple[bool, List[str], List[str]]:
        """Validate file access and formats.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        for file_path in file_paths:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                self.errors.append(f"File not found: {file_path}")
                continue
            
            # Check if readable
            if not path.is_file():
                self.errors.append(f"Path is not a file: {file_path}")
                continue
            
            # Check file extension
            supported_extensions = {'.xlsx', '.xls', '.csv', '.json', '.yaml', '.yml'}
            if path.suffix.lower() not in supported_extensions:
                self.warnings.append(f"File extension '{path.suffix}' may not be supported: {file_path}")
            
            # Check file size
            try:
                file_size = path.stat().st_size
                if file_size > 100 * 1024 * 1024:  # 100MB
                    self.warnings.append(f"Large file ({file_size / 1024 / 1024:.1f} MB): {file_path}")
                elif file_size == 0:
                    self.errors.append(f"Empty file: {file_path}")
            except Exception as e:
                self.warnings.append(f"Cannot check file size: {file_path} - {e}")
        
        is_valid = len(self.errors) == 0 and (not self.fail_on_warnings or len(self.warnings) == 0)
        return is_valid, self.errors.copy(), self.warnings.copy()
    
    def _validate_project_structure(self, project_data: Dict[str, Any]):
        """Validate basic project structure."""
        required_fields = ['id', 'type', 'wbs']
        
        for field in required_fields:
            if field not in project_data:
                self.errors.append(f"Missing required project field: {field}")
        
        # Validate project type
        valid_types = ['TransmissionLine', 'Substation', 'Hybrid']
        if project_data.get('type') not in valid_types:
            self.errors.append(f"Project type must be one of: {valid_types}")
        
        # Validate WBS structure
        wbs = project_data.get('wbs', [])
        if not isinstance(wbs, list) or len(wbs) == 0:
            self.errors.append("WBS must be non-empty list")
    
    def _validate_wbs_items(self, wbs_items: List[WBSItem]):
        """Validate WBS items."""
        codes = []
        
        for i, item in enumerate(wbs_items):
            # Check for duplicate codes
            if item.code in codes:
                self.errors.append(f"WBS item {i}: Duplicate code '{item.code}'")
            codes.append(item.code)
            
            # Validate distributions
            if item.dist_quantity:
                try:
                    validate_distribution_config(item.dist_quantity)
                except Exception as e:
                    self.errors.append(f"WBS {item.code}: Invalid quantity distribution - {e}")
            
            if item.dist_unit_cost:
                try:
                    validate_distribution_config(item.dist_unit_cost)
                except Exception as e:
                    self.errors.append(f"WBS {item.code}: Invalid unit cost distribution - {e}")
            
            # Business logic validation
            if item.quantity <= 0:
                self.errors.append(f"WBS {item.code}: Quantity must be positive")
            
            if item.unit_cost <= 0:
                self.errors.append(f"WBS {item.code}: Unit cost must be positive")
            
            if item.indirect_factor < 0 or item.indirect_factor > 2:
                self.warnings.append(f"WBS {item.code}: Indirect factor {item.indirect_factor} seems unusual (0-2 typical)")
    
    def _validate_escalation_config(self, escalation_data: Dict[str, Any]):
        """Validate escalation configuration."""
        if 'annual_rate_dist' in escalation_data:
            try:
                validate_distribution_config(escalation_data['annual_rate_dist'])
            except Exception as e:
                self.errors.append(f"Escalation: Invalid rate distribution - {e}")
        
        # Validate years
        base_year = escalation_data.get('base_year')
        target_year = escalation_data.get('target_year')
        
        if base_year and target_year:
            try:
                base = int(base_year)
                target = int(target_year)
                
                if target < base:
                    self.errors.append("Target year must be after base year")
                elif target - base > 20:
                    self.warnings.append("Escalation period > 20 years may be unrealistic")
            except ValueError:
                self.errors.append("Invalid year format")
    
    def _validate_project_business_logic(self, project: Project):
        """Validate business logic for project."""
        # Total cost reasonableness
        base_total = project.calculate_base_total()
        
        if base_total <= 0:
            self.errors.append("Total base cost must be positive")
        elif base_total < 10000:
            self.warnings.append("Very low total cost - check units and values")
        elif base_total > 1e9:
            self.warnings.append("Very high total cost - ensure values are correct")
        
        # WBS coverage check
        wbs_types = set()
        for item in project.wbs:
            if any(tag in item.tags for tag in ['structures', 'conductor', 'foundations']):
                wbs_types.add('construction')
            if any(tag in item.tags for tag in ['management', 'engineering']):
                wbs_types.add('soft_costs')
        
        if 'construction' not in wbs_types:
            self.warnings.append("No construction WBS items identified - check tags")
    
    def _validate_risk_distributions(self, risk: RiskItem, index: int):
        """Validate risk distributions."""
        try:
            validate_distribution_config(risk.impact_dist)
        except Exception as e:
            self.errors.append(f"Risk {index} ({risk.id}): Invalid impact distribution - {e}")
        
        if risk.schedule_days_dist:
            try:
                validate_distribution_config(risk.schedule_days_dist)
            except Exception as e:
                self.errors.append(f"Risk {index} ({risk.id}): Invalid schedule distribution - {e}")
    
    def _validate_risk_application(self, risk: RiskItem, index: int):
        """Validate risk application logic."""
        if not risk.applies_to and not risk.applies_by_tag:
            self.errors.append(f"Risk {index} ({risk.id}): Must specify either applies_to or applies_by_tag")
        
        if risk.applies_to and risk.applies_by_tag:
            self.warnings.append(f"Risk {index} ({risk.id}): Specifies both applies_to and applies_by_tag")
    
    def _validate_risk_business_logic(self, risk: RiskItem, index: int):
        """Validate risk business logic."""
        # Probability bounds
        if risk.probability <= 0 or risk.probability > 1:
            self.errors.append(f"Risk {index} ({risk.id}): Probability must be between 0 and 1")
        elif risk.probability > 0.9:
            self.warnings.append(f"Risk {index} ({risk.id}): Very high probability ({risk.probability}) - is this realistic?")
        
        # Impact distribution reasonableness
        if risk.impact_mode == "multiplicative":
            # For multiplicative risks, impacts should typically be around 1.0
            dist = risk.impact_dist
            if dist.get('type') == 'pert':
                min_val = dist.get('min', 1)
                max_val = dist.get('max', 1)
                if min_val < 0.1 or max_val > 10:
                    self.warnings.append(f"Risk {index} ({risk.id}): Extreme multiplicative impact range")
        else:  # additive
            # For additive risks, impacts should be reasonable cost amounts
            dist = risk.impact_dist
            if dist.get('type') == 'pert':
                max_val = dist.get('max', 0)
                if max_val > 10000000:  # $10M
                    self.warnings.append(f"Risk {index} ({risk.id}): Very large additive impact")
    
    def _validate_risk_portfolio(self, risks_data: List[Dict[str, Any]]):
        """Validate risk portfolio as a whole."""
        # Check for risk concentration
        categories = {}
        high_prob_risks = 0
        
        for risk_data in risks_data:
            category = risk_data.get('category', 'Unknown')
            categories[category] = categories.get(category, 0) + 1
            
            if risk_data.get('probability', 0) > 0.7:
                high_prob_risks += 1
        
        # Category concentration
        total_risks = len(risks_data)
        for category, count in categories.items():
            if count / total_risks > 0.6:
                self.warnings.append(f"Risk concentration: {count}/{total_risks} risks in '{category}' category")
        
        # High probability concentration
        if high_prob_risks / total_risks > 0.5:
            self.warnings.append(f"Many high-probability risks ({high_prob_risks}/{total_risks}) - validate realism")


def validate_simulation_inputs(project_data: Dict[str, Any],
                             risks_data: Optional[List[Dict[str, Any]]] = None,
                             correlations_data: Optional[List[Dict[str, Any]]] = None,
                             config_data: Optional[Dict[str, Any]] = None,
                             fail_on_warnings: bool = False) -> Tuple[bool, List[str], List[str]]:
    """Validate all simulation inputs comprehensively.
    
    Args:
        project_data: Project dictionary
        risks_data: Optional risks data
        correlations_data: Optional correlations data
        config_data: Optional configuration data
        fail_on_warnings: Whether to treat warnings as errors
        
    Returns:
        Tuple of (is_valid, all_errors, all_warnings)
    """
    validator = ValidationEngine(fail_on_warnings)
    
    all_errors = []
    all_warnings = []
    
    # Validate project
    valid, errors, warnings = validator.validate_project(project_data)
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    
    # Validate risks
    if risks_data:
        valid, errors, warnings = validator.validate_risks(risks_data)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
    
    # Validate correlations
    if correlations_data:
        valid, errors, warnings = validator.validate_correlations(
            correlations_data, project_data, risks_data
        )
        all_errors.extend(errors)
        all_warnings.extend(warnings)
    
    # Validate configuration
    if config_data:
        valid, errors, warnings = validator.validate_simulation_config(config_data)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
    
    is_valid = len(all_errors) == 0 and (not fail_on_warnings or len(all_warnings) == 0)
    
    return is_valid, all_errors, all_warnings