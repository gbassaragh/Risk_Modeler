"""Escalation and market factor modeling.

Implements time-based escalation and commodity price indexing.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date
from .distributions import DistributionSampler


class EscalationEngine:
    """Handles cost escalation calculations."""

    def __init__(self, random_state: np.random.RandomState):
        """Initialize escalation engine.

        Args:
            random_state: Random number generator
        """
        self.random_state = random_state

    def calculate_escalation_factors(
        self,
        base_year: str,
        target_year: str,
        annual_rate_dist: Dict[str, Any],
        n_samples: int,
    ) -> np.ndarray:
        """Calculate escalation factors from base to target year.

        Args:
            base_year: Base year (e.g., "2025")
            target_year: Target year (e.g., "2027")
            annual_rate_dist: Distribution of annual escalation rates
            n_samples: Number of samples

        Returns:
            Array of cumulative escalation factors
        """
        # Calculate number of years
        years_diff = self._calculate_year_difference(base_year, target_year)

        if years_diff <= 0:
            return np.ones(n_samples)

        # Sample annual rates
        annual_rates = DistributionSampler.sample(
            annual_rate_dist, n_samples, self.random_state
        )

        # Compound escalation
        escalation_factors = np.power(1 + annual_rates, years_diff)

        return escalation_factors

    def apply_escalation_to_costs(
        self,
        base_costs: Dict[str, np.ndarray],
        escalation_factors: np.ndarray,
        escalation_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Apply escalation to WBS costs.

        Args:
            base_costs: Dictionary of base costs by WBS code
            escalation_factors: Escalation factor array
            escalation_mapping: Optional mapping of WBS codes to escalation categories

        Returns:
            Escalated costs
        """
        escalated_costs = {}

        for wbs_code, costs in base_costs.items():
            if escalation_mapping and wbs_code in escalation_mapping:
                # Use specific escalation for this WBS item
                # For now, apply uniform escalation
                escalated_costs[wbs_code] = costs * escalation_factors
            else:
                # Apply uniform escalation
                escalated_costs[wbs_code] = costs * escalation_factors

        return escalated_costs

    @staticmethod
    def _calculate_year_difference(base_year: str, target_year: str) -> float:
        """Calculate difference in years between dates.

        Args:
            base_year: Base year string
            target_year: Target year string

        Returns:
            Difference in years
        """
        try:
            base_date = datetime.strptime(base_year, "%Y")
            target_date = datetime.strptime(target_year, "%Y")
            return (target_date - base_date).days / 365.25
        except ValueError:
            # Try parsing as full dates
            try:
                base_date = datetime.strptime(base_year, "%Y-%m-%d")
                target_date = datetime.strptime(target_year, "%Y-%m-%d")
                return (target_date - base_date).days / 365.25
            except ValueError:
                raise ValueError(f"Cannot parse dates: {base_year}, {target_year}")


class MarketIndexEngine:
    """Handles market index-based cost adjustments."""

    def __init__(self, random_state: np.random.RandomState):
        """Initialize market index engine.

        Args:
            random_state: Random number generator
        """
        self.random_state = random_state

    def simulate_commodity_indices(
        self, index_configs: Dict[str, Dict[str, Any]], n_samples: int
    ) -> Dict[str, np.ndarray]:
        """Simulate commodity price indices.

        Args:
            index_configs: Dictionary of index configurations
            n_samples: Number of samples

        Returns:
            Dictionary of simulated index values
        """
        indices = {}

        for index_name, config in index_configs.items():
            if "distribution" in config:
                # Sample from distribution
                index_values = DistributionSampler.sample(
                    config["distribution"], n_samples, self.random_state
                )

                # Ensure positive values
                index_values = np.maximum(index_values, 0.01)

                indices[index_name] = index_values
            else:
                # Fixed index
                indices[index_name] = np.full(n_samples, config.get("value", 1.0))

        return indices

    def apply_market_factors(
        self,
        base_costs: Dict[str, np.ndarray],
        commodity_indices: Dict[str, np.ndarray],
        wbs_to_commodity_mapping: Dict[str, str],
    ) -> Dict[str, np.ndarray]:
        """Apply market factors to costs based on WBS-commodity mapping.

        Args:
            base_costs: Base costs by WBS code
            commodity_indices: Commodity index values
            wbs_to_commodity_mapping: Mapping of WBS codes to commodity indices

        Returns:
            Market-adjusted costs
        """
        adjusted_costs = {}

        for wbs_code, costs in base_costs.items():
            if wbs_code in wbs_to_commodity_mapping:
                commodity = wbs_to_commodity_mapping[wbs_code]
                if commodity in commodity_indices:
                    # Apply market factor
                    adjusted_costs[wbs_code] = costs * commodity_indices[commodity]
                else:
                    # No market factor available
                    adjusted_costs[wbs_code] = costs
            else:
                # No market factor mapping
                adjusted_costs[wbs_code] = costs

        return adjusted_costs


class ScheduleEscalation:
    """Handles schedule-driven cost escalation."""

    def __init__(self, random_state: np.random.RandomState):
        """Initialize schedule escalation.

        Args:
            random_state: Random number generator
        """
        self.random_state = random_state

    def calculate_schedule_costs(
        self,
        schedule_delays: np.ndarray,
        indirects_per_day: float,
        escalation_engine: EscalationEngine,
        base_year: str,
        annual_rate_dist: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Calculate additional costs due to schedule delays.

        Args:
            schedule_delays: Array of schedule delays in days
            indirects_per_day: Daily indirect cost rate
            escalation_engine: Escalation engine instance
            base_year: Base year for escalation
            annual_rate_dist: Optional escalation rate distribution

        Returns:
            Array of additional schedule-driven costs
        """
        n_samples = len(schedule_delays)

        # Base schedule costs
        schedule_costs = schedule_delays * indirects_per_day

        # Apply escalation if delays push into future years
        if annual_rate_dist:
            # Assume delays translate to fractional year escalation
            delay_years = schedule_delays / 365.25

            # Sample escalation rates
            annual_rates = DistributionSampler.sample(
                annual_rate_dist, n_samples, self.random_state
            )

            # Apply escalation based on delay duration
            escalation_factors = np.power(1 + annual_rates, delay_years)
            schedule_costs *= escalation_factors

        return np.maximum(schedule_costs, 0)


class ProductivityFactors:
    """Handles productivity-related cost adjustments."""

    @staticmethod
    def apply_weather_productivity(
        costs: Dict[str, np.ndarray],
        weather_factors: np.ndarray,
        weather_sensitive_tags: List[str],
        wbs_items: List,
    ) -> Dict[str, np.ndarray]:
        """Apply weather-related productivity factors.

        Args:
            costs: Base costs by WBS code
            weather_factors: Weather productivity factors (e.g., 0.8 for 20% reduction)
            weather_sensitive_tags: Tags indicating weather sensitivity
            wbs_items: List of WBS items to check tags

        Returns:
            Weather-adjusted costs
        """
        adjusted_costs = costs.copy()

        # Create mapping of WBS codes to weather sensitivity
        weather_sensitive_codes = set()
        for item in wbs_items:
            if any(tag in weather_sensitive_tags for tag in item.tags):
                weather_sensitive_codes.add(item.code)

        # Apply weather factors
        for wbs_code in weather_sensitive_codes:
            if wbs_code in adjusted_costs:
                # Weather factors typically increase costs (factor > 1) for adverse weather
                adjusted_costs[wbs_code] *= weather_factors

        return adjusted_costs

    @staticmethod
    def apply_learning_curve(
        costs: Dict[str, np.ndarray], project_sequence: int, learning_rate: float = 0.85
    ) -> Dict[str, np.ndarray]:
        """Apply learning curve cost reductions.

        Args:
            costs: Base costs
            project_sequence: Project number in sequence (1-based)
            learning_rate: Learning curve rate (e.g., 0.85 for 15% cost reduction per doubling)

        Returns:
            Learning-adjusted costs
        """
        if project_sequence <= 1:
            return costs

        # Learning curve: Cost_n = Cost_1 * (n ^ log(learning_rate) / log(2))
        learning_factor = project_sequence ** (np.log(learning_rate) / np.log(2))

        adjusted_costs = {}
        for code, cost_array in costs.items():
            adjusted_costs[code] = cost_array * learning_factor

        return adjusted_costs


def create_default_escalation_config(
    base_year: str = "2025", mean_rate: float = 0.035, std_rate: float = 0.015
) -> Dict[str, Any]:
    """Create default escalation configuration.

    Args:
        base_year: Base year
        mean_rate: Mean annual escalation rate
        std_rate: Standard deviation of escalation rate

    Returns:
        Escalation configuration dictionary
    """
    return {
        "annual_rate_dist": {
            "type": "normal",
            "mean": mean_rate,
            "stdev": std_rate,
            "truncate_low": -0.1,  # Limit deflation
            "truncate_high": 0.15,  # Limit inflation
        },
        "base_year": base_year,
    }


def create_commodity_index_config() -> Dict[str, Dict[str, Any]]:
    """Create default commodity index configurations.

    Returns:
        Commodity index configurations
    """
    return {
        "steel": {"distribution": {"type": "lognormal", "mean": 1.0, "sigma": 0.20}},
        "aluminum": {"distribution": {"type": "lognormal", "mean": 1.0, "sigma": 0.25}},
        "copper": {"distribution": {"type": "lognormal", "mean": 1.0, "sigma": 0.30}},
        "labor": {
            "distribution": {
                "type": "normal",
                "mean": 1.0,
                "stdev": 0.10,
                "truncate_low": 0.8,
                "truncate_high": 1.3,
            }
        },
    }


def create_wbs_commodity_mapping() -> Dict[str, str]:
    """Create default WBS to commodity mapping.

    Returns:
        Dictionary mapping WBS patterns to commodity indices
    """
    return {
        # Transmission line mappings
        "conductor": "aluminum",
        "structures": "steel",
        "foundations": "steel",
        "hardware": "steel",
        "grounding": "copper",
        # Substation mappings
        "transformers": "steel",
        "switchgear": "steel",
        "protection": "steel",
        "control_building": "steel",
        # General
        "construction": "labor",
        "installation": "labor",
        "commissioning": "labor",
    }
