"""Schedule modeling with PERT network and cost linkages.

Implements optional schedule-driven cost calculations.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, validator
from .distributions import DistributionSampler


class Activity(BaseModel):
    """Project activity with duration distribution."""
    id: str
    name: str
    duration_dist: Dict[str, Any]
    predecessors: List[str] = []
    successors: List[str] = []
    
    def get_base_duration(self) -> float:
        """Get base case duration."""
        from .distributions import get_distribution_stats
        mean, _ = get_distribution_stats(self.duration_dist)
        return mean


class Schedule(BaseModel):
    """Project schedule with PERT network."""
    activities: List[Activity]
    
    @validator('activities')
    def validate_activities(cls, v):
        if not v:
            raise ValueError("Schedule must have at least one activity")
        
        # Check for duplicate IDs
        ids = [activity.id for activity in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Activity IDs must be unique")
        
        return v
    
    def get_activity_by_id(self, activity_id: str) -> Optional[Activity]:
        """Get activity by ID."""
        for activity in self.activities:
            if activity.id == activity_id:
                return activity
        return None


class ScheduleSimulator:
    """Simulates project schedules using PERT network."""
    
    def __init__(self, schedule: Schedule, random_state: np.random.RandomState):
        """Initialize schedule simulator.
        
        Args:
            schedule: Schedule model
            random_state: Random number generator
        """
        self.schedule = schedule
        self.random_state = random_state
        self.activity_map = {act.id: act for act in schedule.activities}
    
    def simulate_durations(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Simulate activity durations.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Dictionary mapping activity ID to duration arrays
        """
        durations = {}
        
        for activity in self.schedule.activities:
            duration_samples = DistributionSampler.sample(
                activity.duration_dist, n_samples, self.random_state
            )
            # Ensure positive durations
            durations[activity.id] = np.maximum(duration_samples, 0.1)
        
        return durations
    
    def calculate_project_duration(self, 
                                 activity_durations: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Calculate total project duration using critical path method.
        
        Args:
            activity_durations: Simulated activity durations
            
        Returns:
            Tuple of (total_durations, activity_start_times)
        """
        n_samples = len(next(iter(activity_durations.values())))
        
        # Initialize arrays
        start_times = {act.id: np.zeros(n_samples) for act in self.schedule.activities}
        finish_times = {act.id: np.zeros(n_samples) for act in self.schedule.activities}
        
        # Topological sort of activities (simplified - assumes no cycles)
        sorted_activities = self._topological_sort()
        
        # Forward pass - calculate start and finish times
        for activity in sorted_activities:
            activity_id = activity.id
            
            # Start time is max finish time of predecessors
            if activity.predecessors:
                predecessor_finish_times = []
                for pred_id in activity.predecessors:
                    if pred_id in finish_times:
                        predecessor_finish_times.append(finish_times[pred_id])
                
                if predecessor_finish_times:
                    start_times[activity_id] = np.maximum.reduce(predecessor_finish_times)
            
            # Finish time = start time + duration
            finish_times[activity_id] = (start_times[activity_id] + 
                                       activity_durations[activity_id])
        
        # Project duration is max finish time
        if finish_times:
            project_duration = np.maximum.reduce(list(finish_times.values()))
        else:
            project_duration = np.zeros(n_samples)
        
        return project_duration, start_times
    
    def _topological_sort(self) -> List[Activity]:
        """Perform topological sort of activities.
        
        Returns:
            List of activities in topological order
        """
        # Simplified topological sort
        visited = set()
        sorted_activities = []
        
        def visit(activity):
            if activity.id in visited:
                return
            
            visited.add(activity.id)
            
            # Visit predecessors first
            for pred_id in activity.predecessors:
                pred_activity = self.activity_map.get(pred_id)
                if pred_activity:
                    visit(pred_activity)
            
            sorted_activities.append(activity)
        
        # Start with activities that have no predecessors
        for activity in self.schedule.activities:
            visit(activity)
        
        return sorted_activities


class ScheduleCostCalculator:
    """Calculates schedule-driven costs."""
    
    @staticmethod
    def calculate_indirect_costs(project_durations: np.ndarray,
                               indirects_per_day: float) -> np.ndarray:
        """Calculate indirect costs based on project duration.
        
        Args:
            project_durations: Project duration array (in days)
            indirects_per_day: Daily indirect cost rate
            
        Returns:
            Indirect cost array
        """
        return project_durations * indirects_per_day
    
    @staticmethod
    def calculate_delay_costs(base_duration: float,
                            actual_durations: np.ndarray,
                            delay_cost_per_day: float) -> np.ndarray:
        """Calculate costs due to schedule delays.
        
        Args:
            base_duration: Base case project duration
            actual_durations: Simulated project durations
            delay_cost_per_day: Cost per day of delay
            
        Returns:
            Delay cost array
        """
        delays = np.maximum(actual_durations - base_duration, 0)
        return delays * delay_cost_per_day
    
    @staticmethod
    def calculate_acceleration_costs(base_duration: float,
                                   actual_durations: np.ndarray,
                                   acceleration_cost_curve: Dict[str, float]) -> np.ndarray:
        """Calculate costs for schedule acceleration.
        
        Args:
            base_duration: Base case project duration
            actual_durations: Simulated project durations
            acceleration_cost_curve: Cost curve for acceleration
            
        Returns:
            Acceleration cost array
        """
        n_samples = len(actual_durations)
        acceleration_costs = np.zeros(n_samples)
        
        for i in range(n_samples):
            duration = actual_durations[i]
            if duration < base_duration:
                # Calculate acceleration percentage
                accel_pct = (base_duration - duration) / base_duration
                
                # Apply cost curve (simplified linear model)
                cost_multiplier = acceleration_cost_curve.get('linear_rate', 2.0)
                acceleration_costs[i] = accel_pct * cost_multiplier * base_duration * 1000  # Example cost
        
        return acceleration_costs


class OutageConstraints:
    """Models outage window constraints for energized work."""
    
    @staticmethod
    def apply_outage_constraints(activity_durations: Dict[str, np.ndarray],
                               outage_activities: List[str],
                               max_outage_duration: float,
                               penalty_per_violation: float) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Apply outage window constraints.
        
        Args:
            activity_durations: Simulated activity durations
            outage_activities: List of activities requiring outages
            max_outage_duration: Maximum allowable outage duration
            penalty_per_violation: Penalty cost for exceeding outage window
            
        Returns:
            Tuple of (adjusted_durations, penalty_costs)
        """
        n_samples = len(next(iter(activity_durations.values())))
        penalty_costs = np.zeros(n_samples)
        adjusted_durations = activity_durations.copy()
        
        # Calculate total outage duration
        total_outage_duration = np.zeros(n_samples)
        
        for activity_id in outage_activities:
            if activity_id in activity_durations:
                total_outage_duration += activity_durations[activity_id]
        
        # Calculate penalties for violations
        violations = np.maximum(total_outage_duration - max_outage_duration, 0)
        penalty_costs = violations * penalty_per_violation
        
        # Option 1: Add additional work windows (increases duration)
        for activity_id in outage_activities:
            if activity_id in adjusted_durations:
                # Simple model: extend duration if outage window exceeded
                extension_factor = np.where(violations > 0, 1.5, 1.0)
                adjusted_durations[activity_id] *= extension_factor
        
        return adjusted_durations, penalty_costs


def create_default_schedule() -> Schedule:
    """Create default T&D project schedule.
    
    Returns:
        Default schedule with typical activities
    """
    activities = [
        Activity(
            id="engineering",
            name="Engineering & Design",
            duration_dist={
                "type": "triangular",
                "low": 60,
                "mode": 90,
                "high": 120
            }
        ),
        Activity(
            id="procurement",
            name="Procurement",
            duration_dist={
                "type": "pert",
                "min": 90,
                "most_likely": 180,
                "max": 270
            },
            predecessors=["engineering"]
        ),
        Activity(
            id="construction",
            name="Construction",
            duration_dist={
                "type": "pert",
                "min": 120,
                "most_likely": 200,
                "max": 300
            },
            predecessors=["engineering", "procurement"]
        ),
        Activity(
            id="commissioning",
            name="Testing & Commissioning",
            duration_dist={
                "type": "triangular",
                "low": 20,
                "mode": 30,
                "high": 45
            },
            predecessors=["construction"]
        )
    ]
    
    return Schedule(activities=activities)


def validate_schedule_data(schedule_data: Dict[str, Any]) -> List[str]:
    """Validate schedule data.
    
    Args:
        schedule_data: Schedule dictionary
        
    Returns:
        List of validation errors
    """
    errors = []
    
    try:
        schedule = Schedule(**schedule_data)
        
        # Validate activity distributions
        for activity in schedule.activities:
            try:
                from .distributions import validate_distribution_config
                validate_distribution_config(activity.duration_dist)
            except Exception as e:
                errors.append(f"Activity {activity.id}: Invalid duration distribution - {e}")
        
        # Check for circular dependencies (simplified)
        activity_ids = {act.id for act in schedule.activities}
        
        for activity in schedule.activities:
            for pred_id in activity.predecessors:
                if pred_id not in activity_ids:
                    errors.append(f"Activity {activity.id}: Unknown predecessor {pred_id}")
        
    except Exception as e:
        errors.append(f"Schedule validation: {e}")
    
    return errors