"""Historical data processing for calibration.

Provides tools for cleaning, validating, and analyzing historical
cost data for distribution fitting and model calibration.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
from dataclasses import dataclass, field
from datetime import datetime, date
import logging


@dataclass
class DataQualityReport:
    """Report on data quality assessment."""

    n_total: int
    n_missing: int
    n_zero: int
    n_negative: int
    n_outliers: int

    missing_pct: float
    zero_pct: float
    negative_pct: float
    outlier_pct: float

    # Statistics
    mean: float
    std: float
    min_val: float
    max_val: float
    q25: float
    q50: float
    q75: float

    # Recommendations
    quality_score: float  # 0-100
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary string."""
        lines = [f"Data Quality Report"]
        lines.append(f"Total records: {self.n_total:,}")
        lines.append(f"Missing: {self.n_missing:,} ({self.missing_pct:.1f}%)")
        lines.append(f"Zero values: {self.n_zero:,} ({self.zero_pct:.1f}%)")
        lines.append(f"Negative: {self.n_negative:,} ({self.negative_pct:.1f}%)")
        lines.append(f"Outliers: {self.n_outliers:,} ({self.outlier_pct:.1f}%)")
        lines.append(f"Quality Score: {self.quality_score:.1f}/100")

        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"- {rec}")

        return "\n".join(lines)


@dataclass
class OutlierDetectionResult:
    """Result from outlier detection."""

    outlier_indices: np.ndarray
    outlier_values: np.ndarray
    method: str
    threshold: float
    n_outliers: int
    outlier_pct: float

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Outlier Detection ({self.method})\n"
            f"Found: {self.n_outliers:,} outliers ({self.outlier_pct:.1f}%)\n"
            f"Threshold: {self.threshold:.2f}"
        )


@dataclass
class SeasonalityResult:
    """Result from seasonality analysis."""

    has_seasonality: bool
    seasonal_period: Optional[int]
    seasonal_strength: float  # 0-1 scale
    seasonal_factors: Optional[np.ndarray]  # Monthly/quarterly factors
    trend_component: Optional[np.ndarray]
    seasonal_component: Optional[np.ndarray]
    residual_component: Optional[np.ndarray]

    def summary(self) -> str:
        """Generate summary string."""
        lines = [f"Seasonality Analysis"]
        lines.append(f"Seasonal: {'Yes' if self.has_seasonality else 'No'}")
        lines.append(f"Strength: {self.seasonal_strength:.3f}")

        if self.seasonal_period:
            lines.append(f"Period: {self.seasonal_period}")

        return "\n".join(lines)


class DataQualityChecker:
    """Comprehensive data quality assessment."""

    def __init__(self, outlier_method: str = "iqr", outlier_threshold: float = 1.5):
        """Initialize quality checker.

        Args:
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
            outlier_threshold: Threshold for outlier detection
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold

    def assess_quality(self, data: Union[np.ndarray, pd.Series]) -> DataQualityReport:
        """Comprehensive data quality assessment.

        Args:
            data: Data to assess

        Returns:
            DataQualityReport with assessment results
        """
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.array(data)

        n_total = len(values)

        # Missing values
        if isinstance(data, pd.Series):
            n_missing = data.isna().sum()
            clean_values = data.dropna().values
        else:
            n_missing = np.sum(np.isnan(values))
            clean_values = values[~np.isnan(values)]

        # Zero and negative values
        n_zero = np.sum(clean_values == 0)
        n_negative = np.sum(clean_values < 0)

        # Outliers
        outlier_result = self._detect_outliers(clean_values)
        n_outliers = outlier_result.n_outliers

        # Percentages
        missing_pct = (n_missing / n_total) * 100 if n_total > 0 else 0
        zero_pct = (n_zero / len(clean_values)) * 100 if len(clean_values) > 0 else 0
        negative_pct = (
            (n_negative / len(clean_values)) * 100 if len(clean_values) > 0 else 0
        )
        outlier_pct = (
            (n_outliers / len(clean_values)) * 100 if len(clean_values) > 0 else 0
        )

        # Statistics
        if len(clean_values) > 0:
            mean = np.mean(clean_values)
            std = np.std(clean_values)
            min_val = np.min(clean_values)
            max_val = np.max(clean_values)
            q25, q50, q75 = np.percentile(clean_values, [25, 50, 75])
        else:
            mean = std = min_val = max_val = q25 = q50 = q75 = np.nan

        # Quality score and recommendations
        quality_score, recommendations = self._calculate_quality_score(
            missing_pct, zero_pct, negative_pct, outlier_pct, len(clean_values)
        )

        return DataQualityReport(
            n_total=n_total,
            n_missing=n_missing,
            n_zero=n_zero,
            n_negative=n_negative,
            n_outliers=n_outliers,
            missing_pct=missing_pct,
            zero_pct=zero_pct,
            negative_pct=negative_pct,
            outlier_pct=outlier_pct,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            q25=q25,
            q50=q50,
            q75=q75,
            quality_score=quality_score,
            recommendations=recommendations,
        )

    def _detect_outliers(self, data: np.ndarray) -> OutlierDetectionResult:
        """Detect outliers in data."""
        detector = OutlierDetector(
            method=self.outlier_method, threshold=self.outlier_threshold
        )
        return detector.detect(data)

    def _calculate_quality_score(
        self,
        missing_pct: float,
        zero_pct: float,
        negative_pct: float,
        outlier_pct: float,
        n_valid: int,
    ) -> Tuple[float, List[str]]:
        """Calculate quality score and generate recommendations."""
        score = 100.0
        recommendations = []

        # Sample size penalty
        if n_valid < 30:
            score -= 20
            recommendations.append(
                "Increase sample size (need ≥30 for reliable fitting)"
            )
        elif n_valid < 50:
            score -= 10
            recommendations.append(
                "Consider collecting more data for better reliability"
            )

        # Missing data penalty
        if missing_pct > 20:
            score -= 25
            recommendations.append(
                "High missing data rate - investigate data collection process"
            )
        elif missing_pct > 10:
            score -= 15
            recommendations.append(
                "Consider imputation or data cleaning for missing values"
            )
        elif missing_pct > 5:
            score -= 5

        # Zero values
        if zero_pct > 30:
            score -= 20
            recommendations.append(
                "High rate of zero values - check data recording process"
            )
        elif zero_pct > 15:
            score -= 10
            recommendations.append(
                "Consider separate analysis for zero vs. non-zero values"
            )

        # Negative values (usually problematic for cost data)
        if negative_pct > 10:
            score -= 30
            recommendations.append(
                "Investigate negative values - may indicate data errors"
            )
        elif negative_pct > 0:
            score -= 10
            recommendations.append("Review negative values for accuracy")

        # Outliers
        if outlier_pct > 20:
            score -= 15
            recommendations.append(
                "High outlier rate - consider robust fitting methods"
            )
        elif outlier_pct > 10:
            score -= 10
            recommendations.append(
                "Review outliers - may indicate special conditions or errors"
            )

        # Ensure score doesn't go below 0
        score = max(0, score)

        # Add positive recommendations
        if score >= 80:
            recommendations.append("Data quality is good for distribution fitting")
        elif score >= 60:
            recommendations.append("Data quality is acceptable with some cleaning")
        else:
            recommendations.append("Data requires significant cleaning before fitting")

        return score, recommendations


class OutlierDetector:
    """Outlier detection using various methods."""

    SUPPORTED_METHODS = [
        "iqr",
        "zscore",
        "modified_zscore",
        "isolation_forest",
        "dbscan",
    ]

    def __init__(self, method: str = "iqr", threshold: float = 1.5):
        """Initialize outlier detector.

        Args:
            method: Detection method
            threshold: Threshold parameter (meaning depends on method)
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method must be one of {self.SUPPORTED_METHODS}")

        self.method = method
        self.threshold = threshold

    def detect(self, data: np.ndarray) -> OutlierDetectionResult:
        """Detect outliers in data.

        Args:
            data: Input data

        Returns:
            OutlierDetectionResult with detection results
        """
        data = np.array(data)
        data = data[~np.isnan(data)]  # Remove NaN values

        if len(data) < 4:
            # Not enough data for outlier detection
            return OutlierDetectionResult(
                outlier_indices=np.array([]),
                outlier_values=np.array([]),
                method=self.method,
                threshold=self.threshold,
                n_outliers=0,
                outlier_pct=0.0,
            )

        if self.method == "iqr":
            return self._detect_iqr(data)
        elif self.method == "zscore":
            return self._detect_zscore(data)
        elif self.method == "modified_zscore":
            return self._detect_modified_zscore(data)
        elif self.method == "isolation_forest":
            return self._detect_isolation_forest(data)
        elif self.method == "dbscan":
            return self._detect_dbscan(data)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _detect_iqr(self, data: np.ndarray) -> OutlierDetectionResult:
        """Detect outliers using Interquartile Range method."""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr

        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_indices = np.where(outlier_mask)[0]
        outlier_values = data[outlier_mask]

        return OutlierDetectionResult(
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            method=self.method,
            threshold=self.threshold,
            n_outliers=len(outlier_indices),
            outlier_pct=(len(outlier_indices) / len(data)) * 100,
        )

    def _detect_zscore(self, data: np.ndarray) -> OutlierDetectionResult:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(data))
        outlier_mask = z_scores > self.threshold

        outlier_indices = np.where(outlier_mask)[0]
        outlier_values = data[outlier_mask]

        return OutlierDetectionResult(
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            method=self.method,
            threshold=self.threshold,
            n_outliers=len(outlier_indices),
            outlier_pct=(len(outlier_indices) / len(data)) * 100,
        )

    def _detect_modified_zscore(self, data: np.ndarray) -> OutlierDetectionResult:
        """Detect outliers using Modified Z-score method (more robust)."""
        median = np.median(data)
        mad = np.median(np.abs(data - median))  # Median Absolute Deviation

        # Modified Z-scores
        modified_z_scores = (
            0.6745 * (data - median) / mad if mad > 0 else np.zeros_like(data)
        )
        outlier_mask = np.abs(modified_z_scores) > self.threshold

        outlier_indices = np.where(outlier_mask)[0]
        outlier_values = data[outlier_mask]

        return OutlierDetectionResult(
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            method=self.method,
            threshold=self.threshold,
            n_outliers=len(outlier_indices),
            outlier_pct=(len(outlier_indices) / len(data)) * 100,
        )

    def _detect_isolation_forest(self, data: np.ndarray) -> OutlierDetectionResult:
        """Detect outliers using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest

            # Reshape for sklearn
            X = data.reshape(-1, 1)

            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=self.threshold, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)

            # -1 indicates outlier
            outlier_mask = outlier_labels == -1
            outlier_indices = np.where(outlier_mask)[0]
            outlier_values = data[outlier_mask]

            return OutlierDetectionResult(
                outlier_indices=outlier_indices,
                outlier_values=outlier_values,
                method=self.method,
                threshold=self.threshold,
                n_outliers=len(outlier_indices),
                outlier_pct=(len(outlier_indices) / len(data)) * 100,
            )

        except ImportError:
            warnings.warn(
                "sklearn not available for Isolation Forest. Using IQR method."
            )
            return self._detect_iqr(data)

    def _detect_dbscan(self, data: np.ndarray) -> OutlierDetectionResult:
        """Detect outliers using DBSCAN clustering."""
        try:
            # Reshape for sklearn
            X = data.reshape(-1, 1)

            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Apply DBSCAN
            dbscan = DBSCAN(eps=self.threshold, min_samples=3)
            cluster_labels = dbscan.fit_predict(X_scaled)

            # -1 indicates outlier
            outlier_mask = cluster_labels == -1
            outlier_indices = np.where(outlier_mask)[0]
            outlier_values = data[outlier_mask]

            return OutlierDetectionResult(
                outlier_indices=outlier_indices,
                outlier_values=outlier_values,
                method=self.method,
                threshold=self.threshold,
                n_outliers=len(outlier_indices),
                outlier_pct=(len(outlier_indices) / len(data)) * 100,
            )

        except ImportError:
            warnings.warn("sklearn not available for DBSCAN. Using IQR method.")
            return self._detect_iqr(data)


class SeasonalityAnalyzer:
    """Analyze seasonal patterns in time series data."""

    def __init__(self, significance_level: float = 0.05):
        """Initialize seasonality analyzer.

        Args:
            significance_level: Statistical significance level for tests
        """
        self.significance_level = significance_level

    def analyze(
        self,
        data: Union[pd.Series, np.ndarray],
        dates: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None,
        period: Optional[int] = None,
    ) -> SeasonalityResult:
        """Analyze seasonality in time series data.

        Args:
            data: Time series data
            dates: Corresponding dates (optional)
            period: Expected seasonal period (e.g., 12 for monthly)

        Returns:
            SeasonalityResult with analysis results
        """
        if isinstance(data, pd.Series):
            values = data.values
            if dates is None and hasattr(data, "index"):
                dates = data.index
        else:
            values = np.array(data)

        # Remove NaN values
        valid_mask = ~np.isnan(values)
        values = values[valid_mask]

        if dates is not None:
            dates = np.array(dates)[valid_mask]

        if len(values) < 12:
            # Not enough data for seasonal analysis
            return SeasonalityResult(
                has_seasonality=False,
                seasonal_period=None,
                seasonal_strength=0.0,
                seasonal_factors=None,
                trend_component=None,
                seasonal_component=None,
                residual_component=None,
            )

        # Detect seasonal period if not provided
        if period is None:
            period = self._detect_seasonal_period(values, dates)

        # Decompose time series
        seasonal_strength = 0.0
        seasonal_factors = None
        trend_component = None
        seasonal_component = None
        residual_component = None

        if period and period > 1:
            try:
                # Simple seasonal decomposition
                trend_component, seasonal_component, residual_component = (
                    self._decompose_series(values, period)
                )

                # Calculate seasonal strength
                seasonal_strength = self._calculate_seasonal_strength(
                    values, seasonal_component, residual_component
                )

                # Calculate seasonal factors
                seasonal_factors = self._calculate_seasonal_factors(
                    seasonal_component, period
                )

            except Exception as e:
                warnings.warn(f"Seasonal decomposition failed: {e}")

        # Determine if seasonality is significant
        has_seasonality = seasonal_strength > 0.1 and period is not None and period > 1

        return SeasonalityResult(
            has_seasonality=has_seasonality,
            seasonal_period=period,
            seasonal_strength=seasonal_strength,
            seasonal_factors=seasonal_factors,
            trend_component=trend_component,
            seasonal_component=seasonal_component,
            residual_component=residual_component,
        )

    def _detect_seasonal_period(
        self, values: np.ndarray, dates: Optional[np.ndarray] = None
    ) -> Optional[int]:
        """Detect seasonal period using autocorrelation."""
        if len(values) < 24:
            return None

        # Try common periods
        max_period = min(len(values) // 4, 24)  # Don't exceed 1/4 of data length

        # Calculate autocorrelations
        autocorrs = []
        periods = list(range(2, max_period + 1))

        for period in periods:
            if len(values) > period:
                # Calculate autocorrelation at this lag
                autocorr = np.corrcoef(values[:-period], values[period:])[0, 1]
                if not np.isnan(autocorr):
                    autocorrs.append(autocorr)
                else:
                    autocorrs.append(0)
            else:
                autocorrs.append(0)

        if not autocorrs:
            return None

        # Find period with highest autocorrelation
        max_idx = np.argmax(autocorrs)
        max_autocorr = autocorrs[max_idx]
        best_period = periods[max_idx]

        # Check if autocorrelation is significant
        if max_autocorr > 0.3:  # Arbitrary threshold
            return best_period

        return None

    def _decompose_series(
        self, values: np.ndarray, period: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple additive seasonal decomposition."""
        n = len(values)

        # Calculate trend using moving average
        trend = np.full(n, np.nan)
        half_window = period // 2

        for i in range(half_window, n - half_window):
            trend[i] = np.mean(values[i - half_window : i + half_window + 1])

        # Fill in trend endpoints using linear extrapolation
        for i in range(half_window):
            if not np.isnan(trend[half_window]):
                trend[i] = trend[half_window]

        for i in range(n - half_window, n):
            if not np.isnan(trend[n - half_window - 1]):
                trend[i] = trend[n - half_window - 1]

        # Calculate detrended series
        detrended = values - trend

        # Calculate seasonal component
        seasonal = np.zeros(n)
        for i in range(period):
            # Average values at the same seasonal position
            seasonal_positions = detrended[i::period]
            valid_positions = seasonal_positions[~np.isnan(seasonal_positions)]

            if len(valid_positions) > 0:
                seasonal_value = np.mean(valid_positions)
                seasonal[i::period] = seasonal_value

        # Center seasonal component (sum to zero)
        seasonal = seasonal - np.mean(seasonal)

        # Calculate residual
        residual = values - trend - seasonal

        return trend, seasonal, residual

    def _calculate_seasonal_strength(
        self, original: np.ndarray, seasonal: np.ndarray, residual: np.ndarray
    ) -> float:
        """Calculate strength of seasonality (0-1 scale)."""
        if len(residual) == 0:
            return 0.0

        # Remove NaN values
        valid_mask = ~(np.isnan(seasonal) | np.isnan(residual))
        if np.sum(valid_mask) == 0:
            return 0.0

        seasonal_var = np.var(seasonal[valid_mask])
        residual_var = np.var(residual[valid_mask])

        if seasonal_var + residual_var == 0:
            return 0.0

        seasonal_strength = seasonal_var / (seasonal_var + residual_var)
        return min(1.0, max(0.0, seasonal_strength))

    def _calculate_seasonal_factors(
        self, seasonal: np.ndarray, period: int
    ) -> np.ndarray:
        """Calculate average seasonal factors."""
        factors = np.zeros(period)

        for i in range(period):
            seasonal_values = seasonal[i::period]
            valid_values = seasonal_values[~np.isnan(seasonal_values)]

            if len(valid_values) > 0:
                factors[i] = np.mean(valid_values)

        return factors


class HistoricalDataProcessor:
    """Comprehensive historical data processing."""

    def __init__(self, outlier_method: str = "iqr", outlier_threshold: float = 1.5):
        """Initialize processor.

        Args:
            outlier_method: Method for outlier detection
            outlier_threshold: Threshold for outlier detection
        """
        self.quality_checker = DataQualityChecker(outlier_method, outlier_threshold)
        self.outlier_detector = OutlierDetector(outlier_method, outlier_threshold)
        self.seasonality_analyzer = SeasonalityAnalyzer()

    def process(
        self,
        data: Union[pd.DataFrame, pd.Series, np.ndarray],
        value_column: Optional[str] = None,
        date_column: Optional[str] = None,
        remove_outliers: bool = False,
    ) -> Dict[str, Any]:
        """Process historical data with comprehensive analysis.

        Args:
            data: Input data
            value_column: Name of value column (for DataFrames)
            date_column: Name of date column (for DataFrames)
            remove_outliers: Whether to remove detected outliers

        Returns:
            Dictionary with processed data and analysis results
        """
        # Extract values and dates
        if isinstance(data, pd.DataFrame):
            if value_column is None:
                raise ValueError("value_column must be specified for DataFrames")

            values = data[value_column]
            dates = data[date_column] if date_column else None

        elif isinstance(data, pd.Series):
            values = data
            dates = data.index if hasattr(data, "index") else None

        else:
            values = np.array(data)
            dates = None

        # Quality assessment
        quality_report = self.quality_checker.assess_quality(values)

        # Outlier detection
        clean_values = (
            values.dropna() if hasattr(values, "dropna") else values[~np.isnan(values)]
        )
        outlier_result = self.outlier_detector.detect(clean_values)

        # Process data
        processed_values = values.copy()

        if remove_outliers and outlier_result.n_outliers > 0:
            if hasattr(processed_values, "iloc"):
                # pandas Series
                valid_indices = ~processed_values.index.isin(
                    outlier_result.outlier_indices
                )
                processed_values = processed_values[valid_indices]
            else:
                # numpy array
                processed_values = np.delete(
                    processed_values, outlier_result.outlier_indices
                )

        # Seasonality analysis
        seasonality_result = self.seasonality_analyzer.analyze(processed_values, dates)

        return {
            "processed_values": processed_values,
            "quality_report": quality_report,
            "outlier_result": outlier_result,
            "seasonality_result": seasonality_result,
            "n_original": len(values),
            "n_processed": len(processed_values),
            "processing_notes": self._generate_processing_notes(
                quality_report, outlier_result, remove_outliers
            ),
        }

    def _generate_processing_notes(
        self,
        quality_report: DataQualityReport,
        outlier_result: OutlierDetectionResult,
        removed_outliers: bool,
    ) -> List[str]:
        """Generate processing notes."""
        notes = []

        if quality_report.quality_score < 60:
            notes.append("Low data quality - results may be unreliable")

        if quality_report.missing_pct > 10:
            notes.append(f"High missing data rate ({quality_report.missing_pct:.1f}%)")

        if outlier_result.outlier_pct > 10:
            if removed_outliers:
                notes.append(
                    f"Removed {outlier_result.n_outliers} outliers ({outlier_result.outlier_pct:.1f}%)"
                )
            else:
                notes.append(
                    f"Detected {outlier_result.n_outliers} outliers ({outlier_result.outlier_pct:.1f}%) - not removed"
                )

        if quality_report.negative_pct > 0:
            notes.append(f"Contains {quality_report.n_negative} negative values")

        return notes


# Convenience functions


def process_cost_data(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    value_column: Optional[str] = None,
    date_column: Optional[str] = None,
    clean_data: bool = True,
) -> Dict[str, Any]:
    """Process cost data with default settings.

    Args:
        data: Input data
        value_column: Value column name
        date_column: Date column name
        clean_data: Whether to remove outliers

    Returns:
        Processing results dictionary
    """
    processor = HistoricalDataProcessor()
    return processor.process(data, value_column, date_column, clean_data)


def detect_outliers(
    data: np.ndarray, method: str = "iqr", threshold: float = 1.5
) -> OutlierDetectionResult:
    """Detect outliers in data.

    Args:
        data: Input data
        method: Detection method
        threshold: Detection threshold

    Returns:
        Outlier detection results
    """
    detector = OutlierDetector(method, threshold)
    return detector.detect(data)


def analyze_seasonality(
    data: Union[pd.Series, np.ndarray], dates: Optional[np.ndarray] = None
) -> SeasonalityResult:
    """Analyze seasonality in time series.

    Args:
        data: Time series data
        dates: Optional date index

    Returns:
        Seasonality analysis results
    """
    analyzer = SeasonalityAnalyzer()
    return analyzer.analyze(data, dates)
