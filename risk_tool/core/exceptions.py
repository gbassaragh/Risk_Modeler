"""Custom exceptions for Risk_Modeler.

Provides comprehensive exception hierarchy with detailed error context,
recovery suggestions, and proper error classification.
"""

from typing import Optional, Dict, Any, List
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for categorizing exceptions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    COMPUTATION = "computation"
    IO = "io"
    CONFIGURATION = "configuration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    NETWORK = "network"
    SYSTEM = "system"


class RiskModelingError(Exception):
    """Base exception for all Risk_Modeler errors.
    
    Provides structured error information with context, severity,
    and recovery suggestions.
    """
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize structured error.
        
        Args:
            message: Human-readable error description
            category: Error category for classification
            severity: Error severity level
            error_code: Unique error identifier
            context: Additional error context
            recovery_suggestions: List of recovery suggestions
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.error_code = error_code or self._generate_error_code()
        self.context = context or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.cause = cause
    
    def _generate_error_code(self) -> str:
        """Generate error code from class name."""
        return f"RM_{self.__class__.__name__.upper()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for structured logging."""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': self.context,
            'recovery_suggestions': self.recovery_suggestions,
            'exception_type': self.__class__.__name__,
            'cause': str(self.cause) if self.cause else None
        }


# Data Validation Errors
class ValidationError(RiskModelingError):
    """Raised when input data validation fails."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Any = None,
        validation_rule: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'field_name': field_name,
            'field_value': field_value,
            'validation_rule': validation_rule
        })
        
        recovery_suggestions = kwargs.get('recovery_suggestions', [
            "Check input data format and values",
            "Verify data matches expected schema",
            "Review data dictionary for correct formats"
        ])
        
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestions=recovery_suggestions,
            **kwargs
        )


class SchemaValidationError(ValidationError):
    """Raised when data schema validation fails."""
    
    def __init__(self, message: str, schema_errors: List[str], **kwargs):
        context = kwargs.get('context', {})
        context['schema_errors'] = schema_errors
        
        recovery_suggestions = [
            "Review the data dictionary for correct schema",
            "Check field names and types",
            "Validate required fields are present"
        ]
        
        super().__init__(
            message,
            context=context,
            recovery_suggestions=recovery_suggestions,
            **kwargs
        )


class DistributionParameterError(ValidationError):
    """Raised when distribution parameters are invalid."""
    
    def __init__(
        self,
        message: str,
        distribution_type: str,
        parameter_name: str,
        parameter_value: Any,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'distribution_type': distribution_type,
            'parameter_name': parameter_name,
            'parameter_value': parameter_value
        })
        
        recovery_suggestions = [
            f"Check {parameter_name} parameter for {distribution_type} distribution",
            "Verify parameter constraints (e.g., standard deviation > 0)",
            "Review distribution parameter documentation"
        ]
        
        super().__init__(
            message,
            context=context,
            recovery_suggestions=recovery_suggestions,
            **kwargs
        )


# Computation Errors
class ComputationError(RiskModelingError):
    """Raised when numerical computation fails."""
    
    def __init__(self, message: str, operation: str, **kwargs):
        context = kwargs.get('context', {})
        context['operation'] = operation
        
        recovery_suggestions = kwargs.get('recovery_suggestions', [
            "Check input data for numerical issues",
            "Verify parameters are within valid ranges",
            "Consider reducing problem complexity"
        ])
        
        super().__init__(
            message,
            category=ErrorCategory.COMPUTATION,
            context=context,
            recovery_suggestions=recovery_suggestions,
            **kwargs
        )


class ConvergenceError(ComputationError):
    """Raised when Monte Carlo simulation doesn't converge."""
    
    def __init__(
        self,
        message: str,
        iterations_completed: int,
        convergence_metric: float,
        target_metric: float,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'iterations_completed': iterations_completed,
            'convergence_metric': convergence_metric,
            'target_metric': target_metric
        })
        
        recovery_suggestions = [
            "Increase number of iterations",
            "Check for extreme outliers in data",
            "Verify correlation matrix is positive definite",
            "Consider using variance reduction techniques"
        ]
        
        super().__init__(
            message,
            operation="monte_carlo_convergence",
            context=context,
            recovery_suggestions=recovery_suggestions,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class CorrelationMatrixError(ComputationError):
    """Raised when correlation matrix operations fail."""
    
    def __init__(self, message: str, matrix_property: str, **kwargs):
        context = kwargs.get('context', {})
        context['matrix_property'] = matrix_property
        
        recovery_suggestions = [
            "Verify correlation values are between -1 and 1",
            "Check matrix is symmetric",
            "Ensure matrix is positive semi-definite",
            "Remove highly correlated variables if needed"
        ]
        
        super().__init__(
            message,
            operation="correlation_matrix",
            context=context,
            recovery_suggestions=recovery_suggestions,
            **kwargs
        )


# I/O Errors
class IOError(RiskModelingError):
    """Raised when input/output operations fail."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: str = "unknown",
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'file_path': file_path,
            'operation': operation
        })
        
        recovery_suggestions = kwargs.get('recovery_suggestions', [
            "Check file path exists and is accessible",
            "Verify file permissions",
            "Ensure file is not locked by another process"
        ])
        
        super().__init__(
            message,
            category=ErrorCategory.IO,
            context=context,
            recovery_suggestions=recovery_suggestions,
            **kwargs
        )


class FileFormatError(IOError):
    """Raised when file format is unsupported or corrupted."""
    
    def __init__(
        self,
        message: str,
        file_path: str,
        expected_format: str,
        detected_format: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'expected_format': expected_format,
            'detected_format': detected_format
        })
        
        recovery_suggestions = [
            f"Ensure file is in {expected_format} format",
            "Check file is not corrupted",
            "Verify file extension matches content",
            "Try converting file to supported format"
        ]
        
        super().__init__(
            message,
            file_path=file_path,
            operation="format_detection",
            context=context,
            recovery_suggestions=recovery_suggestions,
            **kwargs
        )


# Configuration Errors
class ConfigurationError(RiskModelingError):
    """Raised when configuration is invalid."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Any = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'config_key': config_key,
            'config_value': config_value
        })
        
        recovery_suggestions = kwargs.get('recovery_suggestions', [
            "Check configuration syntax and format",
            "Verify all required configuration keys are present",
            "Review configuration documentation",
            "Use configuration validation tools"
        ])
        
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            context=context,
            recovery_suggestions=recovery_suggestions,
            **kwargs
        )


class SimulationConfigError(ConfigurationError):
    """Raised when simulation configuration is invalid."""
    
    def __init__(self, message: str, **kwargs):
        recovery_suggestions = [
            "Check simulation parameters (iterations, seed, method)",
            "Verify sampling method is supported (LHS, Sobol, MC)",
            "Ensure number of iterations is positive",
            "Check variance reduction settings"
        ]
        
        super().__init__(
            message,
            recovery_suggestions=recovery_suggestions,
            **kwargs
        )


# Performance Errors
class PerformanceError(RiskModelingError):
    """Raised when performance constraints are violated."""
    
    def __init__(
        self,
        message: str,
        performance_metric: str,
        actual_value: float,
        threshold_value: float,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'performance_metric': performance_metric,
            'actual_value': actual_value,
            'threshold_value': threshold_value
        })
        
        recovery_suggestions = kwargs.get('recovery_suggestions', [
            "Reduce problem complexity",
            "Increase available memory",
            "Use more efficient algorithms",
            "Consider distributed computing"
        ])
        
        super().__init__(
            message,
            category=ErrorCategory.PERFORMANCE,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestions=recovery_suggestions,
            **kwargs
        )


class MemoryError(PerformanceError):
    """Raised when memory usage exceeds limits."""
    
    def __init__(
        self,
        message: str,
        memory_used: float,
        memory_limit: float,
        **kwargs
    ):
        recovery_suggestions = [
            "Reduce number of iterations",
            "Process data in smaller batches",
            "Use memory-efficient data structures",
            "Increase system memory"
        ]
        
        super().__init__(
            message,
            performance_metric="memory_usage",
            actual_value=memory_used,
            threshold_value=memory_limit,
            recovery_suggestions=recovery_suggestions,
            **kwargs
        )


class TimeoutError(PerformanceError):
    """Raised when operation exceeds time limit."""
    
    def __init__(
        self,
        message: str,
        elapsed_time: float,
        timeout_limit: float,
        **kwargs
    ):
        recovery_suggestions = [
            "Increase timeout limit",
            "Optimize algorithm parameters",
            "Use faster hardware",
            "Consider approximate methods"
        ]
        
        super().__init__(
            message,
            performance_metric="execution_time",
            actual_value=elapsed_time,
            threshold_value=timeout_limit,
            recovery_suggestions=recovery_suggestions,
            **kwargs
        )


# Utility functions
def handle_exception(
    exception: Exception,
    logger,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True
) -> Optional[RiskModelingError]:
    """Handle exception with proper logging and conversion.
    
    Args:
        exception: Original exception
        logger: Logger instance
        context: Additional context information
        reraise: Whether to reraise the exception
        
    Returns:
        Converted RiskModelingError if not reraising
        
    Raises:
        RiskModelingError: If reraise is True
    """
    # Convert to RiskModelingError if needed
    if isinstance(exception, RiskModelingError):
        risk_error = exception
    else:
        risk_error = RiskModelingError(
            str(exception),
            context=context,
            cause=exception
        )
    
    # Log the error
    logger.error(
        f"{risk_error.error_code}: {risk_error.message}",
        extra=risk_error.to_dict(),
        exc_info=True
    )
    
    if reraise:
        raise risk_error
    else:
        return risk_error