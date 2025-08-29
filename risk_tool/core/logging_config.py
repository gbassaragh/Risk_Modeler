"""Logging configuration for Risk_Modeler.

Provides comprehensive logging setup with structured output, performance tracking,
and appropriate log levels for different environments.
"""

import logging
import logging.config
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def filter(self, record):
        """Add performance timing to log records."""
        if not hasattr(record, 'performance_time'):
            record.performance_time = time.time()
        return True


class RiskModelingFormatter(logging.Formatter):
    """Custom formatter for Risk_Modeler with structured output."""
    
    def __init__(self, include_performance: bool = True):
        super().__init__()
        self.include_performance = include_performance
        self.start_time = time.time()
    
    def format(self, record):
        """Format log record with structured information."""
        # Base message
        message = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add performance timing if available
        if self.include_performance and hasattr(record, 'performance_time'):
            message['elapsed'] = f"{record.performance_time - self.start_time:.3f}s"
        
        # Add context information
        if hasattr(record, 'simulation_id'):
            message['simulation_id'] = record.simulation_id
        
        if hasattr(record, 'component'):
            message['component'] = record.component
            
        if hasattr(record, 'operation'):
            message['operation'] = record.operation
        
        # Add exception information
        if record.exc_info:
            message['exception'] = self.formatException(record.exc_info)
        
        # Format based on environment
        if self._is_development():
            return json.dumps(message, indent=2)
        else:
            return json.dumps(message)
    
    def _is_development(self) -> bool:
        """Check if running in development environment."""
        return getattr(sys, 'gettrace', None) is not None


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_performance: bool = True,
    enable_structured: bool = False
) -> logging.Logger:
    """Setup comprehensive logging for Risk_Modeler.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        enable_performance: Enable performance tracking
        enable_structured: Enable structured JSON logging
        
    Returns:
        Configured logger instance
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure logging
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'simple': {
                'format': '%(levelname)s: %(message)s'
            },
            'structured': {
                '()': RiskModelingFormatter,
                'include_performance': enable_performance
            }
        },
        'filters': {
            'performance': {
                '()': PerformanceFilter
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'stream': sys.stdout,
                'formatter': 'structured' if enable_structured else 'simple',
                'level': log_level
            }
        },
        'loggers': {
            'risk_tool': {
                'level': log_level,
                'handlers': ['console'],
                'propagate': False
            }
        },
        'root': {
            'level': log_level,
            'handlers': ['console']
        }
    }
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(log_file),
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5,
            'formatter': 'structured' if enable_structured else 'detailed',
            'level': log_level
        }
        config['loggers']['risk_tool']['handlers'].append('file')
        config['root']['handlers'].append('file')
    
    # Add performance filter if enabled
    if enable_performance:
        config['handlers']['console']['filters'] = ['performance']
        if log_file:
            config['handlers']['file']['filters'] = ['performance']
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Get logger instance
    logger = logging.getLogger('risk_tool')
    
    # Log startup message
    logger.info(
        "Risk_Modeler logging initialized",
        extra={
            'component': 'logging',
            'operation': 'initialization',
            'log_level': log_level,
            'structured_logging': enable_structured,
            'performance_tracking': enable_performance
        }
    )
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific component.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"risk_tool.{name}")


class LoggingContext:
    """Context manager for adding structured context to logs."""
    
    def __init__(self, logger: logging.Logger, **context):
        """Initialize logging context.
        
        Args:
            logger: Logger instance
            **context: Context information to add to log records
        """
        self.logger = logger
        self.context = context
        self.old_factory = logging.getLogRecordFactory()
    
    def __enter__(self):
        """Enter context and set up custom log record factory."""
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original log record factory."""
        logging.setLogRecordFactory(self.old_factory)


def log_performance(func):
    """Decorator for automatic performance logging."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            with LoggingContext(logger, 
                              component=func.__module__, 
                              operation=func.__name__):
                logger.debug(f"Starting {func.__name__}")
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"Completed {func.__name__} in {elapsed:.3f}s")
                return result
                
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Failed {func.__name__} after {elapsed:.3f}s: {e}",
                exc_info=True
            )
            raise
    
    return wrapper


# Default logger instance
default_logger = None

def initialize_default_logger():
    """Initialize the default logger if not already done."""
    global default_logger
    if default_logger is None:
        default_logger = setup_logging()
    return default_logger