"""
Logging Utilities
---------------
Provides standardized logging setup for the HLA-ProtBERT system.
"""
import os
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union

def setup_logging(
    level: Union[str, int] = "INFO", 
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """Set up logging for the application
    
    Args:
        level: Logging level (either string like "DEBUG", "INFO", etc. or 
               logging.* constant like logging.DEBUG, logging.INFO)
        log_file: Path to log file (or None for console only)
        log_format: Log message format
        config: Configuration dictionary (overrides other parameters)
        
    Returns:
        Root logger
    """
    # Override parameters with config if provided
    if config:
        level = config.get("logging", {}).get("level", level)
        log_file = config.get("logging", {}).get("file", log_file)
        log_format = config.get("logging", {}).get("format", log_format)
    
    # Default format if not specified
    if not log_format:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Convert level string to logging level constant if it's a string
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        # If level is already a numeric value (int), use it directly
        numeric_level = level
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.error(f"Error setting up log file {log_file}: {e}")
    
    # Create logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized (level={level})")
    
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """Get logger for a specific module
    
    Args:
        name: Module name or logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class LoggingMixin:
    """Mixin that adds logging capabilities to a class"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class
        
        Returns:
            Logger instance
        """
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")
        return self._logger

def log_execution_time(func):
    """Decorator to log function execution time
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {e}")
            raise
            
    return wrapper

def configure_library_loggers(level: str = "WARNING") -> None:
    """Configure logging levels for third-party libraries
    
    Args:
        level: Logging level for libraries
    """
    # List of common libraries to configure
    libraries = [
        "transformers", 
        "torch", 
        "numpy", 
        "matplotlib",
        "sklearn",
        "requests"
    ]
    
    numeric_level = getattr(logging, level.upper(), logging.WARNING)
    
    # Set level for each library
    for lib in libraries:
        logging.getLogger(lib).setLevel(numeric_level)
