"""
Logging Utilities

This module provides utilities for logging research activity for the
Advanced Autonomous Research Agent system.
"""

import os
import logging
import logging.handlers
from typing import Optional, Dict, Any

class ResearchLogger:
    """
    Logger for research activities with configurable outputs.
    
    This class provides a logger for the Advanced Autonomous Research Agent
    with configurable file and console outputs, and optional structured logging.
    """
    
    def __init__(
        self,
        name: str = "research_agent",
        log_level: str = "INFO",
        console_output: bool = True,
        console_level: str = "INFO",
        log_file: Optional[str] = None,
        file_level: str = "DEBUG",
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        log_format: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        date_format: str = "%Y-%m-%d %H:%M:%S"
    ):
        """
        Initialize a ResearchLogger instance.
        
        Args:
            name: Name of the logger
            log_level: Overall logging level
            console_output: Whether to output logs to console
            console_level: Logging level for console output
            log_file: Path to log file (None for no file output)
            file_level: Logging level for file output
            max_file_size: Maximum log file size before rotation
            backup_count: Number of backup log files to keep
            log_format: Format of log messages
            date_format: Format of timestamps in log messages
        """
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self._get_log_level(log_level))
        
        # Clear existing handlers to avoid duplicate logs if logger already exists
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Configure formatter
        formatter = logging.Formatter(log_format, date_format)
        
        # Add console handler if requested
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self._get_log_level(console_level))
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler if a log file is specified
        if log_file:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Set up rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setLevel(self._get_log_level(file_level))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Log creation of logger
        self.debug(f"Initialized {name} logger")
    
    def _get_log_level(self, level_name: str) -> int:
        """
        Convert a log level name to a logging module level.
        
        Args:
            level_name: Name of the log level
            
        Returns:
            Logging module level
        """
        return getattr(logging, level_name.upper(), logging.INFO)
    
    def debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False):
        """Log an error message with optional exception info."""
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str, exc_info: bool = True):
        """Log a critical message with exception info by default."""
        self.logger.critical(message, exc_info=exc_info)
    
    def exception(self, message: str):
        """Log an exception message with exception info."""
        self.logger.exception(message)
    
    def log_dict(self, level: str, message: str, data: Dict[str, Any]):
        """
        Log a message with structured data.
        
        Args:
            level: Log level
            message: Message to log
            data: Structured data to log
        """
        log_level = self._get_log_level(level)
        
        # Format data for display
        formatted_data = "\n".join([f"  {k}: {v}" for k, v in data.items()])
        log_message = f"{message}\n{formatted_data}"
        
        self.logger.log(log_level, log_message)
    
    def get_logger(self):
        """Get the underlying logger instance."""
        return self.logger


# Configure root logger to reduce noise from other libraries
def configure_root_logger(level: str = "WARNING"):
    """
    Configure the root logger to reduce noise from third-party libraries.
    
    Args:
        level: Logging level for the root logger
    """
    # Set root logger level
    logging.getLogger().setLevel(getattr(logging, level.upper(), logging.WARNING))
    
    # Configure known noisy libraries to a higher level
    noisy_loggers = [
        "urllib3",
        "playwright",
        "asyncio",
        "parso",
        "websockets",
        "selenium",
        "boto3",
        "botocore",
        "requests",
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
