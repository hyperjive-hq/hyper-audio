"""Logging utilities for Hyper Audio."""

import logging
import sys
from pathlib import Path
from typing import Optional

from ..config.settings import settings


def setup_logging(
    log_file: Optional[Path] = None,
    log_level: str = "INFO",
    console_output: bool = True
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_file: Path to log file (default from settings)
        log_level: Logging level (default: INFO)
        console_output: Whether to output to console (default: True)
        
    Returns:
        Configured logger
    """
    if log_file is None:
        log_file = settings.log_file

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get root logger
    logger = logging.getLogger("hyper_audio")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"hyper_audio.{name}")


# Setup default logger
default_logger = setup_logging(log_level=settings.log_level)
