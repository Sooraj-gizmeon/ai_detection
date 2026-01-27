# src/utils/logging_utils.py
"""Logging utilities for the video processing pipeline"""

import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime


def setup_logging(level=logging.INFO, log_file=None, log_dir="logs"):
    """
    Setup logging configuration for the application.
    
    Args:
        level: Logging level
        log_file: Specific log file name (optional)
        log_dir: Directory for log files
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Generate log filename if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"video_processing_{timestamp}.log"
    
    log_filepath = log_path / log_file
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_filepath,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Log startup message
    logging.info(f"Logging initialized - Level: {logging.getLevelName(level)}")
    logging.info(f"Log file: {log_filepath}")
    
    return str(log_filepath)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_system_info():
    """Log system information."""
    import platform
    import psutil
    
    logger = get_logger(__name__)
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"CPU Count: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(f"Disk Space: {psutil.disk_usage('/').total / (1024**3):.1f} GB")
    
    # GPU information if available
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA Available: {torch.cuda.is_available()}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB)")
    except ImportError:
        logger.info("PyTorch not available")
    
    logger.info("=== End System Information ===")


def log_processing_start(video_path: str, config: dict):
    """
    Log the start of video processing.
    
    Args:
        video_path: Path to video being processed
        config: Processing configuration
    """
    logger = get_logger(__name__)
    
    logger.info("=== Processing Started ===")
    logger.info(f"Video: {video_path}")
    logger.info(f"Configuration: {config}")
    logger.info("=" * 50)


def log_processing_end(results: dict):
    """
    Log the end of video processing.
    
    Args:
        results: Processing results
    """
    logger = get_logger(__name__)
    
    logger.info("=== Processing Completed ===")
    logger.info(f"Results: {results}")
    logger.info("=" * 50)


def log_error(error_msg: str, exception: Exception = None):
    """
    Log an error with optional exception details.
    
    Args:
        error_msg: Error message
        exception: Optional exception object
    """
    logger = get_logger(__name__)
    
    logger.error(f"ERROR: {error_msg}")
    if exception:
        logger.exception(f"Exception details: {exception}")


def create_performance_logger(name: str) -> logging.Logger:
    """
    Create a specialized logger for performance metrics.
    
    Args:
        name: Logger name
        
    Returns:
        Performance logger
    """
    logger = logging.getLogger(f"performance.{name}")
    
    # Create performance log file
    perf_log_path = Path("logs") / f"performance_{name}.log"
    
    # Performance-specific formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Performance file handler
    perf_handler = logging.handlers.RotatingFileHandler(
        perf_log_path,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(formatter)
    
    logger.addHandler(perf_handler)
    logger.setLevel(logging.INFO)
    
    return logger
