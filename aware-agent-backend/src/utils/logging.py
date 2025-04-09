import logging
import os
from pathlib import Path
from typing import Optional

def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: The logging level (e.g., "INFO", "DEBUG", "ERROR")
        log_file: Optional path to the log file
    """
    # Create log directory if it doesn't exist
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=str(log_file) if log_file else None
    )
    
    # Also log to console if no log file is specified
    if not log_file:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized with level {log_level}") 