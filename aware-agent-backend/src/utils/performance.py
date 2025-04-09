from src.config.settings import Settings
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitors and logs performance metrics for the application."""
    
    def __init__(self, settings: Settings):
        """Initialize the performance monitor.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.metrics: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.log_dir = "performance_logs"
        os.makedirs(self.log_dir, exist_ok=True)
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation.
        
        Args:
            operation: Name of the operation being timed
        """
        if operation not in self.metrics:
            self.metrics[operation] = {
                "count": 0,
                "total_time": 0,
                "min_time": float('inf'),
                "max_time": 0,
                "last_time": 0
            }
        self.start_time = time.time()
    
    def stop_timer(self, operation: str) -> None:
        """Stop timing an operation and record the metrics.
        
        Args:
            operation: Name of the operation being timed
        """
        if self.start_time is None:
            logger.warning(f"Timer not started for operation: {operation}")
            return
            
        duration = time.time() - self.start_time
        self.metrics[operation]["count"] += 1
        self.metrics[operation]["total_time"] += duration
        self.metrics[operation]["min_time"] = min(self.metrics[operation]["min_time"], duration)
        self.metrics[operation]["max_time"] = max(self.metrics[operation]["max_time"], duration)
        self.metrics[operation]["last_time"] = duration
        self.start_time = None
    
    def log_metrics(self) -> None:
        """Log the current performance metrics to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"performance_{timestamp}.json")
        
        with open(log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Performance metrics logged to {log_file}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current performance metrics.
        
        Returns:
            Dictionary containing the performance metrics
        """
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.metrics = {}
        self.start_time = None 