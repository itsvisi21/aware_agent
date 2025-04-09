from typing import Dict, Any
from datetime import datetime

class MonitoringService:
    """Service for monitoring application metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.start_time = datetime.now()
    
    def record_metric(self, name: str, value: Any) -> None:
        """Record a metric value."""
        self.metrics[name] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics."""
        return {
            **self.metrics,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
        } 