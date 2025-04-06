from typing import Dict, Any
import logging
import psutil
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class MonitoringService:
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.start_time = time.time()
        
    async def start(self):
        """Start monitoring service"""
        logger.info("Monitoring service started")
        
    async def stop(self):
        """Stop monitoring service"""
        logger.info("Monitoring service stopped")
        
    async def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            
            # Calculate uptime
            uptime = time.time() - self.start_time
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "cpu_usage": cpu_percent,
                "memory_usage": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent
                },
                "disk_usage": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent
                },
                "uptime": uptime,
                "metrics": self.metrics
            }
        except Exception as e:
            logger.error(f"Failed to get system status: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
            
    def record_metric(self, name: str, value: Any):
        """Record a metric"""
        self.metrics[name] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics"""
        return self.metrics 