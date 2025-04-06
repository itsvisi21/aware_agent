from typing import Dict, Any, Optional
from datetime import datetime
import time
from dataclasses import dataclass
import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class PipelineMetrics:
    """Metrics for the research pipeline."""
    semantic_analysis_time: float
    context_chaining_time: float
    temporal_analysis_time: float
    role_mapping_time: float
    abstraction_processing_time: float
    agent_planning_time: float
    execution_time: float
    memory_operations_time: float
    total_time: float
    success: bool
    error: Optional[str] = None

class MetricsCollector:
    """Collects and manages pipeline metrics."""
    
    def __init__(self):
        self.timers: Dict[str, float] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.metrics_file = Path("metrics.json")
    
    def start_timer(self, name: str):
        """Start a timer for a specific operation."""
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """Stop a timer and return the elapsed time."""
        if name not in self.timers:
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        return elapsed
    
    def record_metrics(self, task_id: str, metrics: PipelineMetrics):
        """Record metrics for a specific task."""
        self.metrics[task_id] = {
            "timestamp": datetime.now().isoformat(),
            **metrics.__dict__
        }
        self._save_metrics()
    
    def get_metrics(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific task."""
        return self.metrics.get(task_id)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a report of all metrics."""
        if not self.metrics:
            return {}
        
        # Calculate averages
        total_tasks = len(self.metrics)
        successful_tasks = sum(1 for m in self.metrics.values() if m["success"])
        
        averages = {
            "semantic_analysis": 0.0,
            "context_chaining": 0.0,
            "temporal_analysis": 0.0,
            "role_mapping": 0.0,
            "abstraction_processing": 0.0,
            "agent_planning": 0.0,
            "execution": 0.0,
            "memory_operations": 0.0,
            "total": 0.0
        }
        
        for metrics in self.metrics.values():
            if metrics["success"]:
                averages["semantic_analysis"] += metrics["semantic_analysis_time"]
                averages["context_chaining"] += metrics["context_chaining_time"]
                averages["temporal_analysis"] += metrics["temporal_analysis_time"]
                averages["role_mapping"] += metrics["role_mapping_time"]
                averages["abstraction_processing"] += metrics["abstraction_processing_time"]
                averages["agent_planning"] += metrics["agent_planning_time"]
                averages["execution"] += metrics["execution_time"]
                averages["memory_operations"] += metrics["memory_operations_time"]
                averages["total"] += metrics["total_time"]
        
        if successful_tasks > 0:
            for key in averages:
                averages[key] /= successful_tasks
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            "average_times": averages,
            "recent_tasks": list(self.metrics.keys())[-10:]  # Last 10 tasks
        }
    
    def _save_metrics(self):
        """Save metrics to file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")
    
    def load_metrics(self):
        """Load metrics from file."""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics: {str(e)}")

# Global metrics collector instance
metrics_collector = MetricsCollector() 