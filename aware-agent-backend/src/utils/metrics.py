import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

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


class MetricsService:
    """Service for collecting and managing metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.timers: Dict[str, float] = {}
        self.metrics_file = Path("metrics.json")
    
    def start_timer(self, name: str) -> None:
        """Start a timer for a metric."""
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """Stop a timer and return the elapsed time."""
        if name not in self.timers:
            return 0.0
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        return elapsed
    
    def record_metric(self, name: str, value: Any) -> None:
        """Record a metric value."""
        self.metrics[name] = value
    
    def get_metric(self, name: str) -> Optional[Any]:
        """Get a metric value."""
        return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics."""
        return self.metrics.copy()
    
    def save_metrics(self) -> None:
        """Save metrics to file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def load_metrics(self) -> None:
        """Load metrics from file."""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")


class MetricsCollector:
    """Collector for pipeline metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, PipelineMetrics] = {}
        self.metrics_file = Path("pipeline_metrics.json")
        self.timers: Dict[str, float] = {}
    
    def start_timer(self, name: str):
        """Start a timer for a metric."""
        if name in self.timers:
            # If timer already exists, stop it first
            self.stop_timer(name)
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """Stop a timer and return the elapsed time."""
        if name not in self.timers:
            return 0.0
        try:
            elapsed = time.time() - self.timers[name]
            return elapsed
        finally:
            # Always clean up the timer
            del self.timers[name]
    
    def record_metrics(self, task_id: str, metrics: PipelineMetrics):
        """Record pipeline metrics for a task."""
        self.metrics[task_id] = metrics
        self._save_metrics()
    
    def get_metrics(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a task."""
        return self.metrics.get(task_id)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a metrics report."""
        report = {
            "total_tasks": len(self.metrics),
            "successful_tasks": sum(1 for m in self.metrics.values() if m.success),
            "failed_tasks": sum(1 for m in self.metrics.values() if not m.success),
            "average_times": {
                "semantic_analysis": sum(m.semantic_analysis_time for m in self.metrics.values()) / len(self.metrics),
                "context_chaining": sum(m.context_chaining_time for m in self.metrics.values()) / len(self.metrics),
                "temporal_analysis": sum(m.temporal_analysis_time for m in self.metrics.values()) / len(self.metrics),
                "role_mapping": sum(m.role_mapping_time for m in self.metrics.values()) / len(self.metrics),
                "abstraction_processing": sum(m.abstraction_processing_time for m in self.metrics.values()) / len(self.metrics),
                "agent_planning": sum(m.agent_planning_time for m in self.metrics.values()) / len(self.metrics),
                "execution": sum(m.execution_time for m in self.metrics.values()) / len(self.metrics),
                "memory_operations": sum(m.memory_operations_time for m in self.metrics.values()) / len(self.metrics),
                "total": sum(m.total_time for m in self.metrics.values()) / len(self.metrics)
            }
        }
        return report
    
    def _save_metrics(self):
        """Save metrics to file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump({k: v.__dict__ for k, v in self.metrics.items()}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def load_metrics(self):
        """Load metrics from file."""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics = {k: PipelineMetrics(**v) for k, v in data.items()}
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")


# Global metrics collector instance
metrics_collector = MetricsCollector()
