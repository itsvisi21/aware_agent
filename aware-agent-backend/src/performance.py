from typing import Dict, Any, List, Optional, Callable
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from functools import wraps
import tracemalloc
import psutil
import gc

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitors and optimizes system performance."""
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.log_dir = Path("performance_logs")
        self.log_dir.mkdir(exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        tracemalloc.start()
    
    def track_performance(self, operation_name: str):
        """Decorator to track performance metrics."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = tracemalloc.get_traced_memory()
                process = psutil.Process()
                start_cpu = process.cpu_percent()
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    end_memory = tracemalloc.get_traced_memory()
                    end_cpu = process.cpu_percent()
                    
                    # Calculate metrics
                    duration = end_time - start_time
                    memory_usage = end_memory[0] - start_memory[0]
                    cpu_usage = end_cpu - start_cpu
                    
                    # Record metrics
                    self._record_metrics(
                        operation_name,
                        duration,
                        memory_usage,
                        cpu_usage
                    )
            return wrapper
        return decorator
    
    def _record_metrics(
        self,
        operation_name: str,
        duration: float,
        memory_usage: int,
        cpu_usage: float
    ):
        """Record performance metrics."""
        if operation_name not in self.metrics:
            self.metrics[operation_name] = []
        
        self.metrics[operation_name].append({
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "memory_usage": memory_usage,
            "cpu_usage": cpu_usage
        })
        
        # Save metrics periodically
        if len(self.metrics[operation_name]) % 10 == 0:
            self._save_metrics()
    
    def _save_metrics(self):
        """Save performance metrics to file."""
        try:
            for operation, metrics in self.metrics.items():
                log_file = self.log_dir / f"{operation}_metrics.json"
                with open(log_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {str(e)}")
    
    async def optimize_memory(self):
        """Optimize memory usage."""
        try:
            # Clear unused memory
            gc.collect()
            
            # Get current memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Log memory optimization
            logger.info(
                f"Memory optimization: RSS={memory_info.rss / 1024 / 1024:.2f}MB, "
                f"VMS={memory_info.vms / 1024 / 1024:.2f}MB"
            )
        except Exception as e:
            logger.error(f"Failed to optimize memory: {str(e)}")
    
    async def parallel_process(
        self,
        tasks: List[Callable],
        max_workers: Optional[int] = None
    ) -> List[Any]:
        """Process tasks in parallel."""
        try:
            if max_workers is None:
                max_workers = min(len(tasks), 4)
            
            # Create semaphore to limit concurrent tasks
            semaphore = asyncio.Semaphore(max_workers)
            
            async def process_with_semaphore(task):
                async with semaphore:
                    return await task
            
            # Process tasks in parallel
            results = await asyncio.gather(
                *[process_with_semaphore(task) for task in tasks]
            )
            
            return results
        except Exception as e:
            logger.error(f"Failed to process tasks in parallel: {str(e)}")
            raise
    
    async def batch_process(
        self,
        items: List[Any],
        process_func: Callable,
        batch_size: int = 10
    ) -> List[Any]:
        """Process items in batches."""
        try:
            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_results = await asyncio.gather(
                    *[process_func(item) for item in batch]
                )
                results.extend(batch_results)
                
                # Optimize memory between batches
                await self.optimize_memory()
            
            return results
        except Exception as e:
            logger.error(f"Failed to process items in batches: {str(e)}")
            raise
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        report = {}
        
        for operation, metrics in self.metrics.items():
            if not metrics:
                continue
            
            durations = [m["duration"] for m in metrics]
            memory_usages = [m["memory_usage"] for m in metrics]
            cpu_usages = [m["cpu_usage"] for m in metrics]
            
            report[operation] = {
                "total_operations": len(metrics),
                "average_duration": np.mean(durations),
                "min_duration": np.min(durations),
                "max_duration": np.max(durations),
                "average_memory_usage": np.mean(memory_usages),
                "max_memory_usage": np.max(memory_usages),
                "average_cpu_usage": np.mean(cpu_usages),
                "max_cpu_usage": np.max(cpu_usages)
            }
        
        return report

# Global performance monitor instance
performance_monitor = PerformanceMonitor() 