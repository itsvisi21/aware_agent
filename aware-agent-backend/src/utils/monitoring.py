import logging
import time
from typing import Dict, Any, Optional
from prometheus_client import Counter, Gauge, Histogram, start_http_server, REGISTRY
from src.config.settings import Settings

logger = logging.getLogger(__name__)

class MonitoringService:
    _instance = None
    _initialized = False

    def __new__(cls, settings: Settings = None):
        if cls._instance is None:
            cls._instance = super(MonitoringService, cls).__new__(cls)
        return cls._instance

    def __init__(self, settings: Settings = None):
        if not self._initialized:
            self.settings = settings or Settings()
            self.metrics: Dict[str, Any] = {}
            self._setup_metrics()
            self._start_server()
            self._initialized = True

    def _setup_metrics(self):
        """Initialize Prometheus metrics."""
        prefix = self.settings.metrics_prefix

        # Clear any existing metrics with the same prefix
        for collector in list(REGISTRY._collector_to_names):
            try:
                # Check if collector has a name attribute and if it starts with our prefix
                if hasattr(collector, '_name') and collector._name and collector._name.startswith(prefix):
                    REGISTRY.unregister(collector)
            except Exception as e:
                logger.warning(f"Failed to unregister collector: {e}")

        # Request metrics
        self.metrics["request_count"] = Counter(
            f"{prefix}_request_total",
            "Total number of requests processed"
        )
        self.metrics["request_latency"] = Histogram(
            f"{prefix}_request_latency_seconds",
            "Request latency in seconds"
        )

        # Agent metrics
        self.metrics["agent_active"] = Gauge(
            f"{prefix}_agent_active",
            "Number of currently active agents"
        )
        self.metrics["agent_errors"] = Counter(
            f"{prefix}_agent_errors_total",
            "Total number of agent errors"
        )

        # Memory metrics
        self.metrics["memory_usage"] = Gauge(
            f"{prefix}_memory_usage_bytes",
            "Current memory usage in bytes"
        )
        self.metrics["memory_items"] = Gauge(
            f"{prefix}_memory_items",
            "Number of items in memory"
        )

        # Cache metrics
        self.metrics["cache_hits"] = Counter(
            f"{prefix}_cache_hits_total",
            "Total number of cache hits"
        )
        self.metrics["cache_misses"] = Counter(
            f"{prefix}_cache_misses_total",
            "Total number of cache misses"
        )

        # WebSocket metrics
        self.metrics["ws_connections"] = Gauge(
            f"{prefix}_ws_connections",
            "Number of active WebSocket connections"
        )
        self.metrics["ws_messages"] = Counter(
            f"{prefix}_ws_messages_total",
            "Total number of WebSocket messages"
        )

    def _start_server(self):
        """Start the Prometheus metrics server."""
        if self.settings.monitoring_enabled:
            try:
                start_http_server(self.settings.monitoring_port)
                logger.info(f"Started monitoring server on port {self.settings.monitoring_port}")
            except Exception as e:
                logger.error(f"Failed to start monitoring server: {e}")

    def track_request(self, duration: float):
        """Track a request and its duration."""
        self.metrics["request_count"].inc()
        self.metrics["request_latency"].observe(duration)

    def track_agent(self, active: bool):
        """Track agent activity."""
        if active:
            self.metrics["agent_active"].inc()
        else:
            self.metrics["agent_active"].dec()

    def track_agent_error(self):
        """Track an agent error."""
        self.metrics["agent_errors"].inc()

    def track_memory(self, usage: int, items: int):
        """Track memory usage."""
        self.metrics["memory_usage"].set(usage)
        self.metrics["memory_items"].set(items)

    def track_cache(self, hit: bool):
        """Track cache hit/miss."""
        if hit:
            self.metrics["cache_hits"].inc()
        else:
            self.metrics["cache_misses"].inc()

    def track_ws_connection(self, connected: bool):
        """Track WebSocket connection."""
        if connected:
            self.metrics["ws_connections"].inc()
        else:
            self.metrics["ws_connections"].dec()

    def track_ws_message(self):
        """Track WebSocket message."""
        self.metrics["ws_messages"].inc()
