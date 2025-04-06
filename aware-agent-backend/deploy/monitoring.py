import logging
import time
from typing import Dict, Any
from prometheus_client import start_http_server, Counter, Gauge, Histogram
from .config import DeploymentConfig

class MonitoringService:
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.setup_metrics()
        self.setup_logging()

    def setup_metrics(self):
        # Message metrics
        self.messages_received = Counter(
            'aware_agent_messages_received_total',
            'Total number of messages received',
            ['agent_type']
        )
        self.messages_processed = Counter(
            'aware_agent_messages_processed_total',
            'Total number of messages processed',
            ['agent_type']
        )
        self.message_processing_time = Histogram(
            'aware_agent_message_processing_seconds',
            'Time spent processing messages',
            ['agent_type']
        )

        # Connection metrics
        self.active_connections = Gauge(
            'aware_agent_active_connections',
            'Number of active WebSocket connections'
        )
        self.connection_errors = Counter(
            'aware_agent_connection_errors_total',
            'Total number of connection errors'
        )

        # Cache metrics
        self.cache_hits = Counter(
            'aware_agent_cache_hits_total',
            'Total number of cache hits'
        )
        self.cache_misses = Counter(
            'aware_agent_cache_misses_total',
            'Total number of cache misses'
        )

        # Agent state metrics
        self.agent_state_size = Gauge(
            'aware_agent_state_size_bytes',
            'Size of agent state in bytes',
            ['agent_type']
        )

    def setup_logging(self):
        logging.basicConfig(
            level=self.config.get('log_level', 'INFO'),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='logs/app.log'
        )
        self.logger = logging.getLogger('aware_agent')

    def start_monitoring(self, port: int = 8001):
        start_http_server(port)
        self.logger.info(f"Monitoring server started on port {port}")

    def track_message(self, agent_type: str, processing_time: float):
        self.messages_received.labels(agent_type=agent_type).inc()
        self.messages_processed.labels(agent_type=agent_type).inc()
        self.message_processing_time.labels(agent_type=agent_type).observe(processing_time)

    def track_connection(self, connected: bool):
        if connected:
            self.active_connections.inc()
        else:
            self.active_connections.dec()

    def track_connection_error(self):
        self.connection_errors.inc()

    def track_cache(self, hit: bool):
        if hit:
            self.cache_hits.inc()
        else:
            self.cache_misses.inc()

    def track_agent_state(self, agent_type: str, state_size: int):
        self.agent_state_size.labels(agent_type=agent_type).set(state_size)

    def log_error(self, error: Exception, context: Dict[str, Any]):
        self.logger.error(f"Error occurred: {str(error)}", extra=context)

    def log_warning(self, message: str, context: Dict[str, Any]):
        self.logger.warning(message, extra=context)

    def log_info(self, message: str, context: Dict[str, Any]):
        self.logger.info(message, extra=context) 