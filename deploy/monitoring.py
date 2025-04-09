import logging
import time
from typing import Dict, Any, Optional
from prometheus_client import start_http_server, Counter, Gauge, Histogram, CollectorRegistry
from .config import deployment_config
from datetime import datetime

logger = logging.getLogger(__name__)

class MonitoringService:
    _instance = None
    _registry = CollectorRegistry()
    _metrics = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MonitoringService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize monitoring service with metrics"""
        self._metrics = {}
        self.config = deployment_config
        
        # Message metrics
        self._metrics['messages_received'] = Counter(
            f'{self.config.metrics_prefix}_messages_received_total',
            'Total number of messages received',
            ['agent_type']
        )
        self._metrics['messages_processed'] = Counter(
            f'{self.config.metrics_prefix}_messages_processed_total',
            'Total number of messages processed',
            ['agent_type']
        )
        self._metrics['message_processing_time'] = Histogram(
            f'{self.config.metrics_prefix}_message_processing_time_seconds',
            'Time taken to process messages',
            ['agent_type']
        )
        
        # Error metrics
        self._metrics['errors'] = Counter(
            f'{self.config.metrics_prefix}_errors_total',
            'Total number of errors',
            ['error_type']
        )
        self._metrics['connection_errors'] = Counter(
            f'{self.config.metrics_prefix}_connection_errors_total',
            'Total number of connection errors'
        )
        
        # Cache metrics
        self._metrics['cache_hits'] = Counter(
            f'{self.config.metrics_prefix}_cache_hits_total',
            'Total number of cache hits'
        )
        self._metrics['cache_misses'] = Counter(
            f'{self.config.metrics_prefix}_cache_misses_total',
            'Total number of cache misses'
        )
        
        # Agent state metrics
        self._metrics['agent_state_size'] = Gauge(
            f'{self.config.metrics_prefix}_state_size_bytes',
            'Size of agent state in bytes',
            ['agent_type']
        )
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=self.config.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='logs/app.log'
        )
        self.logger = logging.getLogger('aware_agent')
        
    def start_monitoring(self):
        """Start the monitoring server"""
        try:
            start_http_server(self.config.monitoring_port)
            self.logger.info(f"Monitoring server started on port {self.config.monitoring_port}")
        except Exception as e:
            self.logger.error(f"Failed to start monitoring server: {str(e)}")
            raise
            
    def track_message(self, agent_type: str, processing_time: float):
        """Track message processing"""
        try:
            self._metrics['messages_received'].labels(agent_type=agent_type).inc()
            self._metrics['messages_processed'].labels(agent_type=agent_type).inc()
            self._metrics['message_processing_time'].labels(agent_type=agent_type).observe(processing_time)
        except Exception as e:
            self.logger.error(f"Failed to track message: {str(e)}")
            
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log error with context"""
        try:
            error_type = error.__class__.__name__
            self._metrics['errors'].labels(error_type=error_type).inc()
            
            if isinstance(error, ConnectionError):
                self._metrics['connection_errors'].inc()
                
            self.logger.error(
                f"Error: {str(error)}",
                extra={"context": context or {}}
            )
        except Exception as e:
            self.logger.error(f"Failed to log error: {str(e)}")
            
    def track_cache_hit(self):
        """Track cache hit"""
        try:
            self._metrics['cache_hits'].inc()
        except Exception as e:
            self.logger.error(f"Failed to track cache hit: {str(e)}")
            
    def track_cache_miss(self):
        """Track cache miss"""
        try:
            self._metrics['cache_misses'].inc()
        except Exception as e:
            self.logger.error(f"Failed to track cache miss: {str(e)}")
            
    def update_agent_state_size(self, agent_type: str, size_bytes: int):
        """Update agent state size"""
        try:
            self._metrics['agent_state_size'].labels(agent_type=agent_type).set(size_bytes)
        except Exception as e:
            self.logger.error(f"Failed to update agent state size: {str(e)}")

    def track_connection(self, connected: bool):
        if connected:
            self._metrics['active_connections'].inc()
        else:
            self._metrics['active_connections'].dec()

    def track_connection_error(self):
        self._metrics['connection_errors'].inc()

    def track_cache(self, hit: bool):
        if hit:
            self._metrics['cache_hits'].inc()
        else:
            self._metrics['cache_misses'].inc()

    def log_warning(self, message: str, context: Dict[str, Any]):
        if hasattr(self, 'logger'):
            self.logger.warning(message, extra=context)

    def log_info(self, message: str, context: Dict[str, Any]):
        if hasattr(self, 'logger'):
            self.logger.info(message, extra=context) 