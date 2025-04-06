import os
from typing import Optional
from dotenv import load_dotenv

class DeploymentConfig:
    def __init__(self):
        load_dotenv()
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Database configuration
        self.db_url = os.getenv("DATABASE_URL")
        self.db_pool_size = int(os.getenv("DB_POOL_SIZE", "20"))
        self.db_max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))
        
        # Cache configuration
        self.redis_url = os.getenv("REDIS_URL")
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
        
        # WebSocket configuration
        self.ws_host = os.getenv("WS_HOST", "0.0.0.0")
        self.ws_port = int(os.getenv("WS_PORT", "8000"))
        self.ws_max_connections = int(os.getenv("WS_MAX_CONNECTIONS", "1000"))
        
        # Monitoring configuration
        self.monitoring_port = int(os.getenv("MONITORING_PORT", "8001"))
        self.metrics_path = os.getenv("METRICS_PATH", "/metrics")
        self.health_path = os.getenv("HEALTH_PATH", "/health")
        
        # Security configuration
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.rate_limit = int(os.getenv("RATE_LIMIT", "100"))
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
        
        # Backup configuration
        self.backup_dir = os.getenv("BACKUP_DIR", "backups")
        self.backup_interval = int(os.getenv("BACKUP_INTERVAL", "86400"))
        self.backup_retention = int(os.getenv("BACKUP_RETENTION", "7"))
        
        # Load balancer configuration
        self.lb_enabled = os.getenv("LOAD_BALANCER_ENABLED", "false").lower() == "true"
        self.lb_host = os.getenv("LOAD_BALANCER_HOST")
        self.lb_port = int(os.getenv("LOAD_BALANCER_PORT", "80"))
        
        # SSL configuration
        self.ssl_enabled = os.getenv("SSL_ENABLED", "false").lower() == "true"
        self.ssl_cert = os.getenv("SSL_CERT")
        self.ssl_key = os.getenv("SSL_KEY")
        
        # Scaling configuration
        self.min_instances = int(os.getenv("MIN_INSTANCES", "1"))
        self.max_instances = int(os.getenv("MAX_INSTANCES", "10"))
        self.scale_up_threshold = float(os.getenv("SCALE_UP_THRESHOLD", "0.8"))
        self.scale_down_threshold = float(os.getenv("SCALE_DOWN_THRESHOLD", "0.2"))
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    @property
    def is_staging(self) -> bool:
        return self.environment == "staging"
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"
    
    def validate(self) -> Optional[str]:
        """Validate configuration and return error message if invalid"""
        if not self.db_url:
            return "DATABASE_URL is required"
        if not self.redis_url:
            return "REDIS_URL is required"
        if self.is_production and not self.auth_secret:
            return "AUTH_SECRET is required in production"
        if self.ssl_enabled and (not self.ssl_cert or not self.ssl_key):
            return "SSL certificate and key are required when SSL is enabled"
        if self.lb_enabled and not self.lb_host:
            return "LOAD_BALANCER_HOST is required when load balancer is enabled"
        return None 