import os
from typing import Optional
from pydantic import BaseSettings

class DeploymentConfig(BaseSettings):
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///aware_agent.db")
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "20"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))

    # Cache
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))

    # WebSocket
    WS_HOST: str = os.getenv("WS_HOST", "0.0.0.0")
    WS_PORT: int = int(os.getenv("WS_PORT", "8000"))
    WS_MAX_CONNECTIONS: int = int(os.getenv("WS_MAX_CONNECTIONS", "1000"))

    # Security
    AUTH_SECRET: str = os.getenv("AUTH_SECRET", "development-secret-key")
    RATE_LIMIT: int = int(os.getenv("RATE_LIMIT", "100"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

    # Monitoring
    MONITORING_PORT: int = int(os.getenv("MONITORING_PORT", "8001"))
    METRICS_PATH: str = os.getenv("METRICS_PATH", "/metrics")
    HEALTH_PATH: str = os.getenv("HEALTH_PATH", "/health")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8" 