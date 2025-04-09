from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class DeploymentConfig(BaseSettings):
    """Deployment configuration."""
    
    # Environment
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=True)
    LOG_LEVEL: str = Field(default="INFO")
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///aware_agent.db")
    DB_POOL_SIZE: int = Field(default=20)
    DB_MAX_OVERFLOW: int = Field(default=10)
    
    # Cache
    REDIS_URL: str = Field(default="redis://localhost:6380/0")
    CACHE_TTL: int = Field(default=3600)
    
    # WebSocket
    WS_HOST: str = Field(default="0.0.0.0")
    WS_PORT: int = Field(default=8000)
    WS_MAX_CONNECTIONS: int = Field(default=1000)
    
    # Security
    AUTH_SECRET: str = Field(default="development-secret-key")
    RATE_LIMIT: int = Field(default=100)
    RATE_LIMIT_WINDOW: int = Field(default=60)
    
    # Monitoring
    MONITORING_PORT: int = Field(default=8001)
    METRICS_PATH: str = Field(default="/metrics")
    HEALTH_PATH: str = Field(default="/health")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )

deployment_config = DeploymentConfig() 