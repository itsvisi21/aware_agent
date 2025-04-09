import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Centralized configuration settings for the application."""
    model_config = SettingsConfigDict(
        env_file=".env",
        protected_namespaces=('settings_',),
        extra='ignore'
    )

    # Application settings
    app_name: str = "Aware Agent"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    development_mode: bool = True  # Enable development mode by default

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # Database settings
    database_url: str = "sqlite:///./aware_agent.db"
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # Cache settings
    cache_type: str = "memory"
    cache_url: Optional[str] = None
    cache_ttl: int = 3600

    # Monitoring settings
    monitoring_enabled: bool = True
    monitoring_port: int = 9090
    metrics_prefix: str = "aware_agent"

    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[Path] = None

    # Agent settings
    agent_timeout: int = 30
    agent_max_retries: int = 3
    agent_concurrency_limit: int = 10

    # Memory settings
    memory_type: str = "local"
    memory_max_size: int = 1000
    memory_cleanup_interval: int = 3600
    memory_storage_path: Path = Path("memory_storage")

    # Semantic settings
    model_name: str = "gpt-3.5-turbo"
    semantic_model: str = "gpt-3.5-turbo"
    semantic_max_tokens: int = 4096
    temperature: float = 0.7
    semantic_temperature: float = 0.7
    openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", ""))

    # Workspace settings
    workspace_dir: Path = Path("workspace")
    workspace_max_size: int = 1000000

    # Security settings
    api_key: Optional[str] = None
    cors_origins: list = ["*"]
    rate_limit: int = 100

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure workspace directory exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        # Set log file if not specified
        if not self.log_file:
            self.log_file = self.workspace_dir / "logs" / "app.log"
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate OpenAI API key
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY environment variable.")

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow
        }

    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return {
            "type": self.cache_type,
            "url": self.cache_url,
            "ttl": self.cache_ttl
        }

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return {
            "enabled": self.monitoring_enabled,
            "port": self.monitoring_port,
            "prefix": self.metrics_prefix
        }

    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            "timeout": self.agent_timeout,
            "max_retries": self.agent_max_retries,
            "concurrency_limit": self.agent_concurrency_limit
        }

    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory configuration."""
        return {
            "type": self.memory_type,
            "max_size": self.memory_max_size,
            "cleanup_interval": self.memory_cleanup_interval
        }

    def get_semantic_config(self) -> Dict[str, Any]:
        """Get semantic configuration."""
        return {
            "model": self.semantic_model,
            "max_tokens": self.semantic_max_tokens,
            "temperature": self.semantic_temperature
        }

# Create a single instance of Settings
settings = Settings()
