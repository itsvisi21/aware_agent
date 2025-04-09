from typing import Dict, Any
from pydantic import BaseModel

class DeploymentConfig(BaseModel):
    """Configuration for deployment settings."""
    environment: str = "development"
    version: str = "1.0.0"
    debug: bool = True
    monitoring_enabled: bool = True
    
    class Config:
        arbitrary_types_allowed = True 