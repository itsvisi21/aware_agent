import os
import sys
import logging
from typing import Optional
from src.config import DeploymentConfig
from src.main import app
from src.monitoring import MonitoringService
from src.database import DatabaseService
from src.cache import CacheService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(os.path.join(log_dir, "application.log"))
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def validate_environment() -> Optional[str]:
    """Validate application environment"""
    config = DeploymentConfig()
    
    # Check required environment variables
    required_vars = [
        "DATABASE_URL",
        "REDIS_URL",
        "AUTH_SECRET"
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            return f"Missing required environment variable: {var}"
    
    return None

async def setup_services():
    """Setup required services"""
    try:
        # Initialize services
        database = DatabaseService()
        cache = CacheService()
        monitoring = MonitoringService()
        
        # Connect to services
        await database.connect()
        await cache.connect()
        await monitoring.start()
        
        return database, cache, monitoring
        
    except Exception as e:
        logger.error(f"Service setup failed: {str(e)}")
        raise

async def cleanup_services(database, cache, monitoring):
    """Cleanup services"""
    try:
        await database.disconnect()
        await cache.disconnect()
        await monitoring.stop()
        
    except Exception as e:
        logger.error(f"Service cleanup failed: {str(e)}")
        raise

def main():
    """Main function"""
    try:
        # Setup logging
        setup_logging()
        logger.info("Starting application...")
        
        # Validate environment
        if error := validate_environment():
            logger.error(f"Environment validation failed: {error}")
            sys.exit(1)
        
        # Setup services
        database, cache, monitoring = setup_services()
        
        # Start application
        import uvicorn
        uvicorn.run(
            app,
            host=os.getenv("WS_HOST", "0.0.0.0"),
            port=int(os.getenv("WS_PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "info").lower()
        )
        
        # Cleanup services
        cleanup_services(database, cache, monitoring)
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 