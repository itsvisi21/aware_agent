import logging
import os
import sys
import asyncio
import uvicorn
from pathlib import Path
from typing import Optional
import warnings
import ssl
import requests
import urllib3
import socket

from src.utils.cache import CacheService
from src.config.settings import Settings
from src.data.database import DatabaseService
from src.main import app
from src.utils.monitoring import MonitoringService
from src.core.services.websocket_manager import websocket_manager
from src.app import create_app
from src.core.services.semantic_abstraction import SemanticAbstractionLayer

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# Disable SSL verification for Hugging Face downloads
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
requests.packages.urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Print startup banner
print("\n" + "="*50)
print("Starting Aware Agent Backend Application")
print("="*50 + "\n")

def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return False
        except socket.error:
            return True

def find_available_port(start_port: int = 8000, max_port: int = 9000) -> int:
    """Find an available port in the given range."""
    for port in range(start_port, max_port + 1):
        if not is_port_in_use(port):
            return port
    raise RuntimeError(f"No available ports found between {start_port} and {max_port}")

def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(log_dir, "application.log"))
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def validate_environment() -> Optional[str]:
    """Validate application environment"""
    settings = Settings()

    # Check required settings
    required_settings = [
        "openai_api_key",
        "model_name",
        "memory_storage_path"
    ]

    for setting in required_settings:
        if not getattr(settings, setting):
            return f"Missing required setting: {setting}"

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


async def main():
    """Main entry point for the application."""
    try:
        print("\nInitializing application...")
        logger.debug("Starting application initialization...")
        
        # Find an available port
        port = find_available_port(8001)  # Start from 8001
        print(f"Using port {port}")
        
        # Initialize settings
        print("Loading settings...")
        logger.debug("Initializing settings...")
        settings = Settings()
        
        # Validate required settings
        print("Validating settings...")
        logger.debug("Validating settings...")
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is required")
        if not settings.model_name:
            raise ValueError("Model name is required")
        if not settings.memory_storage_path:
            raise ValueError("Memory storage path is required")
        
        # Initialize semantic layer
        print("Initializing semantic layer...")
        logger.debug("Initializing semantic layer...")
        semantic_layer = SemanticAbstractionLayer()
        
        # Create FastAPI app
        print("Creating FastAPI application...")
        logger.debug("Creating FastAPI application...")
        app = create_app(settings=settings, semantic_layer=semantic_layer)
        
        # Initialize WebSocket manager
        print("Initializing WebSocket manager...")
        logger.debug("Initializing WebSocket manager...")
        await websocket_manager.initialize()
        
        # Configure uvicorn
        print("Configuring uvicorn server...")
        logger.debug("Configuring uvicorn server...")
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="debug",
            reload=True,
            workers=1,
            loop="asyncio",
            http="h11",
            ws="websockets",
            lifespan="on",
            access_log=True,
            use_colors=True,
            proxy_headers=True,
            server_header=True,
            date_header=True
        )
        
        # Create and run server
        print("\n" + "="*50)
        print("Application is running and ready to accept connections")
        print(f"Server running at http://127.0.0.1:{port}")
        print("="*50 + "\n")
        logger.debug("Starting uvicorn server...")
        server = uvicorn.Server(config)
        await server.serve()
    except Exception as e:
        print(f"\nError: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise
    finally:
        print("\nCleaning up resources...")
        logger.debug("Cleaning up resources...")
        if hasattr(websocket_manager, 'cleanup'):
            await websocket_manager.cleanup()
        print("Cleanup completed")
        logger.debug("Cleanup completed")


if __name__ == "__main__":
    try:
        print("Starting application...")
        logger.debug("Starting application...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
        logger.info("Application stopped by user")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)
        sys.exit(1)
