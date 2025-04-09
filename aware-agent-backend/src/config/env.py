import os
from pathlib import Path
from dotenv import load_dotenv

def load_env():
    """Load environment variables from .env file."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Load environment variables from .env file
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    
    # Set default values for required variables if not set
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "dummy-key-for-testing" 