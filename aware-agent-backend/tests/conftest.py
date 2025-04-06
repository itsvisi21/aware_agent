import os
import pytest
from unittest.mock import patch
from dotenv import load_dotenv
import asyncio
from src.memory.persistence import MemoryStore
from src.interaction.interaction_engine import InteractionEngine
from tests.config import TEST_CONFIG

def pytest_configure(config):
    """Set up test environment before any tests run."""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Set a dummy OpenAI API key for testing if not already set
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "dummy-key-for-testing"

@pytest.fixture(autouse=True)
def mock_openai():
    """Mock OpenAI API calls during tests."""
    with patch("langchain_openai.OpenAI") as mock_openai:
        mock_openai.return_value.temperature = 0.7
        yield mock_openai

@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment variables after tests."""
    yield
    # Clean up after tests if needed
    # We don't remove OPENAI_API_KEY here as it might be needed for other tests 

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def memory_store():
    """Create a memory store instance with test configuration."""
    store = MemoryStore(TEST_CONFIG["database"]["path"])
    yield store
    # Cleanup if needed

@pytest.fixture
async def interaction_engine(memory_store):
    """Create an interaction engine instance with test configuration."""
    engine = InteractionEngine()
    yield engine
    # Cleanup if needed

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["OPENAI_API_KEY"] = "test-key-123"
    yield
    del os.environ["OPENAI_API_KEY"]

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close() 