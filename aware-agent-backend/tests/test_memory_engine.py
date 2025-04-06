"""
Tests for the memory engine.
"""

import pytest
from src.memory_engine import MemoryEngine
from tests.config import TEST_DATA, TEST_CONFIG

@pytest.fixture
def memory_engine():
    """Create a memory engine instance for testing."""
    return MemoryEngine()

def test_context_storage_and_retrieval(memory_engine):
    """Test storing and retrieving context."""
    test_context = {
        "query": TEST_DATA["research_queries"][0]["query"],
        "context": TEST_DATA["research_queries"][0]["context"],
        "semantic_roles": TEST_DATA["semantic_roles"]
    }
    
    # Store context
    task_id = "test_task_1"
    memory_engine.store_context(task_id, test_context)
    
    # Retrieve context
    retrieved_context = memory_engine.retrieve_context(task_id)
    
    assert retrieved_context is not None
    assert retrieved_context["query"] == test_context["query"]
    assert retrieved_context["context"] == test_context["context"]
    assert retrieved_context["semantic_roles"] == test_context["semantic_roles"]

def test_action_storage_and_retrieval(memory_engine):
    """Test storing and retrieving actions."""
    test_actions = [
        {"type": "research", "content": "Find papers on Sanskrit programming"},
        {"type": "analysis", "content": "Analyze quantum cryptography papers"}
    ]
    
    # Store actions
    task_id = "test_task_2"
    for action in test_actions:
        memory_engine.store_action(task_id, action)
    
    # Retrieve actions
    retrieved_actions = memory_engine.retrieve_actions(task_id)
    
    assert len(retrieved_actions) == len(test_actions)
    assert retrieved_actions[0]["type"] == test_actions[0]["type"]
    assert retrieved_actions[1]["content"] == test_actions[1]["content"]

def test_context_cleanup(memory_engine):
    """Test context cleanup after retention period."""
    test_context = {
        "query": TEST_DATA["research_queries"][1]["query"],
        "context": TEST_DATA["research_queries"][1]["context"]
    }
    
    # Store context with short retention period
    task_id = "test_task_3"
    memory_engine.store_context(task_id, test_context, retention_period=1)  # 1 second retention
    
    # Wait for retention period to expire
    import time
    time.sleep(2)
    
    # Attempt to retrieve context
    retrieved_context = memory_engine.retrieve_context(task_id)
    
    assert retrieved_context is None

def test_error_handling(memory_engine):
    """Test error handling in the memory engine."""
    # Test invalid task ID
    assert memory_engine.retrieve_context("nonexistent_task") is None
    assert memory_engine.retrieve_actions("nonexistent_task") == []
    
    # Test invalid context storage
    with pytest.raises(Exception):
        memory_engine.store_context("", None)
    
    # Test invalid action storage
    with pytest.raises(Exception):
        memory_engine.store_action("", None)

def test_concurrent_access(memory_engine):
    """Test concurrent access to the memory engine."""
    import threading
    
    def store_context(task_id, context):
        memory_engine.store_context(task_id, context)
    
    def retrieve_context(task_id):
        return memory_engine.retrieve_context(task_id)
    
    # Create multiple threads
    threads = []
    contexts = []
    
    for i in range(5):
        context = {
            "query": f"Test query {i}",
            "context": {"domain": f"Test domain {i}"}
        }
        contexts.append(context)
        
        t = threading.Thread(
            target=store_context,
            args=(f"concurrent_task_{i}", context)
        )
        threads.append(t)
    
    # Start all threads
    for t in threads:
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    # Verify all contexts were stored correctly
    for i in range(5):
        retrieved = memory_engine.retrieve_context(f"concurrent_task_{i}")
        assert retrieved is not None
        assert retrieved["query"] == contexts[i]["query"]
        assert retrieved["context"] == contexts[i]["context"] 