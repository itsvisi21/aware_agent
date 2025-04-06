"""
Tests for the execution layer.
"""

import pytest
from src.execution_layer import ExecutionLayer
from tests.config import TEST_DATA, TEST_CONFIG

@pytest.fixture
def execution_layer():
    """Create an execution layer instance for testing."""
    return ExecutionLayer()

def test_task_execution(execution_layer):
    """Test basic task execution."""
    test_actions = [
        {
            "type": "research",
            "content": "Find papers on Sanskrit programming",
            "parameters": {
                "max_results": 5,
                "timeframe": "last 5 years"
            }
        }
    ]
    
    context = {
        "query": TEST_DATA["research_queries"][0]["query"],
        "context": TEST_DATA["research_queries"][0]["context"]
    }
    
    result = execution_layer.execute_task(test_actions, context)
    
    assert result.status == "success"
    assert result.content is not None
    assert len(result.results) > 0
    assert result.execution_time > 0

def test_parallel_execution(execution_layer):
    """Test parallel execution of multiple tasks."""
    test_actions = [
        {
            "type": "research",
            "content": "Find quantum computing papers",
            "parameters": {"max_results": 3}
        },
        {
            "type": "analysis",
            "content": "Analyze cryptography impact",
            "parameters": {"depth": "detailed"}
        }
    ]
    
    context = {
        "query": TEST_DATA["research_queries"][1]["query"],
        "context": TEST_DATA["research_queries"][1]["context"]
    }
    
    result = execution_layer.execute_task(test_actions, context)
    
    assert result.status == "success"
    assert len(result.results) == len(test_actions)
    assert all(r["status"] == "completed" for r in result.results)

def test_error_handling(execution_layer):
    """Test error handling in task execution."""
    # Test invalid action type
    with pytest.raises(Exception):
        execution_layer.execute_task([
            {"type": "invalid_type", "content": "test"}
        ], {})
    
    # Test missing required parameters
    with pytest.raises(Exception):
        execution_layer.execute_task([
            {"type": "research", "content": None}
        ], {})
    
    # Test timeout handling
    with pytest.raises(Exception):
        execution_layer.execute_task([
            {
                "type": "research",
                "content": "test",
                "parameters": {"timeout": 0.001}
            }
        ], {})

def test_resource_management(execution_layer):
    """Test resource management during execution."""
    test_actions = [
        {
            "type": "research",
            "content": "Test resource usage",
            "parameters": {
                "max_memory": "1GB",
                "max_cpu": "50%"
            }
        }
    ]
    
    context = {
        "query": "test",
        "context": {"domain": "test"}
    }
    
    result = execution_layer.execute_task(test_actions, context)
    
    assert result.status == "success"
    assert "resource_usage" in result.metadata
    assert result.metadata["resource_usage"]["memory"] <= 1024  # MB
    assert result.metadata["resource_usage"]["cpu"] <= 50  # percentage

def test_caching_mechanism(execution_layer):
    """Test caching mechanism for repeated tasks."""
    test_actions = [
        {
            "type": "research",
            "content": "Cache test query",
            "parameters": {"cache": True}
        }
    ]
    
    context = {
        "query": "test",
        "context": {"domain": "test"}
    }
    
    # First execution
    result1 = execution_layer.execute_task(test_actions, context)
    
    # Second execution (should use cache)
    result2 = execution_layer.execute_task(test_actions, context)
    
    assert result1.status == "success"
    assert result2.status == "success"
    assert result2.metadata["cached"] is True
    assert result2.execution_time < result1.execution_time

def test_progress_tracking(execution_layer):
    """Test progress tracking during execution."""
    test_actions = [
        {
            "type": "research",
            "content": "Track progress",
            "parameters": {"steps": 5}
        }
    ]
    
    context = {
        "query": "test",
        "context": {"domain": "test"}
    }
    
    progress_updates = []
    
    def progress_callback(progress):
        progress_updates.append(progress)
    
    result = execution_layer.execute_task(
        test_actions,
        context,
        progress_callback=progress_callback
    )
    
    assert result.status == "success"
    assert len(progress_updates) > 0
    assert all(0 <= p <= 1 for p in progress_updates)
    assert progress_updates[-1] == 1.0 