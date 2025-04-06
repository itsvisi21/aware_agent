"""
Tests for the main FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app
from tests.config import TEST_DATA
import asyncio

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)

def test_create_research_task(client):
    """Test creating a new research task."""
    test_query = TEST_DATA["research_queries"][0]
    
    response = client.post(
        "/research",
        json={
            "query": test_query["query"],
            "context": test_query["context"]
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "created"
    assert "timestamp" in data

def test_get_task_status(client):
    """Test retrieving task status."""
    # First create a task
    test_query = TEST_DATA["research_queries"][1]
    create_response = client.post(
        "/research",
        json={
            "query": test_query["query"],
            "context": test_query["context"]
        }
    )
    task_id = create_response.json()["task_id"]
    
    # Then get its status
    status_response = client.get(f"/research/{task_id}")
    
    assert status_response.status_code == 200
    data = status_response.json()
    assert data["task_id"] == task_id
    assert "status" in data
    assert "progress" in data
    assert "timestamp" in data

def test_get_task_actions(client):
    """Test retrieving task actions."""
    # First create a task
    test_query = TEST_DATA["research_queries"][0]
    create_response = client.post(
        "/research",
        json={
            "query": test_query["query"],
            "context": test_query["context"]
        }
    )
    task_id = create_response.json()["task_id"]
    
    # Then get its actions
    actions_response = client.get(f"/research/{task_id}/actions")
    
    assert actions_response.status_code == 200
    data = actions_response.json()
    assert data["task_id"] == task_id
    assert "actions" in data

def test_get_task_context(client):
    """Test retrieving task context."""
    # First create a task
    test_query = TEST_DATA["research_queries"][1]
    create_response = client.post(
        "/research",
        json={
            "query": test_query["query"],
            "context": test_query["context"]
        }
    )
    task_id = create_response.json()["task_id"]
    
    # Then get its context
    context_response = client.get(f"/research/{task_id}/context")
    
    assert context_response.status_code == 200
    data = context_response.json()
    assert data["task_id"] == task_id
    assert "context" in data

def test_list_tasks(client):
    """Test listing all tasks."""
    # Create a few tasks
    for query in TEST_DATA["research_queries"]:
        client.post(
            "/research",
            json={
                "query": query["query"],
                "context": query["context"]
            }
        )
    
    # List all tasks
    list_response = client.get("/research")
    
    assert list_response.status_code == 200
    data = list_response.json()
    assert "tasks" in data
    assert len(data["tasks"]) >= len(TEST_DATA["research_queries"])

def test_error_handling(client):
    """Test error handling in the API."""
    # Test invalid task ID
    response = client.get("/research/invalid_task_id")
    assert response.status_code == 404
    
    # Test invalid request body
    response = client.post(
        "/research",
        json={"invalid": "data"}
    )
    assert response.status_code == 422
    
    # Test invalid endpoint
    response = client.get("/invalid_endpoint")
    assert response.status_code == 404

def test_concurrent_requests(client):
    """Test handling of concurrent requests."""
    import threading
    import time
    
    def create_task(query_data):
        response = client.post(
            "/research",
            json={
                "query": query_data["query"],
                "context": query_data["context"]
            }
        )
        return response.json()["task_id"]
    
    # Create multiple threads
    threads = []
    task_ids = []
    
    for query in TEST_DATA["research_queries"]:
        t = threading.Thread(
            target=lambda q: task_ids.append(create_task(q)),
            args=(query,)
        )
        threads.append(t)
    
    # Start all threads
    for t in threads:
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    # Verify all tasks were created
    list_response = client.get("/research")
    created_tasks = {t["task_id"] for t in list_response.json()["tasks"]}
    
    assert all(task_id in created_tasks for task_id in task_ids) 