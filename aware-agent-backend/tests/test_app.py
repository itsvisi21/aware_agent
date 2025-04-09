import pytest
from fastapi.testclient import TestClient
from src.app import create_app
from src.config.settings import Settings
from src.core.services.semantic_abstraction import SemanticAbstractionLayer
from src.core.models.models import ResearchRequest, TaskStatus

@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        environment="test",
        debug=True,
        development_mode=True
    )

@pytest.fixture
def semantic_layer():
    """Create a semantic layer instance."""
    return SemanticAbstractionLayer()

@pytest.fixture
def client(settings, semantic_layer):
    """Create a test client."""
    app = create_app(settings, semantic_layer)
    return TestClient(app)

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "message": "Server is running"
    }

def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Aware Agent API is running"
    }

def test_research_endpoint(client):
    """Test research endpoint."""
    test_request = {
        "query": "test query",
        "context": {"test": "context"},
        "domain": "test_domain",
        "chain_context": True
    }
    
    response = client.post("/api/v1/research", json=test_request)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert "status" in data
    assert data["status"] == TaskStatus.PENDING

def test_task_status_endpoint(client):
    """Test task status endpoint."""
    # First create a task
    test_request = {
        "query": "test query",
        "context": {"test": "context"}
    }
    create_response = client.post("/api/v1/research", json=test_request)
    task_id = create_response.json()["task_id"]
    
    # Then check its status
    response = client.get(f"/api/v1/tasks/{task_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in [status.value for status in TaskStatus]

def test_invalid_request_handling(client):
    """Test handling of invalid requests."""
    # Test invalid JSON
    response = client.post("/api/v1/research", data="invalid json")
    assert response.status_code == 422
    
    # Test missing required fields
    invalid_request = {
        "query": "test query"
        # Missing required 'context' field
    }
    response = client.post("/api/v1/research", json=invalid_request)
    assert response.status_code == 422

def test_error_handling(client):
    """Test error handling."""
    # Test non-existent endpoint
    response = client.get("/non-existent")
    assert response.status_code == 404
    
    # Test non-existent task status
    response = client.get("/api/v1/tasks/non-existent/status")
    assert response.status_code == 404

def test_cors_headers(client):
    """Test CORS headers."""
    response = client.get("/health", headers={"Origin": "http://test.com"})
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "*"

def test_rate_limiting(client):
    """Test rate limiting."""
    # Make multiple requests in quick succession
    for _ in range(5):
        response = client.get("/health")
        assert response.status_code == 200
    
    # The next request should be rate limited
    response = client.get("/health")
    assert response.status_code == 429 