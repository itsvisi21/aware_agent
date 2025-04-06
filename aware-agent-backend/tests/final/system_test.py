import pytest
import asyncio
from src.websocket_manager import WebSocketManager
from src.database import DatabaseService
from src.cache import CacheService
from src.agents import ResearchAgent, BuilderAgent, TeacherAgent, CollaboratorAgent
from src.monitoring import MonitoringService
from src.health import app
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
async def websocket_manager():
    manager = WebSocketManager()
    await manager.start()
    yield manager
    await manager.stop()

@pytest.fixture
async def database():
    db = DatabaseService()
    await db.connect()
    yield db
    await db.disconnect()

@pytest.fixture
async def cache():
    cache = CacheService()
    await cache.connect()
    yield cache
    await cache.disconnect()

@pytest.mark.asyncio
async def test_system_integration(websocket_manager, database, cache):
    """Test complete system flow"""
    # Initialize agents
    research_agent = ResearchAgent()
    builder_agent = BuilderAgent()
    teacher_agent = TeacherAgent()
    collaborator_agent = CollaboratorAgent()
    
    # Test WebSocket connection
    async with websocket_manager.connect() as ws:
        # Test message processing
        await ws.send_json({
            "type": "message",
            "content": "Research quantum computing",
            "agent": "research"
        })
        response = await ws.receive_json()
        assert response["type"] == "response"
        assert "quantum" in response["content"].lower()
        
        # Test agent collaboration
        await ws.send_json({
            "type": "message",
            "content": "Implement a quantum circuit",
            "agent": "builder"
        })
        response = await ws.receive_json()
        assert response["type"] == "response"
        assert "circuit" in response["content"].lower()
    
    # Verify monitoring metrics
    metrics = await MonitoringService.get_metrics()
    assert metrics["messages_processed"] > 0
    assert metrics["active_connections"] == 0

@pytest.mark.asyncio
async def test_performance_under_load(websocket_manager, database, cache):
    """Test system performance under load"""
    # Simulate multiple concurrent connections
    async def simulate_user():
        async with websocket_manager.connect() as ws:
            for _ in range(10):
                await ws.send_json({
                    "type": "message",
                    "content": "Test message",
                    "agent": "research"
                })
                await ws.receive_json()
    
    # Create multiple concurrent users
    tasks = [simulate_user() for _ in range(10)]
    await asyncio.gather(*tasks)
    
    # Verify performance metrics
    metrics = await MonitoringService.get_metrics()
    assert metrics["average_response_time"] < 1.0  # Response time under 1 second
    assert metrics["active_connections"] == 0

@pytest.mark.asyncio
async def test_error_handling(websocket_manager, database, cache):
    """Test system error handling"""
    # Test invalid message format
    async with websocket_manager.connect() as ws:
        await ws.send_json({"invalid": "message"})
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert response["code"] == "1001"
    
    # Test database error
    await database.disconnect()
    async with websocket_manager.connect() as ws:
        await ws.send_json({
            "type": "message",
            "content": "Test message",
            "agent": "research"
        })
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert response["code"] == "1005"
    
    # Test cache error
    await cache.disconnect()
    async with websocket_manager.connect() as ws:
        await ws.send_json({
            "type": "message",
            "content": "Test message",
            "agent": "research"
        })
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert response["code"] == "1005"

@pytest.mark.asyncio
async def test_security(websocket_manager, database, cache):
    """Test system security"""
    # Test unauthorized access
    async with websocket_manager.connect(auth_token="invalid") as ws:
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert response["code"] == "1002"
    
    # Test rate limiting
    async with websocket_manager.connect() as ws:
        for _ in range(150):  # Exceed rate limit
            await ws.send_json({
                "type": "message",
                "content": "Test message",
                "agent": "research"
            })
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert response["code"] == "1004"
    
    # Test malicious input
    async with websocket_manager.connect() as ws:
        await ws.send_json({
            "type": "message",
            "content": "<script>alert('xss')</script>",
            "agent": "research"
        })
        response = await ws.receive_json()
        assert "<script>" not in response["content"]

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert all(data["components"].values())

def test_metrics_endpoint(client):
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "messages_processed" in data
    assert "active_connections" in data
    assert "average_response_time" in data

def test_status_endpoint(client):
    """Test status endpoint"""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "version" in data
    assert "uptime" in data 