import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from src.main import app
from src.websocket_manager import WebSocketManager
from src.database import DatabaseService
from src.cache import CacheService
from src.monitoring import MonitoringService
from src.agents import ResearchAgent, BuilderAgent, TeacherAgent, CollaboratorAgent

# Initialize test client
client = TestClient(app)

@pytest.fixture
async def websocket_manager():
    """WebSocket manager fixture"""
    manager = WebSocketManager()
    yield manager
    await manager.disconnect_all()

@pytest.fixture
async def database():
    """Database service fixture"""
    db = DatabaseService()
    await db.connect()
    yield db
    await db.disconnect()

@pytest.fixture
async def cache():
    """Cache service fixture"""
    cache = CacheService()
    await cache.connect()
    yield cache
    await cache.disconnect()

@pytest.fixture
async def monitoring():
    """Monitoring service fixture"""
    monitor = MonitoringService()
    await monitor.start()
    yield monitor
    await monitor.stop()

@pytest.fixture
async def agents():
    """Agent fixtures"""
    research = ResearchAgent()
    builder = BuilderAgent()
    teacher = TeacherAgent()
    collaborator = CollaboratorAgent()
    yield {
        "research": research,
        "builder": builder,
        "teacher": teacher,
        "collaborator": collaborator
    }

@pytest.mark.asyncio
async def test_websocket_communication(websocket_manager):
    """Test WebSocket communication flow"""
    # Connect to WebSocket
    async with client.websocket_connect("/ws") as websocket:
        # Send message
        await websocket.send_json({
            "type": "message",
            "agent": "research",
            "content": "Test message"
        })
        
        # Receive response
        response = await websocket.receive_json()
        assert response["type"] == "response"
        assert "content" in response

@pytest.mark.asyncio
async def test_database_integration(database):
    """Test database integration"""
    # Test data
    test_data = {
        "id": "test_id",
        "type": "test",
        "content": "Test content"
    }
    
    # Save data
    await database.save("test_collection", test_data)
    
    # Retrieve data
    result = await database.get("test_collection", "test_id")
    assert result == test_data

@pytest.mark.asyncio
async def test_cache_integration(cache):
    """Test cache integration"""
    # Test data
    test_data = {
        "key": "test_key",
        "value": "test_value"
    }
    
    # Set cache
    await cache.set(test_data["key"], test_data["value"])
    
    # Get cache
    result = await cache.get(test_data["key"])
    assert result == test_data["value"]

@pytest.mark.asyncio
async def test_monitoring_integration(monitoring):
    """Test monitoring integration"""
    # Record metrics
    await monitoring.record_metric("test_metric", 100)
    
    # Get metrics
    metrics = await monitoring.get_metrics()
    assert "test_metric" in metrics

@pytest.mark.asyncio
async def test_agent_integration(agents):
    """Test agent integration"""
    # Test each agent
    for agent_type, agent in agents.items():
        response = await agent.process("Test message")
        assert response is not None
        assert isinstance(response, dict)

@pytest.mark.asyncio
async def test_system_health():
    """Test system health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["cache"] == "connected"

@pytest.mark.asyncio
async def test_system_status():
    """Test system status endpoint"""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "connections" in data
    assert "performance" in data

@pytest.mark.asyncio
async def test_end_to_end_flow(websocket_manager, database, cache, monitoring, agents):
    """Test end-to-end system flow"""
    # Connect to WebSocket
    async with client.websocket_connect("/ws") as websocket:
        # Send research request
        await websocket.send_json({
            "type": "message",
            "agent": "research",
            "content": "Research test query"
        })
        
        # Receive response
        response = await websocket.send_json()
        assert response["type"] == "response"
        
        # Verify database update
        result = await database.get("conversations", response["conversation_id"])
        assert result is not None
        
        # Verify cache update
        cached = await cache.get(f"conversation:{response['conversation_id']}")
        assert cached is not None
        
        # Verify metrics
        metrics = await monitoring.get_metrics()
        assert "message_count" in metrics
        assert metrics["message_count"] > 0

@pytest.mark.asyncio
async def test_error_handling(websocket_manager):
    """Test system error handling"""
    # Connect to WebSocket
    async with client.websocket_connect("/ws") as websocket:
        # Send invalid message
        await websocket.send_json({
            "type": "invalid_type",
            "content": "Test message"
        })
        
        # Receive error response
        response = await websocket.receive_json()
        assert response["type"] == "error"
        assert "code" in response
        assert "message" in response

@pytest.mark.asyncio
async def test_concurrent_operations(websocket_manager, database, cache):
    """Test concurrent system operations"""
    async def send_message():
        async with client.websocket_connect("/ws") as websocket:
            await websocket.send_json({
                "type": "message",
                "agent": "research",
                "content": "Concurrent test"
            })
            return await websocket.receive_json()
    
    # Run multiple concurrent operations
    tasks = [send_message() for _ in range(10)]
    results = await asyncio.gather(*tasks)
    
    # Verify all operations completed
    assert len(results) == 10
    for result in results:
        assert result["type"] == "response"
        assert "content" in result 