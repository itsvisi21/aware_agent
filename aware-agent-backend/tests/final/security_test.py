import pytest
import asyncio
from src.websocket_manager import WebSocketManager
from src.database import DatabaseService
from src.cache import CacheService
from src.monitoring import MonitoringService

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
async def test_authentication(websocket_manager, database, cache):
    """Test authentication mechanisms"""
    # Test invalid token
    async with websocket_manager.connect(auth_token="invalid_token") as ws:
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert response["code"] == "1002"
    
    # Test missing token
    async with websocket_manager.connect(auth_token=None) as ws:
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert response["code"] == "1002"
    
    # Test expired token
    async with websocket_manager.connect(auth_token="expired_token") as ws:
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert response["code"] == "1002"

@pytest.mark.asyncio
async def test_input_validation(websocket_manager, database, cache):
    """Test input validation and sanitization"""
    # Test SQL injection
    async with websocket_manager.connect() as ws:
        await ws.send_json({
            "type": "message",
            "content": "'; DROP TABLE conversations; --",
            "agent": "research"
        })
        response = await ws.receive_json()
        assert "DROP TABLE" not in response["content"]
    
    # Test XSS
    async with websocket_manager.connect() as ws:
        await ws.send_json({
            "type": "message",
            "content": "<script>alert('xss')</script>",
            "agent": "research"
        })
        response = await ws.receive_json()
        assert "<script>" not in response["content"]
    
    # Test command injection
    async with websocket_manager.connect() as ws:
        await ws.send_json({
            "type": "message",
            "content": "ls; rm -rf /",
            "agent": "research"
        })
        response = await ws.receive_json()
        assert "rm -rf" not in response["content"]

@pytest.mark.asyncio
async def test_rate_limiting(websocket_manager, database, cache):
    """Test rate limiting mechanisms"""
    # Test message rate limit
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
    
    # Test connection rate limit
    connections = []
    for _ in range(100):  # Exceed connection limit
        try:
            ws = await websocket_manager.connect()
            connections.append(ws)
        except Exception as e:
            assert "rate limit" in str(e).lower()
            break
    
    # Clean up connections
    for ws in connections:
        await ws.close()

@pytest.mark.asyncio
async def test_data_encryption(websocket_manager, database, cache):
    """Test data encryption and security"""
    # Test sensitive data in messages
    sensitive_data = {
        "type": "message",
        "content": "password: 123456, credit card: 1234-5678-9012-3456",
        "agent": "research"
    }
    
    async with websocket_manager.connect() as ws:
        await ws.send_json(sensitive_data)
        response = await ws.receive_json()
        assert "123456" not in response["content"]
        assert "1234-5678-9012-3456" not in response["content"]
    
    # Test data in database
    conversation_id = "test_conversation"
    await database.save_conversation(conversation_id, [sensitive_data])
    conversation = await database.get_conversation(conversation_id)
    assert "123456" not in str(conversation)
    assert "1234-5678-9012-3456" not in str(conversation)

@pytest.mark.asyncio
async def test_access_control(websocket_manager, database, cache):
    """Test access control mechanisms"""
    # Test unauthorized agent access
    async with websocket_manager.connect() as ws:
        await ws.send_json({
            "type": "message",
            "content": "Test message",
            "agent": "unauthorized_agent"
        })
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert response["code"] == "1003"
    
    # Test unauthorized database access
    with pytest.raises(Exception):
        await database.get_conversation("unauthorized_conversation")
    
    # Test unauthorized cache access
    with pytest.raises(Exception):
        await cache.get("unauthorized_key")

@pytest.mark.asyncio
async def test_session_management(websocket_manager, database, cache):
    """Test session management and security"""
    # Test session timeout
    async with websocket_manager.connect() as ws:
        # Wait for session timeout
        await asyncio.sleep(3600)  # 1 hour
        await ws.send_json({
            "type": "message",
            "content": "Test message",
            "agent": "research"
        })
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert response["code"] == "1002"
    
    # Test session hijacking
    valid_session = "valid_session_token"
    async with websocket_manager.connect(auth_token=valid_session) as ws1:
        # Try to use same session token
        async with websocket_manager.connect(auth_token=valid_session) as ws2:
            response = await ws2.receive_json()
            assert response["type"] == "error"
            assert response["code"] == "1002"

@pytest.mark.asyncio
async def test_audit_logging(websocket_manager, database, cache):
    """Test audit logging and monitoring"""
    # Test security event logging
    async with websocket_manager.connect(auth_token="invalid_token") as ws:
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert response["code"] == "1002"
    
    # Verify security event was logged
    logs = await MonitoringService.get_security_logs()
    assert any("authentication failed" in log.lower() for log in logs)
    
    # Test sensitive operation logging
    await database.save_conversation("test_conversation", [])
    logs = await MonitoringService.get_security_logs()
    assert any("database operation" in log.lower() for log in logs) 