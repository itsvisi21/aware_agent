import pytest
import asyncio
import json
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
async def test_authentication_security(websocket_manager, database, cache):
    """Audit authentication security"""
    # Test token validation
    async with websocket_manager.connect(auth_token="invalid_token") as ws:
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert response["code"] == "1002"
    
    # Test token expiration
    async with websocket_manager.connect(auth_token="expired_token") as ws:
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert response["code"] == "1002"
    
    # Test token brute force
    for _ in range(10):
        async with websocket_manager.connect(auth_token="random_token") as ws:
            response = await ws.receive_json()
            assert response["type"] == "error"
            assert response["code"] == "1002"

@pytest.mark.asyncio
async def test_data_security(websocket_manager, database, cache):
    """Audit data security"""
    # Test sensitive data handling
    sensitive_data = {
        "type": "message",
        "content": "password: 123456, credit card: 1234-5678-9012-3456",
        "agent": "research"
    }
    
    # Test message encryption
    async with websocket_manager.connect() as ws:
        await ws.send_json(sensitive_data)
        response = await ws.receive_json()
        assert "123456" not in response["content"]
        assert "1234-5678-9012-3456" not in response["content"]
    
    # Test database encryption
    conversation_id = "test_conversation"
    await database.save_conversation(conversation_id, [sensitive_data])
    conversation = await database.get_conversation(conversation_id)
    assert "123456" not in str(conversation)
    assert "1234-5678-9012-3456" not in str(conversation)
    
    # Test cache encryption
    await cache.set("sensitive_key", sensitive_data)
    cached_data = await cache.get("sensitive_key")
    assert "123456" not in str(cached_data)
    assert "1234-5678-9012-3456" not in str(cached_data)

@pytest.mark.asyncio
async def test_access_control_security(websocket_manager, database, cache):
    """Audit access control security"""
    # Test role-based access
    async with websocket_manager.connect() as ws:
        await ws.send_json({
            "type": "message",
            "content": "Test message",
            "agent": "unauthorized_agent"
        })
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert response["code"] == "1003"
    
    # Test resource access
    with pytest.raises(Exception):
        await database.get_conversation("unauthorized_conversation")
    
    with pytest.raises(Exception):
        await cache.get("unauthorized_key")

@pytest.mark.asyncio
async def test_session_security(websocket_manager, database, cache):
    """Audit session security"""
    # Test session fixation
    valid_session = "valid_session_token"
    async with websocket_manager.connect(auth_token=valid_session) as ws1:
        async with websocket_manager.connect(auth_token=valid_session) as ws2:
            response = await ws2.receive_json()
            assert response["type"] == "error"
            assert response["code"] == "1002"
    
    # Test session timeout
    async with websocket_manager.connect() as ws:
        await asyncio.sleep(3600)  # 1 hour
        await ws.send_json({
            "type": "message",
            "content": "Test message",
            "agent": "research"
        })
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert response["code"] == "1002"

@pytest.mark.asyncio
async def test_input_validation_security(websocket_manager, database, cache):
    """Audit input validation security"""
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
async def test_audit_logging_security(websocket_manager, database, cache):
    """Audit logging security"""
    # Test security event logging
    async with websocket_manager.connect(auth_token="invalid_token") as ws:
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert response["code"] == "1002"
    
    # Verify security logs
    logs = await MonitoringService.get_security_logs()
    assert any("authentication failed" in log.lower() for log in logs)
    
    # Test sensitive operation logging
    await database.save_conversation("test_conversation", [])
    logs = await MonitoringService.get_security_logs()
    assert any("database operation" in log.lower() for log in logs)

@pytest.mark.asyncio
async def test_error_handling_security(websocket_manager, database, cache):
    """Audit error handling security"""
    # Test error message security
    async with websocket_manager.connect() as ws:
        await ws.send_json({"invalid": "message"})
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert "stack trace" not in str(response)
        assert "internal error" not in str(response)
    
    # Test database error security
    await database.disconnect()
    async with websocket_manager.connect() as ws:
        await ws.send_json({
            "type": "message",
            "content": "Test message",
            "agent": "research"
        })
        response = await ws.receive_json()
        assert response["type"] == "error"
        assert "database error" not in str(response)

@pytest.mark.asyncio
async def test_rate_limiting_security(websocket_manager, database, cache):
    """Audit rate limiting security"""
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