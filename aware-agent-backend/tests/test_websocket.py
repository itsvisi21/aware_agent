import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import WebSocket
from src.core.services.websocket_manager import WebSocketManager
from src.core.services.agent_orchestration import AgentOrchestrator

@pytest.fixture
def websocket_manager():
    """Create a WebSocket manager instance."""
    return WebSocketManager()

@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    websocket = AsyncMock(spec=WebSocket)
    websocket.client = Mock(host="test_host", port=1234)
    return websocket

@pytest.mark.asyncio
async def test_websocket_connection(websocket_manager, mock_websocket):
    """Test WebSocket connection management."""
    # Test connection
    await websocket_manager.connect(mock_websocket)
    assert mock_websocket in websocket_manager.active_connections
    
    # Test disconnection
    await websocket_manager.disconnect(mock_websocket)
    assert mock_websocket not in websocket_manager.active_connections

@pytest.mark.asyncio
async def test_websocket_message_handling(websocket_manager, mock_websocket):
    """Test WebSocket message handling."""
    # Mock agent orchestrator
    mock_orchestrator = Mock(spec=AgentOrchestrator)
    websocket_manager.agent_orchestrator = mock_orchestrator
    
    # Test message handling
    test_message = {
        "type": "task",
        "content": "test task"
    }
    
    await websocket_manager.handle_message(mock_websocket, test_message)
    mock_orchestrator.process_message.assert_called_once_with(test_message)

@pytest.mark.asyncio
async def test_websocket_broadcast(websocket_manager, mock_websocket):
    """Test WebSocket broadcast functionality."""
    # Add test connection
    await websocket_manager.connect(mock_websocket)
    
    # Test broadcast
    test_message = "test broadcast"
    await websocket_manager.broadcast(test_message)
    mock_websocket.send_text.assert_called_once_with(test_message)

@pytest.mark.asyncio
async def test_websocket_error_handling(websocket_manager, mock_websocket):
    """Test WebSocket error handling."""
    # Mock send_text to raise an exception
    mock_websocket.send_text = AsyncMock(side_effect=Exception("Test error"))
    
    # Test error handling during broadcast
    await websocket_manager.connect(mock_websocket)
    await websocket_manager.broadcast("test message")
    
    # Verify connection was removed after error
    assert mock_websocket not in websocket_manager.active_connections

@pytest.mark.asyncio
async def test_websocket_connection_limits(websocket_manager):
    """Test WebSocket connection limits."""
    # Create multiple mock connections
    mock_connections = [AsyncMock(spec=WebSocket) for _ in range(5)]
    
    # Connect all mock connections
    for ws in mock_connections:
        await websocket_manager.connect(ws)
    
    # Verify connection count
    assert len(websocket_manager.active_connections) == 5
    
    # Test disconnecting all
    for ws in mock_connections:
        await websocket_manager.disconnect(ws)
    
    # Verify all connections removed
    assert len(websocket_manager.active_connections) == 0 