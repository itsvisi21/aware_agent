import pytest
import asyncio
from src.websocket_manager import WebSocketManager
from src.services.database import DatabaseService

@pytest.fixture
def websocket_manager():
    return WebSocketManager()

@pytest.fixture
def db_service():
    return DatabaseService()

@pytest.mark.asyncio
async def test_websocket_connection(websocket_manager, db_service):
    # Test WebSocket connection establishment
    client_id = 'test_client'
    await websocket_manager.handle_connection(client_id)
    
    assert client_id in websocket_manager.connections
    assert websocket_manager.connections[client_id] is not None

@pytest.mark.asyncio
async def test_message_routing(websocket_manager, db_service):
    # Test message routing to appropriate agent
    client_id = 'test_client'
    await websocket_manager.handle_connection(client_id)
    
    # Test research message
    research_msg = {
        'type': 'message',
        'content': 'Research authentication systems',
        'sender': 'user',
        'agent': 'research'
    }
    response = await websocket_manager.handle_message(client_id, research_msg)
    assert response['agent'] == 'research'
    assert 'authentication' in response['content'].lower()
    
    # Test builder message
    builder_msg = {
        'type': 'message',
        'content': 'Implement authentication system',
        'sender': 'user',
        'agent': 'builder'
    }
    response = await websocket_manager.handle_message(client_id, builder_msg)
    assert response['agent'] == 'builder'
    assert 'implementation' in response['content'].lower()

@pytest.mark.asyncio
async def test_context_persistence_websocket(websocket_manager, db_service):
    # Test context persistence through WebSocket
    client_id = 'test_client'
    await websocket_manager.handle_connection(client_id)
    
    # Initial research message
    research_msg = {
        'type': 'message',
        'content': 'Research OAuth implementation',
        'sender': 'user',
        'agent': 'research'
    }
    research_response = await websocket_manager.handle_message(client_id, research_msg)
    
    # Follow-up builder message with context
    builder_msg = {
        'type': 'message',
        'content': 'Implement OAuth system',
        'sender': 'user',
        'agent': 'builder',
        'context': research_response
    }
    builder_response = await websocket_manager.handle_message(client_id, builder_msg)
    
    assert 'OAuth' in builder_response['content'].lower()
    assert 'research' in builder_response['content'].lower()

@pytest.mark.asyncio
async def test_concurrent_messages(websocket_manager, db_service):
    # Test handling of concurrent messages
    client_id = 'test_client'
    await websocket_manager.handle_connection(client_id)
    
    messages = [
        {
            'type': 'message',
            'content': 'Research topic 1',
            'sender': 'user',
            'agent': 'research'
        },
        {
            'type': 'message',
            'content': 'Research topic 2',
            'sender': 'user',
            'agent': 'research'
        }
    ]
    
    # Send messages concurrently
    tasks = [
        websocket_manager.handle_message(client_id, msg)
        for msg in messages
    ]
    responses = await asyncio.gather(*tasks)
    
    assert len(responses) == 2
    assert all(response['agent'] == 'research' for response in responses)

@pytest.mark.asyncio
async def test_connection_cleanup(websocket_manager, db_service):
    # Test proper cleanup of connections
    client_id = 'test_client'
    await websocket_manager.handle_connection(client_id)
    
    assert client_id in websocket_manager.connections
    
    await websocket_manager.handle_disconnection(client_id)
    assert client_id not in websocket_manager.connections 