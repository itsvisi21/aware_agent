from fastapi.testclient import TestClient
from src.main import app
import json
import pytest

client = TestClient(app)

@pytest.mark.asyncio
async def test_websocket_connection():
    with client.websocket_connect("/ws") as websocket:
        # Send a test message
        test_message = {"text": "Hello, server!"}
        websocket.send_text(json.dumps(test_message))
        
        # Receive response
        response = websocket.receive_text()
        response_data = json.loads(response)
        
        # Verify response structure
        assert "text" in response_data
        assert "semanticContext" in response_data
        assert isinstance(response_data["text"], str)
        assert isinstance(response_data["semanticContext"], dict)

@pytest.mark.asyncio
async def test_websocket_error_handling():
    with client.websocket_connect("/ws") as websocket:
        # Send invalid message
        websocket.send_text("invalid json")
        
        # Receive error response
        response = websocket.receive_text()
        response_data = json.loads(response)
        
        # Verify error response
        assert "error" in response_data
        assert isinstance(response_data["error"], str)

@pytest.mark.asyncio
async def test_websocket_disconnect():
    with client.websocket_connect("/ws") as websocket:
        # Test normal disconnect
        websocket.close()
        assert websocket.client_state.disconnected 