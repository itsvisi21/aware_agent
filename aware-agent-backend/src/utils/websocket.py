from fastapi import WebSocket
from typing import Dict, Set, Optional
import json
from datetime import datetime

class WebSocketService:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.agent_statuses: Dict[str, Dict] = {}
        self.conversation_states: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = set()
        self.active_connections[client_id].add(websocket)

    async def disconnect(self, websocket: WebSocket, client_id: str):
        if client_id in self.active_connections:
            self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]

    async def broadcast_agent_status(self, client_id: str, status: Dict):
        self.agent_statuses[client_id] = status
        if client_id in self.active_connections:
            for connection in self.active_connections[client_id]:
                await connection.send_json({
                    "type": "agent_status",
                    "data": status
                })

    async def broadcast_conversation_update(self, client_id: str, conversation_state: Dict):
        self.conversation_states[client_id] = conversation_state
        if client_id in self.active_connections:
            for connection in self.active_connections[client_id]:
                await connection.send_json({
                    "type": "conversation_update",
                    "data": conversation_state
                })

    async def send_message(self, websocket: WebSocket, message: str):
        await websocket.send_text(message)

    async def receive_message(self, websocket: WebSocket) -> str:
        return await websocket.receive_text()

    def get_agent_status(self, client_id: str) -> Optional[Dict]:
        return self.agent_statuses.get(client_id)

    def get_conversation_state(self, client_id: str) -> Optional[Dict]:
        return self.conversation_states.get(client_id)

# Create a single instance of the WebSocketService
websocket_service = WebSocketService() 