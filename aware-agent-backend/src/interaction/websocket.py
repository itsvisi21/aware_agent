from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set, Any
import json
import asyncio
from .interaction_engine import InteractionEngine

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.interaction_engine = InteractionEngine()

    async def connect(self, websocket: WebSocket, conversation_id: str):
        await websocket.accept()
        if conversation_id not in self.active_connections:
            self.active_connections[conversation_id] = set()
        self.active_connections[conversation_id].add(websocket)

    def disconnect(self, websocket: WebSocket, conversation_id: str):
        self.active_connections[conversation_id].remove(websocket)
        if not self.active_connections[conversation_id]:
            del self.active_connections[conversation_id]

    async def broadcast_agent_status(self, conversation_id: str, agent_type: str, status: Dict[str, Any]):
        if conversation_id in self.active_connections:
            message = {
                "type": "agent_status",
                "agent": agent_type,
                "status": status
            }
            for connection in self.active_connections[conversation_id]:
                try:
                    await connection.send_json(message)
                except WebSocketDisconnect:
                    await self.disconnect(connection, conversation_id)

    async def broadcast_conversation_update(self, conversation_id: str, update: Dict[str, Any]):
        if conversation_id in self.active_connections:
            message = {
                "type": "conversation_update",
                "data": update
            }
            for connection in self.active_connections[conversation_id]:
                try:
                    await connection.send_json(message)
                except WebSocketDisconnect:
                    await self.disconnect(connection, conversation_id)

manager = ConnectionManager()

async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    await manager.connect(websocket, conversation_id)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message["type"] == "agent_status_request":
                    # Get current agent statuses
                    statuses = await manager.interaction_engine.get_agent_statuses(conversation_id)
                    await websocket.send_json({
                        "type": "agent_status_response",
                        "data": statuses
                    })
                elif message["type"] == "conversation_state_request":
                    # Get current conversation state
                    state = await manager.interaction_engine.get_conversation_state(conversation_id)
                    await websocket.send_json({
                        "type": "conversation_state_response",
                        "data": state
                    })
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid message format"
                })
    except WebSocketDisconnect:
        manager.disconnect(websocket, conversation_id) 