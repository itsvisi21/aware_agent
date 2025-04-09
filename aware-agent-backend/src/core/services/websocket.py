import json
from typing import Dict, Set, Any, Union
from abc import ABC, abstractmethod
from fastapi import WebSocket, WebSocketDisconnect
from langchain_openai import OpenAI
from pathlib import Path
from .interaction_engine import InteractionEngine
from .memory_engine import MemoryEngine
from src.config.settings import settings


class WebSocketInterface(ABC):
    """Interface for WebSocket communication."""
    
    @abstractmethod
    async def connect(self, websocket: WebSocket) -> None:
        """Handle new WebSocket connection."""
        pass
    
    @abstractmethod
    async def disconnect(self, websocket: WebSocket) -> None:
        """Handle WebSocket disconnection."""
        pass
    
    @abstractmethod
    async def send_message(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Send a message through the WebSocket."""
        pass
    
    @abstractmethod
    async def handle_message(self, websocket: WebSocket, message: str) -> None:
        """Handle incoming WebSocket message."""
        pass


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.llm = OpenAI(
            model_name=settings.model_name,
            temperature=settings.temperature,
            api_key=settings.openai_api_key
        )
        storage_path = settings.memory_storage_path
        storage_path.mkdir(exist_ok=True)
        self.memory_engine = MemoryEngine(storage_path=storage_path)
        self.interaction_engine = InteractionEngine(self.llm, self.memory_engine)

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
        if client_id in self.active_connections:
            for connection in self.active_connections[client_id]:
                await connection.send_json({
                    "type": "agent_status",
                    "data": status
                })

    async def broadcast_conversation_update(self, client_id: str, conversation_state: Dict):
        if client_id in self.active_connections:
            for connection in self.active_connections[client_id]:
                await connection.send_json({
                    "type": "conversation_update",
                    "data": conversation_state
                })


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
