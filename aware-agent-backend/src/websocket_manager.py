from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
import json
import asyncio
import logging
from datetime import datetime
from src.agent import agent
from src.semantic_abstraction import SemanticAbstractionLayer
from src.memory_engine import MemoryEngine
from src.agent_orchestration import AgentOrchestrator
from .agents.research_agent import ResearchAgent
from .agents.builder_agent import BuilderAgent
from .agents.teacher_agent import TeacherAgent
from .agents.collaborator_agent import CollaboratorAgent

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.agents: Dict[str, Any] = {
            'research': ResearchAgent(),
            'build': BuilderAgent(),
            'teach': TeacherAgent(),
            'collab': CollaboratorAgent(),
            # Add other agents here as they are implemented
        }
        self.active_connections: Dict[int, Dict[str, Any]] = {}
        self.semantic_layer = SemanticAbstractionLayer()
        self.memory_engine = MemoryEngine()
        self.agent_orchestrator = AgentOrchestrator()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        connection_id = id(websocket)
        self.active_connections[connection_id] = {
            'websocket': websocket,
            'current_agent': None
        }
        logger.info(f"New WebSocket connection established. Total connections: {len(self.active_connections)}")
        await self.send_status(websocket, "Connected to server")

    def disconnect(self, websocket: WebSocket):
        connection_id = id(websocket)
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        logger.info(f"WebSocket connection closed. Remaining connections: {len(self.active_connections)}")

    async def send_status(self, websocket: WebSocket, status: str):
        await websocket.send_json({
            'type': 'status',
            'payload': status
        })

    async def send_error(self, websocket: WebSocket, error: str):
        await websocket.send_json({
            'type': 'error',
            'payload': error
        })

    async def send_message(self, websocket: WebSocket, message: Dict[str, Any]):
        await websocket.send_json({
            'type': 'message',
            'payload': message
        })

    async def handle_connection(self, websocket: WebSocket):
        await self.connect(websocket)
        try:
            while True:
                message = await websocket.receive_text()
                await self._handle_message(id(websocket), message)
        except WebSocketDisconnect:
            self.disconnect(websocket)
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {str(e)}")
            self.disconnect(websocket)

    async def _handle_message(self, connection_id: int, message: str):
        try:
            data = json.loads(message)
            if data.get('type') != 'message':
                logger.warning(f"Received non-message type: {data.get('type')}")
                return

            message_data = data['payload']
            mode = message_data.get('metadata', {}).get('mode', 'research')
            
            # Get appropriate agent
            agent = self.agents.get(mode)
            if not agent:
                raise ValueError(f"Unsupported interaction mode: {mode}")

            # Process message with agent
            response = await agent.process_message(message_data)

            # Send response back to client
            await self.send_message(
                self.active_connections[connection_id]['websocket'],
                response
            )

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await self.send_error(
                self.active_connections[connection_id]['websocket'],
                str(e)
            )

    async def _cleanup_connection(self, connection_id: int):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]

# Create a singleton instance
websocket_manager = WebSocketManager() 