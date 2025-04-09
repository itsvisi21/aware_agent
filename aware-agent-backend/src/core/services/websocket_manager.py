import json
import logging
from typing import Dict, Any, Set, Optional
import asyncio
import os
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from openai import OpenAI as OpenAIClient
from langchain_openai import ChatOpenAI, OpenAI
from pathlib import Path

from src.core.services.agent_orchestration import AgentOrchestrator
from src.config.settings import Settings
from src.core.services.memory_engine import MemoryEngine
from src.core.services.semantic_abstraction import SemanticAbstractionLayer
from src.common.agent_types import AgentType, AgentResponse
from src.core.models.models import TaskStatus, Task
from .builder_agent import BuilderAgent
from .collaborator_agent import CollaboratorAgent
from .research_agent import ResearchAgent
from .teacher_agent import TeacherAgent
from src.core.services.execution_layer import ExecutionLayer
from src.core.services.interaction_engine import InteractionEngine
from src.utils.websocket import websocket_service

logger = logging.getLogger(__name__)

settings = Settings()

# Ensure no proxy settings are interfering
if 'http_proxy' in os.environ:
    del os.environ['http_proxy']
if 'https_proxy' in os.environ:
    del os.environ['https_proxy']

class WebSocketManager:
    """Manages WebSocket connections and message handling."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.conversation_ids: Dict[str, str] = {}  # Map client_id to conversation_id
        self.settings = settings
        
        # Initialize LLM
        self.llm = OpenAI(
            model_name=self.settings.model_name,
            temperature=self.settings.temperature,
            api_key=self.settings.openai_api_key
        )
        
        # Initialize components
        self.memory_engine = MemoryEngine(self.settings.memory_storage_path)
        self.interaction_engine = InteractionEngine(llm=self.llm, memory_engine=self.memory_engine)
        self.agent_orchestrator = AgentOrchestrator(llm=self.llm)
        
        # Initialize agents
        self.initialized = False

    async def initialize(self):
        """Initialize the WebSocket manager and its components."""
        if not self.initialized:
            # Initialize memory engine
            await self.memory_engine.initialize()
            
            # Create and register agents
            research_agent = ResearchAgent(llm=self.llm)
            teacher_agent = TeacherAgent(llm=self.llm)
            builder_agent = BuilderAgent(llm=self.llm)
            collaborator_agent = CollaboratorAgent(llm=self.llm)
            
            # Register agents with orchestrator
            self.agent_orchestrator.register_agent(research_agent, AgentType.RESEARCH)
            self.agent_orchestrator.register_agent(teacher_agent, AgentType.TEACHER)
            self.agent_orchestrator.register_agent(builder_agent, AgentType.BUILDER)
            self.agent_orchestrator.register_agent(collaborator_agent, AgentType.COLLABORATOR)
            
            # Initialize agent orchestrator
            await self.agent_orchestrator.initialize_agents()
            
            self.initialized = True

    async def connect(self, websocket: WebSocket):
        """Handle new WebSocket connection."""
        await websocket.accept()
        client_id = str(id(websocket))
        self.active_connections[client_id] = websocket
        
        # Create a new conversation for this client
        conversation_id = await self.interaction_engine.create_conversation({
            "client_id": client_id,
            "created_at": datetime.utcnow().isoformat()
        })
        self.conversation_ids[client_id] = conversation_id
        
        return client_id

    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection."""
        client_id = None
        for cid, ws in self.active_connections.items():
            if ws == websocket:
                client_id = cid
                break
        if client_id:
            del self.active_connections[client_id]
            if client_id in self.conversation_ids:
                del self.conversation_ids[client_id]

    async def handle_message(self, websocket: WebSocket, message: str):
        """Handle incoming WebSocket message."""
        if not self.initialized:
            await self.initialize()
            
        try:
            # Find the client's conversation ID
            client_id = None
            for cid, ws in self.active_connections.items():
                if ws == websocket:
                    client_id = cid
                    break
            
            if not client_id or client_id not in self.conversation_ids:
                await websocket.send_text("Error: No active conversation found")
                return
                
            conversation_id = self.conversation_ids[client_id]
            
            # Process the message with the interaction engine
            response = await self.interaction_engine.process_message(conversation_id, message)
            
            # Send response back to client
            await websocket.send_text(response.content.get("text", "No response generated"))
            
        except Exception as e:
            await websocket.send_text(f"Error: {str(e)}")

    async def broadcast(self, message: str) -> None:
        """Broadcast a message to all connected clients.
        
        Args:
            message: The message to broadcast
            
        Raises:
            Exception: If sending to any client fails
        """
        disconnected_clients = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                print(f"Error broadcasting to client {client_id}: {str(e)}")
                disconnected_clients.append(client_id)
                
        # Remove disconnected clients
        for client_id in disconnected_clients:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
            if client_id in self.conversation_ids:
                del self.conversation_ids[client_id]

# Create a single instance of the WebSocket manager
websocket_manager = WebSocketManager()
