from typing import Dict, Any, Optional, Type
import uuid
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from langchain_openai import OpenAI
from pydantic import BaseModel

from src.core.services.agent_orchestration import AgentOrchestrator, AgentType
from src.config.settings import Settings
from src.core.services.websocket_manager import websocket_manager
from src.core.services.memory_engine import MemoryEngine
from src.core.services.routes import router
from src.core.models.models import tasks
from src.core.services.semantic_layer import SemanticLayer
from src.core.models.requests import ResearchRequest
from src.core.services.semantic_abstraction import SemanticAbstractionLayer
import logging


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Global task storage
tasks: Dict[str, Dict[str, Any]] = {}

logger = logging.getLogger(__name__)

def create_app(settings: Settings, semantic_layer: SemanticAbstractionLayer) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI()
    
    # Store semantic layer in app state
    app.state.semantic_layer = semantic_layer
    
    # Include the router
    app.include_router(router)
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        logger.info("Health check endpoint called")
        return {"status": "healthy", "message": "Server is running"}
    
    # Add root endpoint
    @app.get("/")
    async def root():
        logger.info("Root endpoint called")
        return {"message": "Aware Agent API is running"}
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info(f"Request: {request.method} {request.url}")
        response = await call_next(request)
        logger.info(f"Response: {response.status_code}")
        return response
    
    # WebSocket endpoint
    @app.websocket("/ws/{conversation_id}")
    async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
        logger.info(f"New WebSocket connection for conversation: {conversation_id}")
        await websocket_manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                logger.info(f"Received WebSocket message: {data}")
                await websocket_manager.handle_message(websocket, data)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await websocket_manager.disconnect(websocket)
            logger.info("WebSocket connection closed")
    
    return app
