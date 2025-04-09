import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.core.services.abstraction_engines import AbstractionType, metaphysical_engine, cognitive_engine
from src.core.services.agent_conversation import agent_conversation
from src.core.services.agent_orchestration import AgentOrchestrator
from src.core.services.collaboration import collaboration_manager
from src.config.settings import Settings
from src.core.services.context_chaining import context_chainer
from src.core.services.domain_mappings import domain_mapper
from src.core.services.execution_layer import ExecutionLayer
from src.core.services.interaction_engine import InteractionEngine
from src.core.services.karaka_mapper import KarakaRole
from src.core.services.memory_engine import MemoryEngine
from src.utils.metrics import metrics_collector, PipelineMetrics
from src.core.services.performance import performance_monitor
from src.core.services.semantic_abstraction import SemanticAbstractionLayer
from src.core.services.temporal_reasoning import temporal_reasoner
from src.core.services.visualization import semantic_visualizer
from src.core.services.websocket_manager import websocket_manager
from src.core.services.workspace_integration import WorkspaceType, workspace_exporter
from src.core.services.research_agent import ResearchAgent
from src.core.services.builder_agent import BuilderAgent
from src.core.services.teacher_agent import TeacherAgent
from src.core.services.collaborator_agent import CollaboratorAgent
from src.utils.cache import CacheService
from src.data.database import DatabaseService
from src.core.models.models import (
    ResearchRequest,
    SessionRequest,
    ParticipantRequest,
    ContextUpdate,
    ExportRequest,
    ConversationRequest,
    MessageRequest,
    TaskStatusResponse
)
from src.utils.monitoring import MonitoringService
from src.core.services.websocket_manager import WebSocketManager
from src.app import create_app
from langchain_openai import OpenAI
import uvicorn
from src.utils.logging import setup_logging
from src.common.exceptions import AgentError, ValidationError

# Initialize settings
settings = Settings()

# Setup logging
setup_logging(settings.log_level, settings.log_file)

# Initialize services
database_service = DatabaseService()
cache_service = CacheService()
monitoring_service = MonitoringService(settings)

# Initialize engines
memory_engine = MemoryEngine(settings.memory_storage_path)
interaction_engine = InteractionEngine(
    llm=OpenAI(
        model_name=settings.model_name,
        temperature=settings.temperature,
        openai_api_key=settings.openai_api_key
    ),
    memory_engine=memory_engine
)
semantic_layer = SemanticAbstractionLayer()

# Initialize agents
research_agent = ResearchAgent(settings)
agent_orchestrator = AgentOrchestrator()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Aware Agent Backend API"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
agent_layer: Optional[AgentOrchestrator] = None
execution_layer: Optional[ExecutionLayer] = None

# Task storage - stores task state and results
tasks: Dict[str, Dict[str, Any]] = {}

# Initialize services
llm = OpenAI(
    model_name=settings.model_name,
    temperature=settings.temperature,
    api_key=settings.openai_api_key
)

memory_engine = MemoryEngine(storage_path=settings.memory_storage_path)
interaction_engine = InteractionEngine(llm=llm, memory_engine=memory_engine)
agent_orchestrator = AgentOrchestrator(llm=llm)
research_agent = ResearchAgent(llm=llm)
teacher_agent = TeacherAgent(llm=llm)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources."""
    # Initialize components
    global semantic_layer, agent_layer, interaction_engine, memory_engine, execution_layer, research_agent, builder_agent, teacher_agent, collaborator_agent

    try:
        # Initialize core services first
        await database_service.initialize()
        await cache_service.connect()
        await monitoring_service.start()

        # Initialize engines after services
        semantic_layer = SemanticAbstractionLayer()
        agent_layer = AgentOrchestrator()
        interaction_engine = InteractionEngine()
        memory_engine = MemoryEngine(storage_path=settings.memory_storage_path)
        execution_layer = ExecutionLayer()

        # Initialize agents
        research_agent = ResearchAgent(settings)
        builder_agent = BuilderAgent()
        teacher_agent = TeacherAgent()
        collaborator_agent = CollaboratorAgent()

        # Register agents with the orchestrator
        agent_layer.register_agent(research_agent)
        agent_layer.register_agent(builder_agent)
        agent_layer.register_agent(teacher_agent)
        agent_layer.register_agent(collaborator_agent)

        logger.info("Services initialized successfully")

        yield
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise
    finally:
        # Cleanup
        if execution_layer:
            await execution_layer.close()
        await database_service.close()
        await cache_service.disconnect()
        await monitoring_service.stop()
        logger.info("Services cleaned up successfully")

# Initialize the FastAPI app
settings = Settings()
semantic_layer = SemanticAbstractionLayer()
app = create_app(settings=settings, semantic_layer=semantic_layer)

if __name__ == "__main__":
    # Add debug logging
    logger.info("Starting server with configuration:")
    logger.info(f"Host: 0.0.0.0")
    logger.info(f"Port: 8000")
    logger.info(f"App: src.main:app")
    
    config = uvicorn.Config(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",  # Changed to debug for more verbose logging
        access_log=True,
        proxy_headers=True,
        server_header=True,
        date_header=True
    )
    server = uvicorn.Server(config)
    server.run()
