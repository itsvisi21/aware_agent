from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid
import asyncio
from contextlib import asynccontextmanager
from src.semantic_abstraction import SemanticAbstractionLayer
from src.agent_orchestration import AgentOrchestrator
from src.interaction_engine import InteractionEngine
from src.memory_engine import MemoryEngine
from src.execution_layer import ExecutionLayer
from src.metrics import metrics_collector, PipelineMetrics
from src.workspace_integration import WorkspaceType, workspace_exporter
from src.domain_mappings import DomainType, DomainMapper, domain_mapper
from src.collaboration import CollaborationRole, ResearchSession, collaboration_manager
from src.visualization import semantic_visualizer
from src.context_chaining import ContextNode, ContextChain, context_chainer
from src.temporal_reasoning import TemporalEvent, TemporalSequence, temporal_reasoner
from src.karaka_mapper import KarakaRole
from fastapi.responses import HTMLResponse, Response, JSONResponse
import json
from pathlib import Path
from src.abstraction_engines import AbstractionType, metaphysical_engine, cognitive_engine, logical_engine, causal_engine
from src.agent_conversation import AgentRole, agent_conversation
from src.performance import performance_monitor, PerformanceMonitor
from enum import Enum
import logging
import numpy as np
from src.websocket_manager import websocket_manager
from fastapi.middleware.cors import CORSMiddleware
from .websocket_manager import WebSocketManager
from .database import DatabaseService
from .cache import CacheService
from .monitoring import MonitoringService
from .agents import ResearchAgent, BuilderAgent, TeacherAgent, CollaboratorAgent
from .models import (
    ResearchRequest,
    SessionRequest,
    ParticipantRequest,
    ContextUpdate,
    ExportRequest,
    ConversationRequest,
    MessageRequest,
    TaskStatusResponse
)

logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Global components
semantic_layer: Optional[SemanticAbstractionLayer] = None
agent_layer: Optional[AgentOrchestrator] = None
interaction_engine: Optional[InteractionEngine] = None
memory_engine: Optional[MemoryEngine] = None
execution_layer: Optional[ExecutionLayer] = None

# Task storage
tasks: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources."""
    # Initialize components
    global semantic_layer, agent_layer, interaction_engine, memory_engine, execution_layer
    
    try:
        semantic_layer = SemanticAbstractionLayer()
        agent_layer = AgentOrchestrator()
        interaction_engine = InteractionEngine()
        memory_engine = MemoryEngine()
        execution_layer = ExecutionLayer()
        
        # Initialize database tables
        await memory_engine.initialize_database()
        
        # Initialize services
        await database.connect()
        await cache.connect()
        await monitoring.start()
        
        logger.info("Services initialized successfully")
        
        yield
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise
    finally:
        # Cleanup
        if execution_layer:
            await execution_layer.close()
        await database.disconnect()
        await cache.disconnect()
        await monitoring.stop()
        logger.info("Services cleaned up successfully")

# Initialize FastAPI app
app = FastAPI(
    title="Aware Agent API",
    description="A semantic abstraction engine for AI conversations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
websocket_manager = WebSocketManager()
database = DatabaseService()
cache = CacheService()
monitoring = MonitoringService()

# Initialize agents
research_agent = ResearchAgent()
builder_agent = BuilderAgent()
teacher_agent = TeacherAgent()
collaborator_agent = CollaboratorAgent()

app.post("/research", response_model=Dict[str, Any])
async def create_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Create a new research task."""
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "request": request.model_dump(),
        "created_at": datetime.now().isoformat(),
        "metrics": None
    }
    
    # Start metrics collection
    metrics_collector.start_timer("total")
    
    # Process in background
    background_tasks.add_task(process_research_task, task_id, request)
    
    return {"task_id": task_id, "status": TaskStatus.PENDING}

app.post("/research/domain/mapping")
async def create_domain_mapping(
    domain_name: str,
    role_patterns: Dict[str, List[str]],
    entity_types: List[str],
    relationship_types: List[str],
    context_rules: Dict[str, Any],
    priority_rules: Dict[str, Any]
):
    """Create a new custom domain mapping."""
    try:
        # Convert string role names to KarakaRole enum
        karaka_role_patterns = {
            KarakaRole[role]: patterns
            for role, patterns in role_patterns.items()
        }
        
        mapping = domain_mapper.create_custom_mapping(
            domain_name=domain_name,
            role_patterns=karaka_role_patterns,
            entity_types=entity_types,
            relationship_types=relationship_types,
            context_rules=context_rules,
            priority_rules=priority_rules
        )
        
        return {
            "status": "success",
            "domain": domain_name,
            "mapping": {
                "role_patterns": {
                    role.name: patterns
                    for role, patterns in mapping.role_patterns.items()
                },
                "entity_types": mapping.entity_types,
                "relationship_types": mapping.relationship_types,
                "context_rules": mapping.context_rules,
                "priority_rules": mapping.priority_rules
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to create domain mapping: {str(e)}"
        )

app.get("/research/domain/mappings")
async def get_domain_mappings():
    """Get all available domain mappings."""
    mappings = {}
    for domain, mapping in domain_mapper.domain_mappings.items():
        mappings[domain.value] = {
            "role_patterns": {
                role.name: patterns
                for role, patterns in mapping.role_patterns.items()
            },
            "entity_types": mapping.entity_types,
            "relationship_types": mapping.relationship_types,
            "context_rules": mapping.context_rules,
            "priority_rules": mapping.priority_rules
        }
    return mappings

app.post("/research/sessions")
async def create_research_session(
    request: SessionRequest,
    user_id: str = "default_user"  # In production, this would come from auth
):
    """Create a new collaborative research session."""
    session = await collaboration_manager.create_session(
        owner_id=user_id,
        title=request.title,
        description=request.description,
        initial_context=request.initial_context
    )
    
    return {
        "session_id": session.session_id,
        "title": session.title,
        "description": session.description,
        "created_at": session.created_at,
        "participants": session.participants,
        "is_active": session.is_active
    }

app.post("/research/sessions/{session_id}/participants")
async def add_session_participant(
    session_id: str,
    request: ParticipantRequest,
    user_id: str = "default_user"  # In production, this would come from auth
):
    """Add a participant to a research session."""
    success = await collaboration_manager.add_participant(
        session_id=session_id,
        user_id=request.user_id,
        role=request.role
    )
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Failed to add participant to session"
        )
    
    return {"status": "success"}

app.post("/research/sessions/{session_id}/context")
async def update_session_context(
    session_id: str,
    request: ContextUpdate,
    user_id: str = "default_user"  # In production, this would come from auth
):
    """Update shared context for a research session."""
    success = await collaboration_manager.update_shared_context(
        session_id=session_id,
        user_id=user_id,
        context_updates=request.updates
    )
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Failed to update session context"
        )
    
    return {"status": "success"}

app.post("/research/sessions/{session_id}/tasks")
async def add_session_task(
    session_id: str,
    task_id: str,
    user_id: str = "default_user"  # In production, this would come from auth
):
    """Add a research task to a session."""
    success = await collaboration_manager.add_research_task(
        session_id=session_id,
        task_id=task_id,
        user_id=user_id
    )
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Failed to add task to session"
        )
    
    return {"status": "success"}

app.get("/research/sessions/{session_id}")
async def get_session(
    session_id: str,
    user_id: str = "default_user"  # In production, this would come from auth
):
    """Get details of a research session."""
    session = await collaboration_manager.get_session(session_id, user_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found or access denied"
        )
    
    return {
        "session_id": session.session_id,
        "title": session.title,
        "description": session.description,
        "created_at": session.created_at,
        "participants": session.participants,
        "research_tasks": session.research_tasks,
        "shared_context": session.shared_context,
        "is_active": session.is_active
    }

app.post("/research/sessions/{session_id}/end")
async def end_session(
    session_id: str,
    user_id: str = "default_user"  # In production, this would come from auth
):
    """End a research session."""
    success = await collaboration_manager.end_session(session_id, user_id)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Failed to end session"
        )
    
    return {"status": "success"}

@performance_monitor.track_performance("process_research_task")
async def process_research_task(task_id: str, request: ResearchRequest):
    """Process a research task through the pipeline."""
    try:
        # Get domain-specific mapping if specified
        domain_mapping = None
        if request.domain:
            domain_mapping = domain_mapper.get_mapping(request.domain)
        
        # Check if task is part of a collaborative session
        session_context = None
        for session in collaboration_manager.sessions.values():
            if task_id in session.research_tasks:
                session_context = session.shared_context
                break
        
        # Combine request context with session context if available
        combined_context = request.context or {}
        if session_context:
            combined_context.update(session_context)
        
        # Use parallel processing for semantic analysis
        semantic_tasks = [
            semantic_layer.process(
                request.query,
                combined_context,
                domain_mapping=domain_mapping
            )
        ]
        semantic_results = await performance_monitor.parallel_process(semantic_tasks)
        semantic_result = semantic_results[0]
        
        # Use batch processing for context chaining
        if request.chain_context:
            context_nodes = await performance_monitor.batch_process(
                items=[request.query],
                process_func=lambda q: context_chainer.add_context(
                    content=q,
                    embeddings=semantic_result.get("embeddings", []),
                    parent_id=request.parent_context_id,
                    metadata={
                        "task_id": task_id,
                        "domain": request.domain.value if request.domain else None,
                        "semantic_roles": semantic_result.get("roles", {}),
                        "confidence": semantic_result.get("confidence", 1.0)
                    }
                )
            )
            context_node = context_nodes[0]
        
        # 2. Context Chaining
        if request.chain_context:
            metrics_collector.start_timer("context_chaining")
            
            # Find relevant previous contexts
            relevant_contexts = await context_chainer.find_relevant_contexts(
                query=request.query,
                query_embeddings=semantic_result.get("embeddings", [])
            )
            
            # Get full context chain if parent exists
            context_chain = []
            if request.parent_context_id:
                context_chain = await context_chainer.get_context_chain(
                    request.parent_context_id,
                    include_siblings=True
                )
            
            # Update semantic result with context chain
            semantic_result["context_chain"] = {
                "current_node": context_node.id,
                "relevant_contexts": [
                    {
                        "id": node.id,
                        "content": node.content,
                        "timestamp": node.timestamp.isoformat(),
                        "confidence": node.confidence
                    }
                    for node in relevant_contexts
                ],
                "full_chain": [
                    {
                        "id": node.id,
                        "content": node.content,
                        "timestamp": node.timestamp.isoformat(),
                        "depth": node.depth,
                        "confidence": node.confidence
                    }
                    for node in context_chain
                ]
            }
            
            context_chaining_time = metrics_collector.stop_timer("context_chaining")
        else:
            context_chaining_time = 0.0
        
        # 3. Temporal Analysis
        temporal_results = {}
        if request.analyze_temporal:
            metrics_collector.start_timer("temporal_analysis")
            
            # Extract temporal events
            events = await temporal_reasoner.extract_temporal_events(
                content=request.query,
                embeddings=semantic_result.get("embeddings", []),
                metadata={
                    "task_id": task_id,
                    "domain": request.domain.value if request.domain else None
                }
            )
            
            if events:
                # Create temporal sequence
                sequence = await temporal_reasoner.create_temporal_sequence(
                    events=events,
                    metadata={
                        "task_id": task_id,
                        "query": request.query
                    }
                )
                
                # Analyze temporal patterns
                patterns = await temporal_reasoner.analyze_temporal_patterns(sequence.id)
                
                # Find related events
                time_window = timedelta(seconds=request.time_window) if request.time_window else None
                related_events = []
                for event in events:
                    related = await temporal_reasoner.find_related_events(
                        event,
                        time_window=time_window
                    )
                    related_events.extend(related)
                
                # Predict next event
                predicted_event = await temporal_reasoner.predict_next_event(
                    sequence.id,
                    time_window=time_window or timedelta(days=7)
                )
                
                temporal_results = {
                    "sequence_id": sequence.id,
                    "events": [
                        {
                            "id": event.id,
                            "content": event.content,
                            "start_time": event.start_time.isoformat(),
                            "end_time": event.end_time.isoformat() if event.end_time else None,
                            "duration": event.duration.total_seconds() if event.duration else None,
                            "confidence": event.confidence
                        }
                        for event in events
                    ],
                    "patterns": patterns,
                    "related_events": [
                        {
                            "id": event.id,
                            "content": event.content,
                            "start_time": event.start_time.isoformat(),
                            "confidence": event.confidence
                        }
                        for event in related_events
                    ],
                    "predicted_event": {
                        "id": predicted_event.id,
                        "start_time": predicted_event.start_time.isoformat(),
                        "confidence": predicted_event.confidence
                    } if predicted_event else None
                }
            
            temporal_analysis_time = metrics_collector.stop_timer("temporal_analysis")
        else:
            temporal_analysis_time = 0.0
        
        # 4. Role Mapping
        metrics_collector.start_timer("role_mapping")
        karaka_roles = semantic_layer.karaka_mapper.map_roles(request.query)
        role_mapping_time = metrics_collector.stop_timer("role_mapping")
        
        # 5. Custom Abstraction Processing
        abstraction_results = {}
        if request.abstraction_types:
            metrics_collector.start_timer("abstraction_processing")
            
            for abstraction_type in request.abstraction_types:
                if abstraction_type == AbstractionType.METAPHYSICAL:
                    abstraction_results["metaphysical"] = await metaphysical_engine.process(semantic_result)
                elif abstraction_type == AbstractionType.COGNITIVE:
                    abstraction_results["cognitive"] = await cognitive_engine.process(semantic_result)
            
            abstraction_time = metrics_collector.stop_timer("abstraction_processing")
        else:
            abstraction_time = 0.0
        
        # 6. Agent Planning
        metrics_collector.start_timer("agent_planning")
        agent_plan = await agent_layer.create_plan(semantic_result, request.context)
        planning_time = metrics_collector.stop_timer("agent_planning")
        
        # 7. Execution
        metrics_collector.start_timer("execution")
        execution_result = await execution_layer.execute_plan(agent_plan)
        execution_time = metrics_collector.stop_timer("execution")
        
        # 8. Memory Operations
        metrics_collector.start_timer("memory_operations")
        memory_key = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        await memory_engine.store(memory_key, {
            'request': request.dict(),
            'semantic_result': semantic_result,
            'temporal_results': temporal_results,
            'abstraction_results': abstraction_results,
            'agent_plan': agent_plan,
            'execution_result': execution_result
        })
        memory_time = metrics_collector.stop_timer("memory_operations")
        
        # 9. Interaction
        interaction_response = await interaction_engine.process(
            request.query,
            semantic_result,
            agent_plan,
            execution_result
        )
        
        # Record metrics
        total_time = metrics_collector.stop_timer("total")
        metrics = PipelineMetrics(
            semantic_analysis_time=0.0,
            context_chaining_time=context_chaining_time,
            temporal_analysis_time=temporal_analysis_time,
            role_mapping_time=role_mapping_time,
            abstraction_processing_time=abstraction_time,
            agent_planning_time=planning_time,
            execution_time=execution_time,
            memory_operations_time=memory_time,
            total_time=total_time,
            success=True
        )
        metrics_collector.record_metrics(task_id, metrics)
        
        # Update task status
        tasks[task_id].update({
            "status": TaskStatus.COMPLETED,
            "result": {
                **interaction_response,
                "abstraction_results": abstraction_results,
                "temporal_results": temporal_results,
                "context_node_id": context_node.id if request.chain_context else None
            },
            "metrics": metrics.__dict__
        })
        
        # Create visualizations if requested
        if request.visualize:
            visualization_type = request.visualization_type or "combined"
            visualization_path = semantic_visualizer.create_combined_report(
                semantic_result,
                f"Research Task - {task_id}"
            )
        
        return {
            "status": "success",
            "task_id": task_id,
            "semantic_result": semantic_result,
            "temporal_results": temporal_results,
            "abstraction_results": abstraction_results,
            "context_node_id": context_node.id if request.chain_context else None,
            "visualization_path": visualization_path if request.visualize else None
        }
    except Exception as e:
        # Record failed metrics
        total_time = metrics_collector.stop_timer("total")
        metrics = PipelineMetrics(
            semantic_analysis_time=0.0,
            context_chaining_time=0.0,
            temporal_analysis_time=0.0,
            role_mapping_time=0.0,
            abstraction_processing_time=0.0,
            agent_planning_time=0.0,
            execution_time=0.0,
            memory_operations_time=0.0,
            total_time=total_time,
            success=False,
            error=str(e)
        )
        metrics_collector.record_metrics(task_id, metrics)
        
        # Update task status
        tasks[task_id].update({
            "status": TaskStatus.FAILED,
            "error": str(e),
            "metrics": metrics.__dict__
        })
        
        raise HTTPException(status_code=500, detail=str(e))

app.get("/research/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of a research task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
        
    task = tasks[task_id]
    
    # Retrieve semantic context from memory
    semantic_context = memory_engine.retrieve_context(task_id)
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress", 0.0),
        result=task.get("result"),
        semantic_roles=semantic_context.get("semantic_roles") if semantic_context else None,
        semantic_graph=semantic_context.get("semantic_graph") if semantic_context else None,
        temporal_dimensions=semantic_context.get("temporal_dimensions") if semantic_context else None,
        spatial_dimensions=semantic_context.get("spatial_dimensions") if semantic_context else None,
        domain_context=semantic_context.get("domain_context") if semantic_context else None,
        confidence=task.get("result", {}).get("confidence", 0.0),
        timestamp=task["timestamp"]
    )

app.get("/research/{task_id}/actions")
async def get_task_actions(task_id: str):
    """Get all actions for a research task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
        
    # Retrieve actions from memory
    actions = memory_engine.retrieve_actions(task_id)
    
    return {
        "task_id": task_id,
        "actions": actions
    }

app.get("/research/{task_id}/context")
async def get_task_context(task_id: str):
    """Get semantic context for a research task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
        
    # Retrieve context from memory
    context = memory_engine.retrieve_context(task_id)
    
    return {
        "task_id": task_id,
        "context": context
    }

app.get("/research")
async def list_tasks():
    """List all research tasks."""
    return {
        "tasks": [
            {
                "task_id": task_id,
                "status": task["status"],
                "progress": task.get("progress", 0.0),
                "timestamp": task["timestamp"]
            }
            for task_id, task in tasks.items()
        ]
    }

app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    """Get pipeline performance metrics."""
    return metrics_collector.generate_report()

app.post("/research/export")
async def export_research(request: ExportRequest):
    """Export research results to various workspace platforms."""
    if request.task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[request.task_id]
    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail="Task must be completed before export"
        )
    
    # Prepare research data
    research_data = {
        "title": request.title or f"Research Task - {request.task_id}",
        "request": task["request"],
        "semantic_result": task.get("result", {}).get("semantic_result"),
        "agent_plan": task.get("result", {}).get("agent_plan"),
        "execution_result": task.get("result", {}).get("execution_result"),
        "interaction_response": task.get("result", {}).get("interaction_response")
    }
    
    if request.include_metrics and task.get("metrics"):
        research_data["metrics"] = task["metrics"]
    
    try:
        if request.workspace_type == WorkspaceType.NOTION:
            if not all([request.notion_api_key, request.notion_database_id]):
                raise HTTPException(
                    status_code=400,
                    detail="Notion API key and database ID are required"
                )
            
            result = await workspace_exporter.export_to_notion(
                content=research_data,
                title=research_data["title"],
                api_key=request.notion_api_key,
                database_id=request.notion_database_id,
                parent_page_id=request.notion_parent_page_id
            )
            
        elif request.workspace_type == WorkspaceType.OBSIDIAN:
            if not request.obsidian_vault_path:
                raise HTTPException(
                    status_code=400,
                    detail="Obsidian vault path is required"
                )
            
            result = await workspace_exporter.export_to_obsidian(
                content=research_data,
                title=research_data["title"],
                vault_path=request.obsidian_vault_path,
                folder=request.obsidian_folder
            )
            
        elif request.workspace_type == WorkspaceType.GITHUB:
            if not all([request.github_repo, request.github_path, request.github_token]):
                raise HTTPException(
                    status_code=400,
                    detail="GitHub repository, path, and token are required"
                )
            
            result = await workspace_exporter.export_to_github(
                content=research_data,
                title=research_data["title"],
                repo=request.github_repo,
                path=request.github_path,
                token=request.github_token,
                branch=request.github_branch
            )
            
        else:  # MARKDOWN
            result = await workspace_exporter.export_to_markdown(
                content=research_data,
                title=research_data["title"],
                format=request.format
            )
        
        return {
            "status": "success",
            "workspace_type": request.workspace_type,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Failed to export research: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Export failed: {str(e)}"
        )

app.post("/research/visualize")
async def create_visualization(
    semantic_result: Dict[str, Any],
    visualization_type: str = "combined",
    title: Optional[str] = None
):
    """Create visualization of semantic analysis results."""
    try:
        if not title:
            title = f"Semantic Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        if visualization_type == "graph":
            filepath = semantic_visualizer.create_semantic_graph(semantic_result, title)
        elif visualization_type == "embeddings":
            filepath = semantic_visualizer.create_semantic_embedding_plot(
                semantic_result.get('embeddings', {}),
                title
            )
        elif visualization_type == "roles":
            filepath = semantic_visualizer.create_role_distribution_chart(semantic_result, title)
        elif visualization_type == "timeline":
            if 'context_history' not in semantic_result:
                raise HTTPException(
                    status_code=400,
                    detail="Context history not available for timeline visualization"
                )
            filepath = semantic_visualizer.create_context_timeline(
                semantic_result['context_history'],
                title
            )
        else:  # combined
            filepath = semantic_visualizer.create_combined_report(semantic_result, title)
        
        return {
            "status": "success",
            "visualization_path": filepath,
            "visualization_type": visualization_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.get("/research/visualizations/{filename}")
async def get_visualization(filename: str):
    """Retrieve a specific visualization file."""
    filepath = Path("visualizations") / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    if filepath.suffix == '.html':
        with open(filepath, 'r') as f:
            return HTMLResponse(content=f.read())
    elif filepath.suffix == '.png':
        with open(filepath, 'rb') as f:
            return Response(content=f.read(), media_type="image/png")
    else:
        with open(filepath, 'r') as f:
            return JSONResponse(content=json.load(f))

app.get("/research/context/{node_id}")
async def get_context_chain(node_id: str, include_siblings: bool = False):
    """Get the context chain for a specific node."""
    try:
        chain = await context_chainer.get_context_chain(node_id, include_siblings)
        return {
            "node_id": node_id,
            "chain": [
                {
                    "id": node.id,
                    "content": node.content,
                    "timestamp": node.timestamp.isoformat(),
                    "depth": node.depth,
                    "confidence": node.confidence,
                    "metadata": node.metadata
                }
                for node in chain
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

app.post("/research/context/merge")
async def merge_context_chains(chain_id1: str, chain_id2: str):
    """Merge two context chains."""
    try:
        merged_id = await context_chainer.merge_chains(chain_id1, chain_id2)
        if not merged_id:
            raise HTTPException(
                status_code=400,
                detail="Chains could not be merged (not similar enough or already connected)"
            )
        return {"status": "success", "merged_chain_id": merged_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

app.get("/research/context/analyze/{chain_id}")
async def analyze_context_chain(chain_id: str):
    """Analyze the structure of a context chain."""
    try:
        analysis = await context_chainer.analyze_chain_structure(chain_id)
        return {
            "chain_id": chain_id,
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

app.get("/research/temporal/{sequence_id}")
async def get_temporal_sequence(sequence_id: str):
    """Get details of a temporal sequence."""
    try:
        if sequence_id not in temporal_reasoner.sequences:
            raise HTTPException(status_code=404, detail="Sequence not found")
        
        sequence = temporal_reasoner.sequences[sequence_id]
        return {
            "sequence_id": sequence.id,
            "events": [
                {
                    "id": event.id,
                    "content": event.content,
                    "start_time": event.start_time.isoformat(),
                    "end_time": event.end_time.isoformat() if event.end_time else None,
                    "duration": event.duration.total_seconds() if event.duration else None,
                    "confidence": event.confidence,
                    "metadata": event.metadata
                }
                for event in sequence.events
            ],
            "start_time": sequence.start_time.isoformat(),
            "end_time": sequence.end_time.isoformat(),
            "duration": sequence.duration.total_seconds(),
            "metadata": sequence.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

app.get("/research/temporal/analyze/{sequence_id}")
async def analyze_temporal_sequence(sequence_id: str):
    """Analyze patterns in a temporal sequence."""
    try:
        analysis = await temporal_reasoner.analyze_temporal_patterns(sequence_id)
        return {
            "sequence_id": sequence_id,
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

app.get("/research/temporal/predict/{sequence_id}")
async def predict_next_event(
    sequence_id: str,
    time_window: int = 604800  # Default: 7 days in seconds
):
    """Predict the next event in a sequence."""
    try:
        predicted_event = await temporal_reasoner.predict_next_event(
            sequence_id,
            time_window=timedelta(seconds=time_window)
        )
        
        if not predicted_event:
            raise HTTPException(
                status_code=404,
                detail="Could not predict next event"
            )
        
        return {
            "sequence_id": sequence_id,
            "predicted_event": {
                "id": predicted_event.id,
                "start_time": predicted_event.start_time.isoformat(),
                "confidence": predicted_event.confidence,
                "metadata": predicted_event.metadata
            }
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

app.post("/conversations")
async def create_conversation(request: ConversationRequest):
    """Create a new agent conversation."""
    try:
        conversation_id = agent_conversation.create_conversation(
            goal=request.goal,
            initial_context=request.initial_context
        )
        
        return {
            "status": "success",
            "conversation_id": conversation_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.post("/conversations/{conversation_id}/messages")
async def add_message(
    conversation_id: str,
    request: MessageRequest
):
    """Add a message to a conversation."""
    try:
        message = agent_conversation.add_message(
            conversation_id=conversation_id,
            role=request.role,
            content=request.content,
            reasoning=request.reasoning,
            confidence=request.confidence,
            metadata=request.metadata
        )
        
        return {
            "status": "success",
            "message_id": f"msg_{message.timestamp.strftime('%Y%m%d_%H%M%S')}",
            "timestamp": message.timestamp.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get current state of a conversation."""
    try:
        state = agent_conversation.get_conversation_state(conversation_id)
        
        return {
            "conversation_id": conversation_id,
            "goal": state.goal,
            "current_focus": state.current_focus,
            "confidence": state.confidence,
            "is_active": state.is_active,
            "message_count": len(state.messages),
            "last_message": state.messages[-1].timestamp.isoformat() if state.messages else None
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.post("/conversations/{conversation_id}/end")
async def end_conversation(conversation_id: str):
    """End a conversation and save logs."""
    try:
        agent_conversation.end_conversation(conversation_id)
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.get("/conversations/{conversation_id}/analyze")
async def analyze_conversation(conversation_id: str):
    """Analyze conversation patterns and effectiveness."""
    try:
        analysis = agent_conversation.analyze_conversation(conversation_id)
        return {
            "conversation_id": conversation_id,
            "analysis": analysis
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.get("/performance")
async def get_performance_report():
    """Get system performance report."""
    try:
        report = performance_monitor.get_performance_report()
        return {
            "status": "success",
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.post("/performance/optimize")
async def optimize_performance():
    """Trigger performance optimization."""
    try:
        await performance_monitor.optimize_memory()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for agent communication"""
    try:
        # Accept connection
        await websocket_manager.connect(websocket)
        
        # Process messages
        while True:
            try:
                # Receive message
                message = await websocket.receive_json()
                
                # Validate message
                if "type" not in message or "content" not in message:
                    await websocket.send_json({
                        "type": "error",
                        "code": "1001",
                        "message": "Invalid message format"
                    })
                    continue
                
                # Process message based on type
                if message["type"] == "message":
                    # Get agent type
                    agent_type = message.get("agent", "research")
                    
                    # Select agent
                    agent = {
                        "research": research_agent,
                        "builder": builder_agent,
                        "teacher": teacher_agent,
                        "collaborator": collaborator_agent
                    }.get(agent_type, research_agent)
                    
                    # Process message
                    response = await agent.process(message["content"])
                    
                    # Send response
                    await websocket.send_json({
                        "type": "response",
                        "content": response
                    })
                    
                elif message["type"] == "status":
                    # Get system status
                    status = await monitoring.get_status()
                    
                    # Send status
                    await websocket.send_json({
                        "type": "status",
                        "content": status
                    })
                    
                else:
                    # Invalid message type
                    await websocket.send_json({
                        "type": "error",
                        "code": "1001",
                        "message": "Invalid message type"
                    })
                    
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "code": "1001",
                    "message": "Invalid JSON format"
                })
                
    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket_manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        await database.health_check()
        
        # Check cache connection
        await cache.health_check()
        
        # Get system status
        status = await monitoring.get_status()
        
        return {
            "status": "healthy",
            "database": "connected",
            "cache": "connected",
            "metrics": status
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/status")
async def get_status():
    """Status endpoint"""
    try:
        # Get system status
        status = await monitoring.get_status()
        
        return status
        
    except Exception as e:
        logger.error(f"Status retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Status retrieval failed")

if __name__ == "__main__":
    asyncio.run(
        app.run(
            host="0.0.0.0",
            port=8000
        )
    ) 