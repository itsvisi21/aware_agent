from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class ResearchRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    domain: Optional[str] = None
    chain_context: bool = False
    analyze_temporal: bool = False
    time_window: Optional[int] = None
    abstraction_types: Optional[List[str]] = None
    visualize: bool = False
    visualization_type: Optional[str] = None
    parent_context_id: Optional[str] = None

class SessionRequest(BaseModel):
    title: str
    description: str
    initial_context: Optional[Dict[str, Any]] = None

class ParticipantRequest(BaseModel):
    user_id: str
    role: str

class ContextUpdate(BaseModel):
    updates: Dict[str, Any]

class ExportRequest(BaseModel):
    task_id: str
    title: Optional[str] = None
    workspace_type: str
    include_metrics: bool = False
    notion_api_key: Optional[str] = None
    notion_database_id: Optional[str] = None
    notion_parent_page_id: Optional[str] = None
    obsidian_vault_path: Optional[str] = None
    obsidian_folder: Optional[str] = None
    github_repo: Optional[str] = None
    github_path: Optional[str] = None
    github_token: Optional[str] = None
    github_branch: Optional[str] = None
    format: Optional[str] = None

class ConversationRequest(BaseModel):
    goal: str
    initial_context: Optional[Dict[str, Any]] = None

class MessageRequest(BaseModel):
    role: str
    content: str
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    result: Optional[Dict[str, Any]] = None
    semantic_roles: Optional[Dict[str, Any]] = None
    semantic_graph: Optional[Dict[str, Any]] = None
    temporal_dimensions: Optional[Dict[str, Any]] = None
    spatial_dimensions: Optional[Dict[str, Any]] = None
    domain_context: Optional[Dict[str, Any]] = None
    confidence: float
    timestamp: datetime 