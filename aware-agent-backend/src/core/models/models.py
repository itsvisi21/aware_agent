from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, ConfigDict



class TaskStatus(str, Enum):
    """Enum representing possible task statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentRole(str, Enum):
    """Enum representing possible agent roles."""
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    PLANNER = "planner"
    EXECUTOR = "executor"
    RESEARCHER = "researcher"
    TEACHER = "teacher"
    COLLABORATOR = "collaborator"
    BUILDER = "builder"


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


class Message(BaseModel):
    """Represents a message in a conversation."""
    role: str
    content: str
    type: str = "text"
    status: str = "success"
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = None

    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        super().__init__(**data)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the message, with a default if not found."""
        if hasattr(self, key):
            return getattr(self, key)
        return default

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ConversationState(BaseModel):
    """Represents the state of a conversation."""
    id: str
    messages: List[Message] = []
    context_tree: Dict[str, Any] = {}
    current_goal: Optional[str] = None
    feedback_history: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    created_at: str
    updated_at: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class InteractionResponse(BaseModel):
    """Represents a response from an interaction."""
    content: str
    type: str = "text"
    status: str = "success"
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = None

    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        super().__init__(**data)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Task(BaseModel):
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


class MessageRequest(BaseModel):
    """Represents a request to send a message."""
    content: str
    role: str = "user"
    type: str = "text"
    metadata: Optional[Dict[str, Any]] = None


class ContextNode(BaseModel):
    """Represents a node in a context tree."""
    id: str
    type: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = {}
    children: List['ContextNode'] = []
    parent_id: Optional[str] = None
    created_at: str = None
    updated_at: str = None

    def __init__(self, **data):
        if 'created_at' not in data:
            data['created_at'] = datetime.now().isoformat()
        if 'updated_at' not in data:
            data['updated_at'] = datetime.now().isoformat()
        super().__init__(**data)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ExecutionTask(BaseModel):
    """Represents a task execution in the system."""
    id: str
    type: str
    status: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the task to a dictionary representation."""
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionTask":
        """Create a task from a dictionary representation."""
        return cls(
            id=data["id"],
            type=data["type"],
            status=data["status"],
            input_data=data.get("input_data", {}),
            output_data=data.get("output_data", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"],
            updated_at=datetime.fromisoformat(data["updated_at"]) if isinstance(data["updated_at"], str) else data["updated_at"],
            metadata=data.get("metadata", {})
        )


# Global task storage
tasks: Dict[str, Dict[str, Any]] = {}
