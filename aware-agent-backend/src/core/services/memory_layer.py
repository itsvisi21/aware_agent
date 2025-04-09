import json
import logging
import uuid
from datetime import date
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

import aiosqlite
from pydantic import BaseModel

from .interaction_engine import ConversationState
from .semantic_abstraction import ContextNode
from src.core.models.models import ExecutionTask

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory storage."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryState(Enum):
    """States of memory."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MemoryContext(BaseModel):
    """Context for memory operations."""
    task_id: str
    agent_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class MemoryOperation(Enum):
    """Types of memory operations."""
    STORE = "store"
    RETRIEVE = "retrieve"
    UPDATE = "update"
    DELETE = "delete"


class MemoryResult(BaseModel):
    """Result of a memory operation."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class Task(BaseModel):
    id: str
    type: str
    status: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = {}


class SemanticLog(BaseModel):
    id: str
    timestamp: datetime
    context_snapshot: Dict[str, Any]
    action: str
    details: Dict[str, Any]
    metadata: Dict[str, Any] = {}


class MemoryEntry(BaseModel):
    """Represents a single memory entry."""
    id: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}
    confidence: float = 1.0


class MemoryQuery(BaseModel):
    """Represents a query for memory retrieval."""
    query: str
    filters: Dict[str, Any] = {}
    limit: int = 10
    min_confidence: float = 0.0


class MemoryEngine:
    """Engine responsible for managing persistent storage."""

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self.conn = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the database connection and tables."""
        if self._initialized:
            return

        self.conn = await aiosqlite.connect(self.db_path)
        await self._initialize_db()
        self._initialized = True

    async def _initialize_db(self) -> None:
        """Initialize database tables."""
        async with self.conn.cursor() as cursor:
            # Create conversations table
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    goal TEXT,
                    metadata TEXT
                )
            """)

            # Create messages table
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    type TEXT DEFAULT 'text',
                    status TEXT DEFAULT 'success',
                    metadata TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)

            # Create tasks table
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    config TEXT,
                    result TEXT,
                    error TEXT
                )
            """)

            # Create semantic_logs table
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS semantic_logs (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    context_snapshot TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            # Create conversation_states table
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_states (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    state TEXT NOT NULL
                )
            """)

            # Create indices
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_semantic_logs_timestamp ON semantic_logs(timestamp)")

            await self.conn.commit()

    async def store_task(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """Store a task in the database."""
        await self.initialize()

        async with self.conn.cursor() as cursor:
            await cursor.execute("""
                INSERT OR REPLACE INTO tasks (
                    id, type, status, created_at, updated_at, config, result, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                task_data.get('type', ''),
                task_data.get('status', 'pending'),
                task_data.get('created_at', datetime.now().isoformat()),
                task_data.get('updated_at', datetime.now().isoformat()),
                json.dumps(task_data.get('input_data', {})),
                json.dumps(task_data.get('output_data', {})),
                None
            ))
            await self.conn.commit()

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a task from the database."""
        await self.initialize()

        async with self.conn.cursor() as cursor:
            await cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = await cursor.fetchone()

            if not row:
                return None

            return {
                'id': row[0],
                'type': row[1],
                'status': row[2],
                'created_at': row[3],
                'updated_at': row[4],
                'config': json.loads(row[5]) if row[5] else {},
                'result': json.loads(row[6]) if row[6] else None,
                'error': row[7]
            }

    async def get_semantic_logs(self, start_time: datetime, end_time: datetime) -> List[SemanticLog]:
        """Retrieve semantic logs within a time range."""
        await self.initialize()

        async with self.conn.cursor() as cursor:
            await cursor.execute("""
                SELECT id, timestamp, context_snapshot, action, details, metadata
                FROM semantic_logs
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """, (start_time.isoformat(), end_time.isoformat()))

            rows = await cursor.fetchall()
            logs = []
            for row in rows:
                logs.append(SemanticLog(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    context_snapshot=json.loads(row[2]),
                    action=row[3],
                    details=json.loads(row[4]),
                    metadata=json.loads(row[5]) if row[5] else {}
                ))
            return logs

    async def save_conversation_state(self, state: ConversationState) -> None:
        """Save the current conversation state."""
        await self.initialize()

        async with self.conn.cursor() as cursor:
            await cursor.execute("""
                INSERT INTO conversation_states (id, timestamp, state)
                VALUES (?, ?, ?)
            """, (
                str(uuid.uuid4()),
                datetime.now().isoformat(),
                json.dumps(state.to_dict(), cls=DateTimeEncoder)
            ))
            await self.conn.commit()

    async def get_latest_conversation_state(self) -> Optional[ConversationState]:
        """Retrieve the latest conversation state."""
        await self.initialize()

        async with self.conn.cursor() as cursor:
            await cursor.execute("""
                SELECT state FROM conversation_states
                ORDER BY timestamp DESC LIMIT 1
            """)
            row = await cursor.fetchone()

            if not row:
                return None

            state_data = json.loads(row[0])
            return ConversationState.from_dict(state_data)

    async def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            await self.conn.close()
            self.conn = None
            self._initialized = False


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class TaskManager:
    """Manages task creation, updates, and retrieval."""

    def __init__(self, memory_engine: MemoryEngine):
        self.memory_engine = memory_engine

    async def create_task(self, task_type: str, input_data: Dict[str, Any]) -> ExecutionTask:
        """Create a new task."""
        task_id = str(uuid.uuid4())
        task_data = {
            "id": task_id,
            "type": task_type,
            "status": "pending",
            "input_data": input_data,
            "output_data": {},
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        task = ExecutionTask(**task_data)
        await self.memory_engine.store_task(task_id, task.to_dict())
        return task

    async def update_task(self, task_id: str, status: str, output_data: Dict[str, Any]) -> None:
        """Update an existing task."""
        task_dict = await self.memory_engine.get_task(task_id)
        if task_dict:
            task = ExecutionTask.from_dict(task_dict)
            task.status = status
            task.output_data = output_data
            task.updated_at = datetime.now()
            await self.memory_engine.store_task(task_id, task.to_dict())

    async def get_task_status(self, task_id: str) -> Optional[str]:
        """Get the status of a task."""
        task_dict = await self.memory_engine.get_task(task_id)
        if task_dict:
            task = ExecutionTask.from_dict(task_dict)
            return task.status
        return None


class SemanticLogger:
    """Handles semantic logging of context changes and conversation states."""

    def __init__(self, memory_engine: MemoryEngine):
        self.memory_engine = memory_engine

    async def log_context_change(self, context: ContextNode, action: str, details: Dict[str, Any]) -> None:
        """Log a context change."""
        log_id = str(uuid.uuid4())
        async with self.memory_engine.conn.cursor() as cursor:
            await cursor.execute("""
                INSERT INTO semantic_logs (
                    id, timestamp, context_snapshot, action, details, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                log_id,
                datetime.now().isoformat(),
                json.dumps(context.dict(), cls=DateTimeEncoder),
                action,
                json.dumps(details),
                json.dumps({})
            ))
            await self.memory_engine.conn.commit()

    async def log_conversation_state(self, state: ConversationState) -> None:
        """Log a conversation state."""
        await self.memory_engine.save_conversation_state(state)


class MemoryLayer:
    """Layer responsible for memory operations."""
    
    def __init__(self):
        self.memories: Dict[str, MemoryEntry] = {}
    
    async def store(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Store a new memory entry."""
        entry_id = str(len(self.memories))
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )
        self.memories[entry_id] = entry
        return entry_id
    
    async def retrieve(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Retrieve memories based on a query."""
        results = []
        for entry in self.memories.values():
            if entry.confidence >= query.min_confidence:
                # Simple content matching for now
                if query.query.lower() in entry.content.lower():
                    results.append(entry)
                    if len(results) >= query.limit:
                        break
        return results
    
    async def update(self, entry_id: str, content: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Update an existing memory entry."""
        if entry_id not in self.memories:
            return False
        
        entry = self.memories[entry_id]
        if content is not None:
            entry.content = content
        if metadata is not None:
            entry.metadata.update(metadata)
        entry.timestamp = datetime.now()
        return True
    
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        if entry_id in self.memories:
            del self.memories[entry_id]
            return True
        return False
