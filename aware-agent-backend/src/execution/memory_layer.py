from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel
import sqlite3
import json
from pathlib import Path
from ..semantic_abstraction import ContextNode
from ..interaction import ConversationState

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

class MemoryEngine:
    def __init__(self, db_path: str = "memory/aware_agent.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                input_data TEXT NOT NULL,
                output_data TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                metadata TEXT
            )
        """)

        # Create semantic_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_logs (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                context_snapshot TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT NOT NULL,
                metadata TEXT
            )
        """)

        # Create conversation_states table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_states (
                id TEXT PRIMARY KEY,
                state_data TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                metadata TEXT
            )
        """)

        conn.commit()
        conn.close()

    def store_task(self, task: Task) -> None:
        """Store a task in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO tasks (id, type, status, input_data, output_data, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.id,
            task.type,
            task.status,
            json.dumps(task.input_data),
            json.dumps(task.output_data),
            task.created_at.isoformat(),
            task.updated_at.isoformat(),
            json.dumps(task.metadata)
        ))

        conn.commit()
        conn.close()

    def log_semantic_state(self, log: SemanticLog) -> None:
        """Log a semantic state change."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO semantic_logs (id, timestamp, context_snapshot, action, details, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            log.id,
            log.timestamp.isoformat(),
            json.dumps(log.context_snapshot),
            log.action,
            json.dumps(log.details),
            json.dumps(log.metadata)
        ))

        conn.commit()
        conn.close()

    def save_conversation_state(self, state: ConversationState) -> None:
        """Save a conversation state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        state_id = str(datetime.now().timestamp())
        cursor.execute("""
            INSERT INTO conversation_states (id, state_data, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            state_id,
            state.json(),
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            json.dumps(state.metadata)
        ))

        conn.commit()
        conn.close()

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return Task(
                id=row[0],
                type=row[1],
                status=row[2],
                input_data=json.loads(row[3]),
                output_data=json.loads(row[4]) if row[4] else {},
                created_at=datetime.fromisoformat(row[5]),
                updated_at=datetime.fromisoformat(row[6]),
                metadata=json.loads(row[7]) if row[7] else {}
            )
        return None

    def get_semantic_logs(self, start_time: datetime, end_time: datetime) -> List[SemanticLog]:
        """Retrieve semantic logs within a time range."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM semantic_logs 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        """, (start_time.isoformat(), end_time.isoformat()))

        logs = []
        for row in cursor.fetchall():
            logs.append(SemanticLog(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                context_snapshot=json.loads(row[2]),
                action=row[3],
                details=json.loads(row[4]),
                metadata=json.loads(row[5]) if row[5] else {}
            ))

        conn.close()
        return logs

    def get_latest_conversation_state(self) -> Optional[ConversationState]:
        """Retrieve the most recent conversation state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT state_data FROM conversation_states 
            ORDER BY updated_at DESC LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()

        if row:
            return ConversationState.parse_raw(row[0])
        return None

class TaskManager:
    def __init__(self, memory_engine: MemoryEngine):
        self.memory_engine = memory_engine
        self.active_tasks: Dict[str, Task] = {}

    def create_task(self, task_type: str, input_data: Dict[str, Any]) -> Task:
        """Create and store a new task."""
        task = Task(
            id=str(datetime.now().timestamp()),
            type=task_type,
            status="pending",
            input_data=input_data,
            output_data={},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.memory_engine.store_task(task)
        self.active_tasks[task.id] = task
        return task

    def update_task(self, task_id: str, status: str, output_data: Dict[str, Any]) -> None:
        """Update an existing task."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = status
            task.output_data = output_data
            task.updated_at = datetime.now()
            
            self.memory_engine.store_task(task)

    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get the status of a task."""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].status
        return None

class SemanticLogger:
    def __init__(self, memory_engine: MemoryEngine):
        self.memory_engine = memory_engine

    def log_context_change(self, context: ContextNode, action: str, details: Dict[str, Any]) -> None:
        """Log a change in the semantic context."""
        log = SemanticLog(
            id=str(datetime.now().timestamp()),
            timestamp=datetime.now(),
            context_snapshot=context.dict(),
            action=action,
            details=details
        )
        self.memory_engine.log_semantic_state(log)

    def log_conversation_state(self, state: ConversationState) -> None:
        """Log the current conversation state."""
        self.memory_engine.save_conversation_state(state) 