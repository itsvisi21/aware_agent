from typing import Dict, List, Any, Optional
import sqlite3
import json
from datetime import datetime
from pydantic import BaseModel
from ..semantic_abstraction import ContextNode

class ConversationRecord(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = {}

class MessageRecord(BaseModel):
    id: str
    conversation_id: str
    content: str
    role: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}

class MemoryStore:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    role TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_trees (
                    conversation_id TEXT PRIMARY KEY,
                    tree_data TEXT NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)

    async def create_conversation(self, conversation: ConversationRecord) -> str:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    conversation.id,
                    conversation.title,
                    conversation.created_at.isoformat(),
                    conversation.updated_at.isoformat(),
                    json.dumps(conversation.metadata)
                )
            )
        return conversation.id

    async def add_message(self, message: MessageRecord) -> str:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, content, role, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.conversation_id,
                    message.content,
                    message.role,
                    message.timestamp.isoformat(),
                    json.dumps(message.metadata)
                )
            )
            
            # Update conversation last modified time
            conn.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), message.conversation_id)
            )
        return message.id

    async def get_conversation_history(self, conversation_id: str) -> List[MessageRecord]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, conversation_id, content, role, timestamp, metadata
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                """,
                (conversation_id,)
            )
            
            messages = []
            for row in cursor:
                messages.append(MessageRecord(
                    id=row[0],
                    conversation_id=row[1],
                    content=row[2],
                    role=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    metadata=json.loads(row[5])
                ))
            
            return messages

    async def update_context_tree(self, conversation_id: str, context_tree: ContextNode):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO context_trees (conversation_id, tree_data, updated_at)
                VALUES (?, ?, ?)
                """,
                (
                    conversation_id,
                    json.dumps(context_tree.to_dict()),
                    datetime.utcnow().isoformat()
                )
            )

    async def get_context_tree(self, conversation_id: str) -> Optional[ContextNode]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT tree_data
                FROM context_trees
                WHERE conversation_id = ?
                """,
                (conversation_id,)
            )
            
            row = cursor.fetchone()
            if row:
                tree_data = json.loads(row[0])
                return ContextNode.from_dict(tree_data)
            return None

    async def list_conversations(self, limit: int = 10, offset: int = 0) -> List[ConversationRecord]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, title, created_at, updated_at, metadata
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset)
            )
            
            conversations = []
            for row in cursor:
                conversations.append(ConversationRecord(
                    id=row[0],
                    title=row[1],
                    created_at=datetime.fromisoformat(row[2]),
                    updated_at=datetime.fromisoformat(row[3]),
                    metadata=json.loads(row[4])
                ))
            
            return conversations 