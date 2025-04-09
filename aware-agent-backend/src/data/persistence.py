"""Persistence layer for the Aware Agent application."""
from pathlib import Path
from typing import Dict, Any, List, Optional

import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

import aiosqlite
from src.core.models.models import ContextNode
from pydantic import BaseModel
from src.data.database import DatabaseService
from src.common.exceptions import DatabaseError


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
    def __init__(self, db: Union[str, DatabaseService] = "memory.db"):
        self.db = db
        self._initialized = False
        self._conn = None
        self._is_sqlite = isinstance(db, str)

    async def _get_connection(self):
        if self._conn is None:
            if self._is_sqlite:
                self._conn = await aiosqlite.connect(self.db)
            else:
                self._conn = await self.db.connect()
        return self._conn

    async def _init_db(self):
        if self._initialized:
            return

        conn = await self._get_connection()
        if self._is_sqlite:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    metadata TEXT
                )
            """)

            await conn.execute("""
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

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS context_trees (
                    conversation_id TEXT PRIMARY KEY,
                    tree_data TEXT NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            await conn.commit()
        else:
            await self.db.initialize_schema()

        self._initialized = True

    async def _ensure_db_initialized(self):
        if not self._initialized:
            await self._init_db()

    async def create_conversation(self, conversation: ConversationRecord) -> str:
        await self._ensure_db_initialized()
        conn = await self._get_connection()
        if self._is_sqlite:
            await conn.execute(
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
            await conn.commit()
        else:
            await self.db.execute_command(
                """
                INSERT INTO conversations (id, title, created_at, updated_at, metadata)
                VALUES ($1, $2, $3, $4, $5)
                """,
                conversation.id,
                conversation.title,
                conversation.created_at.isoformat(),
                conversation.updated_at.isoformat(),
                json.dumps(conversation.metadata)
            )
        return conversation.id

    async def add_message(self, message: MessageRecord) -> str:
        await self._ensure_db_initialized()
        conn = await self._get_connection()
        if self._is_sqlite:
            await conn.execute(
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
            await conn.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), message.conversation_id)
            )
            await conn.commit()
        else:
            await self.db.execute_command(
                """
                INSERT INTO messages (id, conversation_id, content, role, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                message.id,
                message.conversation_id,
                message.content,
                message.role,
                message.timestamp.isoformat(),
                json.dumps(message.metadata)
            )

            await self.db.execute_command(
                """
                UPDATE conversations
                SET updated_at = $1
                WHERE id = $2
                """,
                datetime.utcnow().isoformat(),
                message.conversation_id
            )
        return message.id

    async def get_conversation_history(self, conversation_id: str) -> List[MessageRecord]:
        await self._ensure_db_initialized()
        conn = await self._get_connection()
        if self._is_sqlite:
            cursor = await conn.execute(
                """
                SELECT id, conversation_id, content, role, timestamp, metadata
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                """,
                (conversation_id,)
            )

            messages = []
            async for row in cursor:
                messages.append(MessageRecord(
                    id=row[0],
                    conversation_id=row[1],
                    content=row[2],
                    role=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    metadata=json.loads(row[5])
                ))
        else:
            rows = await self.db.execute_query(
                """
                SELECT id, conversation_id, content, role, timestamp, metadata
                FROM messages
                WHERE conversation_id = $1
                ORDER BY timestamp ASC
                """,
                conversation_id
            )

            messages = [
                MessageRecord(
                    id=row[0],
                    conversation_id=row[1],
                    content=row[2],
                    role=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    metadata=json.loads(row[5])
                )
                for row in rows
            ]

        return messages

    async def update_context_tree(self, conversation_id: str, context_tree: ContextNode):
        await self._ensure_db_initialized()
        conn = await self._get_connection()
        if self._is_sqlite:
            await conn.execute(
                """
                INSERT OR REPLACE INTO context_trees (conversation_id, tree_data, updated_at)
                VALUES (?, ?, ?)
                """,
                (
                    conversation_id,
                    json.dumps(context_tree.model_dump()),
                    datetime.utcnow().isoformat()
                )
            )
            await conn.commit()
        else:
            await self.db.execute_command(
                """
                INSERT INTO context_trees (conversation_id, tree_data, updated_at)
                VALUES ($1, $2, $3)
                ON CONFLICT (conversation_id) DO UPDATE
                SET tree_data = $2, updated_at = $3
                """,
                conversation_id,
                json.dumps(context_tree.model_dump()),
                datetime.utcnow().isoformat()
            )

    async def get_context_tree(self, conversation_id: str) -> Optional[ContextNode]:
        await self._ensure_db_initialized()
        conn = await self._get_connection()
        if self._is_sqlite:
            cursor = await conn.execute(
                """
                SELECT tree_data
                FROM context_trees
                WHERE conversation_id = ?
                """,
                (conversation_id,)
            )

            row = await cursor.fetchone()
        else:
            rows = await self.db.execute_query(
                """
                SELECT tree_data
                FROM context_trees
                WHERE conversation_id = $1
                """,
                conversation_id
            )
            row = rows[0] if rows else None

        if row:
            tree_data = json.loads(row[0])
            return ContextNode(**tree_data)
        return None

    async def list_conversations(self, limit: int = 10, offset: int = 0) -> List[ConversationRecord]:
        await self._ensure_db_initialized()
        conn = await self._get_connection()
        if self._is_sqlite:
            cursor = await conn.execute(
                """
                SELECT id, title, created_at, updated_at, metadata
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset)
            )

            conversations = []
            async for row in cursor:
                conversations.append(ConversationRecord(
                    id=row[0],
                    title=row[1],
                    created_at=datetime.fromisoformat(row[2]),
                    updated_at=datetime.fromisoformat(row[3]),
                    metadata=json.loads(row[4])
                ))
        else:
            rows = await self.db.execute_query(
                """
                SELECT id, title, created_at, updated_at, metadata
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset
            )

            conversations = [
                ConversationRecord(
                    id=row[0],
                    title=row[1],
                    created_at=datetime.fromisoformat(row[2]),
                    updated_at=datetime.fromisoformat(row[3]),
                    metadata=json.loads(row[4])
                )
                for row in rows
            ]

        return conversations

    async def close(self):
        if self._conn is not None:
            if self._is_sqlite:
                await self._conn.close()
            else:
                await self.db.disconnect()
            self._conn = None
