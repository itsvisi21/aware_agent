"""
Database service for the Aware Agent application.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
import os
import logging
from datetime import datetime

import aiosqlite

from src.common.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for database operations."""

    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        """Initialize the database service.

        Args:
            db_path (Union[str, Path]): Path to the SQLite database file.
        """
        self.database_path = str(db_path)
        self._connection = None
        self._initialized = False
        
        # Create database directory if it doesn't exist
        if db_path != ":memory:":
            self._ensure_db_dir()

    def _ensure_db_dir(self) -> None:
        """Ensure the database directory exists."""
        if self.database_path != ":memory:":
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)

    @property
    def conn(self):
        """Alias for _connection to maintain compatibility with tests."""
        return self._connection

    async def initialize(self) -> None:
        """Initialize the database connection and create tables."""
        if self._initialized:
            return
        
        try:
            await self.connect()
            await self._create_tables()
            self._initialized = True
        except Exception as e:
            await self.disconnect()
            raise DatabaseError(f"Failed to initialize database: {str(e)}")

    async def connect(self) -> None:
        """Connect to the database."""
        if self._connection is not None:
            return
            
        try:
            self._connection = await aiosqlite.connect(self.database_path)
        except Exception as e:
            raise DatabaseError(f"Error connecting to database: {str(e)}")

    async def disconnect(self) -> None:
        """Disconnect from the database."""
        if self._connection is None:
            return
            
        try:
            await self._connection.close()
        except Exception as e:
            logger.error(f"Error disconnecting from database: {str(e)}")
        finally:
            self._connection = None
            self._initialized = False

    async def close(self) -> None:
        """Alias for disconnect."""
        await self.disconnect()

    async def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            async with self._connection.cursor() as cursor:
                # Create conversations table
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        agent_name TEXT NOT NULL,
                        agent_type TEXT NOT NULL,
                        message TEXT,
                        timestamp TEXT NOT NULL,
                        context_id TEXT,
                        parent_id TEXT,
                        messages TEXT,
                        context_tree TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)

                # Create agent_states table
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS agent_states (
                        id TEXT PRIMARY KEY,
                        agent_type TEXT NOT NULL,
                        state TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)

                # Create tasks table
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tasks (
                        id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        context TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        agent_type TEXT,
                        parent_task_id TEXT,
                        actions TEXT,
                        query TEXT,
                        relationships TEXT,
                        type TEXT
                    )
                """)

                # Create context_tree table
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS context_tree (
                        id TEXT PRIMARY KEY,
                        parent_id TEXT,
                        data TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)

                # Create semantic_logs table
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_logs (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        agent_type TEXT NOT NULL,
                        action TEXT NOT NULL,
                        context TEXT,
                        result TEXT
                    )
                """)

                # Create cache table
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL
                    )
                """)

                # Create metrics table
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        value REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        labels TEXT
                    )
                """)

                await self._connection.commit()
        except Exception as e:
            raise DatabaseError(f"Error creating tables: {str(e)}")

    async def save_conversation(self, conversation: Dict[str, Any]) -> None:
        """Save a conversation to the database."""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
            
        try:
            query = """
                INSERT INTO conversations (
                    id, agent_name, agent_type, message, timestamp,
                    context_id, parent_id, messages, context_tree,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    message = excluded.message,
                    messages = excluded.messages,
                    context_tree = excluded.context_tree,
                    updated_at = excluded.updated_at
            """
            await self.execute(query, (
                conversation['id'],
                conversation['agent_name'],
                conversation['agent_type'],
                conversation['message'],
                conversation['timestamp'],
                conversation['context_id'],
                conversation['parent_id'],
                conversation['messages'],
                conversation['context_tree'],
                conversation['created_at'],
                conversation['updated_at']
            ))
        except Exception as e:
            raise DatabaseError(f"Error saving conversation: {str(e)}")

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID."""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
            
        try:
            query = "SELECT * FROM conversations WHERE id = ?"
            row = await self.fetch_one(query, (conversation_id,))
            if row:
                return {
                    'id': row[0],
                    'agent_name': row[1],
                    'agent_type': row[2],
                    'message': row[3],
                    'timestamp': row[4],
                    'context_id': row[5],
                    'parent_id': row[6],
                    'messages': row[7],
                    'context_tree': row[8],
                    'created_at': row[9],
                    'updated_at': row[10]
                }
            return None
        except Exception as e:
            raise DatabaseError(f"Error getting conversation: {str(e)}")

    async def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations."""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
            
        try:
            query = "SELECT * FROM conversations ORDER BY created_at DESC"
            rows = await self.fetch_all(query)
            return [{
                'id': row[0],
                'agent_name': row[1],
                'agent_type': row[2],
                'message': row[3],
                'timestamp': row[4],
                'context_id': row[5],
                'parent_id': row[6],
                'messages': row[7],
                'context_tree': row[8],
                'created_at': row[9],
                'updated_at': row[10]
            } for row in rows]
        except Exception as e:
            raise DatabaseError(f"Error listing conversations: {str(e)}")

    async def save_agent_state(self, agent_id: str, agent_type: str, state: Dict[str, Any]) -> None:
        """Save agent state to the database."""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
            
        try:
            now = datetime.now().isoformat()
            query = """
                INSERT INTO agent_states (id, agent_type, state, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    state = excluded.state,
                    updated_at = excluded.updated_at
            """
            await self.execute(query, (
                agent_id,
                agent_type,
                json.dumps(state),
                now,
                now
            ))
        except TypeError as e:
            raise TypeError(f"Invalid JSON data in state: {str(e)}")
        except Exception as e:
            raise DatabaseError(f"Error saving agent state: {str(e)}")

    async def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent state from the database."""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
            
        try:
            query = "SELECT state FROM agent_states WHERE id = ?"
            row = await self.fetch_one(query, (agent_id,))
            if row:
                return json.loads(row[0])
            return None
        except Exception as e:
            raise DatabaseError(f"Error getting agent state: {str(e)}")

    async def delete_agent_state(self, agent_id: str) -> None:
        """Delete agent state from the database."""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
            
        try:
            query = "DELETE FROM agent_states WHERE id = ?"
            await self.execute(query, (agent_id,))
        except Exception as e:
            raise DatabaseError(f"Error deleting agent state: {str(e)}")

    async def export_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Export a conversation with all related data."""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
            
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            raise FileNotFoundError(f"Conversation {conversation_id} not found")
            
        return conversation

    async def import_conversation(self, conversation_data: Dict[str, Any]) -> str:
        """Import a conversation with all related data."""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
            
        # Generate a new ID to avoid conflicts
        new_id = f"{conversation_data['id']}_imported_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        conversation_data['id'] = new_id
        
        await self.save_conversation(conversation_data)
        return new_id

    async def execute(self, query: str, params: tuple = ()) -> Any:
        """Execute a SQL query.

        Args:
            query (str): The SQL query to execute.
            params (tuple): Query parameters.

        Returns:
            Any: Query results.

        Raises:
            DatabaseError: If there is an error executing the query.
        """
        try:
            async with self._connection.cursor() as cursor:
                await cursor.execute(query, params)
                await self._connection.commit()
                return await cursor.fetchall()
        except Exception as e:
            raise DatabaseError(f"Error executing query: {str(e)}")

    async def fetch_one(self, query: str, params: tuple = ()) -> Optional[tuple]:
        """Fetch a single row from the database.

        Args:
            query (str): The SQL query to execute.
            params (tuple): Query parameters.

        Returns:
            Optional[tuple]: The fetched row or None.

        Raises:
            DatabaseError: If there is an error fetching the row.
        """
        try:
            async with self._connection.cursor() as cursor:
                await cursor.execute(query, params)
                return await cursor.fetchone()
        except Exception as e:
            raise DatabaseError(f"Error fetching row: {str(e)}")

    async def fetch_all(self, query: str, params: tuple = ()) -> List[tuple]:
        """Fetch all rows from the database.

        Args:
            query (str): The SQL query to execute.
            params (tuple): Query parameters.

        Returns:
            List[tuple]: The fetched rows.

        Raises:
            DatabaseError: If there is an error fetching the rows.
        """
        try:
            async with self._connection.cursor() as cursor:
                await cursor.execute(query, params)
                return await cursor.fetchall()
        except Exception as e:
            raise DatabaseError(f"Error fetching rows: {str(e)}")

    async def insert(self, table: str, data: Dict[str, Any]) -> str:
        """Insert a row into a table.

        Args:
            table (str): The table to insert into.
            data (Dict[str, Any]): The data to insert.

        Returns:
            str: The ID of the inserted row.

        Raises:
            DatabaseError: If there is an error inserting the row.
        """
        try:
            columns = ", ".join(data.keys())
            placeholders = ", ".join("?" * len(data))
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

            async with self._connection.cursor() as cursor:
                await cursor.execute(query, tuple(data.values()))
                await self._connection.commit()
                return data.get("id")
        except Exception as e:
            raise DatabaseError(f"Error inserting row: {str(e)}")

    async def update(self, table: str, id_: str, data: Dict[str, Any]) -> None:
        """Update a row in a table.

        Args:
            table (str): The table to update.
            id_ (str): The ID of the row to update.
            data (Dict[str, Any]): The data to update.

        Raises:
            DatabaseError: If there is an error updating the row.
        """
        try:
            set_clause = ", ".join(f"{k} = ?" for k in data.keys())
            query = f"UPDATE {table} SET {set_clause} WHERE id = ?"

            async with self._connection.cursor() as cursor:
                await cursor.execute(query, tuple(data.values()) + (id_,))
                await self._connection.commit()
        except Exception as e:
            raise DatabaseError(f"Error updating row: {str(e)}")

    async def delete(self, table: str, id_: str) -> None:
        """Delete a row from a table.

        Args:
            table (str): The table to delete from.
            id_ (str): The ID of the row to delete.

        Raises:
            DatabaseError: If there is an error deleting the row.
        """
        try:
            query = f"DELETE FROM {table} WHERE id = ?"

            async with self._connection.cursor() as cursor:
                await cursor.execute(query, (id_,))
                await self._connection.commit()
        except Exception as e:
            raise DatabaseError(f"Error deleting row: {str(e)}")

    def __del__(self):
        """Ensure connection is closed when object is deleted."""
        try:
            if hasattr(self, '_connection') and self._connection is not None:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.disconnect())
                    else:
                        asyncio.run(self.disconnect())
                except Exception:
                    pass
        except Exception:
            pass
