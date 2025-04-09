import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
import json
import os
import aiosqlite

class SemanticLogger:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()
        
    async def _init_db(self):
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        metadata TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                await db.commit()
        except Exception as e:
            logging.error(f"Failed to initialize semantic logs database: {e}")
            raise

    async def log(self, level: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO semantic_logs (timestamp, level, message, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    datetime.utcnow().isoformat(),
                    level,
                    message,
                    json.dumps(metadata) if metadata else None,
                    datetime.utcnow().isoformat()
                ))
                await db.commit()
        except Exception as e:
            logging.error(f"Failed to log message: {e}")
            raise

    async def get_logs(self, level: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            async with aiosqlite.connect(self.db_path) as db:
                if level:
                    async with db.execute("""
                        SELECT * FROM semantic_logs 
                        WHERE level = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (level, limit)) as cursor:
                        rows = await cursor.fetchall()
                else:
                    async with db.execute("""
                        SELECT * FROM semantic_logs 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,)) as cursor:
                        rows = await cursor.fetchall()
                return [{
                    "id": row[0],
                    "timestamp": row[1],
                    "level": row[2],
                    "message": row[3],
                    "metadata": json.loads(row[4]) if row[4] else None,
                    "created_at": row[5]
                } for row in rows]
        except Exception as e:
            logging.error(f"Failed to retrieve logs: {e}")
            return []

    async def get_logs_by_time_range(self, start_time: datetime, end_time: datetime, level: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            async with aiosqlite.connect(self.db_path) as db:
                if level:
                    async with db.execute("""
                        SELECT * FROM semantic_logs 
                        WHERE timestamp BETWEEN ? AND ? 
                        AND level = ?
                        ORDER BY timestamp DESC
                    """, (start_time.isoformat(), end_time.isoformat(), level)) as cursor:
                        rows = await cursor.fetchall()
                else:
                    async with db.execute("""
                        SELECT * FROM semantic_logs 
                        WHERE timestamp BETWEEN ? AND ?
                        ORDER BY timestamp DESC
                    """, (start_time.isoformat(), end_time.isoformat())) as cursor:
                        rows = await cursor.fetchall()
                return [{
                    "id": row[0],
                    "timestamp": row[1],
                    "level": row[2],
                    "message": row[3],
                    "metadata": json.loads(row[4]) if row[4] else None,
                    "created_at": row[5]
                } for row in rows]
        except Exception as e:
            logging.error(f"Failed to retrieve logs by time range: {e}")
            return [] 