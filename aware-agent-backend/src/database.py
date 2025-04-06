from typing import Optional, Dict, Any
import asyncpg
import logging

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        
    async def connect(self):
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                host="localhost",
                port=5432,
                user="postgres",
                password="postgres",
                database="aware_agent"
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {str(e)}")
            raise
            
    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
            
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
            
    async def execute_query(self, query: str, *args) -> list:
        """Execute a database query"""
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)
            
    async def execute_command(self, command: str, *args) -> None:
        """Execute a database command"""
        async with self.pool.acquire() as conn:
            await conn.execute(command, *args) 