import asyncio
from typing import Optional, Dict, Any

import asyncpg
from deploy.config import DeploymentConfig
from deploy.monitoring import MonitoringService
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


class ConnectionPool:
    def __init__(self):
        self.config = DeploymentConfig()
        self.monitoring = MonitoringService()
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.sqlalchemy_engine = None
        self.sqlalchemy_session = None
        self._init_task = None

    async def initialize(self):
        """Initialize connection pools"""
        if not self._init_task:
            self._init_task = asyncio.create_task(self._initialize_pools())
        await self._init_task

    async def _initialize_pools(self):
        """Initialize database connection pools with retry logic"""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                # Initialize PostgreSQL connection pool
                self.pg_pool = await asyncpg.create_pool(
                    self.config.database_url,
                    min_size=5,
                    max_size=20,
                    command_timeout=60
                )

                # Initialize SQLAlchemy engine and session
                self.sqlalchemy_engine = create_async_engine(
                    self.config.database_url,
                    pool_size=20,
                    max_overflow=10,
                    pool_timeout=30,
                    pool_recycle=1800
                )

                self.sqlalchemy_session = sessionmaker(
                    self.sqlalchemy_engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )

                self.monitoring.log_info("Database connection pools established")
                return

            except Exception as e:
                if attempt == max_retries - 1:
                    self.monitoring.log_error(f"Failed to initialize connection pools: {str(e)}")
                    raise
                await asyncio.sleep(retry_delay * (attempt + 1))

    async def get_pg_connection(self):
        """Get a PostgreSQL connection from the pool"""
        try:
            if not self.pg_pool:
                await self.initialize()
            return await self.pg_pool.acquire()
        except Exception as e:
            self.monitoring.log_error(f"Failed to acquire PostgreSQL connection: {str(e)}")
            raise

    async def release_pg_connection(self, conn):
        """Release a PostgreSQL connection back to the pool"""
        try:
            if self.pg_pool:
                await self.pg_pool.release(conn)
        except Exception as e:
            self.monitoring.log_error(f"Failed to release PostgreSQL connection: {str(e)}")

    async def get_sqlalchemy_session(self):
        """Get a SQLAlchemy session"""
        try:
            if not self.sqlalchemy_session:
                await self.initialize()
            return self.sqlalchemy_session()
        except Exception as e:
            self.monitoring.log_error(f"Failed to create SQLAlchemy session: {str(e)}")
            raise

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None):
        """Execute a query using the connection pool"""
        conn = None
        try:
            conn = await self.get_pg_connection()
            return await conn.fetch(query, *(params or {}).values())
        finally:
            if conn:
                await self.release_pg_connection(conn)

    async def execute_transaction(self, queries: list):
        """Execute multiple queries in a transaction"""
        conn = None
        try:
            conn = await self.get_pg_connection()
            async with conn.transaction():
                results = []
                for query, params in queries:
                    result = await conn.fetch(query, *(params or {}).values())
                    results.append(result)
                return results
        finally:
            if conn:
                await self.release_pg_connection(conn)

    async def close(self):
        """Close all connection pools"""
        try:
            if self.pg_pool:
                await self.pg_pool.close()
            if self.sqlalchemy_engine:
                await self.sqlalchemy_engine.dispose()
        except Exception as e:
            self.monitoring.log_error(f"Failed to close connection pools: {str(e)}")
