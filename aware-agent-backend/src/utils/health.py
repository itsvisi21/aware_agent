import logging
import os
from datetime import datetime
from typing import Dict, Any, List

from deploy.config import DeploymentConfig
from deploy.monitoring import MonitoringService
from fastapi import FastAPI, HTTPException
from src.connection_pool import ConnectionPool
from src.message_batcher import MessageBatcher
from src.cache import CacheService


class HealthService:
    def __init__(self):
        self.config = DeploymentConfig()
        self.monitoring = MonitoringService()
        self.cache = CacheService()
        self.db_pool = ConnectionPool()
        self.message_batcher = MessageBatcher()

    async def initialize(self):
        """Initialize services"""
        await self.cache.initialize()
        await self.db_pool.initialize()

    async def check_health(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            # Check cache service
            cache_ok = await self._check_cache()

            # Check database connection
            db_ok = await self._check_database()

            # Check message batcher
            batcher_ok = await self._check_message_batcher()

            # Get system metrics
            metrics = self.monitoring.get_metrics()

            # Check overall health
            healthy = all([cache_ok, db_ok, batcher_ok])

            if not healthy:
                raise HTTPException(status_code=503, detail="Service unhealthy")

            return {
                "status": "healthy",
                "components": {
                    "cache": cache_ok,
                    "database": db_ok,
                    "message_batcher": batcher_ok
                },
                "metrics": metrics
            }

        except Exception as e:
            self.monitoring.log_error(f"Health check failed: {str(e)}")
            raise HTTPException(status_code=503, detail=str(e))

    async def _check_cache(self) -> bool:
        """Check cache service health"""
        try:
            # Test cache operations
            test_key = "health_check"
            test_value = {"timestamp": "now"}

            # Set and get value
            await self.cache.set(test_key, test_value, ttl=5)
            value = await self.cache.get(test_key)

            # Clean up
            await self.cache.delete(test_key)

            return value == test_value
        except Exception as e:
            self.monitoring.log_error(f"Cache health check failed: {str(e)}")
            return False

    async def _check_database(self) -> bool:
        """Check database connection health"""
        try:
            # Test database connection
            conn = await self.db_pool.get_pg_connection()
            try:
                # Execute simple query
                result = await conn.fetch("SELECT 1")
                return len(result) == 1 and result[0][0] == 1
            finally:
                await self.db_pool.release_pg_connection(conn)
        except Exception as e:
            self.monitoring.log_error(f"Database health check failed: {str(e)}")
            return False

    async def _check_message_batcher(self) -> bool:
        """Check message batcher health"""
        try:
            # Test message batching
            test_message = {"type": "health_check", "content": "test"}
            await self.message_batcher.add_message("health_check", test_message)

            # Get queue size
            queue_size = await self.message_batcher.get_queue_size()

            # Clean up
            await self.message_batcher.clear_messages("health_check")

            return "health_check" in queue_size
        except Exception as e:
            self.monitoring.log_error(f"Message batcher health check failed: {str(e)}")
            return False

    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        try:
            return self.monitoring.get_metrics()
        except Exception as e:
            self.monitoring.log_error(f"Failed to get metrics: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            return {
                "status": "operational",
                "version": "1.0.0",
                "uptime": self.monitoring.get_uptime(),
                "last_health_check": self.monitoring.get_last_health_check()
            }
        except Exception as e:
            self.monitoring.log_error(f"Failed to get status: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


# Create FastAPI app and initialize services
app = FastAPI()
health_service = HealthService()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await health_service.initialize()


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return await health_service.check_health()


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get system metrics"""
    return await health_service.get_metrics()


@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get system status"""
    return await health_service.get_status()
