from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import asyncio
from services.cache import CacheService
from services.connection_pool import ConnectionPool
from services.message_batcher import MessageBatcher
from deploy.monitoring import MonitoringService
from deploy.config import DeploymentConfig

app = FastAPI()
config = DeploymentConfig()
monitoring = MonitoringService(config)
cache = CacheService()
db_pool = ConnectionPool()
message_batcher = MessageBatcher()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await cache.initialize()
    await db_pool.initialize()

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check endpoint"""
    try:
        # Check cache service
        cache_ok = await _check_cache()
        
        # Check database connection
        db_ok = await _check_database()
        
        # Check message batcher
        batcher_ok = await _check_message_batcher()
        
        # Get system metrics
        metrics = monitoring.get_metrics()
        
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
        monitoring.log_error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))

async def _check_cache() -> bool:
    """Check cache service health"""
    try:
        # Test cache operations
        test_key = "health_check"
        test_value = {"timestamp": "now"}
        
        # Set and get value
        await cache.set(test_key, test_value, ttl=5)
        value = await cache.get(test_key)
        
        # Clean up
        await cache.delete(test_key)
        
        return value == test_value
    except Exception as e:
        monitoring.log_error(f"Cache health check failed: {str(e)}")
        return False

async def _check_database() -> bool:
    """Check database connection health"""
    try:
        # Test database connection
        conn = await db_pool.get_pg_connection()
        try:
            # Execute simple query
            result = await conn.fetch("SELECT 1")
            return len(result) == 1 and result[0][0] == 1
        finally:
            await db_pool.release_pg_connection(conn)
    except Exception as e:
        monitoring.log_error(f"Database health check failed: {str(e)}")
        return False

async def _check_message_batcher() -> bool:
    """Check message batcher health"""
    try:
        # Test message batching
        test_message = {"type": "health_check", "content": "test"}
        await message_batcher.add_message("health_check", test_message)
        
        # Get queue size
        queue_size = await message_batcher.get_queue_size()
        
        # Clean up
        await message_batcher.clear_messages("health_check")
        
        return "health_check" in queue_size
    except Exception as e:
        monitoring.log_error(f"Message batcher health check failed: {str(e)}")
        return False

@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get system metrics"""
    try:
        return monitoring.get_metrics()
    except Exception as e:
        monitoring.log_error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get system status"""
    try:
        return {
            "status": "operational",
            "version": "1.0.0",
            "uptime": monitoring.get_uptime(),
            "last_health_check": monitoring.get_last_health_check()
        }
    except Exception as e:
        monitoring.log_error(f"Failed to get status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 