import asyncio
import json
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import aioredis
from deploy.config import DeploymentConfig
from deploy.monitoring import MonitoringService

class CacheService:
    def __init__(self):
        self.config = DeploymentConfig()
        self.monitoring = MonitoringService(self.config)
        self.redis = None
        self._init_task = None
        
    async def initialize(self):
        """Initialize Redis connection"""
        if not self._init_task:
            self._init_task = asyncio.create_task(self._initialize_redis())
        await self._init_task
        
    async def _initialize_redis(self):
        """Initialize Redis connection with retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                self.redis = await aioredis.from_url(
                    self.config.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.redis.ping()
                self.monitoring.log_info("Redis connection established")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    self.monitoring.log_error(f"Failed to connect to Redis: {str(e)}")
                    raise
                await asyncio.sleep(retry_delay * (attempt + 1))
                
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if not self.redis:
                await self.initialize()
                
            value = await self.redis.get(key)
            if value:
                self.monitoring.track_cache_hit()
                return json.loads(value)
            self.monitoring.track_cache_miss()
            return None
        except Exception as e:
            self.monitoring.log_error(f"Cache get error: {str(e)}")
            return None
            
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        try:
            if not self.redis:
                await self.initialize()
                
            serialized = json.dumps(value)
            if ttl:
                await self.redis.setex(key, ttl, serialized)
            else:
                await self.redis.set(key, serialized)
            return True
        except Exception as e:
            self.monitoring.log_error(f"Cache set error: {str(e)}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if not self.redis:
                await self.initialize()
                
            await self.redis.delete(key)
            return True
        except Exception as e:
            self.monitoring.log_error(f"Cache delete error: {str(e)}")
            return False
            
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            if not self.redis:
                await self.initialize()
                
            await self.redis.flushdb()
            return True
        except Exception as e:
            self.monitoring.log_error(f"Cache clear error: {str(e)}")
            return False
            
    async def get_many(self, keys: list) -> Dict[str, Any]:
        """Get multiple values from cache"""
        try:
            if not self.redis:
                await self.initialize()
                
            values = await self.redis.mget(keys)
            result = {}
            for key, value in zip(keys, values):
                if value:
                    result[key] = json.loads(value)
            return result
        except Exception as e:
            self.monitoring.log_error(f"Cache get_many error: {str(e)}")
            return {}
            
    async def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache"""
        try:
            if not self.redis:
                await self.initialize()
                
            pipeline = self.redis.pipeline()
            for key, value in items.items():
                serialized = json.dumps(value)
                if ttl:
                    pipeline.setex(key, ttl, serialized)
                else:
                    pipeline.set(key, serialized)
            await pipeline.execute()
            return True
        except Exception as e:
            self.monitoring.log_error(f"Cache set_many error: {str(e)}")
            return False
            
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            self.redis = None 