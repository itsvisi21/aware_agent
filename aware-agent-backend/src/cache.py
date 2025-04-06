from typing import Optional, Dict, Any
import aioredis
import logging
import json

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        
    async def connect(self):
        """Initialize Redis connection"""
        try:
            self.redis = await aioredis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("Redis connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {str(e)}")
            raise
            
    async def disconnect(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")
            
    async def health_check(self) -> bool:
        """Check Redis health"""
        try:
            await self.redis.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return False
            
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = await self.redis.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Failed to get value from cache: {str(e)}")
            return None
            
    async def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Set value in cache"""
        try:
            await self.redis.set(key, json.dumps(value), ex=expire)
            return True
        except Exception as e:
            logger.error(f"Failed to set value in cache: {str(e)}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete value from cache: {str(e)}")
            return False 