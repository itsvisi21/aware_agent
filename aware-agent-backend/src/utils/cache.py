import json
import logging
from typing import Optional, Any, Callable

import redis.asyncio as redis

from src.config.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class CacheService:
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.config = settings

    async def connect(self):
        """Initialize Redis connection"""
        try:
            self.redis = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True,
                socket_timeout=self.config.agent_timeout,
                retry_on_timeout=True
            )
            await self.redis.ping()
            logger.info("Redis connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {str(e)}")
            raise

    async def disconnect(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")

    async def close(self):
        """Alias for disconnect()"""
        await self.disconnect()

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

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        try:
            ttl = ttl or self.config.cache_ttl
            await self.redis.set(key, json.dumps(value), ex=ttl)
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

    async def clear(self) -> bool:
        """Clear all values from cache"""
        try:
            await self.redis.flushdb()
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            return False

    async def get_or_set(self, key: str, value_func: Callable[[], Any], ttl: Optional[int] = None) -> Any:
        """Get value from cache or compute and set if not exists"""
        try:
            value = await self.get(key)
            if value is None:
                value = await value_func()
                await self.set(key, value, ttl)
            return value
        except Exception as e:
            logger.error(f"Failed to get or set value in cache: {str(e)}")
            return None
