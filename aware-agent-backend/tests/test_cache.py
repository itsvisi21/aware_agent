import pytest
import asyncio
from datetime import datetime, timedelta
from src.services.cache import CacheService

@pytest.fixture
def cache_service():
    return CacheService(ttl=1)  # Use 1 second TTL for testing

@pytest.mark.asyncio
async def test_cache_set_get(cache_service):
    # Test basic set and get operations
    await cache_service.set('test_key', 'test_value')
    value = await cache_service.get('test_key')
    assert value == 'test_value'

@pytest.mark.asyncio
async def test_cache_expiration(cache_service):
    # Test cache expiration
    await cache_service.set('test_key', 'test_value')
    await asyncio.sleep(1.1)  # Wait for TTL to expire
    value = await cache_service.get('test_key')
    assert value is None

@pytest.mark.asyncio
async def test_cache_delete(cache_service):
    # Test cache deletion
    await cache_service.set('test_key', 'test_value')
    await cache_service.delete('test_key')
    value = await cache_service.get('test_key')
    assert value is None

@pytest.mark.asyncio
async def test_cache_clear(cache_service):
    # Test cache clearing
    await cache_service.set('key1', 'value1')
    await cache_service.set('key2', 'value2')
    await cache_service.clear()
    assert await cache_service.get('key1') is None
    assert await cache_service.get('key2') is None

@pytest.mark.asyncio
async def test_cache_get_or_set(cache_service):
    # Test get_or_set functionality
    async def value_func():
        return 'computed_value'
    
    # First call should compute and cache the value
    value = await cache_service.get_or_set('test_key', value_func)
    assert value == 'computed_value'
    
    # Second call should return cached value
    value = await cache_service.get_or_set('test_key', value_func)
    assert value == 'computed_value'

@pytest.mark.asyncio
async def test_cache_concurrent_access(cache_service):
    # Test concurrent access to cache
    async def set_value(key: str, value: str):
        await cache_service.set(key, value)
    
    # Set multiple values concurrently
    tasks = [
        set_value(f'key{i}', f'value{i}')
        for i in range(10)
    ]
    await asyncio.gather(*tasks)
    
    # Verify all values were set correctly
    for i in range(10):
        value = await cache_service.get(f'key{i}')
        assert value == f'value{i}'

@pytest.mark.asyncio
async def test_cache_custom_ttl(cache_service):
    # Test custom TTL
    await cache_service.set('test_key', 'test_value', ttl=2)
    await asyncio.sleep(1.1)  # Wait for default TTL to expire
    value = await cache_service.get('test_key')
    assert value == 'test_value'  # Should still be valid
    
    await asyncio.sleep(1)  # Wait for custom TTL to expire
    value = await cache_service.get('test_key')
    assert value is None 