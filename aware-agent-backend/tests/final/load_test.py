import pytest
import asyncio
import time
import psutil
from src.websocket_manager import WebSocketManager
from src.database import DatabaseService
from src.cache import CacheService
from src.monitoring import MonitoringService

@pytest.fixture
async def websocket_manager():
    manager = WebSocketManager()
    await manager.start()
    yield manager
    await manager.stop()

@pytest.fixture
async def database():
    db = DatabaseService()
    await db.connect()
    yield db
    await db.disconnect()

@pytest.fixture
async def cache():
    cache = CacheService()
    await cache.connect()
    yield cache
    await cache.disconnect()

@pytest.mark.asyncio
async def test_high_concurrency(websocket_manager, database, cache):
    """Test system under high concurrency"""
    connection_count = 1000
    messages_per_connection = 10
    
    async def simulate_user():
        async with websocket_manager.connect() as ws:
            for _ in range(messages_per_connection):
                await ws.send_json({
                    "type": "message",
                    "content": "Test message",
                    "agent": "research"
                })
                await ws.receive_json()
    
    # Create concurrent connections
    start_time = time.time()
    tasks = [simulate_user() for _ in range(connection_count)]
    await asyncio.gather(*tasks)
    end_time = time.time()
    
    # Calculate performance metrics
    duration = end_time - start_time
    total_messages = connection_count * messages_per_connection
    throughput = total_messages / duration
    
    # Verify performance
    assert throughput > 1000  # Minimum 1000 messages per second
    assert duration < 60  # Maximum 60 seconds for 10000 messages

@pytest.mark.asyncio
async def test_sustained_load(websocket_manager, database, cache):
    """Test system under sustained load"""
    duration = 300  # 5 minutes
    messages_per_second = 100
    
    async def send_messages():
        async with websocket_manager.connect() as ws:
            while True:
                await ws.send_json({
                    "type": "message",
                    "content": "Test message",
                    "agent": "research"
                })
                await ws.receive_json()
                await asyncio.sleep(1 / messages_per_second)
    
    # Start sending messages
    start_time = time.time()
    task = asyncio.create_task(send_messages())
    
    # Monitor system metrics
    initial_memory = psutil.Process().memory_info().rss
    initial_cpu = psutil.Process().cpu_percent()
    
    # Wait for duration
    await asyncio.sleep(duration)
    
    # Stop sending messages
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    
    # Calculate metrics
    end_time = time.time()
    final_memory = psutil.Process().memory_info().rss
    final_cpu = psutil.Process().cpu_percent()
    
    # Verify performance
    assert final_memory - initial_memory < 500 * 1024 * 1024  # Max 500MB increase
    assert final_cpu < 80  # Max 80% CPU usage

@pytest.mark.asyncio
async def test_database_load(database):
    """Test database under load"""
    # Test bulk operations
    start_time = time.time()
    for i in range(10000):
        await database.save_conversation(f"test_conversation_{i}", [])
    write_duration = time.time() - start_time
    
    # Test concurrent reads
    async def read_conversation(i):
        await database.get_conversation(f"test_conversation_{i}")
    
    start_time = time.time()
    tasks = [read_conversation(i) for i in range(1000)]
    await asyncio.gather(*tasks)
    read_duration = time.time() - start_time
    
    # Verify performance
    assert write_duration < 30  # Max 30 seconds for 10000 writes
    assert read_duration < 10  # Max 10 seconds for 1000 concurrent reads

@pytest.mark.asyncio
async def test_cache_load(cache):
    """Test cache under load"""
    # Test bulk operations
    start_time = time.time()
    for i in range(100000):
        await cache.set(f"test_key_{i}", f"test_value_{i}")
    write_duration = time.time() - start_time
    
    # Test concurrent reads
    async def read_cache(i):
        await cache.get(f"test_key_{i}")
    
    start_time = time.time()
    tasks = [read_cache(i) for i in range(10000)]
    await asyncio.gather(*tasks)
    read_duration = time.time() - start_time
    
    # Verify performance
    assert write_duration < 30  # Max 30 seconds for 100000 writes
    assert read_duration < 5  # Max 5 seconds for 10000 concurrent reads

@pytest.mark.asyncio
async def test_mixed_workload(websocket_manager, database, cache):
    """Test system under mixed workload"""
    async def simulate_user():
        async with websocket_manager.connect() as ws:
            # Mix of different operations
            for _ in range(100):
                # Send message
                await ws.send_json({
                    "type": "message",
                    "content": "Test message",
                    "agent": "research"
                })
                await ws.receive_json()
                
                # Database operation
                await database.save_conversation("test_conversation", [])
                
                # Cache operation
                await cache.set("test_key", "test_value")
    
    # Create concurrent users
    start_time = time.time()
    tasks = [simulate_user() for _ in range(100)]
    await asyncio.gather(*tasks)
    end_time = time.time()
    
    # Calculate performance metrics
    duration = end_time - start_time
    
    # Verify performance
    assert duration < 60  # Max 60 seconds for mixed workload

@pytest.mark.asyncio
async def test_recovery_after_load(websocket_manager, database, cache):
    """Test system recovery after load"""
    # Apply heavy load
    async def apply_load():
        async with websocket_manager.connect() as ws:
            for _ in range(1000):
                await ws.send_json({
                    "type": "message",
                    "content": "Test message",
                    "agent": "research"
                })
                await ws.receive_json()
    
    tasks = [apply_load() for _ in range(10)]
    await asyncio.gather(*tasks)
    
    # Wait for system to stabilize
    await asyncio.sleep(10)
    
    # Test normal operation
    async with websocket_manager.connect() as ws:
        await ws.send_json({
            "type": "message",
            "content": "Test message",
            "agent": "research"
        })
        response = await ws.receive_json()
        assert response["type"] == "response"
    
    # Verify system metrics
    metrics = await MonitoringService.get_metrics()
    assert metrics["active_connections"] == 0
    assert metrics["error_rate"] < 0.01  # Less than 1% error rate 