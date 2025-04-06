import pytest
import asyncio
import time
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
async def test_message_throughput(websocket_manager, database, cache):
    """Test message processing throughput"""
    start_time = time.time()
    message_count = 1000
    
    async def send_messages():
        async with websocket_manager.connect() as ws:
            for _ in range(message_count):
                await ws.send_json({
                    "type": "message",
                    "content": "Test message",
                    "agent": "research"
                })
                await ws.receive_json()
    
    await send_messages()
    end_time = time.time()
    
    # Calculate throughput
    duration = end_time - start_time
    throughput = message_count / duration
    
    # Verify performance
    assert throughput > 100  # Minimum 100 messages per second
    assert duration < 10  # Maximum 10 seconds for 1000 messages

@pytest.mark.asyncio
async def test_concurrent_connections(websocket_manager, database, cache):
    """Test system performance with multiple concurrent connections"""
    connection_count = 50
    messages_per_connection = 20
    
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
    assert throughput > 50  # Minimum 50 messages per second
    assert duration < 20  # Maximum 20 seconds for 1000 messages

@pytest.mark.asyncio
async def test_database_performance(database):
    """Test database performance"""
    # Test write performance
    start_time = time.time()
    for i in range(100):
        await database.save_conversation(f"test_conversation_{i}", [])
    write_duration = time.time() - start_time
    
    # Test read performance
    start_time = time.time()
    for i in range(100):
        await database.get_conversation(f"test_conversation_{i}")
    read_duration = time.time() - start_time
    
    # Verify performance
    assert write_duration < 5  # Maximum 5 seconds for 100 writes
    assert read_duration < 5  # Maximum 5 seconds for 100 reads

@pytest.mark.asyncio
async def test_cache_performance(cache):
    """Test cache performance"""
    # Test write performance
    start_time = time.time()
    for i in range(1000):
        await cache.set(f"test_key_{i}", f"test_value_{i}")
    write_duration = time.time() - start_time
    
    # Test read performance
    start_time = time.time()
    for i in range(1000):
        await cache.get(f"test_key_{i}")
    read_duration = time.time() - start_time
    
    # Verify performance
    assert write_duration < 1  # Maximum 1 second for 1000 writes
    assert read_duration < 1  # Maximum 1 second for 1000 reads

@pytest.mark.asyncio
async def test_memory_usage(websocket_manager, database, cache):
    """Test system memory usage under load"""
    import psutil
    process = psutil.Process()
    
    # Get initial memory usage
    initial_memory = process.memory_info().rss
    
    # Simulate load
    async def simulate_load():
        async with websocket_manager.connect() as ws:
            for _ in range(100):
                await ws.send_json({
                    "type": "message",
                    "content": "Test message",
                    "agent": "research"
                })
                await ws.receive_json()
    
    tasks = [simulate_load() for _ in range(10)]
    await asyncio.gather(*tasks)
    
    # Get final memory usage
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Verify memory usage
    assert memory_increase < 100 * 1024 * 1024  # Maximum 100MB increase

@pytest.mark.asyncio
async def test_response_time(websocket_manager, database, cache):
    """Test system response time"""
    response_times = []
    
    async with websocket_manager.connect() as ws:
        for _ in range(100):
            start_time = time.time()
            await ws.send_json({
                "type": "message",
                "content": "Test message",
                "agent": "research"
            })
            await ws.receive_json()
            end_time = time.time()
            response_times.append(end_time - start_time)
    
    # Calculate statistics
    avg_response_time = sum(response_times) / len(response_times)
    max_response_time = max(response_times)
    
    # Verify response times
    assert avg_response_time < 0.5  # Average response time under 500ms
    assert max_response_time < 1.0  # Maximum response time under 1 second 