import pytest
import asyncio
import os
import shutil
from datetime import datetime
from src.database import DatabaseService
from src.cache import CacheService
from src.monitoring import MonitoringService

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

@pytest.fixture
def backup_dir():
    backup_dir = "test_backups"
    os.makedirs(backup_dir, exist_ok=True)
    yield backup_dir
    shutil.rmtree(backup_dir)

@pytest.mark.asyncio
async def test_database_backup(database, backup_dir):
    """Test database backup procedures"""
    # Create test data
    test_data = {
        "conversation_id": "test_conversation",
        "messages": [
            {"role": "user", "content": "Test message 1"},
            {"role": "assistant", "content": "Test response 1"}
        ]
    }
    
    # Save test data
    await database.save_conversation(test_data["conversation_id"], test_data["messages"])
    
    # Create backup
    backup_file = os.path.join(backup_dir, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    await database.backup(backup_file)
    
    # Verify backup file
    assert os.path.exists(backup_file)
    assert os.path.getsize(backup_file) > 0

@pytest.mark.asyncio
async def test_database_restore(database, backup_dir):
    """Test database restore procedures"""
    # Create and save test data
    test_data = {
        "conversation_id": "test_conversation",
        "messages": [
            {"role": "user", "content": "Test message 1"},
            {"role": "assistant", "content": "Test response 1"}
        ]
    }
    await database.save_conversation(test_data["conversation_id"], test_data["messages"])
    
    # Create backup
    backup_file = os.path.join(backup_dir, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    await database.backup(backup_file)
    
    # Delete test data
    await database.delete_conversation(test_data["conversation_id"])
    
    # Restore from backup
    await database.restore(backup_file)
    
    # Verify restored data
    restored_data = await database.get_conversation(test_data["conversation_id"])
    assert restored_data == test_data["messages"]

@pytest.mark.asyncio
async def test_cache_backup(cache, backup_dir):
    """Test cache backup procedures"""
    # Create test data
    test_data = {
        "key": "test_key",
        "value": "test_value"
    }
    
    # Save test data
    await cache.set(test_data["key"], test_data["value"])
    
    # Create backup
    backup_file = os.path.join(backup_dir, f"cache_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    await cache.backup(backup_file)
    
    # Verify backup file
    assert os.path.exists(backup_file)
    assert os.path.getsize(backup_file) > 0

@pytest.mark.asyncio
async def test_cache_restore(cache, backup_dir):
    """Test cache restore procedures"""
    # Create and save test data
    test_data = {
        "key": "test_key",
        "value": "test_value"
    }
    await cache.set(test_data["key"], test_data["value"])
    
    # Create backup
    backup_file = os.path.join(backup_dir, f"cache_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    await cache.backup(backup_file)
    
    # Delete test data
    await cache.delete(test_data["key"])
    
    # Restore from backup
    await cache.restore(backup_file)
    
    # Verify restored data
    restored_value = await cache.get(test_data["key"])
    assert restored_value == test_data["value"]

@pytest.mark.asyncio
async def test_incremental_backup(database, backup_dir):
    """Test incremental backup procedures"""
    # Create initial data
    initial_data = {
        "conversation_id": "test_conversation_1",
        "messages": [
            {"role": "user", "content": "Test message 1"},
            {"role": "assistant", "content": "Test response 1"}
        ]
    }
    await database.save_conversation(initial_data["conversation_id"], initial_data["messages"])
    
    # Create initial backup
    initial_backup = os.path.join(backup_dir, f"backup_initial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    await database.backup(initial_backup)
    
    # Add more data
    additional_data = {
        "conversation_id": "test_conversation_2",
        "messages": [
            {"role": "user", "content": "Test message 2"},
            {"role": "assistant", "content": "Test response 2"}
        ]
    }
    await database.save_conversation(additional_data["conversation_id"], additional_data["messages"])
    
    # Create incremental backup
    incremental_backup = os.path.join(backup_dir, f"backup_incremental_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    await database.backup(incremental_backup, incremental=True)
    
    # Verify incremental backup
    assert os.path.exists(incremental_backup)
    assert os.path.getsize(incremental_backup) < os.path.getsize(initial_backup)

@pytest.mark.asyncio
async def test_point_in_time_recovery(database, backup_dir):
    """Test point-in-time recovery"""
    # Create sequence of data
    sequence = [
        {
            "conversation_id": f"test_conversation_{i}",
            "messages": [
                {"role": "user", "content": f"Test message {i}"},
                {"role": "assistant", "content": f"Test response {i}"}
            ]
        }
        for i in range(5)
    ]
    
    # Save data and create backups
    backups = []
    for data in sequence:
        await database.save_conversation(data["conversation_id"], data["messages"])
        backup_file = os.path.join(backup_dir, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        await database.backup(backup_file)
        backups.append(backup_file)
    
    # Test recovery to different points
    for i, backup in enumerate(backups):
        # Restore from backup
        await database.restore(backup)
        
        # Verify data at this point
        for j in range(i + 1):
            data = await database.get_conversation(f"test_conversation_{j}")
            assert data == sequence[j]["messages"]
        
        # Verify data after this point doesn't exist
        for j in range(i + 1, len(sequence)):
            with pytest.raises(Exception):
                await database.get_conversation(f"test_conversation_{j}")

@pytest.mark.asyncio
async def test_backup_encryption(database, backup_dir):
    """Test backup encryption"""
    # Create test data
    test_data = {
        "conversation_id": "test_conversation",
        "messages": [
            {"role": "user", "content": "Test message"},
            {"role": "assistant", "content": "Test response"}
        ]
    }
    await database.save_conversation(test_data["conversation_id"], test_data["messages"])
    
    # Create encrypted backup
    backup_file = os.path.join(backup_dir, f"encrypted_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    await database.backup(backup_file, encrypt=True)
    
    # Verify encrypted backup
    assert os.path.exists(backup_file)
    with open(backup_file, 'rb') as f:
        content = f.read()
        assert b'BEGIN ENCRYPTED' in content

@pytest.mark.asyncio
async def test_backup_compression(database, backup_dir):
    """Test backup compression"""
    # Create large test data
    test_data = {
        "conversation_id": "test_conversation",
        "messages": [
            {"role": "user", "content": "Test message " * 1000},
            {"role": "assistant", "content": "Test response " * 1000}
        ]
    }
    await database.save_conversation(test_data["conversation_id"], test_data["messages"])
    
    # Create compressed backup
    backup_file = os.path.join(backup_dir, f"compressed_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    await database.backup(backup_file, compress=True)
    
    # Verify compressed backup
    assert os.path.exists(backup_file)
    assert backup_file.endswith('.gz') 