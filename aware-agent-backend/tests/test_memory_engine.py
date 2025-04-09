import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
from src.core.services.memory_engine import MemoryEngine
from src.core.models.models import Message, ConversationState

@pytest.fixture
def memory_engine(tmp_path):
    """Create a memory engine with a temporary storage path."""
    return MemoryEngine(storage_path=str(tmp_path))

@pytest.mark.asyncio
async def test_memory_engine_initialization(memory_engine, tmp_path):
    """Test memory engine initialization."""
    assert memory_engine.storage_path == Path(tmp_path)
    assert memory_engine.memory_store == {}

@pytest.mark.asyncio
async def test_message_storage(memory_engine):
    """Test storing and retrieving messages."""
    # Create test message
    test_message = Message(
        role="user",
        content="test message",
        type="text",
        status="success"
    )
    
    # Store message
    await memory_engine.store_message("test_conversation", test_message)
    
    # Retrieve message
    messages = await memory_engine.get_messages("test_conversation")
    assert len(messages) == 1
    assert messages[0].content == "test message"
    assert messages[0].role == "user"

@pytest.mark.asyncio
async def test_conversation_state_management(memory_engine):
    """Test conversation state management."""
    # Create test conversation state
    test_state = ConversationState(
        id="test_conversation",
        messages=[],
        context_tree={},
        current_goal="test goal",
        feedback_history=[],
        metadata={},
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    # Store state
    await memory_engine.store_conversation_state(test_state)
    
    # Retrieve state
    retrieved_state = await memory_engine.get_conversation_state("test_conversation")
    assert retrieved_state.id == "test_conversation"
    assert retrieved_state.current_goal == "test goal"

@pytest.mark.asyncio
async def test_memory_persistence(memory_engine, tmp_path):
    """Test memory persistence to disk."""
    # Create and store test data
    test_message = Message(
        role="user",
        content="test message",
        type="text",
        status="success"
    )
    await memory_engine.store_message("test_conversation", test_message)
    
    # Create new instance and verify persistence
    new_engine = MemoryEngine(storage_path=str(tmp_path))
    messages = await new_engine.get_messages("test_conversation")
    assert len(messages) == 1
    assert messages[0].content == "test message"

@pytest.mark.asyncio
async def test_memory_cleanup(memory_engine):
    """Test memory cleanup functionality."""
    # Store test data
    test_message = Message(
        role="user",
        content="test message",
        type="text",
        status="success"
    )
    await memory_engine.store_message("test_conversation", test_message)
    
    # Cleanup memory
    await memory_engine.cleanup()
    
    # Verify cleanup
    messages = await memory_engine.get_messages("test_conversation")
    assert len(messages) == 0

@pytest.mark.asyncio
async def test_error_handling(memory_engine):
    """Test error handling in memory operations."""
    # Test handling of non-existent conversation
    messages = await memory_engine.get_messages("non_existent")
    assert len(messages) == 0
    
    # Test handling of invalid message format
    with pytest.raises(ValueError):
        await memory_engine.store_message("test_conversation", "invalid message")
    
    # Test handling of invalid state format
    with pytest.raises(ValueError):
        await memory_engine.store_conversation_state("invalid state") 