import pytest
from datetime import datetime
from src.memory.persistence import MemoryStore, ConversationRecord, MessageRecord
from src.semantic_abstraction import ContextNode

@pytest.fixture
def memory_store():
    store = MemoryStore(":memory:")  # Use in-memory database for testing
    yield store

@pytest.fixture
def sample_conversation():
    return ConversationRecord(
        id="test-conv-1",
        title="Test Conversation",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        metadata={"goal": "test goal"}
    )

@pytest.fixture
def sample_message(sample_conversation):
    return MessageRecord(
        id="test-msg-1",
        conversation_id=sample_conversation.id,
        content="Test message",
        role="user",
        timestamp=datetime.utcnow(),
        metadata={}
    )

@pytest.fixture
def sample_context_tree():
    return ContextNode(
        id="test-tree-1",
        type="root",
        content={"goal": "test goal"},
        metadata={}
    )

@pytest.mark.asyncio
async def test_create_conversation(memory_store, sample_conversation):
    conversation_id = await memory_store.create_conversation(sample_conversation)
    assert conversation_id == sample_conversation.id

@pytest.mark.asyncio
async def test_add_message(memory_store, sample_conversation, sample_message):
    # First create the conversation
    await memory_store.create_conversation(sample_conversation)
    
    # Then add the message
    message_id = await memory_store.add_message(sample_message)
    assert message_id == sample_message.id

@pytest.mark.asyncio
async def test_get_conversation_history(memory_store, sample_conversation, sample_message):
    # Setup
    await memory_store.create_conversation(sample_conversation)
    await memory_store.add_message(sample_message)
    
    # Test
    history = await memory_store.get_conversation_history(sample_conversation.id)
    assert len(history) == 1
    assert history[0].id == sample_message.id
    assert history[0].content == sample_message.content

@pytest.mark.asyncio
async def test_update_and_get_context_tree(memory_store, sample_conversation, sample_context_tree):
    # Setup
    await memory_store.create_conversation(sample_conversation)
    
    # Test update
    await memory_store.update_context_tree(sample_conversation.id, sample_context_tree)
    
    # Test retrieval
    retrieved_tree = await memory_store.get_context_tree(sample_conversation.id)
    assert retrieved_tree is not None
    assert retrieved_tree.id == sample_context_tree.id
    assert retrieved_tree.type == sample_context_tree.type

@pytest.mark.asyncio
async def test_list_conversations(memory_store, sample_conversation):
    # Setup
    await memory_store.create_conversation(sample_conversation)
    
    # Test
    conversations = await memory_store.list_conversations()
    assert len(conversations) == 1
    assert conversations[0].id == sample_conversation.id
    assert conversations[0].title == sample_conversation.title 