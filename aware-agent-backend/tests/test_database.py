import pytest
import os
import json
import shutil
from src.services.database import DatabaseService

@pytest.fixture
def db_service(tmp_path):
    """Create a database service with a temporary directory."""
    service = DatabaseService(storage_dir=str(tmp_path))
    yield service
    # Cleanup
    shutil.rmtree(tmp_path)

@pytest.mark.asyncio
async def test_save_and_get_conversation(db_service):
    test_conversation = {
        'id': 'test_conv',
        'messages': [
            {'content': 'Hello', 'sender': 'user'},
            {'content': 'Hi there', 'sender': 'agent'}
        ]
    }

    # Save conversation
    conversation_id = await db_service.save_conversation(test_conversation)
    assert conversation_id == 'test_conv'

    # Retrieve conversation
    retrieved = await db_service.get_conversation('test_conv')
    assert retrieved == test_conversation

@pytest.mark.asyncio
async def test_list_conversations(db_service):
    conversations = [
        {'id': 'conv1', 'messages': [{'content': 'Test 1'}]},
        {'id': 'conv2', 'messages': [{'content': 'Test 2'}]}
    ]

    # Save multiple conversations
    for conv in conversations:
        await db_service.save_conversation(conv)

    # List conversations
    listed = await db_service.list_conversations()
    assert len(listed) == 2
    assert any(conv['id'] == 'conv1' for conv in listed)
    assert any(conv['id'] == 'conv2' for conv in listed)

@pytest.mark.asyncio
async def test_save_and_get_agent_state(db_service):
    test_state = {
        'last_action': 'test_action',
        'context': {'key': 'value'}
    }

    # Save agent state
    await db_service.save_agent_state('test_agent', test_state)

    # Retrieve agent state
    retrieved = await db_service.get_agent_state('test_agent')
    assert retrieved == test_state

@pytest.mark.asyncio
async def test_export_and_import_conversation(db_service):
    test_conversation = {
        'id': 'export_test',
        'messages': [{'content': 'Export test'}]
    }

    # Save and export conversation
    await db_service.save_conversation(test_conversation)
    exported = await db_service.export_conversation('export_test')

    # Import conversation
    new_id = await db_service.import_conversation(exported)
    imported = await db_service.get_conversation(new_id)

    assert imported['messages'] == test_conversation['messages']

@pytest.mark.asyncio
async def test_error_handling(db_service):
    # Test non-existent conversation
    with pytest.raises(FileNotFoundError):
        await db_service.get_conversation('non_existent')

    # Test invalid JSON import
    with pytest.raises(json.JSONDecodeError):
        await db_service.import_conversation('invalid json') 