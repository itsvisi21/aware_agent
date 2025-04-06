import pytest
from src.services.semantic_understanding import SemanticUnderstandingService

@pytest.fixture
def semantic_service():
    return SemanticUnderstandingService()

@pytest.mark.asyncio
async def test_analyze_message(semantic_service):
    message = "Create a new User class with authentication"
    result = await semantic_service.analyze_message(message)
    
    assert 'User' in result['entities']
    assert len(result['relationships']) >= 0
    assert 'create' in result['context']['actions']
    assert result['intent'] == 'command'

@pytest.mark.asyncio
async def test_entity_extraction(semantic_service):
    message = "The User and Admin classes need authentication"
    result = await semantic_service.analyze_message(message)
    
    assert 'User' in result['entities']
    assert 'Admin' in result['entities']
    assert len(result['entities']) == 2

@pytest.mark.asyncio
async def test_relationship_identification(semantic_service):
    message = "User inherits from BaseUser"
    result = await semantic_service.analyze_message(message)
    
    assert len(result['relationships']) > 0
    relationship = result['relationships'][0]
    assert relationship['source'] in ['User', 'BaseUser']
    assert relationship['target'] in ['User', 'BaseUser']

@pytest.mark.asyncio
async def test_context_determination(semantic_service):
    message = "Urgently update the authentication system"
    result = await semantic_service.analyze_message(message)
    
    assert 'update' in result['context']['actions']
    assert 'urgent' in result['context']['modifiers']
    assert 'authentication' in result['context']['topics']

@pytest.mark.asyncio
async def test_intent_determination(semantic_service):
    test_cases = [
        ("What is authentication?", 'query'),
        ("Create a new user", 'command'),
        ("Explain the system", 'explanation'),
        ("Is this secure?", 'confirmation'),
        ("The system works", 'statement')
    ]
    
    for message, expected_intent in test_cases:
        result = await semantic_service.analyze_message(message)
        assert result['intent'] == expected_intent

@pytest.mark.asyncio
async def test_cache_updates(semantic_service):
    message = "User authenticates with Password"
    await semantic_service.analyze_message(message)
    
    assert 'User' in semantic_service.entity_cache
    assert 'Password' in semantic_service.entity_cache
    assert len(semantic_service.relationship_cache) > 0
    assert semantic_service.context_cache 