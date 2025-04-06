import pytest
from src.agents.research_agent import ResearchAgent
from src.agents.builder_agent import BuilderAgent
from src.agents.teacher_agent import TeacherAgent
from src.agents.collaborator_agent import CollaboratorAgent
from src.services.database import DatabaseService

@pytest.fixture
def agents():
    return {
        'research': ResearchAgent('research', 'research'),
        'builder': BuilderAgent('builder', 'builder'),
        'teacher': TeacherAgent('teacher', 'teacher'),
        'collaborator': CollaboratorAgent('collaborator', 'collaborator')
    }

@pytest.fixture
def db_service():
    return DatabaseService()

@pytest.mark.asyncio
async def test_research_to_builder_flow(agents, db_service):
    # Research phase
    research_msg = {
        'content': 'Research authentication systems for Python web applications',
        'sender': 'user'
    }
    research_response = await agents['research'].process_message(research_msg)
    
    # Builder phase
    builder_msg = {
        'content': 'Implement the authentication system based on the research',
        'sender': 'user',
        'context': research_response
    }
    builder_response = await agents['builder'].process_message(builder_msg)
    
    assert 'implementation' in builder_response['content'].lower()
    assert 'authentication' in builder_response['content'].lower()

@pytest.mark.asyncio
async def test_teacher_to_collaborator_flow(agents, db_service):
    # Teacher phase
    teacher_msg = {
        'content': 'Explain how to implement secure authentication',
        'sender': 'user'
    }
    teacher_response = await agents['teacher'].process_message(teacher_msg)
    
    # Collaborator phase
    collaborator_msg = {
        'content': 'Coordinate the team to implement the authentication system',
        'sender': 'user',
        'context': teacher_response
    }
    collaborator_response = await agents['collaborator'].process_message(collaborator_msg)
    
    assert 'team' in collaborator_response['content'].lower()
    assert 'tasks' in collaborator_response['content'].lower()

@pytest.mark.asyncio
async def test_semantic_understanding_integration(agents, db_service):
    # Test semantic understanding across agents
    messages = [
        'Research OAuth 2.0 implementation',
        'Explain the OAuth flow',
        'Create an OAuth provider',
        'Coordinate the OAuth implementation'
    ]
    
    for msg in messages:
        semantic_analysis = await agents['research'].analyze_message(msg)
        assert 'OAuth' in semantic_analysis['entities']
        assert len(semantic_analysis['context']['topics']) > 0

@pytest.mark.asyncio
async def test_persistence_integration(agents, db_service):
    # Test persistence across agent interactions
    msg = {
        'content': 'Research and implement secure authentication',
        'sender': 'user'
    }
    
    # Process message through research agent
    research_response = await agents['research'].process_message(msg)
    
    # Verify persistence
    research_state = await db_service.get_agent_state('research')
    assert research_state is not None
    assert 'topics' in research_state
    
    # Process message through builder agent
    builder_msg = {
        'content': 'Implement the authentication system',
        'sender': 'user',
        'context': research_response
    }
    builder_response = await agents['builder'].process_message(builder_msg)
    
    # Verify persistence
    builder_state = await db_service.get_agent_state('builder')
    assert builder_state is not None
    assert 'components' in builder_state

@pytest.mark.asyncio
async def test_context_persistence(agents, db_service):
    # Test context persistence across agent interactions
    initial_msg = {
        'content': 'Research authentication systems',
        'sender': 'user'
    }
    
    # Process through research agent
    research_response = await agents['research'].process_message(initial_msg)
    
    # Process through teacher agent with context
    teacher_msg = {
        'content': 'Explain the authentication system',
        'sender': 'user',
        'context': research_response
    }
    teacher_response = await agents['teacher'].process_message(teacher_msg)
    
    # Verify context persistence
    assert 'authentication' in teacher_response['content'].lower()
    assert 'research' in teacher_response['content'].lower() 