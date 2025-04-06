import pytest
from src.agents.research_agent import ResearchAgent

@pytest.fixture
def research_agent():
    return ResearchAgent()

@pytest.mark.asyncio
async def test_process_message(research_agent):
    test_message = {
        'id': '1',
        'content': 'Research quantum computing',
        'sender': 'user',
        'timestamp': 1234567890,
        'metadata': {
            'agent': 'Researcher',
            'role': 'Research and Analysis',
            'mode': 'research'
        }
    }

    response = await research_agent.process_message(test_message)

    assert response['sender'] == 'agent'
    assert response['metadata']['agent'] == 'Researcher'
    assert response['metadata']['mode'] == 'research'
    assert 'quantum' in research_agent.research_topics
    assert 'computing' in research_agent.research_topics

@pytest.mark.asyncio
async def test_research_topics_tracking(research_agent):
    messages = [
        {
            'id': '1',
            'content': 'Research quantum computing',
            'sender': 'user',
            'timestamp': 1234567890,
            'metadata': {'mode': 'research'}
        },
        {
            'id': '2',
            'content': 'More about quantum mechanics',
            'sender': 'user',
            'timestamp': 1234567891,
            'metadata': {'mode': 'research'}
        }
    ]

    for message in messages:
        await research_agent.process_message(message)

    assert research_agent.research_topics['quantum']['count'] == 2
    assert 'mechanics' in research_agent.research_topics
    assert len(research_agent.conversation_history) == 2

@pytest.mark.asyncio
async def test_context_management(research_agent):
    test_message = {
        'id': '1',
        'content': 'Research AI ethics',
        'sender': 'user',
        'timestamp': 1234567890,
        'metadata': {'mode': 'research'}
    }

    await research_agent.process_message(test_message)
    context = research_agent.get_context()

    assert 'state' in context
    assert 'recent_history' in context
    assert len(context['recent_history']) == 1
    assert context['state']['last_topic'] == 'Research AI ethics'

def test_reset_functionality(research_agent):
    research_agent.research_topics['test'] = {'count': 1, 'last_mentioned': 'test'}
    research_agent.conversation_history.append({'test': 'message'})
    research_agent.state['test'] = 'value'

    research_agent.reset()

    assert len(research_agent.research_topics) == 0
    assert len(research_agent.conversation_history) == 0
    assert len(research_agent.state) == 0 