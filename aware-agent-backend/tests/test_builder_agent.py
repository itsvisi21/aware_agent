import pytest
from src.agents.builder_agent import BuilderAgent

@pytest.fixture
def builder_agent():
    return BuilderAgent()

@pytest.mark.asyncio
async def test_process_message(builder_agent):
    test_message = {
        'id': '1',
        'content': 'Create a new file called main.py',
        'sender': 'user',
        'timestamp': 1234567890,
        'metadata': {
            'agent': 'Builder',
            'role': 'Implementation and Development',
            'mode': 'build'
        }
    }

    response = await builder_agent.process_message(test_message)

    assert response['sender'] == 'agent'
    assert response['metadata']['agent'] == 'Builder'
    assert response['metadata']['mode'] == 'build'
    assert 'main.py' in builder_agent.project_structure
    assert builder_agent.project_structure['main.py']['type'] == 'file'

@pytest.mark.asyncio
async def test_project_structure_tracking(builder_agent):
    messages = [
        {
            'id': '1',
            'content': 'Create a new file called main.py',
            'sender': 'user',
            'timestamp': 1234567890,
            'metadata': {'mode': 'build'}
        },
        {
            'id': '2',
            'content': 'Create a new directory called src',
            'sender': 'user',
            'timestamp': 1234567891,
            'metadata': {'mode': 'build'}
        }
    ]

    for message in messages:
        await builder_agent.process_message(message)

    assert 'main.py' in builder_agent.project_structure
    assert 'src' in builder_agent.project_structure
    assert builder_agent.project_structure['src']['type'] == 'directory'
    assert len(builder_agent.conversation_history) == 2

@pytest.mark.asyncio
async def test_dependency_tracking(builder_agent):
    test_message = {
        'id': '1',
        'content': 'Add fastapi as a dependency',
        'sender': 'user',
        'timestamp': 1234567890,
        'metadata': {'mode': 'build'}
    }

    await builder_agent.process_message(test_message)
    assert 'fastapi' in builder_agent.dependencies

@pytest.mark.asyncio
async def test_context_management(builder_agent):
    test_message = {
        'id': '1',
        'content': 'Create a new file called main.py',
        'sender': 'user',
        'timestamp': 1234567890,
        'metadata': {'mode': 'build'}
    }

    await builder_agent.process_message(test_message)
    context = builder_agent.get_context()

    assert 'state' in context
    assert 'recent_history' in context
    assert len(context['recent_history']) == 1
    assert context['state']['last_task'] == 'Create a new file called main.py'

def test_reset_functionality(builder_agent):
    builder_agent.project_structure['test.py'] = {'type': 'file', 'last_modified': 'test'}
    builder_agent.dependencies.append('test-package')
    builder_agent.conversation_history.append({'test': 'message'})
    builder_agent.state['test'] = 'value'

    builder_agent.reset()

    assert len(builder_agent.project_structure) == 0
    assert len(builder_agent.dependencies) == 0
    assert len(builder_agent.conversation_history) == 0
    assert len(builder_agent.state) == 0 