import pytest
from src.agents.collaborator_agent import CollaboratorAgent

@pytest.fixture
def collaborator_agent():
    return CollaboratorAgent()

@pytest.mark.asyncio
async def test_process_message(collaborator_agent):
    test_message = {
        'id': '1',
        'content': 'Add team member John',
        'sender': 'user',
        'timestamp': 1234567890,
        'metadata': {
            'agent': 'Collaborator',
            'role': 'Team Coordination',
            'mode': 'collab'
        }
    }

    response = await collaborator_agent.process_message(test_message)

    assert response['sender'] == 'agent'
    assert response['metadata']['agent'] == 'Collaborator'
    assert response['metadata']['mode'] == 'collab'
    assert 'John' in collaborator_agent.team_members
    assert collaborator_agent.team_members['John']['role'] == 'team member'

@pytest.mark.asyncio
async def test_task_tracking(collaborator_agent):
    messages = [
        {
            'id': '1',
            'content': 'Create task implement feature X',
            'sender': 'user',
            'timestamp': 1234567890,
            'metadata': {'mode': 'collab'}
        },
        {
            'id': '2',
            'content': 'Update progress on implement feature X',
            'sender': 'user',
            'timestamp': 1234567891,
            'metadata': {'mode': 'collab'}
        }
    ]

    for message in messages:
        await collaborator_agent.process_message(message)

    assert 'implement feature X' in collaborator_agent.tasks
    assert 'implement feature X' in collaborator_agent.progress_tracking
    assert len(collaborator_agent.conversation_history) == 2

@pytest.mark.asyncio
async def test_team_management(collaborator_agent):
    test_message = {
        'id': '1',
        'content': 'Add team member Alice',
        'sender': 'user',
        'timestamp': 1234567890,
        'metadata': {'mode': 'collab'}
    }

    await collaborator_agent.process_message(test_message)
    assert 'Alice' in collaborator_agent.team_members
    assert collaborator_agent.team_members['Alice']['tasks_assigned'] == []

@pytest.mark.asyncio
async def test_context_management(collaborator_agent):
    test_message = {
        'id': '1',
        'content': 'Create task review code',
        'sender': 'user',
        'timestamp': 1234567890,
        'metadata': {'mode': 'collab'}
    }

    await collaborator_agent.process_message(test_message)
    context = collaborator_agent.get_context()

    assert 'state' in context
    assert 'recent_history' in context
    assert len(context['recent_history']) == 1
    assert 'review code' in context['state']['active_tasks']

def test_reset_functionality(collaborator_agent):
    collaborator_agent.team_members['test'] = {'role': 'test', 'tasks_assigned': []}
    collaborator_agent.tasks['test'] = {'status': 'pending', 'assigned_to': None, 'progress': 0}
    collaborator_agent.progress_tracking['test'] = {'last_update': 'test', 'status': 'in_progress'}
    collaborator_agent.conversation_history.append({'test': 'message'})
    collaborator_agent.state['test'] = 'value'

    collaborator_agent.reset()

    assert len(collaborator_agent.team_members) == 0
    assert len(collaborator_agent.tasks) == 0
    assert len(collaborator_agent.progress_tracking) == 0
    assert len(collaborator_agent.conversation_history) == 0
    assert len(collaborator_agent.state) == 0 