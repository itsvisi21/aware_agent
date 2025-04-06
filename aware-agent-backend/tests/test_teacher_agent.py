import pytest
from src.agents.teacher_agent import TeacherAgent

@pytest.fixture
def teacher_agent():
    return TeacherAgent()

@pytest.mark.asyncio
async def test_process_message(teacher_agent):
    test_message = {
        'id': '1',
        'content': 'Explain quantum computing',
        'sender': 'user',
        'timestamp': 1234567890,
        'metadata': {
            'agent': 'Teacher',
            'role': 'Explanation and Education',
            'mode': 'teach'
        }
    }

    response = await teacher_agent.process_message(test_message)

    assert response['sender'] == 'agent'
    assert response['metadata']['agent'] == 'Teacher'
    assert response['metadata']['mode'] == 'teach'
    assert 'quantum computing' in teacher_agent.knowledge_base
    assert 'quantum computing' in teacher_agent.concepts_covered

@pytest.mark.asyncio
async def test_knowledge_base_tracking(teacher_agent):
    messages = [
        {
            'id': '1',
            'content': 'Explain quantum computing',
            'sender': 'user',
            'timestamp': 1234567890,
            'metadata': {'mode': 'teach'}
        },
        {
            'id': '2',
            'content': 'Teach me about machine learning',
            'sender': 'user',
            'timestamp': 1234567891,
            'metadata': {'mode': 'teach'}
        }
    ]

    for message in messages:
        await teacher_agent.process_message(message)

    assert 'quantum computing' in teacher_agent.knowledge_base
    assert 'machine learning' in teacher_agent.knowledge_base
    assert len(teacher_agent.concepts_covered) == 2
    assert len(teacher_agent.conversation_history) == 2

@pytest.mark.asyncio
async def test_learning_path_tracking(teacher_agent):
    test_message = {
        'id': '1',
        'content': 'Create a learning path for Python',
        'sender': 'user',
        'timestamp': 1234567890,
        'metadata': {'mode': 'teach'}
    }

    await teacher_agent.process_message(test_message)
    assert 'Python' in teacher_agent.learning_paths

@pytest.mark.asyncio
async def test_context_management(teacher_agent):
    test_message = {
        'id': '1',
        'content': 'Explain AI ethics',
        'sender': 'user',
        'timestamp': 1234567890,
        'metadata': {'mode': 'teach'}
    }

    await teacher_agent.process_message(test_message)
    context = teacher_agent.get_context()

    assert 'state' in context
    assert 'recent_history' in context
    assert len(context['recent_history']) == 1
    assert context['state']['last_topic'] == 'Explain AI ethics'

def test_reset_functionality(teacher_agent):
    teacher_agent.knowledge_base['test'] = {'times_explained': 1, 'last_explanation': 'test'}
    teacher_agent.learning_paths['test'] = []
    teacher_agent.concepts_covered.append('test')
    teacher_agent.conversation_history.append({'test': 'message'})
    teacher_agent.state['test'] = 'value'

    teacher_agent.reset()

    assert len(teacher_agent.knowledge_base) == 0
    assert len(teacher_agent.learning_paths) == 0
    assert len(teacher_agent.concepts_covered) == 0
    assert len(teacher_agent.conversation_history) == 0
    assert len(teacher_agent.state) == 0 