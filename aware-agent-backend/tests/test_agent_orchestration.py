import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.core.services.agent_orchestration import AgentOrchestrator
from src.core.models.models import AgentType, TaskStatus, Task

@pytest.fixture
def agent_orchestrator():
    """Create an agent orchestrator instance."""
    return AgentOrchestrator()

@pytest.mark.asyncio
async def test_agent_registration(agent_orchestrator):
    """Test agent registration functionality."""
    # Create mock agents
    mock_research_agent = Mock()
    mock_teacher_agent = Mock()
    
    # Register agents
    agent_orchestrator.register_agent(mock_research_agent, AgentType.RESEARCH)
    agent_orchestrator.register_agent(mock_teacher_agent, AgentType.TEACHING)
    
    # Verify registration
    assert AgentType.RESEARCH in agent_orchestrator.agents
    assert AgentType.TEACHING in agent_orchestrator.agents
    assert agent_orchestrator.agents[AgentType.RESEARCH] == mock_research_agent
    assert agent_orchestrator.agents[AgentType.TEACHING] == mock_teacher_agent

@pytest.mark.asyncio
async def test_task_creation(agent_orchestrator):
    """Test task creation and management."""
    # Create a test task
    task_data = {
        "task_id": "test_task",
        "status": TaskStatus.PENDING,
        "progress": 0.0,
        "confidence": 0.0,
        "timestamp": datetime.now()
    }
    task = Task(**task_data)
    
    # Add task
    agent_orchestrator.add_task(task)
    
    # Verify task was added
    assert task.task_id in agent_orchestrator.tasks
    assert agent_orchestrator.tasks[task.task_id] == task

@pytest.mark.asyncio
async def test_task_status_update(agent_orchestrator):
    """Test task status updates."""
    # Create and add a test task
    task = Task(
        task_id="test_task",
        status=TaskStatus.PENDING,
        progress=0.0,
        confidence=0.0,
        timestamp=datetime.now()
    )
    agent_orchestrator.add_task(task)
    
    # Update task status
    await agent_orchestrator.update_task_status(
        task.task_id,
        TaskStatus.PROCESSING,
        progress=0.5
    )
    
    # Verify status update
    updated_task = agent_orchestrator.tasks[task.task_id]
    assert updated_task.status == TaskStatus.PROCESSING
    assert updated_task.progress == 0.5

@pytest.mark.asyncio
async def test_task_execution(agent_orchestrator):
    """Test task execution flow."""
    # Create mock agent
    mock_agent = Mock()
    mock_agent.execute_task = Mock(return_value={"result": "success"})
    agent_orchestrator.register_agent(mock_agent, AgentType.RESEARCH)
    
    # Create and add a test task
    task = Task(
        task_id="test_task",
        status=TaskStatus.PENDING,
        progress=0.0,
        confidence=0.0,
        timestamp=datetime.now()
    )
    agent_orchestrator.add_task(task)
    
    # Execute task
    result = await agent_orchestrator.execute_task(task.task_id, AgentType.RESEARCH)
    
    # Verify execution
    assert result == {"result": "success"}
    mock_agent.execute_task.assert_called_once()
    assert agent_orchestrator.tasks[task.task_id].status == TaskStatus.COMPLETED

@pytest.mark.asyncio
async def test_error_handling(agent_orchestrator):
    """Test error handling in task execution."""
    # Create mock agent that raises an exception
    mock_agent = Mock()
    mock_agent.execute_task = Mock(side_effect=Exception("Test error"))
    agent_orchestrator.register_agent(mock_agent, AgentType.RESEARCH)
    
    # Create and add a test task
    task = Task(
        task_id="test_task",
        status=TaskStatus.PENDING,
        progress=0.0,
        confidence=0.0,
        timestamp=datetime.now()
    )
    agent_orchestrator.add_task(task)
    
    # Attempt execution and verify error handling
    with pytest.raises(Exception) as exc_info:
        await agent_orchestrator.execute_task(task.task_id, AgentType.RESEARCH)
    assert "Test error" in str(exc_info.value)
    assert agent_orchestrator.tasks[task.task_id].status == TaskStatus.FAILED 