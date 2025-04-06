import pytest
import os
import json
from datetime import datetime, timedelta
from src.execution.memory_layer import (
    MemoryEngine,
    TaskManager,
    SemanticLogger,
    Task,
    SemanticLog
)
from src.semantic_abstraction import ContextNode, SemanticDimension, KarakaMapping
from src.interaction import ConversationState
from langchain.schema import HumanMessage, AIMessage

@pytest.fixture
def memory_engine(tmp_path):
    db_path = tmp_path / "test.db"
    return MemoryEngine(str(db_path))

@pytest.fixture
def sample_context():
    dimension = SemanticDimension(
        domain="research",
        role="researcher",
        objective="analyze",
        timeframe="ongoing"
    )
    karaka = KarakaMapping(
        agent="researcher",
        object="data",
        instrument="python"
    )
    return ContextNode("root", dimension, karaka)

@pytest.fixture
def sample_conversation_state(sample_context):
    return ConversationState(
        messages=[
            HumanMessage(content="Initial message"),
            AIMessage(content="Initial response")
        ],
        context_tree=sample_context,
        current_goal="Test goal",
        feedback_history=[]
    )

def test_memory_engine_initialization(memory_engine):
    """Test that the database is properly initialized with required tables."""
    conn = sqlite3.connect(memory_engine.db_path)
    cursor = conn.cursor()
    
    # Check tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    assert "tasks" in tables
    assert "semantic_logs" in tables
    assert "conversation_states" in tables
    
    conn.close()

def test_task_storage_and_retrieval(memory_engine):
    """Test storing and retrieving tasks."""
    task = Task(
        id="test_task",
        type="analysis",
        status="pending",
        input_data={"data": "test"},
        output_data={},
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Store task
    memory_engine.store_task(task)
    
    # Retrieve task
    retrieved_task = memory_engine.get_task("test_task")
    
    assert retrieved_task is not None
    assert retrieved_task.id == task.id
    assert retrieved_task.type == task.type
    assert retrieved_task.status == task.status
    assert retrieved_task.input_data == task.input_data

def test_semantic_logging(memory_engine, sample_context):
    """Test semantic logging functionality."""
    logger = SemanticLogger(memory_engine)
    
    # Log a context change
    logger.log_context_change(
        sample_context,
        "context_update",
        {"change": "test_change"}
    )
    
    # Retrieve logs
    start_time = datetime.now() - timedelta(hours=1)
    end_time = datetime.now() + timedelta(hours=1)
    logs = memory_engine.get_semantic_logs(start_time, end_time)
    
    assert len(logs) > 0
    assert logs[0].action == "context_update"
    assert logs[0].details["change"] == "test_change"

def test_conversation_state_storage(memory_engine, sample_conversation_state):
    """Test storing and retrieving conversation states."""
    # Save state
    memory_engine.save_conversation_state(sample_conversation_state)
    
    # Retrieve state
    retrieved_state = memory_engine.get_latest_conversation_state()
    
    assert retrieved_state is not None
    assert retrieved_state.current_goal == sample_conversation_state.current_goal
    assert len(retrieved_state.messages) == len(sample_conversation_state.messages)

def test_task_manager(memory_engine):
    """Test task management functionality."""
    task_manager = TaskManager(memory_engine)
    
    # Create task
    task = task_manager.create_task(
        "analysis",
        {"data": "test_data"}
    )
    
    assert task is not None
    assert task.status == "pending"
    assert task.input_data["data"] == "test_data"
    
    # Update task
    task_manager.update_task(
        task.id,
        "completed",
        {"result": "success"}
    )
    
    # Check status
    status = task_manager.get_task_status(task.id)
    assert status == "completed"

def test_concurrent_task_handling(memory_engine):
    """Test handling multiple concurrent tasks."""
    task_manager = TaskManager(memory_engine)
    
    # Create multiple tasks
    tasks = []
    for i in range(5):
        task = task_manager.create_task(
            f"task_type_{i}",
            {"data": f"test_data_{i}"}
        )
        tasks.append(task)
    
    # Update tasks
    for task in tasks:
        task_manager.update_task(
            task.id,
            "completed",
            {"result": f"success_{task.id}"}
        )
    
    # Verify all tasks are completed
    for task in tasks:
        status = task_manager.get_task_status(task.id)
        assert status == "completed"

def test_semantic_log_retrieval_by_time_range(memory_engine, sample_context):
    """Test retrieving semantic logs within a specific time range."""
    logger = SemanticLogger(memory_engine)
    
    # Log multiple events at different times
    for i in range(3):
        logger.log_context_change(
            sample_context,
            f"event_{i}",
            {"data": f"test_{i}"}
        )
    
    # Retrieve logs for a specific time range
    start_time = datetime.now() - timedelta(minutes=5)
    end_time = datetime.now() + timedelta(minutes=5)
    logs = memory_engine.get_semantic_logs(start_time, end_time)
    
    assert len(logs) == 3
    assert all(log.timestamp >= start_time and log.timestamp <= end_time for log in logs) 