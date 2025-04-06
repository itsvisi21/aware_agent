import pytest
import asyncio
from datetime import datetime
from src.agent import Agent
from src.agent_orchestration import AgentOrchestrationLayer, AgentType
from src.agent_conversation import AgentConversation, AgentRole
from src.semantic_abstraction import SemanticAbstractionLayer
from src.memory_engine import MemoryEngine
from src.execution_layer import ExecutionLayer, ExecutionStatus

@pytest.fixture
def agent():
    return Agent()

@pytest.fixture
def orchestration():
    return AgentOrchestrationLayer()

@pytest.fixture
def conversation():
    return AgentConversation()

@pytest.fixture
def semantic_layer():
    return SemanticAbstractionLayer()

@pytest.fixture
def memory_engine():
    return MemoryEngine()

@pytest.fixture
def execution_layer():
    return ExecutionLayer()

@pytest.mark.asyncio
async def test_agent_process_input(agent):
    """Test agent's input processing pipeline."""
    input_text = "Research the latest developments in AI and summarize key findings"
    result = await agent.process_input(input_text)
    
    assert result["status"] == "success"
    assert "result" in result
    assert "confidence" in result
    assert "semantic_context" in result

@pytest.mark.asyncio
async def test_orchestration_goal_alignment(orchestration):
    """Test goal alignment in orchestration layer."""
    context_tree = {
        "semantic_roles": {
            "AGENT": [{"entity": "researcher", "properties": {"confidence": 0.9}}],
            "OBJECT": [{"entity": "AI developments", "properties": {"confidence": 0.9}}]
        },
        "semantic_graph": {"nodes": [], "edges": []}
    }
    
    response = orchestration.align_goals(context_tree)
    
    assert response.agent_type == AgentType.GOAL_ALIGNMENT
    assert "goal_structure" in response.response
    assert response.confidence > 0

@pytest.mark.asyncio
async def test_conversation_management(conversation):
    """Test conversation management."""
    conversation_id = conversation.create_conversation(
        goal="Research AI developments",
        initial_context={"domain": "AI"}
    )
    
    message = conversation.add_message(
        conversation_id=conversation_id,
        role=AgentRole.PLANNER,
        content="Planning research strategy",
        reasoning="Breaking down research into manageable steps",
        confidence=0.9
    )
    
    state = conversation.get_conversation_state(conversation_id)
    
    assert state.goal == "Research AI developments"
    assert len(state.messages) == 1
    assert state.messages[0].content == "Planning research strategy"

@pytest.mark.asyncio
async def test_semantic_analysis(semantic_layer):
    """Test semantic analysis."""
    text = "Research the latest developments in AI and summarize key findings"
    result = await semantic_layer.analyze(text)
    
    assert "semantic_roles" in result
    assert "semantic_graph" in result
    assert "temporal_dimensions" in result
    assert "spatial_dimensions" in result
    assert "domain_context" in result

@pytest.mark.asyncio
async def test_memory_operations(memory_engine):
    """Test memory operations."""
    # Store memory
    memory_id = await memory_engine.store(
        content={"type": "research", "topic": "AI"},
        importance=0.8,
        category="research",
        tags=["AI", "research"]
    )
    
    # Retrieve memory
    memories = await memory_engine.retrieve({"type": "research"})
    
    assert len(memories) > 0
    assert memories[0]["content"]["type"] == "research"

@pytest.mark.asyncio
async def test_execution_layer(execution_layer):
    """Test execution layer."""
    plan = {
        "actions": [
            {"type": "research", "parameters": {"topic": "AI"}},
            {"type": "analysis", "parameters": {"data": "research_results"}},
            {"type": "synthesis", "parameters": {"insights": "analysis_results"}}
        ]
    }
    
    result = await execution_layer.execute(plan)
    
    assert result["status"] in ["completed", "failed"]
    assert "execution_id" in result
    assert "results" in result
    assert "metrics" in result

@pytest.mark.asyncio
async def test_agent_self_awareness(agent):
    """Test agent's self-awareness capabilities."""
    # Process multiple inputs to build up execution history
    inputs = [
        "Research AI developments",
        "Analyze the findings",
        "Summarize the key points"
    ]
    
    for input_text in inputs:
        await agent.process_input(input_text)
    
    # Check self-awareness logs
    log_file = agent.workspace_dir / "self_awareness_logs.jsonl"
    assert log_file.exists()
    
    with open(log_file, "r") as f:
        logs = [json.loads(line) for line in f]
    
    assert len(logs) > 0
    assert "metrics" in logs[0]
    assert "state" in logs[0]

@pytest.mark.asyncio
async def test_agent_error_handling(agent):
    """Test agent's error handling."""
    # Test with invalid input
    result = await agent.process_input("")
    
    assert result["status"] == "error"
    assert "error" in result
    
    # Test with malformed input
    result = await agent.process_input(None)
    
    assert result["status"] == "error"
    assert "error" in result

@pytest.mark.asyncio
async def test_agent_performance_monitoring(agent):
    """Test agent's performance monitoring."""
    # Process multiple inputs
    inputs = [
        "Research topic 1",
        "Research topic 2",
        "Research topic 3"
    ]
    
    start_time = datetime.now()
    for input_text in inputs:
        await agent.process_input(input_text)
    end_time = datetime.now()
    
    # Check performance metrics
    execution_time = (end_time - start_time).total_seconds()
    assert execution_time > 0
    
    # Check memory patterns
    patterns = agent.memory.analyze_memory_patterns()
    assert "category_distribution" in patterns
    assert "category_importance" in patterns
    assert "total_memories" in patterns 