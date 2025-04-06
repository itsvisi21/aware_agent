"""
Integration tests for the agent system.
Tests interactions between different components.
"""

import pytest
import asyncio
from datetime import datetime
from src.semantic_abstraction import SemanticAbstractionLayer
from src.agent_orchestration import AgentOrchestrator
from src.execution_layer import ExecutionLayer
from src.memory_engine import MemoryEngine
from src.interaction_engine import InteractionEngine
from src.abstraction_engines import AbstractionType
from tests.config import TEST_DATA

@pytest.fixture
async def test_system():
    """Create test system with all components."""
    semantic_layer = SemanticAbstractionLayer()
    agent_layer = AgentOrchestrator()
    execution_layer = ExecutionLayer()
    memory_engine = MemoryEngine()
    interaction_engine = InteractionEngine()
    
    # Initialize components
    await memory_engine.initialize_database()
    
    return {
        "semantic_layer": semantic_layer,
        "agent_layer": agent_layer,
        "execution_layer": execution_layer,
        "memory_engine": memory_engine,
        "interaction_engine": interaction_engine
    }

@pytest.mark.asyncio
async def test_end_to_end_research_flow(test_system):
    """Test complete research workflow through all components."""
    # 1. Start with semantic analysis
    query = "Analyze the impact of Sanskrit on modern programming languages"
    semantic_result = await test_system["semantic_layer"].process(
        query=query,
        context={"domain": "computer_science"}
    )
    
    assert semantic_result is not None
    assert "embeddings" in semantic_result
    
    # 2. Create agent plan
    agent_plan = await test_system["agent_layer"].create_plan(
        semantic_result=semantic_result,
        context={"max_steps": 3}
    )
    
    assert agent_plan is not None
    assert len(agent_plan["steps"]) > 0
    
    # 3. Execute plan
    execution_result = await test_system["execution_layer"].execute_task(
        task_id=agent_plan["id"],
        progress_callback=lambda p: None
    )
    
    assert execution_result is not None
    assert execution_result.status == "completed"
    
    # 4. Store in memory
    memory_key = f"test_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    await test_system["memory_engine"].store(
        key=memory_key,
        data={
            "semantic_result": semantic_result,
            "agent_plan": agent_plan,
            "execution_result": execution_result
        }
    )
    
    # 5. Verify memory storage
    stored_data = await test_system["memory_engine"].retrieve(memory_key)
    assert stored_data is not None
    assert "semantic_result" in stored_data
    
    # 6. Process interaction
    interaction_response = await test_system["interaction_engine"].process(
        query=query,
        semantic_result=semantic_result,
        agent_plan=agent_plan,
        execution_result=execution_result
    )
    
    assert interaction_response is not None
    assert "response" in interaction_response

@pytest.mark.asyncio
async def test_error_recovery_flow(test_system):
    """Test system's ability to handle and recover from errors."""
    # 1. Test with invalid query
    query = ""  # Empty query should trigger error handling
    
    semantic_result = await test_system["semantic_layer"].process(
        query=query,
        context={"domain": "computer_science"}
    )
    
    assert semantic_result is not None
    assert semantic_result.get("error") is not None
    
    # 2. Test with invalid plan
    agent_plan = await test_system["agent_layer"].create_plan(
        semantic_result={"error": "Invalid semantic result"},
        context={}
    )
    
    assert agent_plan is not None
    assert agent_plan.get("error") is not None
    
    # 3. Verify error is properly stored
    memory_key = f"test_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    await test_system["memory_engine"].store(
        key=memory_key,
        data={
            "error": "test_error",
            "timestamp": datetime.now().isoformat()
        }
    )
    
    stored_error = await test_system["memory_engine"].retrieve(memory_key)
    assert stored_error is not None
    assert "error" in stored_error

@pytest.mark.asyncio
async def test_concurrent_operations(test_system):
    """Test system's ability to handle concurrent operations."""
    # Create multiple concurrent tasks
    queries = [
        "Research quantum computing",
        "Analyze artificial intelligence trends",
        "Study blockchain technology"
    ]
    
    async def process_query(query):
        semantic_result = await test_system["semantic_layer"].process(
            query=query,
            context={"domain": "computer_science"}
        )
        return semantic_result
    
    # Execute queries concurrently
    results = await asyncio.gather(*[
        process_query(query) for query in queries
    ])
    
    assert len(results) == len(queries)
    assert all(result is not None for result in results)

@pytest.mark.asyncio
async def test_component_interaction_flow(test_system):
    """Test specific interactions between components."""
    # 1. Test Semantic Layer -> Agent Layer interaction
    semantic_result = await test_system["semantic_layer"].process(
        query="Test interaction flow",
        context={"test": True}
    )
    
    agent_plan = await test_system["agent_layer"].create_plan(
        semantic_result=semantic_result,
        context={"max_steps": 2}
    )
    
    assert agent_plan is not None
    
    # 2. Test Agent Layer -> Execution Layer interaction
    execution_result = await test_system["execution_layer"].execute_task(
        task_id=agent_plan["id"]
    )
    
    assert execution_result is not None
    
    # 3. Test Memory Engine interactions
    await test_system["memory_engine"].store(
        key="test_interaction",
        data={
            "semantic_result": semantic_result,
            "agent_plan": agent_plan,
            "execution_result": execution_result
        }
    )
    
    retrieved_data = await test_system["memory_engine"].retrieve("test_interaction")
    assert retrieved_data is not None
    assert "semantic_result" in retrieved_data 