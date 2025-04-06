"""
Tests for the agent orchestration layer.
"""

import pytest
from src.agent_orchestration import AgentOrchestrator
from tests.config import TEST_DATA, TEST_CONFIG

@pytest.fixture
def agent_layer():
    """Create an agent orchestration layer instance for testing."""
    return AgentOrchestrator()

def test_goal_alignment(agent_layer):
    """Test the goal alignment process."""
    test_query = TEST_DATA["research_queries"][0]["query"]
    context = TEST_DATA["research_queries"][0]["context"]
    
    response = agent_layer.align_goals({
        "query": test_query,
        "context": context
    })
    
    assert response.status == "success"
    assert "goal" in response.response
    assert "subgoals" in response.response
    assert len(response.response["subgoals"]) > 0

def test_context_analysis(agent_layer):
    """Test the context analysis process."""
    test_query = TEST_DATA["research_queries"][1]["query"]
    context = TEST_DATA["research_queries"][1]["context"]
    
    response = agent_layer.analyze_context({
        "query": test_query,
        "context": context
    })
    
    assert response.status == "success"
    assert "domain_analysis" in response.response
    assert "complexity_assessment" in response.response
    assert "required_expertise" in response.response

def test_strategy_planning(agent_layer):
    """Test the strategy planning process."""
    test_query = TEST_DATA["research_queries"][0]["query"]
    context = TEST_DATA["research_queries"][0]["context"]
    
    goal_response = agent_layer.align_goals({
        "query": test_query,
        "context": context
    })
    
    plan_response = agent_layer.plan_strategy(goal_response, {
        "query": test_query,
        "context": context
    })
    
    assert plan_response.status == "success"
    assert "strategy" in plan_response.response
    assert "steps" in plan_response.response
    assert len(plan_response.response["steps"]) > 0

def test_plan_validation(agent_layer):
    """Test the plan validation process."""
    test_query = TEST_DATA["research_queries"][1]["query"]
    context = TEST_DATA["research_queries"][1]["context"]
    
    goal_response = agent_layer.align_goals({
        "query": test_query,
        "context": context
    })
    
    plan_response = agent_layer.plan_strategy(goal_response, {
        "query": test_query,
        "context": context
    })
    
    validation_response = agent_layer.validate_plan(plan_response, {
        "query": test_query,
        "context": context
    })
    
    assert validation_response.status == "success"
    assert "is_valid" in validation_response.response
    assert "feedback" in validation_response.response
    assert isinstance(validation_response.response["is_valid"], bool)

def test_error_handling(agent_layer):
    """Test error handling in the agent orchestration layer."""
    with pytest.raises(Exception):
        agent_layer.align_goals({})
    
    with pytest.raises(Exception):
        agent_layer.analyze_context({})
    
    with pytest.raises(Exception):
        agent_layer.plan_strategy(None, {})
    
    with pytest.raises(Exception):
        agent_layer.validate_plan(None, {}) 