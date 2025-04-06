import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agent_orchestration.agent_layer import (
    AgentOrchestrator,
    PlannerAgent,
    ResearchAgent,
    ExplainerAgent,
    ValidatorAgent,
    AgentResponse
)
from src.semantic_abstraction import ContextNode, SemanticDimension, KarakaMapping

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.agenerate = AsyncMock(return_value=MagicMock(generations=[[MagicMock(text="Test response")]]))
    return llm

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

@pytest.mark.asyncio
async def test_planner_agent(mock_llm, sample_context):
    agent = PlannerAgent()
    agent.initialize_chain("test template", mock_llm)
    response = await agent.process(sample_context, "Test input")
    
    assert isinstance(response, AgentResponse)
    assert response.content == "Test response"
    assert response.confidence == 0.85
    assert "research" in response.next_steps
    assert "validation" in response.next_steps

@pytest.mark.asyncio
async def test_research_agent(mock_llm, sample_context):
    agent = ResearchAgent()
    agent.initialize_chain("test template", mock_llm)
    response = await agent.process(sample_context, "Test input")
    
    assert isinstance(response, AgentResponse)
    assert response.content == "Test response"
    assert response.confidence == 0.80
    assert "explanation" in response.next_steps
    assert "validation" in response.next_steps

@pytest.mark.asyncio
async def test_explainer_agent(mock_llm, sample_context):
    agent = ExplainerAgent()
    agent.initialize_chain("test template", mock_llm)
    response = await agent.process(sample_context, "Test input")
    
    assert isinstance(response, AgentResponse)
    assert response.content == "Test response"
    assert response.confidence == 0.90
    assert "validation" in response.next_steps

@pytest.mark.asyncio
async def test_validator_agent(mock_llm, sample_context):
    agent = ValidatorAgent()
    agent.initialize_chain("test template", mock_llm)
    response = await agent.process(sample_context, "Test input")
    
    assert isinstance(response, AgentResponse)
    assert response.content == "Test response"
    assert response.confidence == 0.95
    assert "planning" in response.next_steps
    assert "research" in response.next_steps

@pytest.mark.asyncio
async def test_agent_orchestrator(mock_llm, sample_context):
    orchestrator = AgentOrchestrator(mock_llm)
    responses = await orchestrator.process_query(sample_context, "Test input")
    
    assert len(responses) == 4
    assert all(isinstance(response, AgentResponse) for response in responses.values())
    assert all(agent_name in responses for agent_name in ["planner", "research", "explainer", "validator"]) 