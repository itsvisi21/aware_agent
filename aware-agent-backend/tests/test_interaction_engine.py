"""
Tests for the interaction engine.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.interaction.interaction_engine import (
    InteractionEngine,
    ConversationState,
    PromptBuilder,
    ResponseTranslator,
    FeedbackIntegrator
)
from src.agent_orchestration import AgentResponse
from src.semantic_abstraction import ContextNode, SemanticDimension, KarakaMapping
from langchain.schema import HumanMessage, AIMessage
from src.interaction_engine import InteractionEngine
from tests.config import TEST_DATA, TEST_CONFIG
from datetime import datetime
from src.memory.persistence import ConversationRecord, MessageRecord

@pytest.fixture
def interaction_engine():
    """Create an interaction engine instance for testing."""
    return InteractionEngine()

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
def sample_agent_responses():
    return {
        "planner": AgentResponse(
            content="Plan content",
            reasoning="Planning reasoning",
            confidence=0.85,
            next_steps=["research"]
        ),
        "research": AgentResponse(
            content="Research content",
            reasoning="Research reasoning",
            confidence=0.80,
            next_steps=["explanation"]
        ),
        "explainer": AgentResponse(
            content="Explanation content",
            reasoning="Explanation reasoning",
            confidence=0.90,
            next_steps=["validation"]
        ),
        "validator": AgentResponse(
            content="Validation content",
            reasoning="Validation reasoning",
            confidence=0.95,
            next_steps=["planning"]
        )
    }

@pytest.fixture
def sample_conversation():
    return ConversationRecord(
        id="test-conv-1",
        title="Test Conversation",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        metadata={"goal": "test goal"}
    )

def test_prompt_construction(interaction_engine):
    """Test the prompt construction process."""
    test_query = TEST_DATA["research_queries"][0]["query"]
    context = TEST_DATA["research_queries"][0]["context"]
    
    prompt = interaction_engine.construct_prompt(test_query, context)
    
    assert prompt is not None
    assert isinstance(prompt, str)
    assert test_query in prompt
    assert context["domain"] in prompt
    assert "role" in prompt.lower()
    assert "context" in prompt.lower()

def test_response_translation(interaction_engine):
    """Test the response translation process."""
    test_response = {
        "content": "Here's a research plan for Sanskrit-based programming...",
        "actions": [
            {"type": "research", "content": "Find papers on Paninian grammar"},
            {"type": "analysis", "content": "Compare with formal logic"}
        ]
    }
    
    translated = interaction_engine.translate_response(test_response, {
        "query": TEST_DATA["research_queries"][0]["query"],
        "context": TEST_DATA["research_queries"][0]["context"]
    })
    
    assert translated.status == "success"
    assert translated.content is not None
    assert len(translated.actions) == len(test_response["actions"])
    assert translated.confidence > 0.0

def test_feedback_integration(interaction_engine):
    """Test the feedback integration process."""
    test_feedback = {
        "rating": 4,
        "comments": "Good analysis, but needs more technical depth",
        "suggestions": ["Add more code examples", "Include performance metrics"]
    }
    
    updated_context = interaction_engine.integrate_feedback(
        test_feedback,
        {
            "query": TEST_DATA["research_queries"][1]["query"],
            "context": TEST_DATA["research_queries"][1]["context"]
        }
    )
    
    assert updated_context is not None
    assert "feedback" in updated_context
    assert updated_context["feedback"]["rating"] == test_feedback["rating"]
    assert len(updated_context["feedback"]["suggestions"]) == len(test_feedback["suggestions"])

def test_error_handling(interaction_engine):
    """Test error handling in the interaction engine."""
    # Test invalid prompt construction
    with pytest.raises(Exception):
        interaction_engine.construct_prompt("", None)
    
    # Test invalid response translation
    with pytest.raises(Exception):
        interaction_engine.translate_response(None, {})
    
    # Test invalid feedback integration
    with pytest.raises(Exception):
        interaction_engine.integrate_feedback(None, {})

def test_context_awareness(interaction_engine):
    """Test context awareness in responses."""
    test_query = TEST_DATA["research_queries"][1]["query"]
    context = TEST_DATA["research_queries"][1]["context"]
    
    # First interaction
    prompt1 = interaction_engine.construct_prompt(test_query, context)
    response1 = interaction_engine.translate_response({
        "content": "Initial analysis...",
        "actions": [{"type": "research", "content": "Find quantum papers"}]
    }, {"query": test_query, "context": context})
    
    # Second interaction with context from first
    context["previous_interaction"] = response1.content
    prompt2 = interaction_engine.construct_prompt(test_query, context)
    
    assert prompt2 != prompt1
    assert response1.content in prompt2
    assert "previous_interaction" in prompt2.lower()

def test_action_validation(interaction_engine):
    """Test action validation in responses."""
    valid_actions = [
        {"type": "research", "content": "Find papers"},
        {"type": "analysis", "content": "Analyze results"}
    ]
    
    invalid_actions = [
        {"type": "invalid_type", "content": ""},
        {"type": "research", "content": None}
    ]
    
    # Test valid actions
    response = interaction_engine.translate_response({
        "content": "Test content",
        "actions": valid_actions
    }, {"query": "test", "context": {}})
    
    assert response.status == "success"
    assert len(response.actions) == len(valid_actions)
    
    # Test invalid actions
    with pytest.raises(Exception):
        interaction_engine.translate_response({
            "content": "Test content",
            "actions": invalid_actions
        }, {"query": "test", "context": {}})

def test_prompt_builder_goal_clarification(sample_context):
    builder = PromptBuilder()
    state = ConversationState(
        messages=[
            HumanMessage(content="Initial message"),
            AIMessage(content="Initial response")
        ],
        context_tree=sample_context,
        current_goal="Test goal",
        feedback_history=[]
    )
    
    prompt = builder.build_goal_clarification_prompt(state)
    assert "Context" in prompt
    assert "History" in prompt
    assert "Initial message" in prompt
    assert "Initial response" in prompt

def test_prompt_builder_response_synthesis(sample_agent_responses):
    builder = PromptBuilder()
    prompt = builder.build_response_synthesis_prompt(sample_agent_responses)
    
    assert "Planner" in prompt
    assert "Research" in prompt
    assert "Explainer" in prompt
    assert "Validator" in prompt
    assert "Plan content" in prompt
    assert "Research content" in prompt

def test_response_translator():
    translator = ResponseTranslator()
    test_response = "Test response content"
    
    # Test markdown format
    markdown_result = translator.translate(test_response, "markdown")
    assert markdown_result["type"] == "markdown"
    assert markdown_result["content"] == test_response
    
    # Test conversational format
    conversational_result = translator.translate(test_response, "conversational")
    assert conversational_result["type"] == "conversational"
    assert conversational_result["content"] == test_response
    
    # Test structured format
    structured_result = translator.translate(test_response, "structured")
    assert structured_result["type"] == "structured"
    assert "main_points" in structured_result["content"]
    assert "supporting_evidence" in structured_result["content"]
    assert "next_steps" in structured_result["content"]

def test_feedback_integrator(sample_context):
    integrator = FeedbackIntegrator()
    state = ConversationState(
        messages=[],
        context_tree=sample_context,
        current_goal="Test goal",
        feedback_history=[]
    )
    
    # Test clarification feedback
    clarification_feedback = {
        "type": "clarification",
        "content": ["Clarification point 1", "Clarification point 2"]
    }
    updated_state = integrator.integrate(clarification_feedback, state)
    assert "clarifications" in updated_state.context_tree.dimension.metadata
    
    # Test preference feedback
    preference_feedback = {
        "type": "preference",
        "content": {"format": "markdown", "detail_level": "high"}
    }
    updated_state = integrator.integrate(preference_feedback, state)
    assert "preferences" in updated_state.metadata

@pytest.mark.asyncio
async def test_interaction_engine(sample_context, sample_agent_responses):
    engine = InteractionEngine()
    engine.initialize_conversation(sample_context, "Test goal")
    
    # Test processing interaction
    response = await engine.process_interaction(
        "Test user input",
        sample_agent_responses,
        "conversational"
    )
    
    assert response["type"] == "conversational"
    assert len(engine.conversation_state.messages) == 2
    assert isinstance(engine.conversation_state.messages[0], HumanMessage)
    assert isinstance(engine.conversation_state.messages[1], AIMessage)
    
    # Test feedback integration
    feedback = {
        "type": "clarification",
        "content": ["Test clarification"]
    }
    engine.integrate_feedback(feedback)
    assert len(engine.conversation_state.feedback_history) == 1
    assert "clarifications" in engine.conversation_state.context_tree.dimension.metadata

@pytest.mark.asyncio
async def test_create_conversation(interaction_engine):
    conversation_id = await interaction_engine.create_conversation(
        title="Test Conversation",
        initial_goal="test goal"
    )
    assert conversation_id is not None
    assert len(conversation_id) > 0

@pytest.mark.asyncio
async def test_process_message(interaction_engine):
    # Create a conversation first
    conversation_id = await interaction_engine.create_conversation(
        title="Test Conversation",
        initial_goal="test goal"
    )
    
    # Process a test message
    response = await interaction_engine.process_message(
        conversation_id=conversation_id,
        content="Hello, this is a test message"
    )
    
    assert response is not None
    assert "responses" in response
    assert "context" in response

@pytest.mark.asyncio
async def test_get_conversation_history(interaction_engine):
    # Create a conversation
    conversation_id = await interaction_engine.create_conversation(
        title="Test Conversation",
        initial_goal="test goal"
    )
    
    # Process a message
    await interaction_engine.process_message(
        conversation_id=conversation_id,
        content="Test message"
    )
    
    # Get history
    history = await interaction_engine.get_conversation_history(conversation_id)
    assert len(history) > 0
    assert history[0].content == "Test message"

@pytest.mark.asyncio
async def test_get_context_tree(interaction_engine):
    # Create a conversation
    conversation_id = await interaction_engine.create_conversation(
        title="Test Conversation",
        initial_goal="test goal"
    )
    
    # Get context tree
    context_tree = await interaction_engine.get_context_tree(conversation_id)
    assert context_tree is not None
    assert "goal" in context_tree["content"]

@pytest.mark.asyncio
async def test_list_conversations(interaction_engine):
    # Create a conversation
    await interaction_engine.create_conversation(
        title="Test Conversation",
        initial_goal="test goal"
    )
    
    # List conversations
    conversations = await interaction_engine.list_conversations()
    assert len(conversations) > 0
    assert conversations[0].title == "Test Conversation"

@pytest.mark.asyncio
async def test_get_agent_statuses(interaction_engine):
    # Create a conversation
    conversation_id = await interaction_engine.create_conversation(
        title="Test Conversation",
        initial_goal="test goal"
    )
    
    # Get agent statuses
    statuses = await interaction_engine.get_agent_statuses(conversation_id)
    assert statuses is not None
    assert "planner" in statuses
    assert "researcher" in statuses
    assert "explainer" in statuses
    assert "validator" in statuses 