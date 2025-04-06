"""
Tests for the semantic abstraction layer.
"""

import pytest
from src.semantic_abstraction import SemanticAbstractionLayer
from tests.config import TEST_DATA, TEST_CONFIG

@pytest.fixture
def semantic_layer():
    """Create a semantic abstraction layer instance for testing."""
    return SemanticAbstractionLayer()

def test_tokenization_and_annotation(semantic_layer):
    """Test the tokenization and annotation process."""
    test_query = TEST_DATA["research_queries"][0]["query"]
    context = TEST_DATA["research_queries"][0]["context"]
    
    result = semantic_layer.tokenize_and_annotate(test_query, context=context)
    
    assert "tokens" in result
    assert "annotations" in result
    assert len(result["tokens"]) > 0
    assert "domain" in result["annotations"]
    assert result["annotations"]["domain"] == "AI + Sanskrit"

def test_context_tree_building(semantic_layer):
    """Test the context tree building process."""
    test_query = TEST_DATA["research_queries"][1]["query"]
    context = TEST_DATA["research_queries"][1]["context"]
    
    tokenized = semantic_layer.tokenize_and_annotate(test_query, context=context)
    context_tree = semantic_layer.build_context_tree(tokenized, context)
    
    assert "query" in context_tree
    assert "semantic_roles" in context_tree
    assert "temporal_dimensions" in context_tree
    assert "domain_context" in context_tree
    assert context_tree["domain_context"]["domain"] == "Quantum Computing + Cryptography"

def test_semantic_role_assignment(semantic_layer):
    """Test the semantic role assignment process."""
    test_query = TEST_DATA["research_queries"][0]["query"]
    context = TEST_DATA["research_queries"][0]["context"]
    
    tokenized = semantic_layer.tokenize_and_annotate(test_query, context=context)
    roles = semantic_layer.assign_semantic_roles(tokenized)
    
    assert "agent" in roles
    assert "object" in roles
    assert "instrument" in roles
    assert any(role in roles["agent"] for role in TEST_DATA["semantic_roles"]["agent"])
    assert any(role in roles["object"] for role in TEST_DATA["semantic_roles"]["object"])

def test_temporal_constraint_processing(semantic_layer):
    """Test the processing of temporal constraints."""
    test_query = TEST_DATA["research_queries"][1]["query"]
    temporal_constraints = TEST_DATA["temporal_constraints"]
    
    tokenized = semantic_layer.tokenize_and_annotate(test_query)
    temporal_info = semantic_layer.process_temporal_constraints(tokenized, temporal_constraints)
    
    assert "timeframe" in temporal_info
    assert "urgency" in temporal_info
    assert temporal_info["timeframe"] in TEST_DATA["temporal_constraints"]["timeframe"]
    assert temporal_info["urgency"] in TEST_DATA["temporal_constraints"]["urgency"]

def test_error_handling(semantic_layer):
    """Test error handling in the semantic abstraction layer."""
    with pytest.raises(Exception):
        semantic_layer.tokenize_and_annotate("")
    
    with pytest.raises(Exception):
        semantic_layer.build_context_tree({}, None)
    
    with pytest.raises(Exception):
        semantic_layer.assign_semantic_roles({}) 