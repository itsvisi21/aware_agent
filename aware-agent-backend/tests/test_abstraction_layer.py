import pytest
from src.semantic_abstraction.abstraction_layer import (
    SemanticAbstractionLayer,
    SemanticDimension,
    KarakaMapping
)

@pytest.fixture
def abstraction_layer():
    return SemanticAbstractionLayer()

@pytest.fixture
def sample_dimensions():
    return SemanticDimension(
        domain="research",
        role="researcher",
        objective="analyze",
        timeframe="ongoing"
    )

def test_tokenize_and_annotate(abstraction_layer):
    text = "The researcher analyzes the data using Python"
    tokens = abstraction_layer.tokenize_and_annotate(text)
    
    assert len(tokens) > 0
    assert all(key in tokens[0] for key in ["text", "pos", "dep", "lemma"])

def test_map_karaka_roles(abstraction_layer):
    text = "The researcher analyzes the data using Python"
    tokens = abstraction_layer.tokenize_and_annotate(text)
    karaka = abstraction_layer.map_karaka_roles(tokens)
    
    assert isinstance(karaka, KarakaMapping)
    assert "researcher" in karaka.agent.lower()
    assert "data" in karaka.object.lower()
    assert "python" in karaka.instrument.lower()

def test_build_context_tree(abstraction_layer, sample_dimensions):
    query = "Analyze the research paper about AI"
    context_tree = abstraction_layer.build_context_tree(query, sample_dimensions)
    
    assert context_tree is not None
    assert context_tree.name == "root"
    assert len(context_tree.children) == 2  # tokens and semantic_roles nodes

def test_visualize_context_tree(abstraction_layer, sample_dimensions):
    query = "Analyze the research paper about AI"
    abstraction_layer.build_context_tree(query, sample_dimensions)
    visualization = abstraction_layer.visualize_context_tree()
    
    assert isinstance(visualization, str)
    assert "root" in visualization
    assert "tokens" in visualization
    assert "semantic_roles" in visualization 