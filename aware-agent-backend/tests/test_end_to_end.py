import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any
from src.main import app, ResearchRequest, TaskStatus
from src.semantic_abstraction import SemanticAbstractionLayer
from src.agent_orchestration import AgentOrchestrator
from src.interaction_engine import InteractionEngine
from src.memory_engine import MemoryEngine
from src.execution_layer import ExecutionLayer
from src.karaka_mapper import KarakaRole

@pytest.fixture
async def pipeline_components():
    """Initialize all pipeline components for testing."""
    semantic_layer = SemanticAbstractionLayer()
    agent_layer = AgentOrchestrator()
    interaction_engine = InteractionEngine()
    memory_engine = MemoryEngine()
    execution_layer = ExecutionLayer()
    
    yield {
        'semantic_layer': semantic_layer,
        'agent_layer': agent_layer,
        'interaction_engine': interaction_engine,
        'memory_engine': memory_engine,
        'execution_layer': execution_layer
    }
    
    # Cleanup
    await execution_layer.close()

@pytest.mark.asyncio
async def test_sanskrit_paper_research_pipeline(pipeline_components):
    """Test the complete pipeline for researching a Sanskrit-based programming paper."""
    # 1. Create research request
    request = ResearchRequest(
        query="Help me design an AI paper based on Sanskrit-based programming principles",
        context={
            "domain": "AI + Sanskrit",
            "output_type": "Academic Paper",
            "timeframe": "2 weeks"
        }
    )
    
    # 2. Process through semantic abstraction layer
    semantic_result = await pipeline_components['semantic_layer'].process(request.query, request.context)
    assert semantic_result is not None
    assert 'dimensions' in semantic_result
    assert 'tokens' in semantic_result
    
    # 3. Verify Karaka role mapping
    karaka_roles = pipeline_components['semantic_layer'].karaka_mapper.map_roles(request.query)
    assert KarakaRole.AGENT in karaka_roles
    assert KarakaRole.OBJECT in karaka_roles
    assert KarakaRole.PURPOSE in karaka_roles
    
    # 4. Process through agent orchestration
    agent_plan = await pipeline_components['agent_layer'].create_plan(
        semantic_result,
        request.context
    )
    assert agent_plan is not None
    assert 'steps' in agent_plan
    assert len(agent_plan['steps']) > 0
    
    # 5. Execute the plan
    execution_result = await pipeline_components['execution_layer'].execute_plan(agent_plan)
    assert execution_result is not None
    assert 'actions' in execution_result
    assert len(execution_result['actions']) > 0
    
    # 6. Verify memory storage
    memory_key = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    await pipeline_components['memory_engine'].store(
        memory_key,
        {
            'request': request.dict(),
            'semantic_result': semantic_result,
            'agent_plan': agent_plan,
            'execution_result': execution_result
        }
    )
    
    # 7. Retrieve and verify stored memory
    stored_memory = await pipeline_components['memory_engine'].retrieve(memory_key)
    assert stored_memory is not None
    assert stored_memory['request']['query'] == request.query
    
    # 8. Test interaction engine response
    interaction_response = await pipeline_components['interaction_engine'].process(
        request.query,
        semantic_result,
        agent_plan,
        execution_result
    )
    assert interaction_response is not None
    assert 'response' in interaction_response
    assert 'next_steps' in interaction_response

@pytest.mark.asyncio
async def test_pipeline_error_handling(pipeline_components):
    """Test error handling in the pipeline."""
    # Test with invalid input
    request = ResearchRequest(
        query="",  # Empty query
        context={}
    )
    
    # Should handle gracefully
    semantic_result = await pipeline_components['semantic_layer'].process(request.query, request.context)
    assert semantic_result is not None
    assert 'error' in semantic_result or 'dimensions' in semantic_result

@pytest.mark.asyncio
async def test_pipeline_performance(pipeline_components):
    """Test pipeline performance metrics."""
    import time
    
    request = ResearchRequest(
        query="Analyze the relationship between Paninian grammar and modern programming languages",
        context={
            "domain": "Linguistics + Programming",
            "output_type": "Analysis",
            "timeframe": "1 week"
        }
    )
    
    # Measure processing time
    start_time = time.time()
    
    # Run through pipeline
    semantic_result = await pipeline_components['semantic_layer'].process(request.query, request.context)
    agent_plan = await pipeline_components['agent_layer'].create_plan(semantic_result, request.context)
    execution_result = await pipeline_components['execution_layer'].execute_plan(agent_plan)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Assert reasonable processing time (adjust threshold as needed)
    assert processing_time < 5.0  # Should complete within 5 seconds
    
    # Verify results
    assert semantic_result is not None
    assert agent_plan is not None
    assert execution_result is not None 