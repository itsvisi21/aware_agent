"""
Test configuration and test data for the Agent-Based Research Pipeline.
"""

TEST_DATA = {
    "research_queries": [
        {
            "query": "Help me design an AI paper based on Sanskrit-based programming principles",
            "context": {
                "domain": "AI + Sanskrit",
                "output_type": "Academic Paper",
                "complexity": "Advanced"
            }
        },
        {
            "query": "Analyze the impact of quantum computing on cryptography",
            "context": {
                "domain": "Quantum Computing + Cryptography",
                "output_type": "Research Analysis",
                "complexity": "Intermediate"
            }
        }
    ],
    "semantic_roles": {
        "agent": ["user", "researcher", "system"],
        "object": ["paper", "analysis", "research"],
        "instrument": ["AI", "quantum computer", "algorithm"]
    },
    "temporal_constraints": {
        "timeframe": ["immediate", "short_term", "long_term"],
        "urgency": ["high", "medium", "low"]
    },
    "conversations": [
        {
            "id": "test-conv-1",
            "title": "Test Conversation 1",
            "goal": "Test goal 1"
        },
        {
            "id": "test-conv-2",
            "title": "Test Conversation 2",
            "goal": "Test goal 2"
        }
    ],
    "messages": [
        {
            "id": "test-msg-1",
            "conversation_id": "test-conv-1",
            "content": "Hello, this is a test message",
            "role": "user"
        },
        {
            "id": "test-msg-2",
            "conversation_id": "test-conv-1",
            "content": "This is a test response",
            "role": "assistant"
        }
    ],
    "context_trees": [
        {
            "id": "test-tree-1",
            "conversation_id": "test-conv-1",
            "type": "root",
            "content": {
                "goal": "Test goal 1"
            }
        }
    ]
}

TEST_CONFIG = {
    "model_settings": {
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9
    },
    "memory_settings": {
        "max_context_length": 1000,
        "retention_period": 3600  # 1 hour in seconds
    },
    "execution_settings": {
        "timeout": 30,  # seconds
        "max_retries": 3
    },
    "database": {
        "path": ":memory:",  # Use in-memory database for tests
        "echo": False
    },
    "agents": {
        "planner": {
            "model": "test-model",
            "temperature": 0.7
        },
        "researcher": {
            "model": "test-model",
            "temperature": 0.7
        },
        "explainer": {
            "model": "test-model",
            "temperature": 0.7
        },
        "validator": {
            "model": "test-model",
            "temperature": 0.7
        }
    }
} 