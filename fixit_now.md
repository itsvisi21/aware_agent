aware-agent-backend/
├── src/
│   ├── core/                    # Core application components
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration management
│   │   ├── exceptions.py       # Custom exceptions
│   │   └── models.py           # Core data models
│   │
│   ├── agents/                 # Agent-related code
│   │   ├── __init__.py
│   │   ├── base.py            # Base agent class
│   │   ├── factory.py         # Agent factory
│   │   ├── types.py           # Agent types and enums
│   │   └── specialized/       # Specialized agent implementations
│   │       ├── __init__.py
│   │       ├── research.py
│   │       ├── builder.py
│   │       └── ...
│   │
│   ├── orchestration/          # Agent orchestration
│   │   ├── __init__.py
│   │   ├── orchestrator.py    # Main orchestrator
│   │   ├── scheduler.py       # Task scheduling
│   │   └── coordination.py    # Inter-agent coordination
│   │
│   ├── services/              # Core services
│   │   ├── __init__.py
│   │   ├── monitoring.py     # Monitoring service
│   │   ├── database.py       # Database service
│   │   ├── cache.py          # Caching service
│   │   └── semantic.py       # Semantic service
│   │
│   ├── memory/               # Memory management
│   │   ├── __init__.py
│   │   ├── engine.py        # Memory engine
│   │   ├── storage.py       # Memory storage
│   │   └── retrieval.py     # Memory retrieval
│   │
│   ├── execution/           # Execution layer
│   │   ├── __init__.py
│   │   ├── engine.py       # Execution engine
│   │   ├── planner.py      # Task planning
│   │   └── validator.py    # Task validation
│   │
│   ├── semantic/           # Semantic processing
│   │   ├── __init__.py
│   │   ├── abstraction.py  # Semantic abstraction
│   │   ├── mapping.py      # Semantic mapping
│   │   └── reasoning.py    # Semantic reasoning
│   │
│   ├── api/               # API layer
│   │   ├── __init__.py
│   │   ├── routes.py      # API routes
│   │   ├── handlers.py    # Request handlers
│   │   └── middleware.py  # API middleware
│   │
│   ├── utils/            # Utility functions
│   │   ├── __init__.py
│   │   ├── logging.py    # Logging utilities
│   │   ├── metrics.py    # Metrics utilities
│   │   └── validation.py # Validation utilities
│   │
│   └── main.py          # Application entry point
│
├── tests/               # Test directory
│   ├── __init__.py
│   ├── core/
│   ├── agents/
│   ├── services/
│   └── ...
│
└── docs/               # Documentation
    ├── architecture.md
    ├── api.md
    └── ...