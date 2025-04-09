# Aware Agent Backend Documentation

## Overview

The Aware Agent backend is a Python-based service that powers the AI agent system. It handles natural language processing, agent orchestration, memory management, and provides a REST API and WebSocket interface for the frontend.

## Tech Stack

### Core Technologies
- **Python 3.11+**: Programming language
- **FastAPI**: Web framework
- **LangChain**: LLM orchestration
- **spaCy**: NLP processing
- **SQLAlchemy**: Database ORM
- **Redis**: Caching
- **ChromaDB**: Vector database

### Key Dependencies
```txt
fastapi>=0.100.0
uvicorn>=0.22.0
langchain>=0.1.0
spacy>=3.7.0
scikit-learn>=1.3.0
sqlalchemy>=2.0.0
redis>=4.5.0
chromadb>=0.4.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.6
```

## Project Structure

```
aware-agent-backend/
├── src/
│   ├── api/                 # API routes and endpoints
│   │   ├── v1/             # API version 1
│   │   └── websocket/      # WebSocket handlers
│   ├── core/               # Core business logic
│   │   ├── agents/        # Agent implementations
│   │   ├── memory/        # Memory management
│   │   └── processing/    # NLP processing
│   ├── models/            # Data models
│   │   ├── database/     # Database models
│   │   └── pydantic/     # Pydantic models
│   ├── services/          # Service layer
│   │   ├── auth/         # Authentication
│   │   ├── cache/        # Caching
│   │   └── storage/      # Storage services
│   └── utils/            # Utility functions
├── tests/                # Test files
├── alembic/             # Database migrations
└── requirements.txt     # Project dependencies
```

## Core Components

### 1. Semantic Abstraction Layer
- Text tokenization and annotation
- Semantic role mapping
- Context extraction
- Domain-specific analysis

### 2. Agent Orchestration
- Agent coordination
- Task planning
- Error handling
- Parallel execution

### 3. Memory Management
- Context storage
- Action history
- Retention policies
- Concurrent access

### 4. Interaction Engine
- Prompt construction
- Response translation
- Feedback integration
- Context awareness

## Database Schema

### Main Tables
```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE TABLE messages (
    id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE context (
    id UUID PRIMARY KEY,
    key VARCHAR(255) NOT NULL,
    value JSONB NOT NULL,
    expiry TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);
```

## API Architecture

### REST API
- FastAPI-based endpoints
- OpenAPI documentation
- JWT authentication
- Rate limiting

### WebSocket
- Real-time updates
- Bi-directional communication
- Connection management
- Error handling

## Agent Types

### 1. Research Agent
- Information gathering
- Source analysis
- Knowledge synthesis
- Context building

### 2. Builder Agent
- Code generation
- Project setup
- Technical guidance
- Implementation support

### 3. Teacher Agent
- Concept explanation
- Learning path management
- Knowledge assessment
- Educational guidance

### 4. Collaborator Agent
- Task management
- Progress tracking
- Team coordination
- Project planning

## Memory System

### Storage Types
1. **Short-term Memory**
   - In-memory cache
   - Session storage
   - Temporary context

2. **Long-term Memory**
   - Database storage
   - Vector embeddings
   - Persistent context

### Retrieval Methods
- Semantic search
- Context matching
- Time-based retrieval
- Priority-based access

## Development Setup

### Prerequisites
- Python 3.11+
- PostgreSQL 14+
- Redis 6+
- ChromaDB

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start development server
uvicorn src.main:app --reload
```

## Testing Strategy

### Test Types
1. **Unit Tests**
   - Component testing
   - Function testing
   - Mock testing

2. **Integration Tests**
   - API testing
   - Database testing
   - Service testing

3. **Performance Tests**
   - Load testing
   - Stress testing
   - Benchmarking

## Security Measures

### Authentication
- JWT token-based auth
- Token refresh mechanism
- Role-based access control

### Data Protection
- Input validation
- Output sanitization
- Encryption at rest
- Secure communication

## Deployment

### Containerization
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aware-agent-backend
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: aware-agent
        image: aware-agent-backend:latest
        ports:
        - containerPort: 8000
```

## Monitoring

### Metrics
- API response times
- Error rates
- Resource usage
- Agent performance

### Logging
- Structured logging
- Log aggregation
- Error tracking
- Audit trails

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [spaCy Documentation](https://spacy.io/usage)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Redis Documentation](https://redis.io/documentation)
- [ChromaDB Documentation](https://docs.trychroma.com/) 