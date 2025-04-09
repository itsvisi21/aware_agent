# Aware Agent API Documentation

## Overview

The Aware Agent API is built using FastAPI and provides endpoints for interacting with the AI agent system. This documentation covers all available endpoints, their request/response formats, and authentication requirements.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

All API endpoints require authentication using JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

## Endpoints

### 1. Authentication

#### POST /auth/token
Generate a new JWT token for authentication.

**Request Body:**
```json
{
    "username": "string",
    "password": "string"
}
```

**Response:**
```json
{
    "access_token": "string",
    "token_type": "bearer",
    "expires_in": 3600
}
```

**Error Responses:**
- 401 Unauthorized: Invalid credentials
- 400 Bad Request: Missing required fields

### 2. Conversation Management

#### POST /conversations
Start a new conversation with the AI agent.

**Request Body:**
```json
{
    "message": "What is semantic abstraction?",
    "context": {
        "domain": "AI",
        "mode": "research",
        "preferences": {
            "detail_level": "high",
            "format": "technical"
        }
    }
}
```

**Response:**
```json
{
    "conversation_id": "conv_123456",
    "response": "Semantic abstraction is a process of...",
    "context": {
        "semantic_roles": [
            {
                "role": "subject",
                "value": "semantic abstraction",
                "confidence": 0.95
            }
        ],
        "entities": [
            {
                "type": "concept",
                "value": "semantic abstraction",
                "description": "AI concept"
            }
        ]
    },
    "metadata": {
        "processing_time": 0.45,
        "model_used": "gpt-4",
        "tokens_used": 150
    }
}
```

**Error Responses:**
- 400 Bad Request: Invalid message format
- 429 Too Many Requests: Rate limit exceeded
- 500 Internal Server Error: Processing error

#### GET /conversations/{conversation_id}
Retrieve conversation history.

**Response:**
```json
{
    "conversation_id": "conv_123456",
    "messages": [
        {
            "role": "user",
            "content": "What is semantic abstraction?",
            "timestamp": "2024-03-15T10:30:00Z"
        },
        {
            "role": "assistant",
            "content": "Semantic abstraction is a process of...",
            "timestamp": "2024-03-15T10:30:01Z",
            "metadata": {
                "processing_time": 0.45,
                "model_used": "gpt-4"
            }
        }
    ],
    "summary": {
        "topic": "Semantic Abstraction",
        "key_points": ["definition", "importance", "applications"],
        "duration": "5 minutes"
    }
}
```

**Error Responses:**
- 404 Not Found: Conversation not found
- 403 Forbidden: Not authorized to access conversation

#### DELETE /conversations/{conversation_id}
Delete a conversation and its associated data.

**Response:**
```json
{
    "status": "success",
    "message": "Conversation deleted successfully"
}
```

### 3. Memory Management

#### POST /memory/context
Store or update context information.

**Request Body:**
```json
{
    "key": "user_preferences_123",
    "value": {
        "language": "en",
        "technical_level": "advanced",
        "preferred_agents": ["research", "explainer"],
        "recent_topics": ["AI", "Machine Learning"]
    },
    "expiry": "2024-04-15T00:00:00Z",
    "metadata": {
        "source": "user_settings",
        "priority": "high"
    }
}
```

**Response:**
```json
{
    "status": "success",
    "key": "user_preferences_123",
    "expires_at": "2024-04-15T00:00:00Z"
}
```

#### GET /memory/context/{key}
Retrieve stored context.

**Response:**
```json
{
    "value": {
        "language": "en",
        "technical_level": "advanced",
        "preferred_agents": ["research", "explainer"],
        "recent_topics": ["AI", "Machine Learning"]
    },
    "metadata": {
        "created_at": "2024-03-15T10:00:00Z",
        "updated_at": "2024-03-15T11:00:00Z",
        "source": "user_settings",
        "priority": "high"
    }
}
```

### 4. Agent Control

#### POST /agents/execute
Execute a specific agent task.

**Request Body:**
```json
{
    "agent_type": "research",
    "task": "analyze_topic",
    "parameters": {
        "topic": "quantum computing",
        "depth": "comprehensive",
        "sources": ["academic", "technical"],
        "format": "structured"
    },
    "context": {
        "user_level": "advanced",
        "previous_knowledge": ["quantum mechanics basics"]
    }
}
```

**Response:**
```json
{
    "task_id": "task_789012",
    "status": "completed",
    "result": {
        "analysis": {
            "overview": "Quantum computing is...",
            "key_concepts": ["qubits", "superposition", "entanglement"],
            "current_state": "developing",
            "applications": ["cryptography", "optimization"]
        },
        "sources": [
            {
                "title": "Quantum Computing for Computer Scientists",
                "type": "academic",
                "relevance": 0.95
            }
        ],
        "recommendations": [
            {
                "topic": "quantum algorithms",
                "priority": "high",
                "reason": "foundational knowledge"
            }
        ]
    },
    "metadata": {
        "processing_time": 2.5,
        "models_used": ["gpt-4", "claude-2"],
        "tokens_used": 500
    }
}
```

## Error Responses

All endpoints may return the following error responses:

```json
{
    "detail": "Error message description",
    "status_code": 400,
    "error_code": "INVALID_REQUEST",
    "timestamp": "2024-03-15T12:00:00Z"
}
```

Common status codes and their meanings:
- 400: Bad Request - Invalid input parameters
- 401: Unauthorized - Invalid or missing authentication
- 403: Forbidden - Insufficient permissions
- 404: Not Found - Resource not found
- 429: Too Many Requests - Rate limit exceeded
- 500: Internal Server Error - Server-side error

## Rate Limiting

API requests are limited to:
- 100 requests per minute per IP address
- 1000 requests per hour per authenticated user

Rate limit headers in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1615910400
```

## WebSocket Endpoints

### /ws/conversations/{conversation_id}
Establish a WebSocket connection for real-time conversation updates.

**Connection:**
```
ws://localhost:8000/ws/conversations/{conversation_id}?token={jwt_token}
```

**Message Format:**
```json
{
    "type": "message|status|error",
    "data": {
        "content": "string",
        "metadata": {}
    },
    "timestamp": "ISO-8601"
}
```

**Event Types:**
- `message`: New message in conversation
- `status`: Processing status update
- `error`: Error notification
- `complete`: Task completion

## Examples

### Python Client Example

```python
import requests
from typing import Dict, Any

class AwareAgentClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def start_conversation(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/conversations",
            json={
                "message": message,
                "context": context
            },
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/conversations/{conversation_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def execute_agent_task(self, agent_type: str, task: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/agents/execute",
            json={
                "agent_type": agent_type,
                "task": task,
                "parameters": parameters
            },
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage example
client = AwareAgentClient(
    base_url="http://localhost:8000/api/v1",
    token="your_jwt_token"
)

# Start a conversation
response = client.start_conversation(
    message="What is semantic abstraction?",
    context={
        "domain": "AI",
        "mode": "research"
    }
)
```

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAPI Specification](http://localhost:8000/docs)
- [ReDoc Documentation](http://localhost:8000/redoc)
- [API Changelog](CHANGELOG.md)
- [Rate Limiting Policy](RATE_LIMITS.md) 