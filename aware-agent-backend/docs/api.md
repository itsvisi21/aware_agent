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
    "token_type": "bearer"
}
```

### 2. Conversation Management

#### POST /conversations
Start a new conversation with the AI agent.

**Request Body:**
```json
{
    "message": "string",
    "context": {
        "domain": "string",
        "mode": "string"
    }
}
```

**Response:**
```json
{
    "conversation_id": "string",
    "response": "string",
    "context": {
        "semantic_roles": [],
        "entities": []
    }
}
```

#### GET /conversations/{conversation_id}
Retrieve conversation history.

**Response:**
```json
{
    "messages": [
        {
            "role": "string",
            "content": "string",
            "timestamp": "string"
        }
    ]
}
```

### 3. Memory Management

#### POST /memory/context
Store or update context information.

**Request Body:**
```json
{
    "key": "string",
    "value": {},
    "expiry": "string"
}
```

#### GET /memory/context/{key}
Retrieve stored context.

**Response:**
```json
{
    "value": {},
    "metadata": {
        "created_at": "string",
        "updated_at": "string"
    }
}
```

### 4. Agent Control

#### POST /agents/execute
Execute a specific agent task.

**Request Body:**
```json
{
    "agent_type": "string",
    "task": "string",
    "parameters": {}
}
```

**Response:**
```json
{
    "task_id": "string",
    "status": "string",
    "result": {}
}
```

## Error Responses

All endpoints may return the following error responses:

```json
{
    "detail": "string",
    "status_code": number
}
```

Common status codes:
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 500: Internal Server Error

## Rate Limiting

API requests are limited to:
- 100 requests per minute per IP address
- 1000 requests per hour per authenticated user

## WebSocket Endpoints

### /ws/conversations/{conversation_id}
Establish a WebSocket connection for real-time conversation updates.

**Message Format:**
```json
{
    "type": "string",
    "data": {}
}
```

## Examples

### Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"
TOKEN = "your_jwt_token"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Start a conversation
response = requests.post(
    f"{BASE_URL}/conversations",
    json={
        "message": "What is semantic abstraction?",
        "context": {
            "domain": "AI",
            "mode": "research"
        }
    },
    headers=headers
)
```

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAPI Specification](http://localhost:8000/docs)
- [ReDoc Documentation](http://localhost:8000/redoc) 