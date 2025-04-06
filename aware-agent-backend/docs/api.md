# Aware Agent API Documentation

## Overview
The Aware Agent API provides a WebSocket-based interface for interacting with various specialized agents (Research, Builder, Teacher, Collaborator) and includes REST endpoints for system monitoring and management.

## WebSocket API

### Connection
- **URL**: `ws://api.aware-agent.com/ws`
- **Protocol**: WebSocket
- **Authentication**: Bearer token in query parameter

### Message Format
```json
{
    "type": "message",
    "content": "Your message here",
    "sender": "user_id",
    "agent": "research|builder|teacher|collaborator"
}
```

### Response Format
```json
{
    "type": "response",
    "content": "Agent response",
    "agent": "agent_type",
    "timestamp": "ISO-8601 timestamp"
}
```

### Error Format
```json
{
    "type": "error",
    "code": "error_code",
    "message": "Error description",
    "timestamp": "ISO-8601 timestamp"
}
```

## REST API

### Health Check
- **Endpoint**: `GET /health`
- **Description**: Comprehensive system health check
- **Response**:
```json
{
    "status": "healthy",
    "components": {
        "cache": true,
        "database": true,
        "message_batcher": true
    },
    "metrics": {
        "uptime": "duration",
        "message_count": 123,
        "error_count": 0
    }
}
```

### Metrics
- **Endpoint**: `GET /metrics`
- **Description**: Get system metrics
- **Response**:
```json
{
    "messages_processed": 1234,
    "active_connections": 10,
    "cache_hits": 500,
    "cache_misses": 50,
    "average_response_time": 0.123
}
```

### Status
- **Endpoint**: `GET /status`
- **Description**: Get system status
- **Response**:
```json
{
    "status": "operational",
    "version": "1.0.0",
    "uptime": "2h 30m",
    "last_health_check": "2024-01-01T12:00:00Z"
}
```

## Agent Types

### Research Agent
- **Purpose**: Information gathering and analysis
- **Capabilities**:
  - Topic research
  - Information synthesis
  - Context building
- **Example Usage**:
```json
{
    "type": "message",
    "content": "Research quantum computing",
    "agent": "research"
}
```

### Builder Agent
- **Purpose**: Code generation and implementation
- **Capabilities**:
  - Code generation
  - Project setup
  - Technical guidance
- **Example Usage**:
```json
{
    "type": "message",
    "content": "Create a Python web server",
    "agent": "builder"
}
```

### Teacher Agent
- **Purpose**: Educational guidance
- **Capabilities**:
  - Concept explanation
  - Learning path management
  - Knowledge assessment
- **Example Usage**:
```json
{
    "type": "message",
    "content": "Explain machine learning",
    "agent": "teacher"
}
```

### Collaborator Agent
- **Purpose**: Team coordination
- **Capabilities**:
  - Task management
  - Progress tracking
  - Team coordination
- **Example Usage**:
```json
{
    "type": "message",
    "content": "Track project progress",
    "agent": "collaborator"
}
```

## Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| 1001 | Invalid message format | Check message structure |
| 1002 | Unauthorized access | Provide valid authentication |
| 1003 | Agent not available | Try different agent or retry |
| 1004 | Rate limit exceeded | Wait before sending more messages |
| 1005 | System error | Contact support |

## Best Practices

1. **Connection Management**
   - Implement reconnection logic
   - Handle connection errors gracefully
   - Monitor connection status

2. **Message Handling**
   - Validate message format
   - Implement timeout handling
   - Process responses asynchronously

3. **Error Handling**
   - Implement retry logic
   - Log errors appropriately
   - Provide user feedback

4. **Performance**
   - Batch messages when possible
   - Cache responses when appropriate
   - Monitor response times

## Rate Limits

- **WebSocket**: 100 messages per minute
- **REST API**: 100 requests per minute
- **Burst**: 200 requests in 5 seconds

## Security

1. **Authentication**
   - Use Bearer tokens
   - Rotate tokens regularly
   - Implement token validation

2. **Data Protection**
   - Use HTTPS/WSS
   - Encrypt sensitive data
   - Validate input data

3. **Access Control**
   - Implement role-based access
   - Monitor access patterns
   - Log security events 