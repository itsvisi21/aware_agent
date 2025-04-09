# Aware Agent Integration Guide

## Overview
This guide provides step-by-step instructions for integrating the Aware Agent system into your application.

## Prerequisites
- Python 3.8+
- WebSocket client library
- AsyncIO support
- Database access (if using persistence)

## Installation

### Backend Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables:
   ```bash
   export DATABASE_URL=your_database_url
   export CACHE_TTL=300
   ```

### Frontend Setup
1. Install WebSocket client:
   ```bash
   npm install websocket
   ```
2. Import required components:
   ```javascript
   import { WebSocketService } from './services/websocket';
   import { ConversationProvider } from './contexts/ConversationContext';
   ```

## Integration Steps

### 1. WebSocket Connection
```javascript
const wsService = new WebSocketService('ws://your-server/ws');

wsService.connect()
  .then(() => {
    console.log('Connected to WebSocket server');
  })
  .catch((error) => {
    console.error('Connection failed:', error);
  });
```

### 2. Message Handling
```javascript
// Send message
const message = {
  type: 'message',
  content: 'Your message here',
  sender: 'user',
  agent: 'research'
};

wsService.sendMessage(message)
  .then((response) => {
    console.log('Response:', response);
  })
  .catch((error) => {
    console.error('Error:', error);
  });

// Receive messages
wsService.onMessage((message) => {
  console.log('Received:', message);
});
```

### 3. Conversation Management
```javascript
// Wrap your app with ConversationProvider
function App() {
  return (
    <ConversationProvider>
      {/* Your app components */}
    </ConversationProvider>
  );
}

// Use conversation context
const { conversations, addMessage } = useConversation();

// Add message to conversation
addMessage({
  content: 'Message content',
  sender: 'user',
  timestamp: Date.now()
});
```

### 4. Agent Selection
```javascript
// Select appropriate agent based on task
function selectAgent(task) {
  switch (task.type) {
    case 'research':
      return 'research';
    case 'implementation':
      return 'builder';
    case 'explanation':
      return 'teacher';
    case 'coordination':
      return 'collaborator';
    default:
      return 'research';
  }
}
```

## Best Practices

### 1. Error Handling
```javascript
try {
  const response = await wsService.sendMessage(message);
  // Handle response
} catch (error) {
  if (error.code === 'CONNECTION_ERROR') {
    // Handle connection error
  } else if (error.code === 'MESSAGE_ERROR') {
    // Handle message error
  }
}
```

### 2. State Management
```javascript
// Use appropriate caching
const cachedResponse = await cache.get(key);
if (!cachedResponse) {
  const response = await wsService.sendMessage(message);
  await cache.set(key, response, ttl);
}
```

### 3. Performance Optimization
```javascript
// Batch messages when possible
const messages = [...];
const responses = await Promise.all(
  messages.map(msg => wsService.sendMessage(msg))
);
```

## Common Use Cases

### 1. Research Task
```javascript
async function handleResearchTask(topic) {
  const message = {
    type: 'message',
    content: `Research ${topic}`,
    sender: 'user',
    agent: 'research'
  };
  
  const response = await wsService.sendMessage(message);
  // Process research results
}
```

### 2. Implementation Task
```javascript
async function handleImplementationTask(requirements) {
  const message = {
    type: 'message',
    content: JSON.stringify(requirements),
    sender: 'user',
    agent: 'builder'
  };
  
  const response = await wsService.sendMessage(message);
  // Process implementation results
}
```

### 3. Educational Task
```javascript
async function handleEducationalTask(topic) {
  const message = {
    type: 'message',
    content: `Explain ${topic}`,
    sender: 'user',
    agent: 'teacher'
  };
  
  const response = await wsService.sendMessage(message);
  // Process educational content
}
```

## Troubleshooting

### Common Issues
1. **Connection Issues**
   - Check WebSocket URL
   - Verify network connectivity
   - Check server status

2. **Message Errors**
   - Validate message format
   - Check agent availability
   - Verify permissions

3. **Performance Issues**
   - Optimize message size
   - Implement caching
   - Handle concurrent requests

### Debugging
```javascript
// Enable debug logging
wsService.setDebug(true);

// Monitor WebSocket events
wsService.on('open', () => console.log('Connection opened'));
wsService.on('close', () => console.log('Connection closed'));
wsService.on('error', (error) => console.error('Error:', error));
```

## Support
For additional support:
- Check documentation
- Review error logs
- Contact support team 