# Aware Agent Frontend Documentation

## Overview

The Aware Agent frontend is a modern web application built with Next.js 13+ that provides an intuitive interface for interacting with AI agents. It features a semantic conversation canvas, real-time updates, and context-aware interactions.

## Tech Stack

### Core Technologies
- **Next.js 13+**: React framework with App Router
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **React Query**: Data fetching and state management
- **Zustand**: Lightweight state management
- **Shadcn/ui**: Reusable UI components

### Key Dependencies
```json
{
  "dependencies": {
    "next": "^14.0.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "typescript": "^5.0.0",
    "tailwindcss": "^3.3.0",
    "@tanstack/react-query": "^5.0.0",
    "zustand": "^4.4.0",
    "@radix-ui/react-dialog": "^1.0.0",
    "@radix-ui/react-dropdown-menu": "^1.0.0",
    "socket.io-client": "^4.7.0",
    "date-fns": "^2.30.0",
    "react-markdown": "^8.0.0"
  }
}
```

## Project Structure

```
aware-agent-frontend/
├── src/
│   ├── app/                    # Next.js app directory
│   │   ├── (auth)/            # Authentication routes
│   │   ├── (dashboard)/       # Dashboard routes
│   │   └── layout.tsx         # Root layout
│   ├── components/            # Reusable components
│   │   ├── ui/               # Shadcn/ui components
│   │   ├── chat/             # Chat interface components
│   │   └── agents/           # Agent-specific components
│   ├── lib/                  # Utility functions
│   │   ├── api/             # API client
│   │   ├── hooks/           # Custom hooks
│   │   └── utils/           # Helper functions
│   ├── styles/              # Global styles
│   └── types/               # TypeScript types
├── public/                  # Static assets
└── package.json            # Project dependencies
```

## Key Features

### 1. Semantic Chat Interface
- Real-time message updates
- Context-aware responses
- Markdown support
- Code highlighting
- File attachments

### 2. Agent Management
- Multiple agent types (Research, Builder, Teacher, Collaborator)
- Agent switching
- Agent-specific settings
- Performance metrics

### 3. Context Visualization
- Semantic relationship graphs
- Conversation history
- Context breadcrumbs
- Topic clustering

### 4. User Experience
- Dark/Light mode
- Responsive design
- Keyboard shortcuts
- Accessibility support

## Component Architecture

### Core Components

1. **Chat Interface**
   - MessageList
   - MessageInput
   - ContextPanel
   - AgentSelector

2. **Agent Components**
   - ResearchAgent
   - BuilderAgent
   - TeacherAgent
   - CollaboratorAgent

3. **UI Components**
   - Modal
   - Dropdown
   - Button
   - Input
   - Card

## State Management

### Global State (Zustand)
```typescript
interface AppState {
  conversations: Conversation[];
  currentAgent: AgentType;
  settings: UserSettings;
  theme: 'light' | 'dark';
}
```

### API State (React Query)
```typescript
const { data: conversation } = useQuery({
  queryKey: ['conversation', id],
  queryFn: () => fetchConversation(id)
});
```

## API Integration

### REST API Client
```typescript
class ApiClient {
  private baseUrl: string;
  private token: string;

  async startConversation(message: string, context: Context) {
    // Implementation
  }

  async getConversation(id: string) {
    // Implementation
  }
}
```

### WebSocket Integration
```typescript
const socket = io(WS_URL, {
  auth: {
    token: userToken
  }
});

socket.on('message', (data) => {
  // Handle real-time updates
});
```

## Styling System

### Tailwind Configuration
```javascript
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: '#0070f3',
        secondary: '#7928ca'
      }
    }
  }
}
```

### CSS Modules
```css
.chatContainer {
  @apply flex flex-col h-full;
}

.message {
  @apply p-4 rounded-lg;
}
```

## Development Setup

### Prerequisites
- Node.js 18.x or later
- npm or yarn

### Installation
```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test
```

## Testing Strategy

### Unit Tests
- Component testing with React Testing Library
- Hook testing with React Hooks Testing Library
- Utility function testing

### Integration Tests
- API integration testing
- WebSocket connection testing
- End-to-end user flows

## Performance Optimization

### Techniques
- Code splitting
- Image optimization
- Caching strategies
- Lazy loading

### Monitoring
- Performance metrics
- Error tracking
- User analytics

## Security

### Measures
- JWT authentication
- CSRF protection
- XSS prevention
- Input validation

## Deployment

### Environments
- Development
- Staging
- Production

### CI/CD Pipeline
```yaml
name: CI/CD
on:
  push:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
      - run: npm install
      - run: npm run build
      - run: npm test
```

## Additional Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [React Query Documentation](https://tanstack.com/query/latest)
- [Zustand Documentation](https://github.com/pmndrs/zustand)
- [Shadcn/ui Documentation](https://ui.shadcn.com/) 