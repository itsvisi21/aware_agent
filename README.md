# Aware Agent

An intelligent agent-based research pipeline that powers smart workspaces with AI conversations. This project combines a modern Next.js frontend with a powerful Python backend to create a comprehensive AI-assisted research and development environment.

## Project Overview

Aware Agent is designed to help users conduct research, analyze information, and develop ideas through intelligent conversation with AI agents. The system uses semantic abstraction and agent orchestration to provide meaningful, context-aware assistance.

### Frontend (Next.js)
The frontend is built with Next.js 13+ and provides a modern, interactive interface for users to:
- Engage in natural language conversations with AI agents
- Visualize semantic relationships and context
- Manage research tasks and workflows
- View real-time analysis and insights

**Tech Stack:**
- Next.js 13+ (App Router)
- TypeScript
- Tailwind CSS
- React Query
- Zustand (State Management)
- Shadcn/ui (UI Components)

### Backend (Python)
The backend is a robust Python service that handles:
- Natural Language Processing and semantic analysis
- Agent orchestration and task management
- Memory and context management
- Data persistence and retrieval

**Tech Stack:**
- FastAPI (Web Framework)
- LangChain (LLM Orchestration)
- spaCy (NLP)
- scikit-learn (Machine Learning)
- SQLAlchemy (Database ORM)
- Redis (Caching)
- ChromaDB (Vector Database)

## Features

- **Semantic Abstraction Layer**: Processes and structures natural language input using semantic dimensions and karaka roles.
- **Agent Orchestration Layer**: Coordinates multiple specialized agents (Planner, Research, Explainer, Validator) for comprehensive task handling.
- **Interaction Engine**: Manages conversation state, builds prompts, translates responses, and integrates feedback.
- **Execution & Memory Layer**: Handles task management, semantic logging, and persistent storage.
- **Modern UI**: Built with Next.js and Tailwind CSS, featuring a semantic conversation canvas, task manager, and context visualization.

## Getting Started

### Prerequisites

- Node.js 18.x or later
- Python 3.9+
- npm or yarn
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/aware-agent.git
   cd aware-agent
   ```

2. Frontend Setup:
   ```bash
   cd aware-agent-frontend
   npm install
   # or
   yarn install
   ```

3. Backend Setup:
   ```bash
   cd aware-agent-backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. Start Development Servers:
   - Frontend (from aware-agent-frontend directory):
     ```bash
     npm run dev
     # or
     yarn dev
     ```
   - Backend (from aware-agent-backend directory):
     ```bash
     uvicorn src.main:app --reload
     ```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
aware-agent/
├── aware-agent-frontend/     # Next.js frontend application
│   ├── src/
│   │   ├── app/             # Next.js app directory
│   │   ├── components/      # React components
│   │   └── lib/            # Utility functions
│   └── package.json
│
├── aware-agent-backend/      # Python backend service
│   ├── src/
│   │   ├── api/            # FastAPI routes
│   │   ├── core/           # Core business logic
│   │   ├── models/         # Data models
│   │   └── services/       # Service layer
│   └── requirements.txt
│
└── docs/                    # Project documentation
```

## Documentation

- [Development Setup Guide](dev_setup.md) - Detailed setup instructions
- [Architecture Guide](guide.md) - System architecture and design decisions
- [API Documentation](aware-agent-backend/docs/api.md) - Backend API specifications
- [Frontend Guide](docs/frontend.md) - Frontend development guidelines

## Development

### Running Tests

Frontend:
```bash
cd aware-agent-frontend
npm test
# or
yarn test
```

Backend:
```bash
cd aware-agent-backend
pytest
```

### Building for Production

Frontend:
```bash
cd aware-agent-frontend
npm run build
# or
yarn build
```

Backend:
```bash
cd aware-agent-backend
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Next.js](https://nextjs.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://github.com/hwchase17/langchain)
- [Tailwind CSS](https://tailwindcss.com/)
- [spaCy](https://spacy.io/) 