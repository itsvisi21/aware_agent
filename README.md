# Aware Agent

An intelligent agent-based research pipeline that powers smart workspaces with AI conversations.

## Features

- **Semantic Abstraction Layer**: Processes and structures natural language input using semantic dimensions and karaka roles.
- **Agent Orchestration Layer**: Coordinates multiple specialized agents (Planner, Research, Explainer, Validator) for comprehensive task handling.
- **Interaction Engine**: Manages conversation state, builds prompts, translates responses, and integrates feedback.
- **Execution & Memory Layer**: Handles task management, semantic logging, and persistent storage.
- **Modern UI**: Built with Next.js and Tailwind CSS, featuring a semantic conversation canvas, task manager, and context visualization.

## Getting Started

### Prerequisites

- Node.js 18.x or later
- npm or yarn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/aware-agent.git
   cd aware-agent
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

3. Start the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
aware-agent/
├── src/
│   ├── app/                  # Next.js app directory
│   │   ├── components/       # React components
│   │   ├── globals.css       # Global styles
│   │   ├── layout.tsx        # Root layout
│   │   └── page.tsx          # Main page
│   ├── semantic_abstraction/ # Semantic abstraction layer
│   ├── agent_orchestration/  # Agent orchestration layer
│   ├── interaction/          # Interaction engine
│   ├── execution/            # Execution & memory layer
│   └── types/                # TypeScript type definitions
├── tests/                    # Test files
├── public/                   # Static assets
└── package.json              # Project dependencies
```

## Development

### Running Tests

```bash
npm test
# or
yarn test
```

### Building for Production

```bash
npm run build
# or
yarn build
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
- [Tailwind CSS](https://tailwindcss.com/)
- [LangChain](https://github.com/hwchase17/langchain) 