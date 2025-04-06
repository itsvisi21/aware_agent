import { LocalLLM } from './lib/llm/LocalLLM';
import { SemanticMemory } from './lib/memory/SemanticMemory';
import { KarakaMapper } from './lib/semantic/KarakaMapper';
import { AgentOrchestrator } from './lib/agents/AgentOrchestrator';
import { AgentContext } from './lib/types/agent';
import { ChatMessage } from './lib/types/chat';

export class AwareAgent {
  private llm: LocalLLM;
  private memory: SemanticMemory;
  private karakaMapper: KarakaMapper;
  private orchestrator: AgentOrchestrator;
  private context: AgentContext;

  constructor(
    llmModel: string = 'mistral',
    memoryPath: string = './memory',
    contextPath: string = './context'
  ) {
    // Initialize core components
    this.llm = new LocalLLM(llmModel);
    this.memory = new SemanticMemory(memoryPath);
    this.karakaMapper = new KarakaMapper(this.llm);

    // Initialize agent context
    this.context = {
      llm: this.llm,
      memory: this.memory,
      agents: [],
      feedback: {
        history: [],
        preferences: {}
      }
    };

    // Initialize orchestrator
    this.orchestrator = new AgentOrchestrator(
      'main_orchestrator',
      this.context,
      this.karakaMapper,
      this.memory
    );
  }

  public async initialize(): Promise<void> {
    try {
      // Initialize LLM
      await this.llm.initialize();

      // Load memory
      await this.memory.load();

      console.log('AwareAgent initialized successfully');
    } catch (error) {
      console.error('Failed to initialize AwareAgent:', error);
      throw error;
    }
  }

  public async processMessage(message: string): Promise<string> {
    try {
      const chatMessage: ChatMessage = {
        content: message,
        metadata: {
          timestamp: new Date(),
          source: 'user'
        }
      };

      const response = await this.orchestrator.generateResponse(chatMessage);
      return response.content;
    } catch (error) {
      console.error('Error processing message:', error);
      return `Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
    }
  }

  public async exportSession(format: 'json' | 'markdown' = 'json'): Promise<string> {
    return await this.orchestrator.exportSession(format);
  }

  public getState(): any {
    return this.orchestrator.getState();
  }
}

// Example usage
async function main() {
  const agent = new AwareAgent();
  
  try {
    await agent.initialize();
    
    // Example conversation
    const messages = [
      "Help me design an AI paper based on Sanskrit-based programming principles.",
      "I want to focus on the relationship between Paninian grammar and formal logic.",
      "Can you suggest some key references for this topic?"
    ];

    for (const message of messages) {
      console.log('\nUser:', message);
      const response = await agent.processMessage(message);
      console.log('Agent:', response);
    }

    // Export session
    const sessionLog = await agent.exportSession('markdown');
    console.log('\nSession Log:\n', sessionLog);
  } catch (error) {
    console.error('Error in main:', error);
  }
}

// Run the example if this file is executed directly
if (require.main === module) {
  main().catch(console.error);
} 