import { AwareAgent } from '../app';
import { LocalLLM } from '../lib/llm/LocalLLM';
import { SemanticMemory } from '../lib/memory/SemanticMemory';
import { KarakaMapper } from '../lib/semantic/KarakaMapper';
import { TaskManager } from '../lib/tasks/TaskManager';
import { WebSocketService } from '../lib/websocket/WebSocketService';
import { AgentOrchestrator } from '../lib/agents/AgentOrchestrator';
import { ChatMessage } from '../lib/types/chat';
import { AgentContext } from '../lib/types/agent';
import { expect } from '@jest/globals';

jest.mock('../lib/llm/LocalLLM');
jest.mock('../lib/memory/SemanticMemory');
jest.mock('../lib/semantic/KarakaMapper');
jest.mock('../lib/tasks/TaskManager');
jest.mock('../lib/websocket/WebSocketService');
jest.mock('../lib/agents/AgentOrchestrator');

describe('AwareAgent', () => {
  let agent: AwareAgent;
  let mockLLM: jest.Mocked<LocalLLM>;
  let mockMemory: jest.Mocked<SemanticMemory>;
  let mockKarakaMapper: jest.Mocked<KarakaMapper>;
  let mockTaskManager: jest.Mocked<TaskManager>;
  let mockWebSocket: jest.Mocked<WebSocketService>;
  let mockOrchestrator: jest.Mocked<AgentOrchestrator>;

  beforeEach(() => {
    mockLLM = new LocalLLM({ model: 'test', endpoint: 'test' }) as jest.Mocked<LocalLLM>;
    mockMemory = new SemanticMemory('test') as jest.Mocked<SemanticMemory>;
    mockKarakaMapper = new KarakaMapper(mockLLM) as jest.Mocked<KarakaMapper>;
    mockTaskManager = new TaskManager(mockLLM, mockMemory, mockKarakaMapper, {
      maxConcurrentTasks: 2,
      retryAttempts: 1,
      retryDelay: 100
    }) as jest.Mocked<TaskManager>;
    mockWebSocket = new WebSocketService('ws://test') as jest.Mocked<WebSocketService>;

    const context: AgentContext = {
      currentGoal: null,
      conversationHistory: [],
      mode: 'exploration',
      metadata: {},
      role: 'test',
      memory: mockMemory,
      llm: mockLLM,
      agents: [],
      feedback: {
        sentiment: 'neutral',
        impact: 'medium',
        lastUpdate: new Date()
      }
    };

    mockOrchestrator = new AgentOrchestrator(
      'test',
      context,
      mockKarakaMapper,
      mockMemory,
      mockLLM
    ) as jest.Mocked<AgentOrchestrator>;

    // Mock initialization methods
    mockLLM.initialize = jest.fn().mockResolvedValue(undefined);
    mockMemory.load = jest.fn().mockResolvedValue(undefined);
    mockWebSocket.connect = jest.fn().mockResolvedValue(undefined);
    mockOrchestrator.process = jest.fn().mockResolvedValue({ 
      content: 'Test response', 
      confidence: 1, 
      reasoning: '', 
      nextSteps: [] 
    });

    // Create agent with string parameters
    agent = new AwareAgent('test-model', 'test-memory', 'test-context');
  });

  describe('initialize', () => {
    it('should initialize all components successfully', async () => {
      await agent.initialize();

      expect(mockLLM.initialize).toHaveBeenCalled();
      expect(mockMemory.load).toHaveBeenCalled();
    });

    it('should handle initialization errors', async () => {
      mockLLM.initialize = jest.fn().mockRejectedValue(new Error('LLM initialization failed'));

      try {
        await agent.initialize();
        fail('Expected initialization to fail');
      } catch (error) {
        expect(error).toBeInstanceOf(Error);
        if (error instanceof Error) {
          expect(error.message).toBe('LLM initialization failed');
        }
      }
    });
  });

  describe('processMessage', () => {
    beforeEach(async () => {
      await agent.initialize();
    });

    it('should process message successfully', async () => {
      const message = 'Test message';
      const expectedResponse = 'Test response';

      mockOrchestrator.process = jest.fn().mockResolvedValue({ 
        content: expectedResponse,
        confidence: 1,
        reasoning: '',
        nextSteps: []
      });

      const response = await agent.processMessage(message);
      expect(response).toBe(expectedResponse);
      expect(mockOrchestrator.process).toHaveBeenCalledWith(expect.objectContaining({
        content: message,
        metadata: expect.any(Object)
      }));
    });

    it('should handle processing errors', async () => {
      const message = 'Test message';
      mockOrchestrator.process = jest.fn().mockRejectedValue(new Error('Processing failed'));

      const response = await agent.processMessage(message);
      expect(response).toContain('Error: Processing failed');
    });
  });
}); 