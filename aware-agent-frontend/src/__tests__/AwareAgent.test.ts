import { AwareAgent } from '../app';
import { LocalLLM } from '../lib/llm/LocalLLM';
import { SemanticMemory } from '../lib/memory/SemanticMemory';
import { KarakaMapper } from '../lib/semantic/KarakaMapper';
import { TaskManager } from '../lib/tasks/TaskManager';
import { WebSocketService } from '../lib/websocket/WebSocketService';

jest.mock('../lib/llm/LocalLLM');
jest.mock('../lib/memory/SemanticMemory');
jest.mock('../lib/semantic/KarakaMapper');
jest.mock('../lib/tasks/TaskManager');
jest.mock('../lib/websocket/WebSocketService');

describe('AwareAgent', () => {
  let agent: AwareAgent;
  let mockLLM: jest.Mocked<LocalLLM>;
  let mockMemory: jest.Mocked<SemanticMemory>;
  let mockKarakaMapper: jest.Mocked<KarakaMapper>;
  let mockTaskManager: jest.Mocked<TaskManager>;
  let mockWebSocket: jest.Mocked<WebSocketService>;

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

    // Mock initialization methods
    mockLLM.initialize = jest.fn().mockResolvedValue(undefined);
    mockMemory.load = jest.fn().mockResolvedValue(undefined);
    mockWebSocket.connect = jest.fn().mockResolvedValue(undefined);

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

      // Mock the generate method instead of generateResponse
      mockLLM.generate = jest.fn().mockResolvedValue(expectedResponse);

      const response = await agent.processMessage(message);
      expect(response).toBe(expectedResponse);
      expect(mockLLM.generate).toHaveBeenCalledWith(message);
    });

    it('should handle processing errors', async () => {
      const message = 'Test message';
      mockLLM.generate = jest.fn().mockRejectedValue(new Error('Processing failed'));

      const response = await agent.processMessage(message);
      expect(response).toContain('Error: Processing failed');
    });
  });
}); 