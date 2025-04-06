import { TaskHandlers } from '../lib/tasks/TaskHandlers';
import { LocalLLM } from '../lib/llm/LocalLLM';
import { SemanticMemory } from '../lib/memory/SemanticMemory';
import { KarakaMapper } from '../lib/semantic/KarakaMapper';
import { 
  ResearchTask, 
  AnalysisTask, 
  GenerationTask, 
  ComparisonTask, 
  OrganizationTask 
} from '../lib/tasks/TaskTypes';
import { LLMConfig } from '../lib/types/llm';

jest.mock('../lib/llm/LocalLLM');
jest.mock('../lib/memory/SemanticMemory');
jest.mock('../lib/semantic/KarakaMapper');

describe('TaskHandlers', () => {
  let handlers: TaskHandlers;
  let mockLLM: jest.Mocked<LocalLLM>;
  let mockMemory: jest.Mocked<SemanticMemory>;
  let mockKarakaMapper: jest.Mocked<KarakaMapper>;

  beforeEach(() => {
    const llmConfig: LLMConfig = {
      model: 'test-model',
      endpoint: 'http://localhost:11434/api/generate'
    };

    mockLLM = new LocalLLM(llmConfig) as jest.Mocked<LocalLLM>;
    mockMemory = new SemanticMemory('./test-memory') as jest.Mocked<SemanticMemory>;
    mockKarakaMapper = new KarakaMapper(mockLLM) as jest.Mocked<KarakaMapper>;

    handlers = new TaskHandlers(mockLLM, mockMemory, mockKarakaMapper);
  });

  describe('handleTask', () => {
    it('should handle research tasks', async () => {
      const task: ResearchTask = {
        id: 'test-research',
        type: 'research',
        description: 'Test research task',
        status: 'pending',
        priority: 'medium',
        query: 'Test query',
        metadata: {
          created: new Date(),
          updated: new Date()
        }
      };

      mockLLM.generate.mockResolvedValue('Test research results');
      mockMemory.addMessage.mockResolvedValue(undefined);

      const result = await handlers.handleTask(task);

      expect(result.status).toBe('completed');
      expect(result.output).toBeDefined();
      expect(mockLLM.generate).toHaveBeenCalled();
      expect(mockMemory.addMessage).toHaveBeenCalled();
    });

    it('should handle analysis tasks', async () => {
      const task: AnalysisTask = {
        id: 'test-analysis',
        type: 'analysis',
        description: 'Test analysis task',
        status: 'pending',
        priority: 'medium',
        target: 'Test target',
        method: 'semantic',
        metadata: {
          created: new Date(),
          updated: new Date()
        }
      };

      mockLLM.generate.mockResolvedValue('Test analysis results');
      mockMemory.searchSemantic.mockResolvedValue([]);

      const result = await handlers.handleTask(task);

      expect(result.status).toBe('completed');
      expect(result.output).toBeDefined();
      expect(mockLLM.generate).toHaveBeenCalled();
      expect(mockMemory.searchSemantic).toHaveBeenCalled();
    });

    it('should handle generation tasks', async () => {
      const task: GenerationTask = {
        id: 'test-generation',
        type: 'generation',
        description: 'Test generation task',
        status: 'pending',
        priority: 'medium',
        template: 'Test template',
        context: { test: 'context' },
        format: 'markdown',
        metadata: {
          created: new Date(),
          updated: new Date()
        }
      };

      mockLLM.generate.mockResolvedValue('Test generated content');

      const result = await handlers.handleTask(task);

      expect(result.status).toBe('completed');
      expect(result.output).toBeDefined();
      expect(mockLLM.generate).toHaveBeenCalled();
    });

    it('should handle comparison tasks', async () => {
      const task: ComparisonTask = {
        id: 'test-comparison',
        type: 'comparison',
        description: 'Test comparison task',
        status: 'pending',
        priority: 'medium',
        items: ['item1', 'item2'],
        criteria: ['criterion1', 'criterion2'],
        method: 'semantic',
        metadata: {
          created: new Date(),
          updated: new Date()
        }
      };

      mockLLM.generate.mockResolvedValue('Test comparison results');
      mockMemory.searchSemantic.mockResolvedValue([]);

      const result = await handlers.handleTask(task);

      expect(result.status).toBe('completed');
      expect(result.output).toBeDefined();
      expect(mockLLM.generate).toHaveBeenCalled();
      expect(mockMemory.searchSemantic).toHaveBeenCalledTimes(2);
    });

    it('should handle organization tasks', async () => {
      const task: OrganizationTask = {
        id: 'test-organization',
        type: 'organization',
        description: 'Test organization task',
        status: 'pending',
        priority: 'medium',
        action: 'categorize',
        target: 'Test target',
        metadata: {
          created: new Date(),
          updated: new Date()
        }
      };

      mockLLM.generate.mockResolvedValue('{"categories": ["cat1", "cat2"]}');
      mockMemory.searchSemantic.mockResolvedValue([]);

      const result = await handlers.handleTask(task);

      expect(result.status).toBe('completed');
      expect(result.output).toBeDefined();
      expect(mockLLM.generate).toHaveBeenCalled();
      expect(mockMemory.searchSemantic).toHaveBeenCalled();
    });

    it('should handle task errors', async () => {
      const task: ResearchTask = {
        id: 'test-error',
        type: 'research',
        description: 'Test error task',
        status: 'pending',
        priority: 'medium',
        query: 'Test query',
        metadata: {
          created: new Date(),
          updated: new Date()
        }
      };

      mockLLM.generate.mockRejectedValue(new Error('Test error'));

      const result = await handlers.handleTask(task);

      expect(result.status).toBe('failed');
      expect(result.error).toBe('Test error');
    });
  });
}); 