import { TaskManager } from '../lib/tasks/TaskManager';
import { LocalLLM } from '../lib/llm/LocalLLM';
import { SemanticMemory } from '../lib/memory/SemanticMemory';
import { KarakaMapper } from '../lib/semantic/KarakaMapper';
import { ResearchTask, TaskStatus } from '../lib/tasks/TaskTypes';

jest.mock('../lib/llm/LocalLLM');
jest.mock('../lib/memory/SemanticMemory');
jest.mock('../lib/semantic/KarakaMapper');
jest.mock('../lib/tasks/TaskHandlers');

describe('TaskManager', () => {
  let manager: TaskManager;
  let mockLLM: jest.Mocked<LocalLLM>;
  let mockMemory: jest.Mocked<SemanticMemory>;
  let mockKarakaMapper: jest.Mocked<KarakaMapper>;

  beforeEach(() => {
    mockLLM = new LocalLLM({ model: 'test', endpoint: 'test' }) as jest.Mocked<LocalLLM>;
    mockMemory = new SemanticMemory('test') as jest.Mocked<SemanticMemory>;
    mockKarakaMapper = new KarakaMapper(mockLLM) as jest.Mocked<KarakaMapper>;

    manager = new TaskManager(mockLLM, mockMemory, mockKarakaMapper, {
      maxConcurrentTasks: 2,
      retryAttempts: 1,
      retryDelay: 100
    });
  });

  describe('addTask', () => {
    it('should execute task immediately when under concurrency limit', async () => {
      const task: ResearchTask = {
        id: 'test-task',
        type: 'research',
        description: 'Test task',
        query: 'Test query',
        status: 'pending',
        priority: 'medium',
        metadata: {
          created: new Date('2025-04-05T22:15:23.677Z'),
          updated: new Date('2025-04-05T22:15:23.677Z'),
          tags: []
        }
      };

      await manager.addTask(task);
      await new Promise(resolve => setTimeout(resolve, 0));

      expect(manager.getActiveTasks()).toContainEqual(task);
    });

    it('should queue task when at concurrency limit', async () => {
      const tasks: ResearchTask[] = [
        {
          id: 'task-1',
          type: 'research',
          description: 'Task 1',
          query: 'Query 1',
          status: 'pending',
          priority: 'medium',
          metadata: {
            created: new Date(),
            updated: new Date(),
            tags: []
          }
        },
        {
          id: 'task-2',
          type: 'research',
          description: 'Task 2',
          query: 'Query 2',
          status: 'pending',
          priority: 'medium',
          metadata: {
            created: new Date(),
            updated: new Date(),
            tags: []
          }
        },
        {
          id: 'task-3',
          type: 'research',
          description: 'Task 3',
          query: 'Query 3',
          status: 'pending',
          priority: 'medium',
          metadata: {
            created: new Date(),
            updated: new Date(),
            tags: []
          }
        }
      ];

      // Add tasks up to concurrency limit
      await manager.addTask(tasks[0]);
      await manager.addTask(tasks[1]);
      await manager.addTask(tasks[2]);

      await new Promise(resolve => setTimeout(resolve, 0));

      expect(manager.getActiveTasks()).toHaveLength(2);
      expect(manager.getQueuedTasks()).toHaveLength(1);
    });
  });

  describe('task status and results', () => {
    it('should track task status', async () => {
      const task: ResearchTask = {
        id: 'test-task',
        type: 'research',
        description: 'Test task',
        query: 'Test query',
        status: 'pending',
        priority: 'medium',
        metadata: {
          created: new Date(),
          updated: new Date(),
          tags: []
        }
      };

      await manager.addTask(task);
      expect(manager.getTaskStatus(task.id)).toBe('pending');

      // Wait for task to complete
      await new Promise(resolve => setTimeout(resolve, 0));
      expect(manager.getTaskStatus(task.id)).toBe('completed');
    });

    it('should retrieve task results', async () => {
      const task: ResearchTask = {
        id: 'test-task',
        type: 'research',
        description: 'Test task',
        query: 'Test query',
        status: 'pending',
        priority: 'medium',
        metadata: {
          created: new Date(),
          updated: new Date(),
          tags: []
        }
      };

      await manager.addTask(task);
      await new Promise(resolve => setTimeout(resolve, 0));

      const result = manager.getTaskResult(task.id);
      expect(result).toBeDefined();
      expect(result?.taskId).toBe(task.id);
    });
  });

  describe('task cancellation', () => {
    it('should cancel active task', async () => {
      const task: ResearchTask = {
        id: 'test-task',
        type: 'research',
        description: 'Test task',
        query: 'Test query',
        status: 'pending',
        priority: 'medium',
        metadata: {
          created: new Date(),
          updated: new Date(),
          tags: []
        }
      };

      await manager.addTask(task);
      await new Promise(resolve => setTimeout(resolve, 0));

      const cancelled = await manager.cancelTask(task.id);
      expect(cancelled).toBe(true);
      expect(manager.getActiveTasks()).not.toContainEqual(task);
    });
  });

  describe('task queue management', () => {
    it('should process next task when active task completes', async () => {
      const tasks: ResearchTask[] = [
        {
          id: 'task-1',
          type: 'research',
          description: 'Task 1',
          query: 'Query 1',
          status: 'pending',
          priority: 'medium',
          metadata: {
            created: new Date(),
            updated: new Date(),
            tags: []
          }
        },
        {
          id: 'task-2',
          type: 'research',
          description: 'Task 2',
          query: 'Query 2',
          status: 'pending',
          priority: 'medium',
          metadata: {
            created: new Date(),
            updated: new Date(),
            tags: []
          }
        },
        {
          id: 'task-3',
          type: 'research',
          description: 'Task 3',
          query: 'Query 3',
          status: 'pending',
          priority: 'medium',
          metadata: {
            created: new Date(),
            updated: new Date(),
            tags: []
          }
        }
      ];

      // Add tasks up to concurrency limit
      await manager.addTask(tasks[0]);
      await manager.addTask(tasks[1]);
      await manager.addTask(tasks[2]);

      await new Promise(resolve => setTimeout(resolve, 0));

      expect(manager.getActiveTasks()).toHaveLength(2);
      expect(manager.getQueuedTasks()).toHaveLength(1);

      // Wait for tasks to complete
      await new Promise(resolve => setTimeout(resolve, 100));

      expect(manager.getActiveTasks()).toHaveLength(0);
      expect(manager.getQueuedTasks()).toHaveLength(0);
    });
  });
}); 