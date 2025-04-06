import { render, screen, fireEvent, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MemoryVisualizer } from '../../components/MemoryVisualizer';
import { MockMemory } from '../mocks/MockMemory';
import { MemoryNode } from '../../lib/memory/types';
import { SemanticMemory } from '../../lib/memory/SemanticMemory';

jest.mock('../mocks/MockMemory');

describe('MemoryVisualizer', () => {
  let mockMemory: SemanticMemory;
  let mockNodes: MemoryNode[];

  beforeEach(() => {
    mockNodes = [
      {
        id: '1',
        content: 'Test Node 1',
        type: 'concept',
        metadata: {
          timestamp: new Date(),
          source: 'test'
        },
        tags: ['test'],
        timestamp: new Date(),
        context: 'main'
      },
      {
        id: '2',
        content: 'Test Node 2',
        type: 'concept',
        metadata: {
          timestamp: new Date(),
          source: 'test'
        },
        tags: ['test'],
        timestamp: new Date(),
        context: 'main'
      }
    ];

    const mock = new MockMemory();
    mock.getNodes = jest.fn().mockResolvedValue(mockNodes);
    mock.addNode = jest.fn().mockImplementation((node) => Promise.resolve({
      id: '3',
      content: node.content || '',
      type: node.type || 'concept',
      metadata: node.metadata || {},
      tags: node.tags || [],
      timestamp: node.timestamp || new Date(),
      context: node.context || 'main'
    }));
    mock.branches = new Map();
    mock.rootNode = null;
    mock.conversationId = 'test';
    mock.conversationHistory = [];
    mock.currentContext = 'main';
    mock.isLoaded = true;
    mock.storagePath = 'test';
    mock.getConnections = jest.fn().mockResolvedValue([]);
    mock.search = jest.fn().mockResolvedValue(mockNodes);
    mock.removeNode = jest.fn().mockResolvedValue(true);
    mock.addConnection = jest.fn().mockResolvedValue({ id: '1', source: '1', target: '2', type: 'test' });
    mock.removeConnection = jest.fn().mockResolvedValue(true);
    mock.filter = jest.fn().mockResolvedValue(mockNodes);
    mock.addMessage = jest.fn().mockResolvedValue({ id: '1', role: 'user', content: 'test', timestamp: new Date() });
    mock.getMessages = jest.fn().mockResolvedValue([]);
    mock.createBranch = jest.fn().mockResolvedValue(true);
    mock.mergeBranch = jest.fn().mockResolvedValue(true);
    mock.switchContext = jest.fn().mockResolvedValue(true);
    mock.getContext = jest.fn().mockResolvedValue('main');
    mock.initialize = jest.fn().mockResolvedValue(undefined);
    mock.save = jest.fn().mockResolvedValue(undefined);
    mock.load = jest.fn().mockResolvedValue(undefined);
    mock.clear = jest.fn().mockResolvedValue(undefined);
    mock.exportMemory = jest.fn().mockResolvedValue({ nodes: mockNodes });
    mock.importMemory = jest.fn().mockResolvedValue(true);
    mock.analyze = jest.fn().mockResolvedValue({ nodeCount: 2 });
    mock.prune = jest.fn().mockResolvedValue(true);
    mock.backup = jest.fn().mockResolvedValue(true);
    mock.restore = jest.fn().mockResolvedValue(true);
    mock.validate = jest.fn().mockResolvedValue(true);
    mock.findRelevantNodes = jest.fn().mockImplementation(async (query) => {
      return mockNodes.filter(node => node.content.toLowerCase().includes(query.toLowerCase()));
    });
    mock.storeNode = jest.fn().mockImplementation(async (node) => {
      const newNode = {
        id: Date.now().toString(),
        timestamp: new Date(),
        ...node,
      };
      mockNodes.push(newNode);
      return newNode;
    });

    mockMemory = mock as unknown as SemanticMemory;
  });

  it('renders memory nodes', async () => {
    render(<MemoryVisualizer memory={mockMemory} />);

    // Wait for the nodes to be loaded and rendered
    const node1 = await screen.findByText('Test Node 1', { exact: false });
    const node2 = await screen.findByText('Test Node 2', { exact: false });

    expect(node1).toBeInTheDocument();
    expect(node2).toBeInTheDocument();
  });

  it('filters nodes based on search input', async () => {
    render(<MemoryVisualizer memory={mockMemory} />);

    // Wait for initial nodes to load
    await screen.findByText('Test Node 1', { exact: false });

    const searchInput = screen.getByPlaceholderText('Search memory...');
    const searchButton = screen.getByRole('button', { name: 'Search' });

    await act(async () => {
      fireEvent.change(searchInput, { target: { value: 'Node 1' } });
      fireEvent.click(searchButton);
    });

    const node1 = screen.getByText('Test Node 1', { exact: false });
    const node2 = screen.queryByText('Test Node 2', { exact: false });

    expect(node1).toBeInTheDocument();
    expect(node2).toBe(null);
  });

  it('exports memory to file', async () => {
    render(<MemoryVisualizer memory={mockMemory} />);

    // Wait for initial nodes to load
    await screen.findByText('Test Node 1', { exact: false });

    const exportButton = screen.getByRole('button', { name: /Export Memory/ });

    // Mock the file download
    const mockUrl = 'blob:test';
    const createObjectURL = jest.fn().mockReturnValue(mockUrl);
    global.URL.createObjectURL = createObjectURL;

    await act(async () => {
      fireEvent.click(exportButton);
    });

    expect(createObjectURL).toHaveBeenCalled();
  });

  it('imports memory from file', async () => {
    render(<MemoryVisualizer memory={mockMemory} />);

    // Wait for initial nodes to load
    await screen.findByText('Test Node 1', { exact: false });

    const file = new File(['{"nodes": []}'], 'memory.json', { type: 'application/json' });
    const input = screen.getByLabelText('Import Memory');

    await act(async () => {
      Object.defineProperty(input, 'files', { value: [file] });
      fireEvent.change(input);
    });

    expect(mockMemory.addNode).toHaveBeenCalled();
  });
}); 