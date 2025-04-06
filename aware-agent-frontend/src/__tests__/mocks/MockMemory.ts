import { vi } from 'vitest';
import { SemanticMemory, MemoryNode, Message, MemoryGraph, Connection } from '../../lib/memory/types';
import { ChatMessage } from '../../lib/types/chat';

export class MockMemory implements SemanticMemory {
  public branches: Map<string, string[]> = new Map();
  public rootNode: MemoryNode | null = null;
  public conversationId: string = `mock_${Date.now()}`;
  public conversationHistory: Message[] = [];
  public currentContext: string = 'main';
  public isLoaded: boolean = true;
  public storagePath: string = 'mock-storage';

  private nodes: Map<string, MemoryNode> = new Map();
  private connections: Map<string, Connection> = new Map();

  constructor() {
    this.rootNode = null;
    this.conversationId = 'test-conversation';
    this.conversationHistory = [];
    this.currentContext = 'main';
    this.isLoaded = false;
  }

  async getNodes(): Promise<MemoryNode[]> {
    return Array.from(this.nodes.values());
  }

  async getConnections(): Promise<Connection[]> {
    return Array.from(this.connections.values());
  }

  async search(query: string): Promise<MemoryNode[]> {
    return Array.from(this.nodes.values()).filter(node => 
      node.content.toLowerCase().includes(query.toLowerCase())
    );
  }

  async addNode(node: Partial<MemoryNode>): Promise<MemoryNode> {
    const newNode: MemoryNode = {
      id: node.id || `node_${Date.now()}`,
      content: node.content || '',
      type: node.type || 'concept',
      metadata: node.metadata || {},
      tags: node.tags || [],
      timestamp: node.timestamp || new Date(),
      context: node.context || this.currentContext
    };
    this.nodes.set(newNode.id, newNode);
    return newNode;
  }

  async removeNode(id: string): Promise<boolean> {
    return this.nodes.delete(id);
  }

  async addConnection(connection: Partial<Connection>): Promise<Connection> {
    const newConnection: Connection = {
      id: connection.id || `conn_${Date.now()}`,
      source: connection.source || '',
      target: connection.target || '',
      type: connection.type || 'related',
      metadata: connection.metadata || {}
    };
    this.connections.set(newConnection.id, newConnection);
    return newConnection;
  }

  async removeConnection(id: string): Promise<boolean> {
    return this.connections.delete(id);
  }

  async filter(type: string): Promise<MemoryNode[]> {
    return Array.from(this.nodes.values()).filter(node => node.type === type);
  }

  async addMessage(message: Message): Promise<Message> {
    const newMessage: Message = {
      id: message.id || `msg_${Date.now()}`,
      role: message.role,
      content: message.content,
      timestamp: message.timestamp || new Date(),
      metadata: message.metadata || {}
    };
    this.conversationHistory.push(newMessage);
    return newMessage;
  }

  async getMessages(): Promise<Message[]> {
    return this.conversationHistory;
  }

  async createBranch(branchId: string, parentId: string): Promise<boolean> {
    if (!this.branches.has(branchId)) {
      this.branches.set(branchId, [parentId]);
      return true;
    }
    return false;
  }

  async mergeBranch(sourceBranchId: string, targetBranchId: string): Promise<boolean> {
    const sourceNodes = this.branches.get(sourceBranchId);
    const targetNodes = this.branches.get(targetBranchId);
    if (sourceNodes && targetNodes) {
      this.branches.set(targetBranchId, [...targetNodes, ...sourceNodes]);
      this.branches.delete(sourceBranchId);
      return true;
    }
    return false;
  }

  async switchContext(contextId: string): Promise<boolean> {
    this.currentContext = contextId;
    return true;
  }

  async getContext(): Promise<string> {
    return this.currentContext;
  }

  async initialize(): Promise<void> {
    this.isLoaded = true;
  }

  async save(): Promise<void> {
    // Mock save implementation
  }

  async load(): Promise<void> {
    this.isLoaded = true;
  }

  async clear(): Promise<void> {
    this.nodes.clear();
    this.connections.clear();
    this.branches.clear();
    this.conversationHistory = [];
    this.rootNode = null;
  }

  async exportMemory(): Promise<any> {
    return {
      nodes: Array.from(this.nodes.values()),
      connections: Array.from(this.connections.values()),
      branches: Array.from(this.branches.entries()),
      conversationHistory: this.conversationHistory
    };
  }

  async importMemory(data: any): Promise<boolean> {
    if (data.nodes) {
      this.nodes = new Map(data.nodes.map((node: MemoryNode) => [node.id, node]));
    }
    if (data.connections) {
      this.connections = new Map(data.connections.map((conn: Connection) => [conn.id, conn]));
    }
    if (data.branches) {
      this.branches = new Map(data.branches);
    }
    if (data.conversationHistory) {
      this.conversationHistory = data.conversationHistory;
    }
    return true;
  }

  async analyze(): Promise<Record<string, number>> {
    return {
      nodeCount: this.nodes.size,
      connectionCount: this.connections.size,
      branchCount: this.branches.size,
      messageCount: this.conversationHistory.length
    };
  }

  async prune(): Promise<boolean> {
    // Mock prune implementation
    return true;
  }

  async backup(): Promise<boolean> {
    // Mock backup implementation
    return true;
  }

  async restore(): Promise<boolean> {
    // Mock restore implementation
    return true;
  }

  async validate(): Promise<boolean> {
    // Mock validate implementation
    return true;
  }
} 