export interface Message {
  id: string;
  content: string;
  sender: 'user' | 'agent';
  timestamp: string;
  metadata?: {
    agent?: string;
    role?: string;
    mode?: 'research' | 'build' | 'teach' | 'collab';
    context?: any;
    semanticContext?: any;
  };
}

export interface Branch {
  id: string;
  name: string;
  parentId?: string;
  metadata?: Record<string, any>;
}

export interface ConversationContext {
  goal: string;
  branches: Branch[];
  activeBranch?: string;
  metadata?: Record<string, any>;
}

export type AgentStatus = 'idle' | 'thinking' | 'active' | 'error';

export interface AgentState {
  status: AgentStatus;
  message?: string;
} 