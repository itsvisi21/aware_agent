import React, { useState, useEffect } from 'react';
import { SemanticKernel, SemanticBreadcrumb, Branch, ConversationMode } from '../lib/kernel/SemanticKernel';
import { ChatMessage, ChatMessageMetadata } from '../lib/types/chat';
import ConversationGraph from './ConversationGraph';
import { ModelSettings } from './ModelSettings';
import { LLMConfig } from '../lib/types/llm';
import { MemoryVisualizer } from './MemoryVisualizer';
import { MemorySearch } from './MemorySearch';
import { MarkdownEditor } from './MarkdownEditor';
import { CodeEditor } from './CodeEditor';
import { TaskManager } from './TaskManager';
import { TeachingMode } from './TeachingMode';
import { TeamMode } from './TeamMode';
import { PlanningMode } from './PlanningMode';
import { Task } from '../lib/types/task';
import { ToolManager } from './ToolManager';
import { ToolChain } from './ToolChain';

interface ConversationCanvasProps {
  kernel: SemanticKernel;
}

interface Bookmark {
  id: string;
  messageId: string;
  title: string;
  description: string;
  timestamp: Date;
}

export const ConversationCanvas: React.FC<ConversationCanvasProps> = ({ kernel }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [breadcrumbs, setBreadcrumbs] = useState<SemanticBreadcrumb[]>([]);
  const [currentBranch, setCurrentBranch] = useState<string | null>(null);
  const [branches, setBranches] = useState<Branch[]>([]);
  const [showGraph, setShowGraph] = useState(false);
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);
  const [selectedMessage, setSelectedMessage] = useState<string | null>(null);
  const [currentMode, setCurrentMode] = useState<ConversationMode>('exploration');
  const [modeContext, setModeContext] = useState<Record<string, any>>({});
  const [showSettings, setShowSettings] = useState(false);
  const [showMemory, setShowMemory] = useState(false);
  const [showEditor, setShowEditor] = useState(false);
  const [editorContent, setEditorContent] = useState('');
  const [showCodeEditor, setShowCodeEditor] = useState(false);
  const [codeContent, setCodeContent] = useState('');
  const [codeLanguage, setCodeLanguage] = useState('javascript');
  const [showTaskManager, setShowTaskManager] = useState(false);
  const [showMemorySearch, setShowMemorySearch] = useState(false);
  const [showToolManager, setShowToolManager] = useState(false);
  const [showToolChain, setShowToolChain] = useState(false);

  useEffect(() => {
    const loadConversation = async () => {
      const history = await kernel.getConversationHistory();
      setMessages(history);
      
      // Load semantic breadcrumbs
      const semanticPath = await kernel.getSemanticPath();
      setBreadcrumbs(semanticPath);
      
      // Load available branches
      const conversationBranches = await kernel.getConversationBranches();
      setBranches(conversationBranches);
      setCurrentBranch(kernel.getCurrentBranch()?.id || null);
      
      // Load conversation mode
      const modeContext = await kernel.getConversationMode();
      setCurrentMode(modeContext.mode);
      setModeContext(modeContext.metadata);
    };
    loadConversation();
  }, [kernel]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isProcessing) return;

    setIsProcessing(true);
    const metadata: ChatMessageMetadata = {
      timestamp: new Date(),
      branch: currentBranch || undefined
    };

    const userMessage: ChatMessage = {
      role: 'user',
      content: input,
      metadata
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');

    try {
      const response = await kernel.processMessage(input);
      const assistantMetadata: ChatMessageMetadata = {
        timestamp: new Date(),
        branch: currentBranch || undefined
      };

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response,
        metadata: assistantMetadata
      };
      setMessages(prev => [...prev, assistantMessage]);
      
      // Update semantic breadcrumbs
      const updatedPath = await kernel.getSemanticPath();
      setBreadcrumbs(updatedPath);
    } catch (error) {
      console.error('Error processing message:', error);
      const errorMetadata: ChatMessageMetadata = {
        timestamp: new Date(),
        error: true,
        branch: currentBranch || undefined
      };

      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error while processing your message.',
        metadata: errorMetadata
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleCreateBranch = async () => {
    const branchName = prompt('Enter branch name:');
    if (branchName) {
      const newBranch = await kernel.createConversationBranch(branchName);
      setBranches(prev => [...prev, newBranch]);
      setCurrentBranch(newBranch.id);
    }
  };

  const handleSwitchBranch = async (branchId: string) => {
    await kernel.switchConversationBranch(branchId);
    setCurrentBranch(branchId);
    const history = await kernel.getConversationHistory();
    setMessages(history);
  };

  const handleCreateBookmark = async (messageId: string) => {
    const message = messages.find(m => m.metadata.timestamp.toString() === messageId);
    if (!message) return;

    const title = prompt('Enter bookmark title:');
    if (!title) return;

    const description = prompt('Enter bookmark description:');
    if (!description) return;

    const bookmark: Bookmark = {
      id: Date.now().toString(),
      messageId,
      title,
      description,
      timestamp: new Date()
    };

    setBookmarks(prev => [...prev, bookmark]);
    await kernel.storeBookmark(bookmark);
  };

  const handleExportBranch = async () => {
    if (!currentBranch) return;

    const branch = branches.find(b => b.id === currentBranch);
    if (!branch) return;

    const branchMessages = messages.filter(m => m.metadata.branch === currentBranch);
    const exportContent = branchMessages.map(m => `${m.role}: ${m.content}`).join('\n\n');

    // Create a downloadable file
    const blob = new Blob([exportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation-${branch.name}-${new Date().toISOString()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleResumeFromMessage = async (messageId: string) => {
    const message = messages.find(m => m.metadata.timestamp.toString() === messageId);
    if (!message) return;

    // Create a new branch from this point
    const branchName = prompt('Enter name for new branch:');
    if (!branchName) return;

    const newBranch = await kernel.createConversationBranch(branchName);
    setBranches(prev => [...prev, newBranch]);
    setCurrentBranch(newBranch.id);

    // Set the context to this message
    await kernel.setContextFromMessage(message);
  };

  const handleModeChange = async (mode: ConversationMode) => {
    let metadata = {};
    
    if (mode === 'team') {
      const teamName = prompt('Enter team name:');
      if (!teamName) return;
      
      const teamMembers = prompt('Enter team members (comma-separated):');
      if (!teamMembers) return;
      
      metadata = {
        teamContext: {
          name: teamName,
          members: teamMembers.split(',').map(m => m.trim())
        }
      };
    }
    
    await kernel.setConversationMode(mode, metadata);
    setCurrentMode(mode);
    setModeContext(metadata);
  };

  const handleConfigUpdate = async (config: Partial<LLMConfig>) => {
    await kernel.getLLM().updateConfig(config);
  };

  const handleEditorSave = async () => {
    if (!editorContent.trim()) return;

    const message: ChatMessage = {
      role: 'user',
      content: editorContent,
      metadata: {
        timestamp: new Date(),
        type: 'markdown',
        branch: currentBranch || undefined
      }
    };

    await kernel.processMessage(editorContent);
    setEditorContent('');
    setShowEditor(false);
  };

  const handleCodeExecute = async () => {
    if (!codeContent.trim()) return;

    const message: ChatMessage = {
      role: 'user',
      content: codeContent,
      metadata: {
        timestamp: new Date(),
        type: 'code',
        language: codeLanguage,
        branch: currentBranch || undefined
      }
    };

    await kernel.processMessage(codeContent);
    setCodeContent('');
    setShowCodeEditor(false);
  };

  const handleTaskCreate = async (task: Omit<Task, 'id' | 'createdAt' | 'updatedAt'>) => {
    const message: ChatMessage = {
      role: 'user',
      content: `Task created: ${task.title}\nDescription: ${task.description}\nStatus: ${task.status}\nPriority: ${task.priority}`,
      metadata: {
        timestamp: new Date(),
        type: 'task',
        branch: currentBranch || undefined
      }
    };

    await kernel.processMessage(message.content);
  };

  const handleTaskUpdate = async (taskId: string, updates: Partial<Task>) => {
    const message: ChatMessage = {
      role: 'user',
      content: `Task updated: ${Object.entries(updates)
        .map(([key, value]) => `${key}: ${value}`)
        .join('\n')}`,
      metadata: {
        timestamp: new Date(),
        type: 'task',
        branch: currentBranch || undefined
      }
    };

    await kernel.processMessage(message.content);
  };

  const handleTaskDelete = async (taskId: string) => {
    const message: ChatMessage = {
      role: 'user',
      content: `Task deleted: ${taskId}`,
      metadata: {
        timestamp: new Date(),
        type: 'task',
        branch: currentBranch || undefined
      }
    };

    await kernel.processMessage(message.content);
  };

  const handleTeachingMessage = async (message: ChatMessage) => {
    await kernel.processMessage(message.content);
  };

  const handleTeachingQuestion = async (question: string) => {
    const message: ChatMessage = {
      role: 'user',
      content: `Question: ${question}`,
      metadata: {
        timestamp: new Date(),
        type: 'teaching',
        branch: currentBranch || undefined
      }
    };
    await kernel.processMessage(message.content);
  };

  const handleTeachingExample = async (example: string) => {
    const message: ChatMessage = {
      role: 'user',
      content: `Example request: ${example}`,
      metadata: {
        timestamp: new Date(),
        type: 'teaching',
        branch: currentBranch || undefined
      }
    };
    await kernel.processMessage(message.content);
  };

  const handleTeachingExercise = async (exercise: string) => {
    const message: ChatMessage = {
      role: 'user',
      content: `Exercise request: ${exercise}`,
      metadata: {
        timestamp: new Date(),
        type: 'teaching',
        branch: currentBranch || undefined
      }
    };
    await kernel.processMessage(message.content);
  };

  const handleTeamMessage = async (message: ChatMessage) => {
    await kernel.processMessage(message.content);
  };

  const handleTaskAssign = async (task: string, assignee: string) => {
    const message: ChatMessage = {
      role: 'user',
      content: `Task assigned: ${task}\nAssignee: ${assignee}`,
      metadata: {
        timestamp: new Date(),
        type: 'task',
        branch: currentBranch || undefined
      }
    };
    await kernel.processMessage(message.content);
  };

  const handleStatusUpdate = async (status: string) => {
    const message: ChatMessage = {
      role: 'user',
      content: `Status update: ${status}`,
      metadata: {
        timestamp: new Date(),
        type: 'text',
        branch: currentBranch || undefined
      }
    };
    await kernel.processMessage(message.content);
  };

  const handleMeetingStart = async (agenda: string) => {
    const message: ChatMessage = {
      role: 'user',
      content: `Meeting started with agenda: ${agenda}`,
      metadata: {
        timestamp: new Date(),
        type: 'text',
        branch: currentBranch || undefined
      }
    };
    await kernel.processMessage(message.content);
  };

  const handlePlanningMessage = async (message: ChatMessage) => {
    await kernel.processMessage(message.content);
  };

  const handleGoalSet = async (goal: string) => {
    const message: ChatMessage = {
      role: 'user',
      content: `Goal set: ${goal}`,
      metadata: {
        timestamp: new Date(),
        type: 'text',
        branch: currentBranch || undefined
      }
    };
    await kernel.processMessage(message.content);
  };

  const handleMilestoneCreate = async (milestone: string) => {
    const message: ChatMessage = {
      role: 'user',
      content: `Milestone created: ${milestone}`,
      metadata: {
        timestamp: new Date(),
        type: 'text',
        branch: currentBranch || undefined
      }
    };
    await kernel.processMessage(message.content);
  };

  const handleResourceAdd = async (resource: string) => {
    const message: ChatMessage = {
      role: 'user',
      content: `Resource added: ${resource}`,
      metadata: {
        timestamp: new Date(),
        type: 'text',
        branch: currentBranch || undefined
      }
    };
    await kernel.processMessage(message.content);
  };

  const handleTimelineUpdate = async (timeline: string) => {
    const message: ChatMessage = {
      role: 'user',
      content: `Timeline updated: ${timeline}`,
      metadata: {
        timestamp: new Date(),
        type: 'text',
        branch: currentBranch || undefined
      }
    };
    await kernel.processMessage(message.content);
  };

  const handleMemoryResultSelect = (message: ChatMessage) => {
    // Handle memory search result selection
    setSelectedMessage(message.metadata.timestamp.toString());
    setShowMemorySearch(false);
  };

  return (
    <div className="flex flex-col h-[600px]">
      {/* Semantic Breadcrumbs */}
      <div className="bg-gray-100 p-2 flex items-center space-x-2 text-sm">
        {breadcrumbs.map((crumb, index) => (
          <React.Fragment key={crumb.id}>
            {index > 0 && <span className="text-gray-400">/</span>}
            <span className={`px-2 py-1 rounded ${
              crumb.type === 'goal' ? 'bg-blue-100 text-blue-800' :
              crumb.type === 'milestone' ? 'bg-green-100 text-green-800' :
              crumb.type === 'task' ? 'bg-yellow-100 text-yellow-800' :
              'bg-purple-100 text-purple-800'
            }`}>
              {crumb.label}
            </span>
          </React.Fragment>
        ))}
      </div>

      {/* Branch Controls */}
      <div className="bg-gray-50 p-2 flex items-center space-x-4 border-b">
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium">Branch:</span>
          <select
            value={currentBranch || ''}
            onChange={(e) => handleSwitchBranch(e.target.value)}
            className="text-sm border rounded p-1"
          >
            {branches.map(branch => (
              <option key={branch.id} value={branch.id}>
                {branch.name}
              </option>
            ))}
          </select>
        </div>
        <button
          onClick={handleCreateBranch}
          className="text-sm px-2 py-1 bg-purple-100 text-purple-800 rounded hover:bg-purple-200"
        >
          New Branch
        </button>
        <button
          onClick={() => setShowGraph(!showGraph)}
          className="text-sm px-2 py-1 bg-blue-100 text-blue-800 rounded hover:bg-blue-200"
        >
          {showGraph ? 'Hide Graph' : 'Show Graph'}
        </button>
      </div>

      {/* Mode Controls */}
      <div className="bg-gray-50 p-2 flex items-center space-x-4 border-b">
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium">Mode:</span>
          <select
            value={currentMode}
            onChange={(e) => handleModeChange(e.target.value as ConversationMode)}
            className="text-sm border rounded p-1"
          >
            <option value="exploration">Exploration</option>
            <option value="teaching">Teaching</option>
            <option value="team">Team</option>
            <option value="planning">Planning</option>
          </select>
        </div>
        {currentMode === 'team' && (
          <div className="text-sm text-gray-600">
            Team: {modeContext.teamContext?.name || 'Not set'}
          </div>
        )}
      </div>

      {/* Bookmarks and Export Controls */}
      <div className="bg-gray-50 p-2 flex items-center space-x-4 border-b">
        <button
          onClick={handleExportBranch}
          className="text-sm px-2 py-1 bg-green-100 text-green-800 rounded hover:bg-green-200"
        >
          Export Branch
        </button>
        <div className="flex-1" />
        <div className="text-sm text-gray-600">
          {bookmarks.length} bookmarks
        </div>
      </div>

      {/* Memory Button */}
      <div className="bg-gray-50 p-2 flex items-center space-x-4 border-b">
        <button
          onClick={() => setShowMemory(!showMemory)}
          className="text-sm px-2 py-1 bg-purple-100 text-purple-800 rounded hover:bg-purple-200"
        >
          {showMemory ? 'Hide Memory' : 'Show Memory'}
        </button>
      </div>

      {/* Settings Button */}
      <div className="bg-gray-50 p-2 flex items-center space-x-4 border-b">
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="text-sm px-2 py-1 bg-gray-100 text-gray-800 rounded hover:bg-gray-200"
        >
          {showSettings ? 'Hide Settings' : 'Show Settings'}
        </button>
      </div>

      {/* Editor Button */}
      <div className="bg-gray-50 p-2 flex items-center space-x-4 border-b">
        <button
          onClick={() => setShowEditor(!showEditor)}
          className="text-sm px-2 py-1 bg-blue-100 text-blue-800 rounded hover:bg-blue-200"
        >
          {showEditor ? 'Hide Editor' : 'Show Editor'}
        </button>
      </div>

      {/* Code Editor Button */}
      <div className="bg-gray-50 p-2 flex items-center space-x-4 border-b">
        <button
          onClick={() => setShowCodeEditor(!showCodeEditor)}
          className="text-sm px-2 py-1 bg-blue-100 text-blue-800 rounded hover:bg-blue-200"
        >
          {showCodeEditor ? 'Hide Code Editor' : 'Show Code Editor'}
        </button>
      </div>

      {/* Task Manager Button */}
      <div className="bg-gray-50 p-2 flex items-center space-x-4 border-b">
        <button
          onClick={() => setShowTaskManager(!showTaskManager)}
          className="text-sm px-2 py-1 bg-blue-100 text-blue-800 rounded hover:bg-blue-200"
        >
          {showTaskManager ? 'Hide Tasks' : 'Show Tasks'}
        </button>
      </div>

      {/* Memory Search Button */}
      <div className="bg-gray-50 p-2 flex items-center space-x-4 border-b">
        <button
          onClick={() => setShowMemorySearch(!showMemorySearch)}
          className="text-sm px-2 py-1 bg-blue-100 text-blue-800 rounded hover:bg-blue-200"
        >
          {showMemorySearch ? 'Hide Memory Search' : 'Show Memory Search'}
        </button>
      </div>

      {/* Tool Manager Button */}
      <div className="bg-gray-50 p-2 flex items-center space-x-4 border-b">
        <button
          onClick={() => setShowToolManager(!showToolManager)}
          className="text-sm px-2 py-1 bg-blue-100 text-blue-800 rounded hover:bg-blue-200"
        >
          {showToolManager ? 'Hide Tools' : 'Show Tools'}
        </button>
      </div>

      {/* Tool Chain Button */}
      <div className="bg-gray-50 p-2 flex items-center space-x-4 border-b">
        <button
          onClick={() => setShowToolChain(!showToolChain)}
          className="text-sm px-2 py-1 bg-blue-100 text-blue-800 rounded hover:bg-blue-200"
        >
          {showToolChain ? 'Hide Tool Chains' : 'Show Tool Chains'}
        </button>
      </div>

      {/* Teaching Mode Panel */}
      {currentMode === 'teaching' && (
        <div className="border-b h-[300px]">
          <TeachingMode
            onMessage={handleTeachingMessage}
            onQuestion={handleTeachingQuestion}
            onExample={handleTeachingExample}
            onExercise={handleTeachingExercise}
          />
        </div>
      )}

      {/* Planning Mode Panel */}
      {currentMode === 'planning' && (
        <div className="border-b h-[300px]">
          <PlanningMode
            onMessage={handlePlanningMessage}
            onGoalSet={handleGoalSet}
            onMilestoneCreate={handleMilestoneCreate}
            onResourceAdd={handleResourceAdd}
            onTimelineUpdate={handleTimelineUpdate}
          />
        </div>
      )}

      {/* Settings Panel */}
      {showSettings && (
        <div className="border-b">
          <ModelSettings
            llm={kernel.getLLM()}
            onConfigUpdate={handleConfigUpdate}
          />
        </div>
      )}

      {/* Memory Panel */}
      {showMemory && (
        <div className="border-b">
          <MemoryVisualizer memory={kernel.getMemory()} />
        </div>
      )}

      {/* Memory Search Panel */}
      {showMemorySearch && (
        <div className="border-b h-[300px]">
          <MemorySearch
            memory={kernel.getMemory()}
            onResultSelect={handleMemoryResultSelect}
          />
        </div>
      )}

      {/* Editor Panel */}
      {showEditor && (
        <div className="border-b h-[300px]">
          <MarkdownEditor
            content={editorContent}
            onChange={setEditorContent}
            onSave={handleEditorSave}
          />
        </div>
      )}

      {/* Code Editor Panel */}
      {showCodeEditor && (
        <div className="border-b h-[300px]">
          <CodeEditor
            code={codeContent}
            language={codeLanguage}
            onChange={setCodeContent}
            onExecute={handleCodeExecute}
          />
        </div>
      )}

      {/* Task Manager Panel */}
      {showTaskManager && (
        <div className="border-b h-[300px]">
          <TaskManager
            onTaskCreate={handleTaskCreate}
            onTaskUpdate={handleTaskUpdate}
            onTaskDelete={handleTaskDelete}
          />
        </div>
      )}

      {/* Tool Manager Panel */}
      {showToolManager && (
        <div className="border-b h-[300px]">
          <ToolManager kernel={kernel} />
        </div>
      )}

      {/* Tool Chain Panel */}
      {showToolChain && (
        <div className="border-b h-[300px]">
          <ToolChain
            kernel={kernel}
            tools={kernel.getTools()}
          />
        </div>
      )}

      {/* Conversation Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${
              message.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            <div
              className={`max-w-[80%] rounded-lg p-4 ${
                message.role === 'user'
                  ? 'bg-blue-500 text-white'
                  : message.metadata.error
                  ? 'bg-red-100 text-red-800'
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              <div className="whitespace-pre-wrap">{message.content}</div>
              <div className="text-xs mt-1 opacity-70 flex items-center space-x-2">
                <span>{message.metadata.timestamp.toLocaleTimeString()}</span>
                {message.metadata.branch && (
                  <span className="px-1.5 py-0.5 bg-purple-200 text-purple-800 rounded">
                    {branches.find(b => b.id === message.metadata.branch)?.name}
                  </span>
                )}
                <button
                  onClick={() => handleCreateBookmark(message.metadata.timestamp.toString())}
                  className="ml-2 text-xs px-1.5 py-0.5 bg-yellow-100 text-yellow-800 rounded hover:bg-yellow-200"
                >
                  Bookmark
                </button>
                <button
                  onClick={() => handleResumeFromMessage(message.metadata.timestamp.toString())}
                  className="ml-2 text-xs px-1.5 py-0.5 bg-blue-100 text-blue-800 rounded hover:bg-blue-200"
                >
                  Resume From Here
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Conversation Graph */}
      {showGraph && (
        <div className="h-[300px] border-t">
          <ConversationGraph
            conversationId={kernel.getCurrentConversationId()}
            onNodeSelect={(nodeId) => {
              // Handle node selection - e.g., scroll to message
              console.log('Selected node:', nodeId);
            }}
          />
        </div>
      )}

      {/* Input Area */}
      <div className="border-t border-gray-200 p-4">
        <form onSubmit={handleSubmit} className="flex space-x-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            disabled={isProcessing}
          />
          <button
            type="submit"
            disabled={isProcessing}
            className={`px-4 py-2 rounded-md ${
              isProcessing
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-500 hover:bg-blue-600'
            } text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2`}
          >
            {isProcessing ? 'Processing...' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  );
}; 