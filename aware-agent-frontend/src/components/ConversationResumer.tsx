import React, { useState, useEffect } from 'react';
import { SemanticKernel } from '../lib/kernel/SemanticKernel';
import { ChatMessage } from '../lib/types/chat';

interface ConversationResumerProps {
  kernel: SemanticKernel;
}

export const ConversationResumer: React.FC<ConversationResumerProps> = ({ kernel }) => {
  const [conversations, setConversations] = useState<Array<{
    id: string;
    title: string;
    lastMessage: string;
    timestamp: Date;
  }>>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadConversations();
  }, [kernel]);

  const loadConversations = async () => {
    setIsLoading(true);
    try {
      const history = await kernel.getConversationHistory();
      const uniqueConversations = history.reduce((acc, msg) => {
        const conversationId = msg.metadata?.conversationId || 'default';
        if (!acc[conversationId]) {
          acc[conversationId] = {
            id: conversationId,
            title: msg.metadata?.title || 'Untitled Conversation',
            lastMessage: msg.content,
            timestamp: new Date(msg.metadata?.timestamp || Date.now())
          };
        }
        return acc;
      }, {} as Record<string, any>);

      setConversations(Object.values(uniqueConversations));
    } catch (error) {
      console.error('Error loading conversations:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleResume = async (conversationId: string) => {
    try {
      await kernel.resumeConversation(conversationId);
      // The conversation will be loaded in the ConversationCanvas component
    } catch (error) {
      console.error('Error resuming conversation:', error);
    }
  };

  return (
    <div className="relative">
      <button
        onClick={() => loadConversations()}
        className="px-3 py-1 rounded-md bg-gray-500 hover:bg-gray-600 text-white focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2"
      >
        Resume
      </button>

      {isLoading && (
        <div className="absolute top-0 right-0 mt-10 w-64 bg-white rounded-lg shadow-lg p-4">
          <div className="text-center text-gray-600">Loading conversations...</div>
        </div>
      )}

      {!isLoading && conversations.length > 0 && (
        <div className="absolute top-0 right-0 mt-10 w-64 bg-white rounded-lg shadow-lg p-4">
          <h3 className="text-sm font-semibold mb-2">Previous Conversations</h3>
          <div className="space-y-2">
            {conversations.map((conv) => (
              <button
                key={conv.id}
                onClick={() => handleResume(conv.id)}
                className="w-full text-left p-2 hover:bg-gray-100 rounded-md"
              >
                <div className="font-medium text-sm">{conv.title}</div>
                <div className="text-xs text-gray-500 truncate">
                  {conv.lastMessage}
                </div>
                <div className="text-xs text-gray-400">
                  {conv.timestamp.toLocaleDateString()}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}; 