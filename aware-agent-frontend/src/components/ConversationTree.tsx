import React, { useState } from 'react';
import { useConversation } from '../contexts/ConversationContext';
import { Conversation } from '../services/conversation';

interface ConversationNodeProps {
    conversation: Conversation;
    level: number;
    onSelect: (id: string) => void;
    onFork: (id: string, title: string) => void;
}

const ConversationNode: React.FC<ConversationNodeProps> = ({
    conversation,
    level,
    onSelect,
    onFork,
}) => {
    const [isExpanded, setIsExpanded] = useState(true);
    const [forkTitle, setForkTitle] = useState('');

    const handleFork = () => {
        if (forkTitle.trim()) {
            onFork(conversation.id, forkTitle);
            setForkTitle('');
        }
    };

    return (
        <div className="ml-4">
            <div
                className="flex items-center space-x-2 p-2 hover:bg-gray-100 rounded cursor-pointer"
                onClick={() => onSelect(conversation.id)}
            >
                <button
                    onClick={(e) => {
                        e.stopPropagation();
                        setIsExpanded(!isExpanded);
                    }}
                    className="text-gray-500 hover:text-gray-700"
                >
                    {isExpanded ? '▼' : '▶'}
                </button>
                <span className="font-medium">{conversation.title}</span>
                <span className="text-sm text-gray-500">
                    ({conversation.messages.length} messages)
                </span>
            </div>

            {isExpanded && (
                <div className="ml-8 space-y-2">
                    <div className="flex space-x-2">
                        <input
                            type="text"
                            value={forkTitle}
                            onChange={(e) => setForkTitle(e.target.value)}
                            placeholder="New conversation title"
                            className="flex-1 p-1 border rounded"
                            onClick={(e) => e.stopPropagation()}
                        />
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                handleFork();
                            }}
                            className="px-2 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
                        >
                            Fork
                        </button>
                    </div>
                    {conversation.children.map((childId) => (
                        <ConversationNode
                            key={childId}
                            conversation={conversation}
                            level={level + 1}
                            onSelect={onSelect}
                            onFork={onFork}
                        />
                    ))}
                </div>
            )}
        </div>
    );
};

export const ConversationTree: React.FC = () => {
    const {
        conversations,
        currentConversation,
        setCurrentConversation,
        forkConversation,
    } = useConversation();

    return (
        <div className="w-64 bg-white rounded-lg shadow-lg p-4">
            <h3 className="font-semibold text-gray-700 mb-4">Conversations</h3>
            <div className="space-y-2">
                {conversations.map((conversation) => (
                    <ConversationNode
                        key={conversation.id}
                        conversation={conversation}
                        level={0}
                        onSelect={setCurrentConversation}
                        onFork={forkConversation}
                    />
                ))}
            </div>
        </div>
    );
}; 