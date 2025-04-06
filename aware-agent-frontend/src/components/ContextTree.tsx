import React, { useState } from 'react';
import { Message } from '../types/chat';

interface ContextNode {
    id: string;
    content: string;
    children: ContextNode[];
    expanded?: boolean;
}

interface ContextTreeProps {
    messages: Message[];
}

export const ContextTree: React.FC<ContextTreeProps> = ({ messages }) => {
    const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());

    const toggleNode = (nodeId: string) => {
        setExpandedNodes(prev => {
            const newSet = new Set(prev);
            if (newSet.has(nodeId)) {
                newSet.delete(nodeId);
            } else {
                newSet.add(nodeId);
            }
            return newSet;
        });
    };

    const buildContextTree = (messages: Message[]): ContextNode[] => {
        return messages.map(message => ({
            id: message.id,
            content: message.content,
            children: [],
            expanded: expandedNodes.has(message.id)
        }));
    };

    const contextTree = buildContextTree(messages);

    const renderNode = (node: ContextNode) => {
        const isExpanded = expandedNodes.has(node.id);
        
        return (
            <div key={node.id} className="ml-4">
                <div 
                    className="flex items-center cursor-pointer hover:bg-gray-100 p-1 rounded"
                    onClick={() => toggleNode(node.id)}
                >
                    {node.children.length > 0 && (
                        <span className="mr-2">
                            {isExpanded ? '▼' : '▶'}
                        </span>
                    )}
                    <span className="text-sm text-gray-700 truncate">
                        {node.content}
                    </span>
                </div>
                {isExpanded && node.children.length > 0 && (
                    <div className="ml-4">
                        {node.children.map(child => renderNode(child))}
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="w-64 h-full bg-white rounded-lg shadow-lg p-4 overflow-y-auto">
            <h3 className="font-semibold text-gray-700 mb-4">Conversation Context</h3>
            <div className="space-y-2">
                {contextTree.map(node => renderNode(node))}
            </div>
        </div>
    );
}; 