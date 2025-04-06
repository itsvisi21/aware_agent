'use client';

import { useState, useEffect } from 'react';
import { MemoryNode } from '@/lib/memory/types';
import { StorageService } from '@/lib/storage/StorageService';

interface ConversationVisualizerProps {
  conversationId: string;
  onNodeSelect?: (nodeId: string) => void;
  searchQuery?: string;
  selectedTags?: string[];
}

export default function ConversationVisualizer({
  conversationId,
  onNodeSelect,
  searchQuery = '',
  selectedTags = [],
}: ConversationVisualizerProps) {
  const [nodes, setNodes] = useState<Map<string, MemoryNode>>(new Map());
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());

  useEffect(() => {
    const stored = StorageService.loadConversation(conversationId);
    if (stored) {
      setNodes(stored.nodes);
      // Expand the root node by default
      if (stored.rootNodeId) {
        setExpandedNodes(new Set([stored.rootNodeId]));
      }
    }
  }, [conversationId]);

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

  const getTagColor = (tag: string): string => {
    // Generate a consistent color based on the tag
    const colors = [
      'bg-blue-100 text-blue-800',
      'bg-green-100 text-green-800',
      'bg-yellow-100 text-yellow-800',
      'bg-purple-100 text-purple-800',
      'bg-pink-100 text-pink-800',
      'bg-indigo-100 text-indigo-800',
    ];
    const index = tag.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    return colors[index % colors.length];
  };

  const getImportanceColor = (importance: number) => {
    if (importance > 0.8) return 'bg-red-100 text-red-800';
    if (importance > 0.6) return 'bg-orange-100 text-orange-800';
    if (importance > 0.4) return 'bg-yellow-100 text-yellow-800';
    return 'bg-gray-100 text-gray-800';
  };

  const getConfidenceIndicator = (confidence: number) => {
    const width = `${confidence * 100}%`;
    return (
      <div className="w-full h-1 bg-gray-200 rounded-full overflow-hidden">
        <div
          className="h-full bg-blue-500 transition-all duration-300"
          style={{ width }}
        />
      </div>
    );
  };

  const renderNode = (node: MemoryNode, level: number = 0) => {
    const isSelected = node.id === selectedNode;
    const isExpanded = expandedNodes.has(node.id);
    const isBranchPoint = node.childrenIds.length > 1;
    const isVisible = 
      (!searchQuery || node.content.toLowerCase().includes(searchQuery.toLowerCase())) &&
      (selectedTags.length === 0 || selectedTags.some(tag => node.tags.includes(tag)));

    if (!isVisible) return null;

    return (
      <div key={node.id} className="ml-4">
        <div
          className={`p-3 rounded-lg cursor-pointer transition-colors ${
            isSelected
              ? 'bg-blue-500 text-white'
              : 'bg-white hover:bg-gray-50 border border-gray-200'
          }`}
          onClick={() => {
            setSelectedNode(node.id);
            onNodeSelect?.(node.id);
          }}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  toggleNode(node.id);
                }}
                className="p-1 hover:bg-gray-200 rounded"
              >
                {isExpanded ? '▼' : '▶'}
              </button>
              <span className="text-sm opacity-75">
                {new Date(node.timestamp).toLocaleTimeString()}
              </span>
            </div>
            {isBranchPoint && (
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded-full text-xs">
                Branch Point
              </span>
            )}
          </div>
          
          <div className="mt-2 flex flex-wrap gap-1">
            {node.tags.map((tag) => (
              <span
                key={tag}
                className={`px-2 py-1 rounded-full text-xs ${getTagColor(tag)}`}
              >
                {tag}
              </span>
            ))}
          </div>

          <div className="mt-2">{node.content}</div>

          <div className="mt-2 flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-xs opacity-75">Importance:</span>
              <span className={`px-2 py-1 rounded-full text-xs ${getImportanceColor(node.metadata.importance)}`}>
                {Math.round(node.metadata.importance * 100)}%
              </span>
            </div>
            <div className="flex-1">
              <span className="text-xs opacity-75 block mb-1">Confidence:</span>
              {getConfidenceIndicator(node.metadata.confidence)}
            </div>
          </div>
        </div>

        {isExpanded && node.childrenIds.map((childId: string) => {
          const child = nodes.get(childId);
          return child ? renderNode(child, level + 1) : null;
        })}
      </div>
    );
  };

  const rootNodes = Array.from(nodes.values()).filter(
    (node) => !node.parentId
  );

  return (
    <div className="h-full overflow-y-auto p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold">Conversation Structure</h2>
        <div className="flex gap-2">
          <button
            onClick={() => setZoom(zoom * 1.2)}
            className="p-1 bg-gray-200 rounded-lg hover:bg-gray-300"
          >
            Zoom In
          </button>
          <button
            onClick={() => setZoom(zoom / 1.2)}
            className="p-1 bg-gray-200 rounded-lg hover:bg-gray-300"
          >
            Zoom Out
          </button>
        </div>
      </div>
      <div style={{ transform: `scale(${zoom})` }}>
        {rootNodes.map((node) => renderNode(node))}
      </div>
    </div>
  );
} 