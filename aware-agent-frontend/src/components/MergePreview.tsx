'use client';

import { useState, useEffect } from 'react';
import { MemoryNode } from '@/lib/memory/SemanticMemory';
import { StorageService } from '@/lib/storage/StorageService';
import { ConversationMerger } from '@/lib/memory/ConversationMerger';

interface MergePreviewProps {
  sourceId: string;
  targetId: string;
  mergeStrategy: 'append' | 'merge' | 'branch';
  onPreviewComplete: (result: {
    conflicts: { nodeId: string; reason: string }[];
    previewNodes: MemoryNode[];
  }) => void;
}

export default function MergePreview({
  sourceId,
  targetId,
  mergeStrategy,
  onPreviewComplete,
}: MergePreviewProps) {
  const [previewNodes, setPreviewNodes] = useState<MemoryNode[]>([]);
  const [conflicts, setConflicts] = useState<{ nodeId: string; reason: string }[]>([]);

  useEffect(() => {
    const simulateMerge = async () => {
      const source = StorageService.loadConversation(sourceId);
      const target = StorageService.loadConversation(targetId);

      if (!source || !target) {
        setConflicts([{
          nodeId: sourceId,
          reason: 'Source or target conversation not found'
        }]);
        return;
      }

      const newNodes: MemoryNode[] = [];
      const newConflicts: { nodeId: string; reason: string }[] = [];

      switch (mergeStrategy) {
        case 'append': {
          const lastTargetNode = Array.from(target.nodes.values())
            .filter((node: MemoryNode) => !node.childrenIds.length)
            .sort((a: MemoryNode, b: MemoryNode) => b.timestamp.getTime() - a.timestamp.getTime())[0];

          if (!lastTargetNode) {
            newConflicts.push({
              nodeId: target.id,
              reason: 'No valid target node found for appending'
            });
            break;
          }

          Array.from(source.nodes.values()).forEach((node: MemoryNode) => {
            const newNode = { ...node };
            if (!node.parentId) {
              newNode.parentId = lastTargetNode.id;
            }
            newNodes.push(newNode);
          });
          break;
        }

        case 'merge': {
          Array.from(source.nodes.values()).forEach((sourceNode: MemoryNode) => {
            const similarNode = ConversationMerger.findSimilarNode(sourceNode, target.nodes);
            if (similarNode) {
              newNodes.push(ConversationMerger.mergeNodeContent(similarNode, sourceNode));
            } else {
              newNodes.push(sourceNode);
            }
          });
          break;
        }

        case 'branch': {
          const mostRelevantNode = ConversationMerger.findMostRelevantNode(source, target);
          if (!mostRelevantNode) {
            newConflicts.push({
              nodeId: source.id,
              reason: 'No relevant node found for branching'
            });
            break;
          }

          const branchNode: MemoryNode = {
            id: `preview_branch_${Date.now()}`,
            content: `Branch from: ${mostRelevantNode.content}`,
            timestamp: new Date(),
            tags: [...mostRelevantNode.tags, 'branch'],
            parentId: mostRelevantNode.id,
            childrenIds: [],
            metadata: {
              importance: mostRelevantNode.metadata.importance,
              confidence: mostRelevantNode.metadata.confidence,
              context: {
                ...mostRelevantNode.metadata.context,
                branchSource: source.id
              }
            }
          };
          newNodes.push(branchNode);
          break;
        }
      }

      setPreviewNodes(newNodes);
      setConflicts(newConflicts);
      onPreviewComplete({ conflicts: newConflicts, previewNodes: newNodes });
    };

    simulateMerge();
  }, [sourceId, targetId, mergeStrategy, onPreviewComplete]);

  return (
    <div className="p-4">
      <h3 className="text-lg font-semibold mb-4">Merge Preview</h3>
      
      {conflicts.length > 0 && (
        <div className="mb-4 p-4 bg-red-100 rounded-lg">
          <h4 className="font-semibold text-red-700 mb-2">Potential Conflicts</h4>
          <ul className="list-disc list-inside">
            {conflicts.map((conflict) => (
              <li key={conflict.nodeId} className="text-red-600">
                {conflict.reason}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="space-y-4">
        {previewNodes.map((node) => (
          <div
            key={node.id}
            className="p-4 bg-gray-50 rounded-lg border border-gray-200"
          >
            <div className="flex justify-between items-start mb-2">
              <span className="font-medium">{node.content.substring(0, 50)}...</span>
              <span className="text-sm text-gray-500">
                {new Date(node.timestamp).toLocaleString()}
              </span>
            </div>
            <div className="flex flex-wrap gap-2">
              {node.tags.map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs"
                >
                  {tag}
                </span>
              ))}
            </div>
            {node.parentId && (
              <div className="mt-2 text-sm text-gray-600">
                Parent: {node.parentId}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
} 