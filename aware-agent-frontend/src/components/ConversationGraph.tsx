'use client';

import { useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { MemoryNode } from '@/lib/memory/SemanticMemory';
import { StorageService } from '@/lib/storage/StorageService';

// Dynamically import ForceGraph2D to avoid SSR issues
const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
  ssr: false,
});

interface GraphNode {
  id: string;
  name: string;
  val: number;
  color: string;
  tags: string[];
  timestamp: Date;
  x?: number;
  y?: number;
  hidden?: boolean;
}

interface GraphLink {
  source: string;
  target: string;
  hidden?: boolean;
}

interface ConversationGraphProps {
  conversationId: string;
  onNodeSelect?: (nodeId: string) => void;
  searchQuery?: string;
  selectedTags?: string[];
}

export default function ConversationGraph({
  conversationId,
  onNodeSelect,
  searchQuery = '',
  selectedTags = [],
}: ConversationGraphProps) {
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; links: GraphLink[] }>({
    nodes: [],
    links: [],
  });
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  useEffect(() => {
    const stored = StorageService.loadConversation(conversationId);
    if (stored) {
      const nodes: GraphNode[] = [];
      const links: GraphLink[] = [];

      // Convert MemoryNodes to graph nodes
      stored.nodes.forEach((node: MemoryNode) => {
        const isVisible =
          (!searchQuery || node.content.toLowerCase().includes(searchQuery.toLowerCase())) &&
          (selectedTags.length === 0 || selectedTags.some(tag => node.tags.includes(tag)));

        nodes.push({
          id: node.id,
          name: node.content.substring(0, 30) + (node.content.length > 30 ? '...' : ''),
          val: node.metadata.importance * 10,
          color: node.tags.includes('branch') ? '#ff6b6b' : '#4dabf7',
          tags: node.tags,
          timestamp: node.timestamp,
          hidden: !isVisible,
        });

        // Create links for parent-child relationships
        if (node.parentId) {
          links.push({
            source: node.parentId,
            target: node.id,
            hidden: !isVisible,
          });
        }
      });

      setGraphData({ nodes, links });
    }
  }, [conversationId, searchQuery, selectedTags]);

  const handleNodeClick = useCallback((node: { id: string | number | undefined }) => {
    if (typeof node.id === 'string') {
      setSelectedNode(node.id);
      onNodeSelect?.(node.id);
    }
  }, [onNodeSelect]);

  return (
    <div className="w-full h-full">
      <ForceGraph2D
        graphData={graphData}
        nodeLabel="name"
        nodeAutoColorBy="tags"
        nodeVal="val"
        linkDirectionalParticles={2}
        linkDirectionalParticleSpeed={0.005}
        onNodeClick={handleNodeClick}
        nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
          if (node.hidden) return;
          
          const label = (node as GraphNode).name;
          const fontSize = 12/globalScale;
          ctx.font = `${fontSize}px Sans-Serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = (node as GraphNode).color;
          ctx.fillText(label, node.x, node.y);
        }}
        linkVisibility={(link: any) => !link.hidden}
        nodeVisibility={(node: any) => !node.hidden}
      />
    </div>
  );
} 