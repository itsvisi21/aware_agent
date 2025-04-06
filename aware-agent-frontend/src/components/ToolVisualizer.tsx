import React, { useEffect, useRef } from 'react';
import { Tool } from '../lib/types/tool';
import { ToolNode, ToolEdge, ToolVisualizerProps } from '../lib/types/toolVisualization';

const ToolVisualizer: React.FC<ToolVisualizerProps> = ({
  tools,
  relationships,
  performanceData,
  onNodeClick
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Initialize nodes
    const nodes: ToolNode[] = tools.map((tool, index) => {
      const x = (index % 5) * 150 + 100;
      const y = Math.floor(index / 5) * 150 + 100;
      const performance = performanceData?.get(tool.id)?.[0];
      return {
        id: tool.id,
        name: tool.name,
        x,
        y,
        performance: performance ? {
          executionTime: performance.executionTime,
          memoryUsage: performance.memoryUsage,
          errorRate: performance.errorRate
        } : undefined
      };
    });

    // Draw edges
    relationships.forEach(relationship => {
      const source = nodes.find(n => n.id === relationship.source);
      const target = nodes.find(n => n.id === relationship.target);
      if (!source || !target) return;

      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.strokeStyle = relationship.type === 'dependency' ? '#666' : '#999';
      ctx.stroke();
    });

    // Draw nodes
    nodes.forEach(node => {
      ctx.beginPath();
      ctx.arc(node.x, node.y, 30, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.stroke();

      // Draw performance indicators
      if (node.performance) {
        const { executionTime, memoryUsage, errorRate } = node.performance;
        const isGoodPerformance = executionTime < 100 && memoryUsage < 100 && errorRate < 0.1;
        ctx.fillStyle = isGoodPerformance ? '#0f0' : '#f00';
        ctx.beginPath();
        ctx.arc(node.x, node.y, 10, 0, Math.PI * 2);
        ctx.fill();
      }

      // Draw tool name
      ctx.fillStyle = '#000';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(node.name, node.x, node.y + 50);
    });

    // Add click handler
    const handleClick = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      nodes.forEach(node => {
        const dx = x - node.x;
        const dy = y - node.y;
        if (Math.sqrt(dx * dx + dy * dy) <= 30) {
          onNodeClick?.(node.id);
        }
      });
    };

    canvas.addEventListener('click', handleClick);
    return () => canvas.removeEventListener('click', handleClick);
  }, [tools, relationships, performanceData, onNodeClick]);

  return (
    <div className="tool-visualizer">
      <canvas
        ref={canvasRef}
        width={800}
        height={600}
      />
      <div className="tool-visualizer-legend">
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: '#fff', border: '1px solid #000' }}></div>
          <span>Tool Node</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: '#0f0' }}></div>
          <span>Good Performance</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: '#f00' }}></div>
          <span>Poor Performance</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: '#666' }}></div>
          <span>Dependency</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: '#999' }}></div>
          <span>Collaboration</span>
        </div>
      </div>
    </div>
  );
};

export default ToolVisualizer; 