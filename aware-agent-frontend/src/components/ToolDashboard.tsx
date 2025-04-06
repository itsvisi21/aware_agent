import React, { useState } from 'react';
import { Tool } from '../lib/types/tool';
import { PerformanceMetric, UsageMetric, SecurityMetric, CollaborationMetric } from '../lib/types/tool';
import { ToolDashboardProps } from '../lib/types/toolVisualization';
import ToolVisualizer from './ToolVisualizer';

const ToolDashboard: React.FC<ToolDashboardProps> = ({
  tools,
  performanceData,
  usageData,
  securityData,
  collaborationData,
  onToolSelect
}) => {
  const [selectedTool, setSelectedTool] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'day' | 'week' | 'month'>('week');

  const handleToolSelect = (toolId: string) => {
    setSelectedTool(toolId);
    onToolSelect?.(toolId);
  };

  const generateToolRelationships = () => {
    const relationships: Array<{ source: string; target: string; type: 'dependency' | 'collaboration' }> = [];

    // Add dependencies based on usage patterns
    tools.forEach(tool => {
      const usage = usageData.get(tool.id);
      if (usage && usage.length > 0) {
        const lastUsage = usage[usage.length - 1];
        if (lastUsage.context) {
          const contextTools = tools.filter(t => 
            t.id !== tool.id && lastUsage.context.includes(t.name)
          );
          contextTools.forEach(contextTool => {
            relationships.push({
              source: tool.id,
              target: contextTool.id,
              type: 'dependency'
            });
          });
        }
      }
    });

    // Add collaborations based on collaboration data
    tools.forEach(tool => {
      const collaboration = collaborationData.get(tool.id);
      if (collaboration && collaboration.length > 0) {
        const lastCollaboration = collaboration[collaboration.length - 1];
        lastCollaboration.participants.forEach(participant => {
          const participantTool = tools.find(t => t.name === participant);
          if (participantTool) {
            relationships.push({
              source: tool.id,
              target: participantTool.id,
              type: 'collaboration'
            });
          }
        });
      }
    });

    return relationships;
  };

  const renderToolMetrics = () => {
    if (!selectedTool) return null;

    const tool = tools.find(t => t.id === selectedTool);
    if (!tool) return null;

    const performance = performanceData.get(tool.id);
    const usage = usageData.get(tool.id);
    const security = securityData.get(tool.id);
    const collaboration = collaborationData.get(tool.id);

    return (
      <div className="tool-metrics">
        <h2>{tool.name}</h2>
        <p>{tool.description}</p>

        {performance && performance.length > 0 && (
          <div className="metric-section">
            <h3>Performance</h3>
            <ul>
              <li>Execution Time: {performance[0].executionTime}ms</li>
              <li>Memory Usage: {performance[0].memoryUsage}MB</li>
              <li>Error Rate: {(performance[0].errorRate * 100).toFixed(2)}%</li>
            </ul>
          </div>
        )}

        {usage && usage.length > 0 && (
          <div className="metric-section">
            <h3>Usage</h3>
            <ul>
              <li>Usage Count: {usage[0].count}</li>
              <li>Success Rate: {(usage[0].successRate * 100).toFixed(2)}%</li>
              <li>Last Used: {usage[0].timestamp.toLocaleDateString()}</li>
            </ul>
          </div>
        )}

        {security && security.length > 0 && (
          <div className="metric-section">
            <h3>Security</h3>
            <ul>
              <li>Risk Score: {security[0].riskScore}/10</li>
              <li>Vulnerabilities: {security[0].vulnerabilities.length}</li>
              <li>Permissions: {security[0].permissions.length}</li>
            </ul>
          </div>
        )}

        {collaboration && collaboration.length > 0 && (
          <div className="metric-section">
            <h3>Collaboration</h3>
            <ul>
              <li>Participants: {collaboration[0].participants.length}</li>
              <li>Changes: {collaboration[0].changes}</li>
              <li>Notes: {collaboration[0].notes}</li>
            </ul>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="tool-dashboard">
      <div className="dashboard-header">
        <h1>Tool Dashboard</h1>
        <div className="time-range-selector">
          <button
            className={timeRange === 'day' ? 'active' : ''}
            onClick={() => setTimeRange('day')}
          >
            Day
          </button>
          <button
            className={timeRange === 'week' ? 'active' : ''}
            onClick={() => setTimeRange('week')}
          >
            Week
          </button>
          <button
            className={timeRange === 'month' ? 'active' : ''}
            onClick={() => setTimeRange('month')}
          >
            Month
          </button>
        </div>
      </div>

      <div className="dashboard-content">
        <div className="visualization-section">
          <ToolVisualizer
            tools={tools}
            relationships={generateToolRelationships()}
            performanceData={performanceData}
            onNodeClick={handleToolSelect}
          />
        </div>

        <div className="metrics-section">
          {selectedTool ? renderToolMetrics() : (
            <div className="no-tool-selected">
              <p>Select a tool to view its metrics</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ToolDashboard; 