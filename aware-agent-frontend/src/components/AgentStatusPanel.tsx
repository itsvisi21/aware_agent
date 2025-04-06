'use client';

import { useAgentStore } from '@/lib/store/agentStore';
import AgentStatus from './AgentStatus';

export default function AgentStatusPanel() {
  const { status } = useAgentStore();

  return (
    <div className="space-y-2">
      <h3 className="text-lg font-semibold mb-2">Agent Status</h3>
      <div className="grid grid-cols-1 gap-2">
        <AgentStatus
          agentType="planner"
          status={status.planner.status}
          message={status.planner.message}
        />
        <AgentStatus
          agentType="researcher"
          status={status.researcher.status}
          message={status.researcher.message}
        />
        <AgentStatus
          agentType="explainer"
          status={status.explainer.status}
          message={status.explainer.message}
        />
        <AgentStatus
          agentType="validator"
          status={status.validator.status}
          message={status.validator.message}
        />
      </div>
    </div>
  );
} 