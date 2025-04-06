import React from 'react';

interface AgentRoleIndicatorProps {
    agent: string;
    role: string;
}

export const AgentRoleIndicator: React.FC<AgentRoleIndicatorProps> = ({ agent, role }) => {
    return (
        <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span className="font-semibold text-gray-700">{agent}</span>
            <span className="text-gray-500">•</span>
            <span className="text-gray-600">{role}</span>
        </div>
    );
}; 