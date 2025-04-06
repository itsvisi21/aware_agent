'use client';

import { useState, useEffect } from 'react';
import { ClockIcon, CheckCircleIcon, ExclamationCircleIcon } from '@heroicons/react/24/outline';

interface AgentStatusProps {
  agentType: 'planner' | 'researcher' | 'explainer' | 'validator';
  status: 'idle' | 'thinking' | 'active' | 'error';
  message?: string;
}

const agentColors = {
  planner: {
    bg: 'bg-purple-100',
    text: 'text-purple-800',
    border: 'border-purple-300',
  },
  researcher: {
    bg: 'bg-green-100',
    text: 'text-green-800',
    border: 'border-green-300',
  },
  explainer: {
    bg: 'bg-yellow-100',
    text: 'text-yellow-800',
    border: 'border-yellow-300',
  },
  validator: {
    bg: 'bg-red-100',
    text: 'text-red-800',
    border: 'border-red-300',
  },
};

const statusIcons = {
  idle: null,
  thinking: <ClockIcon className="h-5 w-5 animate-spin" />,
  active: <CheckCircleIcon className="h-5 w-5" />,
  error: <ExclamationCircleIcon className="h-5 w-5" />,
};

export default function AgentStatus({ agentType, status, message }: AgentStatusProps) {
  const colors = agentColors[agentType];
  const icon = statusIcons[status];

  return (
    <div className={`flex items-center space-x-2 p-2 rounded-lg border ${colors.bg} ${colors.border}`}>
      <div className={`font-medium ${colors.text}`}>
        {agentType.charAt(0).toUpperCase() + agentType.slice(1)}
      </div>
      {icon && <div className={colors.text}>{icon}</div>}
      {message && (
        <div className={`text-sm ${colors.text} opacity-75`}>
          {message}
        </div>
      )}
    </div>
  );
} 