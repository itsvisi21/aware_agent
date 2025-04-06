import React, { useState } from 'react';
import { SemanticKernel } from '../lib/kernel/SemanticKernel';
import { GoalVisualizer } from './GoalVisualizer';
import { GoalCreator } from './GoalCreator';
import { ProgressTracker } from './ProgressTracker';
import { ConversationCanvas } from './ConversationCanvas';
import { ConversationExporter } from './ConversationExporter';
import { ConversationResumer } from './ConversationResumer';

interface MainLayoutProps {
  kernel: SemanticKernel;
}

export const MainLayout: React.FC<MainLayoutProps> = ({ kernel }) => {
  const [activeTab, setActiveTab] = useState<'goal' | 'conversation'>('goal');
  const [showGoalCreator, setShowGoalCreator] = useState(false);

  const handleGoalCreated = () => {
    setShowGoalCreator(false);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-gray-900">Aware Agent</h1>
            <div className="flex space-x-4">
              <button
                onClick={() => setActiveTab('goal')}
                className={`px-4 py-2 rounded-md ${
                  activeTab === 'goal'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Goal Management
              </button>
              <button
                onClick={() => setActiveTab('conversation')}
                className={`px-4 py-2 rounded-md ${
                  activeTab === 'conversation'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Conversation
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'goal' ? (
          <div className="space-y-8">
            {/* Goal Management Section */}
            <div className="flex justify-between items-center">
              <h2 className="text-xl font-semibold">Goal Management</h2>
              <button
                onClick={() => setShowGoalCreator(true)}
                className="bg-green-500 text-white px-4 py-2 rounded-md hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
              >
                Create New Goal
              </button>
            </div>

            {showGoalCreator ? (
              <GoalCreator kernel={kernel} onGoalCreated={handleGoalCreated} />
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div>
                  <GoalVisualizer kernel={kernel} />
                </div>
                <div>
                  <ProgressTracker kernel={kernel} />
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-8">
            {/* Conversation Section */}
            <div className="flex justify-between items-center">
              <h2 className="text-xl font-semibold">Conversation</h2>
              <div className="flex space-x-4">
                <ConversationExporter kernel={kernel} />
                <ConversationResumer kernel={kernel} />
              </div>
            </div>

            <div className="bg-white rounded-lg shadow">
              <ConversationCanvas kernel={kernel} />
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <p className="text-center text-gray-500 text-sm">
            © {new Date().getFullYear()} Aware Agent. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}; 