import React, { useState, useEffect } from 'react';
import { SemanticKernel } from '../lib/kernel/SemanticKernel';
import { Goal } from '../lib/goal/GoalManager';

interface ProgressTrackerProps {
  kernel: SemanticKernel;
}

export const ProgressTracker: React.FC<ProgressTrackerProps> = ({ kernel }) => {
  const [progressReport, setProgressReport] = useState<string>('');
  const [currentGoal, setCurrentGoal] = useState<Goal | null>(null);

  useEffect(() => {
    const loadProgress = async () => {
      const goal = kernel.getCurrentGoal();
      setCurrentGoal(goal);
      if (goal) {
        const report = await kernel.generateProgressReport();
        setProgressReport(report);
      }
    };
    loadProgress();
  }, [kernel]);

  const handleExportReport = () => {
    const blob = new Blob([progressReport], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `progress-report-${new Date().toISOString().split('T')[0]}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (!currentGoal) {
    return (
      <div className="p-4 bg-gray-100 rounded-lg">
        <p className="text-gray-600">No active goal. Start by setting a goal.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Progress Report</h2>
        <button
          onClick={handleExportReport}
          className="bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        >
          Export Report
        </button>
      </div>

      <div className="bg-white rounded-lg shadow p-4">
        <div className="mb-4">
          <h3 className="text-lg font-semibold mb-2">Goal Overview</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-600">Title</p>
              <p className="font-medium">{currentGoal.title}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Status</p>
              <p className="font-medium capitalize">{currentGoal.status}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Progress</p>
              <div className="flex items-center space-x-2">
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div
                    className="bg-blue-600 h-2.5 rounded-full"
                    style={{ width: `${currentGoal.progress}%` }}
                  ></div>
                </div>
                <span className="text-sm font-medium">{currentGoal.progress}%</span>
              </div>
            </div>
            <div>
              <p className="text-sm text-gray-600">Last Updated</p>
              <p className="font-medium">
                {new Date(currentGoal.updatedAt).toLocaleString()}
              </p>
            </div>
          </div>
        </div>

        <div className="mb-4">
          <h3 className="text-lg font-semibold mb-2">Milestones</h3>
          <div className="space-y-4">
            {currentGoal.milestones.map((milestone) => (
              <div key={milestone.id} className="border rounded-lg p-4">
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h4 className="font-medium">{milestone.title}</h4>
                    <p className="text-sm text-gray-600">{milestone.description}</p>
                  </div>
                  <span className={`px-2 py-1 rounded text-sm ${
                    milestone.status === 'completed' ? 'bg-green-100 text-green-800' :
                    milestone.status === 'in_progress' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {milestone.status.replace('_', ' ')}
                  </span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-600">Tasks</p>
                    <p className="font-medium">
                      {milestone.tasks.filter(t => t.status === 'completed').length} / {milestone.tasks.length} completed
                    </p>
                  </div>
                  {milestone.dueDate && (
                    <div>
                      <p className="text-sm text-gray-600">Due Date</p>
                      <p className="font-medium">
                        {new Date(milestone.dueDate).toLocaleDateString()}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-lg font-semibold mb-2">Next Steps</h3>
          <div className="bg-gray-50 rounded-lg p-4">
            <pre className="whitespace-pre-wrap font-sans">
              {progressReport.split('## Next Steps')[1]?.trim() || 'No next steps defined.'}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}; 