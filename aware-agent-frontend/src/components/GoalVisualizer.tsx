import React, { useState, useEffect } from 'react';
import { Goal, Milestone, Task } from '../lib/goal/GoalManager';
import { SemanticKernel } from '../lib/kernel/SemanticKernel';

interface GoalVisualizerProps {
  kernel: SemanticKernel;
}

interface MilestoneProgress {
  completed: number;
  total: number;
  percentage: number;
}

interface GoalAlignment {
  score: number;
  feedback: string;
  suggestions: string[];
}

export const GoalVisualizer: React.FC<GoalVisualizerProps> = ({ kernel }) => {
  const [currentGoal, setCurrentGoal] = useState<Goal | null>(null);
  const [selectedMilestone, setSelectedMilestone] = useState<Milestone | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [milestoneProgress, setMilestoneProgress] = useState<Record<string, MilestoneProgress>>({});
  const [goalAlignment, setGoalAlignment] = useState<GoalAlignment | null>(null);

  useEffect(() => {
    const loadGoal = async () => {
      const goal = kernel.getCurrentGoal();
      setCurrentGoal(goal);
      if (goal) {
        setProgress(goal.progress);
        calculateMilestoneProgress(goal.milestones);
        const alignment = await kernel.analyzeGoalAlignment(goal);
        setGoalAlignment(alignment);
      }
    };
    loadGoal();
  }, [kernel]);

  const calculateMilestoneProgress = (milestones: Milestone[]) => {
    const progress: Record<string, MilestoneProgress> = {};
    milestones.forEach(milestone => {
      const completed = milestone.tasks.filter(t => t.status === 'completed').length;
      const total = milestone.tasks.length;
      progress[milestone.id] = {
        completed,
        total,
        percentage: total > 0 ? (completed / total) * 100 : 0
      };
    });
    setMilestoneProgress(progress);
  };

  const handleProgressUpdate = async (newProgress: number) => {
    await kernel.updateGoalProgress(newProgress);
    setProgress(newProgress);
    const updatedGoal = kernel.getCurrentGoal();
    setCurrentGoal(updatedGoal);
    if (updatedGoal) {
      const alignment = await kernel.analyzeGoalAlignment(updatedGoal);
      setGoalAlignment(alignment);
    }
  };

  const handleMilestoneClick = (milestone: Milestone) => {
    setSelectedMilestone(milestone);
  };

  const handleTaskStatusUpdate = async (taskId: string, newStatus: Task['status']) => {
    if (!currentGoal || !selectedMilestone) return;

    const updatedMilestone: Milestone = {
      ...selectedMilestone,
      tasks: selectedMilestone.tasks.map(task =>
        task.id === taskId ? { ...task, status: newStatus } : task
      )
    };

    const updatedGoal: Goal = {
      ...currentGoal,
      milestones: currentGoal.milestones.map(milestone =>
        milestone.id === selectedMilestone.id ? updatedMilestone : milestone
      )
    };

    await kernel.setGoal(updatedGoal);
    setCurrentGoal(updatedGoal);
    setSelectedMilestone(updatedMilestone);
    calculateMilestoneProgress(updatedGoal.milestones);
    const alignment = await kernel.analyzeGoalAlignment(updatedGoal);
    setGoalAlignment(alignment);
  };

  if (!currentGoal) {
    return (
      <div className="p-4 bg-gray-100 rounded-lg">
        <p className="text-gray-600">No active goal. Start by setting a goal.</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4">
      {/* Goal Overview */}
      <div className="col-span-1 md:col-span-3 bg-white rounded-lg shadow p-4">
        <h2 className="text-2xl font-bold mb-2">{currentGoal.title}</h2>
        <p className="text-gray-600 mb-4">{currentGoal.description}</p>
        
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Overall Progress: {progress}%
          </label>
          <div className="relative pt-1">
            <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-gray-200">
              <div
                style={{ width: `${progress}%` }}
                className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-blue-500 transition-all duration-500"
              />
            </div>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            value={progress}
            onChange={(e) => handleProgressUpdate(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        {/* Goal Alignment Feedback */}
        {goalAlignment && (
          <div className={`p-4 rounded-lg mb-4 ${
            goalAlignment.score >= 0.7 ? 'bg-green-50 text-green-800' :
            goalAlignment.score >= 0.4 ? 'bg-yellow-50 text-yellow-800' :
            'bg-red-50 text-red-800'
          }`}>
            <h3 className="font-semibold mb-2">Goal Alignment</h3>
            <p className="mb-2">{goalAlignment.feedback}</p>
            {goalAlignment.suggestions.length > 0 && (
              <div>
                <h4 className="font-medium mb-1">Suggestions:</h4>
                <ul className="list-disc list-inside">
                  {goalAlignment.suggestions.map((suggestion, index) => (
                    <li key={index}>{suggestion}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Milestones List */}
      <div className="col-span-1 bg-white rounded-lg shadow p-4">
        <h3 className="text-xl font-semibold mb-4">Milestones</h3>
        <div className="space-y-4">
          {currentGoal.milestones.map((milestone) => {
            const progress = milestoneProgress[milestone.id];
            return (
              <div
                key={milestone.id}
                className={`p-4 rounded-lg cursor-pointer transition-all duration-200 ${
                  selectedMilestone?.id === milestone.id
                    ? 'bg-blue-100 border-blue-500 border-2'
                    : 'bg-gray-50 hover:bg-gray-100 border-2 border-transparent'
                }`}
                onClick={() => handleMilestoneClick(milestone)}
              >
                <h4 className="font-medium mb-2">{milestone.title}</h4>
                <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
                  <span>{milestone.status}</span>
                  <span>{progress.completed} / {progress.total} tasks</span>
                </div>
                <div className="relative pt-1">
                  <div className="overflow-hidden h-2 text-xs flex rounded bg-gray-200">
                    <div
                      style={{ width: `${progress.percentage}%` }}
                      className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center transition-all duration-500 ${
                        progress.percentage === 100 ? 'bg-green-500' :
                        progress.percentage >= 50 ? 'bg-blue-500' :
                        'bg-yellow-500'
                      }`}
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Selected Milestone Details */}
      <div className="col-span-2 bg-white rounded-lg shadow p-4">
        {selectedMilestone ? (
          <>
            <h3 className="text-xl font-semibold mb-4">{selectedMilestone.title}</h3>
            <p className="text-gray-600 mb-4">{selectedMilestone.description}</p>
            
            <div className="space-y-4">
              {selectedMilestone.tasks.map((task) => (
                <div key={task.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div className="flex-1">
                    <h4 className="font-medium mb-1">{task.title}</h4>
                    <p className="text-sm text-gray-600 mb-2">{task.description}</p>
                    <div className="flex items-center space-x-2">
                      <span className={`text-xs px-2 py-1 rounded ${
                        task.priority === 'high' ? 'bg-red-100 text-red-800' :
                        task.priority === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-green-100 text-green-800'
                      }`}>
                        {task.priority}
                      </span>
                      <span className={`text-xs px-2 py-1 rounded ${
                        task.status === 'completed' ? 'bg-green-100 text-green-800' :
                        task.status === 'in_progress' ? 'bg-blue-100 text-blue-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {task.status}
                      </span>
                    </div>
                  </div>
                  <select
                    value={task.status}
                    onChange={(e) => handleTaskStatusUpdate(task.id, e.target.value as Task['status'])}
                    className="ml-4 p-2 border rounded bg-white"
                  >
                    <option value="pending">Pending</option>
                    <option value="in_progress">In Progress</option>
                    <option value="completed">Completed</option>
                  </select>
                </div>
              ))}
            </div>
          </>
        ) : (
          <p className="text-gray-600">Select a milestone to view its tasks</p>
        )}
      </div>
    </div>
  );
}; 