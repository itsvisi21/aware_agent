import React, { useState } from 'react';
import { SemanticKernel } from '../lib/kernel/SemanticKernel';
import { Goal, Milestone, Task } from '../lib/goal/GoalManager';

interface GoalCreatorProps {
  kernel: SemanticKernel;
  onGoalCreated: () => void;
}

export const GoalCreator: React.FC<GoalCreatorProps> = ({ kernel, onGoalCreated }) => {
  const [step, setStep] = useState<'goal' | 'milestones' | 'tasks'>('goal');
  const [goal, setGoal] = useState<Partial<Goal>>({
    title: '',
    description: '',
    constraints: []
  });
  const [milestones, setMilestones] = useState<Partial<Milestone>[]>([]);
  const [currentMilestone, setCurrentMilestone] = useState<Partial<Milestone>>({
    title: '',
    description: '',
    tasks: []
  });
  const [currentTask, setCurrentTask] = useState<Partial<Task>>({
    title: '',
    description: '',
    priority: 'medium'
  });

  const handleGoalSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStep('milestones');
  };

  const handleAddMilestone = () => {
    if (currentMilestone.title) {
      setMilestones([...milestones, currentMilestone]);
      setCurrentMilestone({
        title: '',
        description: '',
        tasks: []
      });
    }
  };

  const handleAddTask = () => {
    if (currentTask.title) {
      const newTask: Task = {
        id: Date.now().toString(),
        title: currentTask.title,
        description: currentTask.description || '',
        status: 'pending',
        priority: currentTask.priority || 'medium',
        dependencies: [],
        estimatedTime: currentTask.estimatedTime
      };

      setCurrentMilestone({
        ...currentMilestone,
        tasks: [...(currentMilestone.tasks || []), newTask]
      });
      setCurrentTask({
        title: '',
        description: '',
        priority: 'medium'
      });
    }
  };

  const handleCreateGoal = async () => {
    const newGoal = await kernel.setGoal({
      ...goal,
      milestones: milestones.map(m => ({
        id: Date.now().toString(),
        title: m.title || '',
        description: m.description || '',
        status: 'pending',
        tasks: m.tasks?.map(t => ({
          id: Date.now().toString(),
          title: t.title || '',
          description: t.description || '',
          status: 'pending',
          priority: t.priority || 'medium',
          dependencies: [],
          estimatedTime: t.estimatedTime
        })) || []
      }))
    });
    onGoalCreated();
  };

  return (
    <div className="p-4 bg-white rounded-lg shadow">
      {step === 'goal' && (
        <form onSubmit={handleGoalSubmit} className="space-y-4">
          <h2 className="text-2xl font-bold">Create New Goal</h2>
          
          <div>
            <label className="block text-sm font-medium text-gray-700">Title</label>
            <input
              type="text"
              value={goal.title}
              onChange={(e) => setGoal({ ...goal, title: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">Description</label>
            <textarea
              value={goal.description}
              onChange={(e) => setGoal({ ...goal, description: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              rows={3}
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">Constraints</label>
            <input
              type="text"
              value={goal.constraints?.join(', ')}
              onChange={(e) => setGoal({ ...goal, constraints: e.target.value.split(', ') })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              placeholder="Enter constraints separated by commas"
            />
          </div>

          <button
            type="submit"
            className="w-full bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          >
            Next: Add Milestones
          </button>
        </form>
      )}

      {step === 'milestones' && (
        <div className="space-y-4">
          <h2 className="text-2xl font-bold">Add Milestones</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">Milestone Title</label>
              <input
                type="text"
                value={currentMilestone.title}
                onChange={(e) => setCurrentMilestone({ ...currentMilestone, title: e.target.value })}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">Description</label>
              <textarea
                value={currentMilestone.description}
                onChange={(e) => setCurrentMilestone({ ...currentMilestone, description: e.target.value })}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                rows={2}
              />
            </div>

            <button
              onClick={handleAddMilestone}
              className="w-full bg-green-500 text-white py-2 px-4 rounded-md hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
            >
              Add Milestone
            </button>
          </div>

          <div className="mt-4">
            <h3 className="text-lg font-semibold">Added Milestones</h3>
            <div className="space-y-2">
              {milestones.map((m, index) => (
                <div key={index} className="p-3 bg-gray-50 rounded-lg">
                  <h4 className="font-medium">{m.title}</h4>
                  <p className="text-sm text-gray-600">{m.description}</p>
                </div>
              ))}
            </div>
          </div>

          <button
            onClick={() => setStep('tasks')}
            className="w-full bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          >
            Next: Add Tasks
          </button>
        </div>
      )}

      {step === 'tasks' && (
        <div className="space-y-4">
          <h2 className="text-2xl font-bold">Add Tasks</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">Task Title</label>
              <input
                type="text"
                value={currentTask.title}
                onChange={(e) => setCurrentTask({ ...currentTask, title: e.target.value })}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">Description</label>
              <textarea
                value={currentTask.description}
                onChange={(e) => setCurrentTask({ ...currentTask, description: e.target.value })}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                rows={2}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">Priority</label>
              <select
                value={currentTask.priority}
                onChange={(e) => setCurrentTask({ ...currentTask, priority: e.target.value as Task['priority'] })}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>

            <button
              onClick={handleAddTask}
              className="w-full bg-green-500 text-white py-2 px-4 rounded-md hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
            >
              Add Task
            </button>
          </div>

          <div className="mt-4">
            <h3 className="text-lg font-semibold">Added Tasks</h3>
            <div className="space-y-2">
              {currentMilestone.tasks?.map((t, index) => (
                <div key={index} className="p-3 bg-gray-50 rounded-lg">
                  <h4 className="font-medium">{t.title}</h4>
                  <p className="text-sm text-gray-600">{t.description}</p>
                  <span className={`text-xs px-2 py-1 rounded ${
                    t.priority === 'high' ? 'bg-red-100 text-red-800' :
                    t.priority === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    {t.priority}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <button
            onClick={handleCreateGoal}
            className="w-full bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          >
            Create Goal
          </button>
        </div>
      )}
    </div>
  );
}; 