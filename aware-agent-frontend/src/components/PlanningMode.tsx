import React, { useState } from 'react';
import { ChatMessage } from '../lib/types/chat';

interface PlanningModeProps {
  onMessage: (message: ChatMessage) => void;
  onGoalSet: (goal: string) => void;
  onMilestoneCreate: (milestone: string) => void;
  onResourceAdd: (resource: string) => void;
  onTimelineUpdate: (timeline: string) => void;
}

export const PlanningMode: React.FC<PlanningModeProps> = ({
  onMessage,
  onGoalSet,
  onMilestoneCreate,
  onResourceAdd,
  onTimelineUpdate
}) => {
  const [currentGoal, setCurrentGoal] = useState('');
  const [milestone, setMilestone] = useState('');
  const [resource, setResource] = useState('');
  const [timeline, setTimeline] = useState('');
  const [showMilestoneForm, setShowMilestoneForm] = useState(false);
  const [showResourceForm, setShowResourceForm] = useState(false);
  const [showTimelineForm, setShowTimelineForm] = useState(false);

  const handleGoalSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentGoal.trim()) return;

    onGoalSet(currentGoal);
    setCurrentGoal('');
  };

  const handleMilestoneSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!milestone.trim()) return;

    onMilestoneCreate(milestone);
    setMilestone('');
    setShowMilestoneForm(false);
  };

  const handleResourceSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!resource.trim()) return;

    onResourceAdd(resource);
    setResource('');
    setShowResourceForm(false);
  };

  const handleTimelineSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!timeline.trim()) return;

    onTimelineUpdate(timeline);
    setTimeline('');
    setShowTimelineForm(false);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between p-2 border-b">
        <h2 className="text-lg font-semibold">Planning Mode</h2>
        <div className="flex space-x-2">
          <button
            onClick={() => setShowMilestoneForm(!showMilestoneForm)}
            className="px-3 py-1 bg-blue-100 text-blue-800 rounded hover:bg-blue-200"
          >
            Add Milestone
          </button>
          <button
            onClick={() => setShowResourceForm(!showResourceForm)}
            className="px-3 py-1 bg-green-100 text-green-800 rounded hover:bg-green-200"
          >
            Add Resource
          </button>
          <button
            onClick={() => setShowTimelineForm(!showTimelineForm)}
            className="px-3 py-1 bg-purple-100 text-purple-800 rounded hover:bg-purple-200"
          >
            Update Timeline
          </button>
        </div>
      </div>

      <div className="p-4 space-y-4">
        <form onSubmit={handleGoalSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Project Goal</label>
            <textarea
              value={currentGoal}
              onChange={(e) => setCurrentGoal(e.target.value)}
              placeholder="Enter your project goal..."
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              rows={3}
            />
          </div>
          <button
            type="submit"
            className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Set Goal
          </button>
        </form>

        {showMilestoneForm && (
          <form onSubmit={handleMilestoneSubmit} className="mt-4 p-4 border rounded">
            <h3 className="text-lg font-medium mb-2">Add Milestone</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Milestone Description</label>
                <textarea
                  value={milestone}
                  onChange={(e) => setMilestone(e.target.value)}
                  placeholder="Describe the milestone..."
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  rows={3}
                />
              </div>
              <div className="flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setShowMilestoneForm(false)}
                  className="px-3 py-1 bg-gray-100 text-gray-800 rounded hover:bg-gray-200"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
                >
                  Add
                </button>
              </div>
            </div>
          </form>
        )}

        {showResourceForm && (
          <form onSubmit={handleResourceSubmit} className="mt-4 p-4 border rounded">
            <h3 className="text-lg font-medium mb-2">Add Resource</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Resource Description</label>
                <textarea
                  value={resource}
                  onChange={(e) => setResource(e.target.value)}
                  placeholder="Describe the resource..."
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  rows={3}
                />
              </div>
              <div className="flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setShowResourceForm(false)}
                  className="px-3 py-1 bg-gray-100 text-gray-800 rounded hover:bg-gray-200"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600"
                >
                  Add
                </button>
              </div>
            </div>
          </form>
        )}

        {showTimelineForm && (
          <form onSubmit={handleTimelineSubmit} className="mt-4 p-4 border rounded">
            <h3 className="text-lg font-medium mb-2">Update Timeline</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Timeline Update</label>
                <textarea
                  value={timeline}
                  onChange={(e) => setTimeline(e.target.value)}
                  placeholder="Enter timeline update..."
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  rows={3}
                />
              </div>
              <div className="flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setShowTimelineForm(false)}
                  className="px-3 py-1 bg-gray-100 text-gray-800 rounded hover:bg-gray-200"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-3 py-1 bg-purple-500 text-white rounded hover:bg-purple-600"
                >
                  Update
                </button>
              </div>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}; 