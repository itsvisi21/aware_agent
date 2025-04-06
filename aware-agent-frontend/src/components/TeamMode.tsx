import React, { useState } from 'react';
import { ChatMessage } from '../lib/types/chat';

interface TeamMember {
  id: string;
  name: string;
  role: string;
  status: 'active' | 'inactive';
}

interface TeamModeProps {
  onMessage: (message: ChatMessage) => void;
  onTaskAssign: (task: string, assignee: string) => void;
  onStatusUpdate: (status: string) => void;
  onMeetingStart: (agenda: string) => void;
}

export const TeamMode: React.FC<TeamModeProps> = ({
  onMessage,
  onTaskAssign,
  onStatusUpdate,
  onMeetingStart
}) => {
  const [teamMembers, setTeamMembers] = useState<TeamMember[]>([
    { id: '1', name: 'Alice', role: 'Developer', status: 'active' },
    { id: '2', name: 'Bob', role: 'Designer', status: 'active' },
    { id: '3', name: 'Charlie', role: 'QA', status: 'active' }
  ]);
  const [showTaskForm, setShowTaskForm] = useState(false);
  const [showMeetingForm, setShowMeetingForm] = useState(false);
  const [newTask, setNewTask] = useState({ description: '', assignee: '' });
  const [meetingAgenda, setMeetingAgenda] = useState('');

  const handleTaskSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTask.description.trim() || !newTask.assignee) return;

    onTaskAssign(newTask.description, newTask.assignee);
    setNewTask({ description: '', assignee: '' });
    setShowTaskForm(false);
  };

  const handleMeetingSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!meetingAgenda.trim()) return;

    onMeetingStart(meetingAgenda);
    setMeetingAgenda('');
    setShowMeetingForm(false);
  };

  const handleStatusUpdate = (memberId: string, status: TeamMember['status']) => {
    setTeamMembers(prev =>
      prev.map(member =>
        member.id === memberId ? { ...member, status } : member
      )
    );
    onStatusUpdate(`Member ${memberId} status updated to ${status}`);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between p-2 border-b">
        <h2 className="text-lg font-semibold">Team Collaboration</h2>
        <div className="flex space-x-2">
          <button
            onClick={() => setShowTaskForm(!showTaskForm)}
            className="px-3 py-1 bg-blue-100 text-blue-800 rounded hover:bg-blue-200"
          >
            Assign Task
          </button>
          <button
            onClick={() => setShowMeetingForm(!showMeetingForm)}
            className="px-3 py-1 bg-green-100 text-green-800 rounded hover:bg-green-200"
          >
            Start Meeting
          </button>
        </div>
      </div>

      <div className="p-4 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {teamMembers.map(member => (
            <div
              key={member.id}
              className="p-4 border rounded-lg"
            >
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium">{member.name}</h3>
                  <p className="text-sm text-gray-600">{member.role}</p>
                </div>
                <div className="flex items-center space-x-2">
                  <span
                    className={`w-2 h-2 rounded-full ${
                      member.status === 'active' ? 'bg-green-500' : 'bg-gray-300'
                    }`}
                  />
                  <select
                    value={member.status}
                    onChange={(e) => handleStatusUpdate(member.id, e.target.value as TeamMember['status'])}
                    className="text-sm border rounded p-1"
                  >
                    <option value="active">Active</option>
                    <option value="inactive">Inactive</option>
                  </select>
                </div>
              </div>
            </div>
          ))}
        </div>

        {showTaskForm && (
          <form onSubmit={handleTaskSubmit} className="mt-4 p-4 border rounded">
            <h3 className="text-lg font-medium mb-2">Assign New Task</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Task Description</label>
                <textarea
                  value={newTask.description}
                  onChange={(e) => setNewTask(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="Describe the task..."
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  rows={3}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Assign To</label>
                <select
                  value={newTask.assignee}
                  onChange={(e) => setNewTask(prev => ({ ...prev, assignee: e.target.value }))}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                >
                  <option value="">Select team member</option>
                  {teamMembers.map(member => (
                    <option key={member.id} value={member.id}>
                      {member.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setShowTaskForm(false)}
                  className="px-3 py-1 bg-gray-100 text-gray-800 rounded hover:bg-gray-200"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
                >
                  Assign
                </button>
              </div>
            </div>
          </form>
        )}

        {showMeetingForm && (
          <form onSubmit={handleMeetingSubmit} className="mt-4 p-4 border rounded">
            <h3 className="text-lg font-medium mb-2">Start Team Meeting</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Meeting Agenda</label>
                <textarea
                  value={meetingAgenda}
                  onChange={(e) => setMeetingAgenda(e.target.value)}
                  placeholder="Enter meeting agenda..."
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  rows={3}
                />
              </div>
              <div className="flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setShowMeetingForm(false)}
                  className="px-3 py-1 bg-gray-100 text-gray-800 rounded hover:bg-gray-200"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600"
                >
                  Start Meeting
                </button>
              </div>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}; 