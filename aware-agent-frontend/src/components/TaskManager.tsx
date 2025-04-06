import React, { useState, useEffect } from 'react';
import { createTask, updateTask, deleteTask, getTasks } from '../lib/api';
import { Task } from '../types/Task';

export const TaskManager: React.FC = () => {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null);
  const [newTask, setNewTask] = useState<Partial<Task>>({
    title: '',
    description: '',
    status: 'pending',
    priority: 'medium'
  });
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [sortBy, setSortBy] = useState<string>('createdAt');

  useEffect(() => {
    const loadTasks = async () => {
      try {
        const loadedTasks = await getTasks();
        setTasks(loadedTasks);
      } catch (error) {
        console.error('Failed to load tasks:', error);
      }
    };
    loadTasks();
  }, []);

  const handleCreateTask = async () => {
    if (!newTask.title?.trim()) return;

    try {
      const createdTask = await createTask(newTask);
      setTasks(prev => [...prev, createdTask]);
      setNewTask({
        title: '',
        description: '',
        status: 'pending',
        priority: 'medium'
      });
      setShowCreateForm(false);
    } catch (error) {
      console.error('Failed to create task:', error);
    }
  };

  const handleUpdateTask = async (taskId: string, updates: Partial<Task>) => {
    try {
      const updatedTask = await updateTask(taskId, updates);
      setTasks(prev =>
        prev.map(task => (task.id === taskId ? updatedTask : task))
      );
    } catch (error) {
      console.error('Failed to update task:', error);
    }
  };

  const handleDeleteTask = async (taskId: string) => {
    try {
      await deleteTask(taskId);
      setTasks(prev => prev.filter(task => task.id !== taskId));
      setShowDeleteConfirm(null);
    } catch (error) {
      console.error('Failed to delete task:', error);
    }
  };

  const filteredTasks = tasks.filter(task => 
    filterStatus === 'all' || task.status === filterStatus
  );

  const sortedTasks = [...filteredTasks].sort((a, b) => {
    if (sortBy === 'priority') {
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    }
    return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
  });

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between p-2 border-b">
        <h2 className="text-lg font-semibold">Task Manager</h2>
        <button
          onClick={() => setShowCreateForm(!showCreateForm)}
          className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          {showCreateForm ? 'Cancel' : 'New Task'}
        </button>
      </div>

      {showCreateForm && (
        <div className="p-4 border-b">
          <div className="space-y-4">
            <div>
              <label htmlFor="title" className="block text-sm font-medium text-gray-700">Title</label>
              <input
                id="title"
                type="text"
                value={newTask.title}
                onChange={(e) => setNewTask(prev => ({ ...prev, title: e.target.value }))}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>
            <div>
              <label htmlFor="description" className="block text-sm font-medium text-gray-700">Description</label>
              <textarea
                id="description"
                value={newTask.description}
                onChange={(e) => setNewTask(prev => ({ ...prev, description: e.target.value }))}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                rows={3}
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="status" className="block text-sm font-medium text-gray-700">Status</label>
                <select
                  id="status"
                  value={newTask.status}
                  onChange={(e) => setNewTask(prev => ({ ...prev, status: e.target.value as Task['status'] }))}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                >
                  <option value="pending">Pending</option>
                  <option value="in_progress">In Progress</option>
                  <option value="completed">Completed</option>
                  <option value="cancelled">Cancelled</option>
                </select>
              </div>
              <div>
                <label htmlFor="priority" className="block text-sm font-medium text-gray-700">Priority</label>
                <select
                  id="priority"
                  value={newTask.priority}
                  onChange={(e) => setNewTask(prev => ({ ...prev, priority: e.target.value as Task['priority'] }))}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>
            </div>
            <button
              onClick={handleCreateTask}
              className="w-full px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
            >
              Save
            </button>
          </div>
        </div>
      )}

      <div className="p-4 space-y-4">
        <div className="flex items-center space-x-4">
          <div className="flex-1">
            <label htmlFor="filter-status" className="block text-sm font-medium text-gray-700">Filter by Status</label>
            <select
              id="filter-status"
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            >
              <option value="all">All</option>
              <option value="pending">Pending</option>
              <option value="in_progress">In Progress</option>
              <option value="completed">Completed</option>
              <option value="cancelled">Cancelled</option>
            </select>
          </div>
          <div className="flex-1">
            <label htmlFor="sort-by" className="block text-sm font-medium text-gray-700">Sort by</label>
            <select
              id="sort-by"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            >
              <option value="createdAt">Created Date</option>
              <option value="priority">Priority</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" role="list">
          {sortedTasks.length === 0 ? (
            <div className="col-span-full text-center text-gray-500">
              No tasks available
            </div>
          ) : (
            sortedTasks.map(task => (
              <div
                key={task.id}
                className="p-4 rounded-lg border border-gray-200 hover:border-gray-300"
                role="listitem"
              >
                <div className="flex items-start justify-between">
                  <h3 className="font-medium">{task.title}</h3>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => handleUpdateTask(task.id, { status: 'completed' })}
                      className="text-green-600 hover:text-green-700"
                      aria-label="Complete task"
                    >
                      ✓
                    </button>
                    <button
                      onClick={() => setShowDeleteConfirm(task.id)}
                      className="text-red-600 hover:text-red-700"
                      aria-label="Delete task"
                    >
                      ✕
                    </button>
                  </div>
                </div>
                <p className="mt-2 text-sm text-gray-600">{task.description}</p>
                <div className="mt-4 flex items-center justify-between">
                  <span
                    className={`px-2 py-1 text-xs rounded-full ${
                      task.priority === 'high'
                        ? 'bg-red-100 text-red-800'
                        : task.priority === 'medium'
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-green-100 text-green-800'
                    }`}
                  >
                    {task.priority}
                  </span>
                  <span className="text-xs text-gray-500">
                    {new Date(task.updatedAt).toLocaleDateString()}
                  </span>
                </div>
                <div className="mt-2">
                  <button
                    onClick={() => handleUpdateTask(task.id, { 
                      status: task.status === 'pending' ? 'in_progress' : 
                             task.status === 'in_progress' ? 'completed' : 'pending'
                    })}
                    className="text-xs text-gray-500 hover:text-gray-700"
                  >
                    Status: {task.status}
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Delete Confirmation Dialog */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
          <div className="bg-white p-6 rounded-lg shadow-lg">
            <h3 className="text-lg font-medium mb-4">Delete Task</h3>
            <p className="text-gray-600 mb-6">Are you sure you want to delete this task?</p>
            <div className="flex justify-end space-x-4">
              <button
                onClick={() => setShowDeleteConfirm(null)}
                className="px-4 py-2 text-gray-600 hover:text-gray-700"
              >
                Cancel
              </button>
              <button
                onClick={() => showDeleteConfirm && handleDeleteTask(showDeleteConfirm)}
                className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
              >
                Confirm
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};