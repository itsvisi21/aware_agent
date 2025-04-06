import React, { useState } from 'react';
import { ChatMessage } from '../lib/types/chat';

interface TeachingModeProps {
  onMessage: (message: ChatMessage) => void;
  onQuestion: (question: string) => void;
  onExample: (example: string) => void;
  onExercise: (exercise: string) => void;
}

export const TeachingMode: React.FC<TeachingModeProps> = ({
  onMessage,
  onQuestion,
  onExample,
  onExercise
}) => {
  const [currentTopic, setCurrentTopic] = useState('');
  const [difficulty, setDifficulty] = useState<'beginner' | 'intermediate' | 'advanced'>('beginner');
  const [teachingStyle, setTeachingStyle] = useState<'socratic' | 'explanation' | 'interactive'>('socratic');
  const [showQuestionForm, setShowQuestionForm] = useState(false);
  const [showExampleForm, setShowExampleForm] = useState(false);
  const [showExerciseForm, setShowExerciseForm] = useState(false);

  const handleTopicSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentTopic.trim()) return;

    const message: ChatMessage = {
      role: 'user',
      content: `Let's learn about: ${currentTopic}`,
      metadata: {
        timestamp: new Date(),
        type: 'teaching',
        topic: currentTopic,
        difficulty,
        style: teachingStyle
      }
    };

    onMessage(message);
    setCurrentTopic('');
  };

  const handleQuestionSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const question = (e.target as HTMLFormElement).question.value;
    if (!question.trim()) return;

    onQuestion(question);
    setShowQuestionForm(false);
  };

  const handleExampleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const example = (e.target as HTMLFormElement).example.value;
    if (!example.trim()) return;

    onExample(example);
    setShowExampleForm(false);
  };

  const handleExerciseSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const exercise = (e.target as HTMLFormElement).exercise.value;
    if (!exercise.trim()) return;

    onExercise(exercise);
    setShowExerciseForm(false);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between p-2 border-b">
        <h2 className="text-lg font-semibold">Teaching Mode</h2>
        <div className="flex space-x-2">
          <button
            onClick={() => setShowQuestionForm(!showQuestionForm)}
            className="px-3 py-1 bg-blue-100 text-blue-800 rounded hover:bg-blue-200"
          >
            Ask Question
          </button>
          <button
            onClick={() => setShowExampleForm(!showExampleForm)}
            className="px-3 py-1 bg-green-100 text-green-800 rounded hover:bg-green-200"
          >
            Request Example
          </button>
          <button
            onClick={() => setShowExerciseForm(!showExerciseForm)}
            className="px-3 py-1 bg-purple-100 text-purple-800 rounded hover:bg-purple-200"
          >
            Get Exercise
          </button>
        </div>
      </div>

      <div className="p-4 space-y-4">
        <form onSubmit={handleTopicSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Topic</label>
            <input
              type="text"
              value={currentTopic}
              onChange={(e) => setCurrentTopic(e.target.value)}
              placeholder="What would you like to learn about?"
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">Difficulty</label>
              <select
                value={difficulty}
                onChange={(e) => setDifficulty(e.target.value as typeof difficulty)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">Teaching Style</label>
              <select
                value={teachingStyle}
                onChange={(e) => setTeachingStyle(e.target.value as typeof teachingStyle)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                <option value="socratic">Socratic Dialogue</option>
                <option value="explanation">Detailed Explanation</option>
                <option value="interactive">Interactive Learning</option>
              </select>
            </div>
          </div>

          <button
            type="submit"
            className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Start Learning
          </button>
        </form>

        {showQuestionForm && (
          <form onSubmit={handleQuestionSubmit} className="mt-4 p-4 border rounded">
            <h3 className="text-lg font-medium mb-2">Ask a Question</h3>
            <textarea
              name="question"
              placeholder="Type your question..."
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              rows={3}
            />
            <div className="mt-2 flex justify-end space-x-2">
              <button
                type="button"
                onClick={() => setShowQuestionForm(false)}
                className="px-3 py-1 bg-gray-100 text-gray-800 rounded hover:bg-gray-200"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Ask
              </button>
            </div>
          </form>
        )}

        {showExampleForm && (
          <form onSubmit={handleExampleSubmit} className="mt-4 p-4 border rounded">
            <h3 className="text-lg font-medium mb-2">Request an Example</h3>
            <textarea
              name="example"
              placeholder="What kind of example would you like?"
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              rows={3}
            />
            <div className="mt-2 flex justify-end space-x-2">
              <button
                type="button"
                onClick={() => setShowExampleForm(false)}
                className="px-3 py-1 bg-gray-100 text-gray-800 rounded hover:bg-gray-200"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600"
              >
                Request
              </button>
            </div>
          </form>
        )}

        {showExerciseForm && (
          <form onSubmit={handleExerciseSubmit} className="mt-4 p-4 border rounded">
            <h3 className="text-lg font-medium mb-2">Get an Exercise</h3>
            <textarea
              name="exercise"
              placeholder="What kind of exercise would you like?"
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              rows={3}
            />
            <div className="mt-2 flex justify-end space-x-2">
              <button
                type="button"
                onClick={() => setShowExerciseForm(false)}
                className="px-3 py-1 bg-gray-100 text-gray-800 rounded hover:bg-gray-200"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-3 py-1 bg-purple-500 text-white rounded hover:bg-purple-600"
              >
                Get Exercise
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}; 