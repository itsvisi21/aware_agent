import React, { useState } from 'react';
import { SemanticKernel } from '../lib/kernel/SemanticKernel';

interface Tool {
  id: string;
  name: string;
  description: string;
  parameters: ToolParameter[];
  code: string;
}

interface ToolParameter {
  name: string;
  type: string;
  description: string;
  required: boolean;
}

interface ToolManagerProps {
  kernel: SemanticKernel;
}

export const ToolManager: React.FC<ToolManagerProps> = ({ kernel }) => {
  const [tools, setTools] = useState<Tool[]>([]);
  const [newTool, setNewTool] = useState<Partial<Tool>>({
    name: '',
    description: '',
    parameters: [],
    code: ''
  });
  const [showEditor, setShowEditor] = useState(false);
  const [selectedTool, setSelectedTool] = useState<Tool | null>(null);

  const handleCreateTool = () => {
    if (!newTool.name || !newTool.description || !newTool.code) return;

    const tool: Tool = {
      id: `tool_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name: newTool.name,
      description: newTool.description,
      parameters: newTool.parameters || [],
      code: newTool.code
    };

    setTools(prev => [...prev, tool]);
    setNewTool({
      name: '',
      description: '',
      parameters: [],
      code: ''
    });
    setShowEditor(false);
  };

  const handleAddParameter = () => {
    setNewTool(prev => ({
      ...prev,
      parameters: [
        ...(prev.parameters || []),
        {
          name: '',
          type: 'string',
          description: '',
          required: true
        }
      ]
    }));
  };

  const handleParameterChange = (index: number, field: keyof ToolParameter, value: string | boolean) => {
    setNewTool(prev => ({
      ...prev,
      parameters: (prev.parameters || []).map((param, i) =>
        i === index ? { ...param, [field]: value } : param
      )
    }));
  };

  const handleEditTool = (tool: Tool) => {
    setSelectedTool(tool);
    setNewTool(tool);
    setShowEditor(true);
  };

  const handleDeleteTool = (toolId: string) => {
    setTools(prev => prev.filter(tool => tool.id !== toolId));
  };

  const handleExecuteTool = async (tool: Tool) => {
    try {
      // Execute the tool's code
      const result = await kernel.executeTool(tool);
      console.log('Tool execution result:', result);
    } catch (error) {
      console.error('Error executing tool:', error);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 space-y-4">
        <div className="flex justify-between items-center">
          <h2 className="text-lg font-semibold">Custom Tools</h2>
          <button
            onClick={() => setShowEditor(true)}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Create New Tool
          </button>
        </div>

        {showEditor && (
          <div className="border rounded-lg p-4 space-y-4">
            <div className="space-y-2">
              <label className="block text-sm font-medium">Tool Name</label>
              <input
                type="text"
                value={newTool.name}
                onChange={(e) => setNewTool(prev => ({ ...prev, name: e.target.value }))}
                className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium">Description</label>
              <textarea
                value={newTool.description}
                onChange={(e) => setNewTool(prev => ({ ...prev, description: e.target.value }))}
                className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                rows={3}
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <label className="block text-sm font-medium">Parameters</label>
                <button
                  onClick={handleAddParameter}
                  className="px-2 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
                >
                  Add Parameter
                </button>
              </div>
              {newTool.parameters?.map((param, index) => (
                <div key={index} className="flex space-x-4">
                  <input
                    type="text"
                    value={param.name}
                    onChange={(e) => handleParameterChange(index, 'name', e.target.value)}
                    placeholder="Parameter name"
                    className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  />
                  <select
                    value={param.type}
                    onChange={(e) => handleParameterChange(index, 'type', e.target.value)}
                    className="rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  >
                    <option value="string">String</option>
                    <option value="number">Number</option>
                    <option value="boolean">Boolean</option>
                    <option value="object">Object</option>
                  </select>
                  <input
                    type="text"
                    value={param.description}
                    onChange={(e) => handleParameterChange(index, 'description', e.target.value)}
                    placeholder="Description"
                    className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  />
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={param.required}
                      onChange={(e) => handleParameterChange(index, 'required', e.target.checked)}
                      className="rounded border-gray-300 text-blue-500 focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm">Required</span>
                  </label>
                </div>
              ))}
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium">Code</label>
              <textarea
                value={newTool.code}
                onChange={(e) => setNewTool(prev => ({ ...prev, code: e.target.value }))}
                className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 font-mono"
                rows={10}
              />
            </div>

            <div className="flex justify-end space-x-4">
              <button
                onClick={() => setShowEditor(false)}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateTool}
                className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
              >
                {selectedTool ? 'Update Tool' : 'Create Tool'}
              </button>
            </div>
          </div>
        )}

        <div className="space-y-4">
          {tools.map(tool => (
            <div key={tool.id} className="border rounded-lg p-4">
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="font-medium">{tool.name}</h3>
                  <p className="text-sm text-gray-600">{tool.description}</p>
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => handleEditTool(tool)}
                    className="px-2 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => handleDeleteTool(tool.id)}
                    className="px-2 py-1 bg-red-100 text-red-700 rounded hover:bg-red-200"
                  >
                    Delete
                  </button>
                  <button
                    onClick={() => handleExecuteTool(tool)}
                    className="px-2 py-1 bg-green-100 text-green-700 rounded hover:bg-green-200"
                  >
                    Execute
                  </button>
                </div>
              </div>
              {tool.parameters.length > 0 && (
                <div className="mt-2">
                  <h4 className="text-sm font-medium">Parameters:</h4>
                  <ul className="text-sm text-gray-600">
                    {tool.parameters.map((param, index) => (
                      <li key={index}>
                        {param.name} ({param.type}) - {param.description}
                        {param.required && ' *'}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}; 