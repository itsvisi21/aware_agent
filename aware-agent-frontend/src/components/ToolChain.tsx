import React, { useState } from 'react';
import { SemanticKernel } from '../lib/kernel/SemanticKernel';
import { Tool } from '../lib/types/tool';

interface ToolChainProps {
  kernel: SemanticKernel;
  tools: Tool[];
}

interface ToolChain {
  id: string;
  name: string;
  description: string;
  tools: Tool[];
  executionOrder: string[];
  dependencies: Record<string, string[]>;
}

export const ToolChain: React.FC<ToolChainProps> = ({ kernel, tools }) => {
  const [chains, setChains] = useState<ToolChain[]>([]);
  const [newChain, setNewChain] = useState<Partial<ToolChain>>({
    name: '',
    description: '',
    tools: [],
    executionOrder: [],
    dependencies: {}
  });
  const [showEditor, setShowEditor] = useState(false);
  const [selectedChain, setSelectedChain] = useState<ToolChain | null>(null);

  const handleCreateChain = () => {
    if (!newChain.name || !newChain.description || !newChain.tools?.length) return;

    const chain: ToolChain = {
      id: `chain_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name: newChain.name,
      description: newChain.description,
      tools: newChain.tools,
      executionOrder: newChain.executionOrder || [],
      dependencies: newChain.dependencies || {}
    };

    setChains(prev => [...prev, chain]);
    setNewChain({
      name: '',
      description: '',
      tools: [],
      executionOrder: [],
      dependencies: {}
    });
    setShowEditor(false);
  };

  const handleAddTool = (tool: Tool) => {
    setNewChain(prev => ({
      ...prev,
      tools: [...(prev.tools || []), tool],
      executionOrder: [...(prev.executionOrder || []), tool.id]
    }));
  };

  const handleRemoveTool = (toolId: string) => {
    setNewChain(prev => ({
      ...prev,
      tools: (prev.tools || []).filter(tool => tool.id !== toolId),
      executionOrder: (prev.executionOrder || []).filter(id => id !== toolId),
      dependencies: Object.fromEntries(
        Object.entries(prev.dependencies || {}).filter(([key]) => key !== toolId)
      )
    }));
  };

  const handleAddDependency = (toolId: string, dependsOn: string) => {
    setNewChain(prev => ({
      ...prev,
      dependencies: {
        ...(prev.dependencies || {}),
        [toolId]: [...(prev.dependencies?.[toolId] || []), dependsOn]
      }
    }));
  };

  const handleRemoveDependency = (toolId: string, dependsOn: string) => {
    setNewChain(prev => ({
      ...prev,
      dependencies: {
        ...(prev.dependencies || {}),
        [toolId]: (prev.dependencies?.[toolId] || []).filter(id => id !== dependsOn)
      }
    }));
  };

  const handleEditChain = (chain: ToolChain) => {
    setSelectedChain(chain);
    setNewChain(chain);
    setShowEditor(true);
  };

  const handleDeleteChain = (chainId: string) => {
    setChains(prev => prev.filter(chain => chain.id !== chainId));
  };

  const handleExecuteChain = async (chain: ToolChain) => {
    try {
      // Execute tools in order, respecting dependencies
      const results: Record<string, any> = {};
      const executed = new Set<string>();

      const executeTool = async (toolId: string) => {
        if (executed.has(toolId)) return;

        // Check dependencies
        const dependencies = chain.dependencies[toolId] || [];
        for (const depId of dependencies) {
          if (!executed.has(depId)) {
            await executeTool(depId);
          }
        }

        // Execute tool
        const tool = chain.tools.find(t => t.id === toolId);
        if (tool) {
          const result = await kernel.executeTool(tool);
          results[toolId] = result;
          executed.add(toolId);
        }
      };

      // Execute tools in order
      for (const toolId of chain.executionOrder) {
        await executeTool(toolId);
      }

      console.log('Chain execution results:', results);
    } catch (error) {
      console.error('Error executing tool chain:', error);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 space-y-4">
        <div className="flex justify-between items-center">
          <h2 className="text-lg font-semibold">Tool Chains</h2>
          <button
            onClick={() => setShowEditor(true)}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Create New Chain
          </button>
        </div>

        {showEditor && (
          <div className="border rounded-lg p-4 space-y-4">
            <div className="space-y-2">
              <label className="block text-sm font-medium">Chain Name</label>
              <input
                type="text"
                value={newChain.name}
                onChange={(e) => setNewChain(prev => ({ ...prev, name: e.target.value }))}
                className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium">Description</label>
              <textarea
                value={newChain.description}
                onChange={(e) => setNewChain(prev => ({ ...prev, description: e.target.value }))}
                className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                rows={3}
              />
            </div>

            <div className="space-y-2">
              <label className="block text-sm font-medium">Available Tools</label>
              <div className="grid grid-cols-2 gap-4">
                {tools.map(tool => (
                  <div
                    key={tool.id}
                    className="border rounded-lg p-2 flex items-center justify-between"
                  >
                    <div>
                      <h4 className="font-medium">{tool.name}</h4>
                      <p className="text-sm text-gray-600">{tool.description}</p>
                    </div>
                    {newChain.tools?.some(t => t.id === tool.id) ? (
                      <button
                        onClick={() => handleRemoveTool(tool.id)}
                        className="px-2 py-1 bg-red-100 text-red-700 rounded hover:bg-red-200"
                      >
                        Remove
                      </button>
                    ) : (
                      <button
                        onClick={() => handleAddTool(tool)}
                        className="px-2 py-1 bg-green-100 text-green-700 rounded hover:bg-green-200"
                      >
                        Add
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {newChain.tools && newChain.tools.length > 0 && (
              <div className="space-y-2">
                <label className="block text-sm font-medium">Tool Dependencies</label>
                {newChain.tools.map(tool => (
                  <div key={tool.id} className="border rounded-lg p-2">
                    <h4 className="font-medium">{tool.name}</h4>
                    <div className="mt-2 space-y-2">
                      {newChain.tools
                        .filter(t => t.id !== tool.id)
                        .map(depTool => (
                          <div key={depTool.id} className="flex items-center space-x-2">
                            <input
                              type="checkbox"
                              checked={newChain.dependencies?.[tool.id]?.includes(depTool.id)}
                              onChange={(e) => {
                                if (e.target.checked) {
                                  handleAddDependency(tool.id, depTool.id);
                                } else {
                                  handleRemoveDependency(tool.id, depTool.id);
                                }
                              }}
                              className="rounded border-gray-300 text-blue-500 focus:ring-blue-500"
                            />
                            <span>{depTool.name}</span>
                          </div>
                        ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            <div className="flex justify-end space-x-4">
              <button
                onClick={() => setShowEditor(false)}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateChain}
                className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
              >
                {selectedChain ? 'Update Chain' : 'Create Chain'}
              </button>
            </div>
          </div>
        )}

        <div className="space-y-4">
          {chains.map(chain => (
            <div key={chain.id} className="border rounded-lg p-4">
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="font-medium">{chain.name}</h3>
                  <p className="text-sm text-gray-600">{chain.description}</p>
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => handleEditChain(chain)}
                    className="px-2 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => handleDeleteChain(chain.id)}
                    className="px-2 py-1 bg-red-100 text-red-700 rounded hover:bg-red-200"
                  >
                    Delete
                  </button>
                  <button
                    onClick={() => handleExecuteChain(chain)}
                    className="px-2 py-1 bg-green-100 text-green-700 rounded hover:bg-green-200"
                  >
                    Execute
                  </button>
                </div>
              </div>
              <div className="mt-2">
                <h4 className="text-sm font-medium">Tools:</h4>
                <ul className="text-sm text-gray-600">
                  {chain.tools.map(tool => (
                    <li key={tool.id}>
                      {tool.name}
                      {chain.dependencies[tool.id]?.length > 0 && (
                        <span className="text-xs text-gray-400">
                          {' '}
                          (depends on:{' '}
                          {chain.dependencies[tool.id]
                            .map(depId => chain.tools.find(t => t.id === depId)?.name)
                            .join(', ')}
                          )
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}; 