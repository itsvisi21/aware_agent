import React, { useState, useEffect } from 'react';
import { SemanticMemory } from '../lib/memory/SemanticMemory';
import { MemoryNode } from '../lib/memory/types';

interface MemoryVisualizerProps {
  memory: SemanticMemory;
}

export const MemoryVisualizer: React.FC<MemoryVisualizerProps> = ({ memory }) => {
  const [nodes, setNodes] = useState<MemoryNode[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedNode, setSelectedNode] = useState<MemoryNode | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [isImporting, setIsImporting] = useState(false);

  useEffect(() => {
    const loadNodes = async () => {
      const memoryNodes = await memory.getNodes();
      setNodes(memoryNodes);
    };
    loadNodes();
  }, [memory]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      const allNodes = await memory.getNodes();
      setNodes(allNodes);
      return;
    }

    const results = await memory.findRelevantNodes(searchQuery);
    setNodes(results.map(r => r.node));
  };

  const handleExport = async () => {
    setIsExporting(true);
    try {
      const exportData = {
        nodes,
        timestamp: new Date().toISOString()
      };

      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `memory-export-${new Date().toISOString()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error exporting memory:', error);
    } finally {
      setIsExporting(false);
    }
  };

  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsImporting(true);
    try {
      const content = await file.text();
      const data = JSON.parse(content);
      
      if (Array.isArray(data.nodes)) {
        for (const node of data.nodes) {
          await memory.storeNode(node);
        }
        setNodes(await memory.getNodes());
      }
    } catch (error) {
      console.error('Error importing memory:', error);
    } finally {
      setIsImporting(false);
    }
  };

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Memory Visualization</h2>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleExport}
            disabled={isExporting}
            className="px-3 py-1 bg-green-100 text-green-800 rounded hover:bg-green-200 disabled:opacity-50"
          >
            {isExporting ? 'Exporting...' : 'Export Memory'}
          </button>
          <label className="px-3 py-1 bg-blue-100 text-blue-800 rounded hover:bg-blue-200 cursor-pointer">
            {isImporting ? 'Importing...' : 'Import Memory'}
            <input
              type="file"
              accept=".json"
              onChange={handleImport}
              className="hidden"
              disabled={isImporting}
            />
          </label>
        </div>
      </div>

      <div className="flex space-x-4">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search memory..."
          className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
        />
        <button
          onClick={handleSearch}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Search
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {nodes.map((node) => (
          <div
            key={node.id}
            onClick={() => setSelectedNode(node)}
            className={`p-4 rounded-lg border cursor-pointer ${
              selectedNode?.id === node.id
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            <div className="font-medium">{node.content.substring(0, 100)}...</div>
            <div className="text-sm text-gray-500 mt-2">
              {new Date(node.timestamp).toLocaleString()}
            </div>
            <div className="flex flex-wrap gap-1 mt-2">
              {node.tags.map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-1 bg-gray-100 text-gray-800 rounded-full text-xs"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {selectedNode && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-xl font-semibold">Memory Node Details</h3>
              <button
                onClick={() => setSelectedNode(null)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>
            <div className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">Content</h4>
                <p className="whitespace-pre-wrap">{selectedNode.content}</p>
              </div>
              <div>
                <h4 className="font-medium mb-2">Metadata</h4>
                <pre className="bg-gray-50 p-2 rounded text-sm overflow-x-auto">
                  {JSON.stringify(selectedNode.metadata, null, 2)}
                </pre>
              </div>
              <div>
                <h4 className="font-medium mb-2">Relationships</h4>
                <div className="space-y-2">
                  {selectedNode.parentId && (
                    <div>
                      <span className="text-gray-500">Parent:</span>{' '}
                      {nodes.find(n => n.id === selectedNode.parentId)?.content.substring(0, 50)}...
                    </div>
                  )}
                  {selectedNode.childrenIds.length > 0 && (
                    <div>
                      <span className="text-gray-500">Children:</span>
                      <ul className="list-disc list-inside mt-1">
                        {selectedNode.childrenIds.map(childId => {
                          const child = nodes.find(n => n.id === childId);
                          return child ? (
                            <li key={childId}>
                              {child.content.substring(0, 50)}...
                            </li>
                          ) : null;
                        })}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}; 