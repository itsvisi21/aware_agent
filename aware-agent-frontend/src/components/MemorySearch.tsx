import React, { useState } from 'react';
import { SemanticMemory } from '../lib/memory/SemanticMemory';
import { ChatMessage } from '../lib/types/chat';

interface MemorySearchProps {
  memory: SemanticMemory;
  onResultSelect: (message: ChatMessage) => void;
}

export const MemorySearch: React.FC<MemorySearchProps> = ({
  memory,
  onResultSelect
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<ChatMessage[]>([]);
  const [searchType, setSearchType] = useState<'semantic' | 'keyword' | 'temporal'>('semantic');
  const [dateRange, setDateRange] = useState<{ start: Date | null; end: Date | null }>({
    start: null,
    end: null
  });
  const [isSearching, setIsSearching] = useState(false);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    try {
      let results: ChatMessage[] = [];

      switch (searchType) {
        case 'semantic':
          results = await memory.searchSemantic(searchQuery);
          break;
        case 'keyword':
          results = await memory.searchKeyword(searchQuery);
          break;
        case 'temporal':
          results = await memory.searchTemporal(searchQuery, dateRange);
          break;
      }

      setSearchResults(results);
    } catch (error) {
      console.error('Error searching memory:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleExport = async () => {
    try {
      const exportData = await memory.exportMemory();
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
    }
  };

  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const content = e.target?.result as string;
        const importData = JSON.parse(content);
        await memory.importMemory(importData);
      };
      reader.readAsText(file);
    } catch (error) {
      console.error('Error importing memory:', error);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 space-y-4">
        <div className="flex items-center space-x-4">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search memory..."
            className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
          />
          <select
            value={searchType}
            onChange={(e) => setSearchType(e.target.value as typeof searchType)}
            className="rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
          >
            <option value="semantic">Semantic Search</option>
            <option value="keyword">Keyword Search</option>
            <option value="temporal">Temporal Search</option>
          </select>
          <button
            onClick={handleSearch}
            disabled={isSearching}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400"
          >
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </div>

        {searchType === 'temporal' && (
          <div className="flex items-center space-x-4">
            <input
              type="date"
              onChange={(e) => setDateRange(prev => ({ ...prev, start: e.target.value ? new Date(e.target.value) : null }))}
              className="rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            />
            <span>to</span>
            <input
              type="date"
              onChange={(e) => setDateRange(prev => ({ ...prev, end: e.target.value ? new Date(e.target.value) : null }))}
              className="rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            />
          </div>
        )}

        <div className="flex justify-end space-x-4">
          <button
            onClick={handleExport}
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
          >
            Export Memory
          </button>
          <label className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 cursor-pointer">
            Import Memory
            <input
              type="file"
              accept=".json"
              onChange={handleImport}
              className="hidden"
            />
          </label>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {searchResults.map((result, index) => (
          <div
            key={index}
            onClick={() => onResultSelect(result)}
            className="p-4 mb-4 border rounded-lg hover:bg-gray-50 cursor-pointer"
          >
            <div className="text-sm text-gray-500">
              {result.metadata.timestamp.toLocaleString()}
            </div>
            <div className="mt-2">
              {result.content}
            </div>
            {result.metadata.type && (
              <div className="mt-2 text-xs text-gray-400">
                Type: {result.metadata.type}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}; 