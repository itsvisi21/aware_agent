import React, { useState, useEffect } from 'react';
import { LocalLLM } from '../lib/llm/LocalLLM';
import { LLMConfig } from '../lib/types/llm';

interface ModelSettingsProps {
  llm: LocalLLM;
  onConfigUpdate: (config: Partial<LLMConfig>) => void;
}

export const ModelSettings: React.FC<ModelSettingsProps> = ({ llm, onConfigUpdate }) => {
  const [config, setConfig] = useState<LLMConfig>(llm.getConfig());
  const [isConnected, setIsConnected] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const checkConnection = async () => {
      setIsLoading(true);
      try {
        const connected = await llm.checkConnection();
        setIsConnected(connected);
        setError(null);
      } catch (err) {
        setError('Failed to check connection');
        setIsConnected(false);
      } finally {
        setIsLoading(false);
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, [llm]);

  const handleConfigChange = (key: keyof LLMConfig, value: any) => {
    const newConfig = { ...config, [key]: value };
    setConfig(newConfig);
    onConfigUpdate(newConfig);
  };

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Model Settings</h2>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-sm">
            {isLoading ? 'Checking...' : isConnected ? 'Connected' : 'Offline'}
          </span>
        </div>
      </div>

      {error && (
        <div className="p-2 bg-red-100 text-red-800 rounded text-sm">
          {error}
        </div>
      )}

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">Model</label>
          <input
            type="text"
            value={config.model}
            onChange={(e) => handleConfigChange('model', e.target.value)}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">Endpoint</label>
          <input
            type="text"
            value={config.endpoint}
            onChange={(e) => handleConfigChange('endpoint', e.target.value)}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Temperature ({config.temperature})
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={config.temperature}
            onChange={(e) => handleConfigChange('temperature', parseFloat(e.target.value))}
            className="mt-1 block w-full"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Max Tokens
          </label>
          <input
            type="number"
            value={config.maxTokens}
            onChange={(e) => handleConfigChange('maxTokens', parseInt(e.target.value))}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Top P ({config.topP})
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={config.topP}
            onChange={(e) => handleConfigChange('topP', parseFloat(e.target.value))}
            className="mt-1 block w-full"
          />
        </div>
      </div>

      <div className="pt-4 border-t">
        <button
          onClick={() => onConfigUpdate(config)}
          className="w-full bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Save Settings
        </button>
      </div>
    </div>
  );
}; 