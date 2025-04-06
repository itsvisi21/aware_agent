import React, { useState } from 'react';
import { SemanticKernel } from '../lib/kernel/SemanticKernel';
import { ChatMessage } from '../lib/types/chat';

interface ConversationExporterProps {
  kernel: SemanticKernel;
}

export const ConversationExporter: React.FC<ConversationExporterProps> = ({ kernel }) => {
  const [exportFormat, setExportFormat] = useState<'markdown' | 'json' | 'text'>('markdown');
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = async () => {
    setIsExporting(true);
    try {
      const messages = await kernel.getConversationHistory();
      let content = '';

      switch (exportFormat) {
        case 'markdown':
          content = messages.map(msg => {
            const timestamp = new Date(msg.metadata?.timestamp || Date.now()).toLocaleString();
            return `### ${msg.role.toUpperCase()} (${timestamp})\n\n${msg.content}\n\n`;
          }).join('\n---\n\n');
          break;
        case 'json':
          content = JSON.stringify(messages, null, 2);
          break;
        case 'text':
          content = messages.map(msg => {
            const timestamp = new Date(msg.metadata?.timestamp || Date.now()).toLocaleString();
            return `${msg.role.toUpperCase()} (${timestamp}):\n${msg.content}\n\n`;
          }).join('---\n\n');
          break;
      }

      const blob = new Blob([content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `conversation-${new Date().toISOString().split('T')[0]}.${exportFormat}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error exporting conversation:', error);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="flex items-center space-x-2">
      <select
        value={exportFormat}
        onChange={(e) => setExportFormat(e.target.value as 'markdown' | 'json' | 'text')}
        className="rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
      >
        <option value="markdown">Markdown</option>
        <option value="json">JSON</option>
        <option value="text">Text</option>
      </select>
      <button
        onClick={handleExport}
        disabled={isExporting}
        className={`px-3 py-1 rounded-md ${
          isExporting
            ? 'bg-gray-400 cursor-not-allowed'
            : 'bg-blue-500 hover:bg-blue-600'
        } text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2`}
      >
        {isExporting ? 'Exporting...' : 'Export'}
      </button>
    </div>
  );
}; 