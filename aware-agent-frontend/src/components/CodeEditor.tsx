import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface CodeEditorProps {
  code: string;
  language: string;
  onChange: (code: string) => void;
  onExecute?: () => void;
  readOnly?: boolean;
}

export const CodeEditor: React.FC<CodeEditorProps> = ({
  code,
  language,
  onChange,
  onExecute,
  readOnly = false
}) => {
  const [isExecuting, setIsExecuting] = useState(false);
  const [output, setOutput] = useState<string>('');

  const handleExecute = async () => {
    if (!onExecute) return;

    setIsExecuting(true);
    try {
      const result = await onExecute();
      setOutput(result);
    } catch (error) {
      setOutput(`Error: ${error.message}`);
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between p-2 border-b">
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium">Language:</span>
          <select
            value={language}
            onChange={(e) => onChange(code)}
            className="text-sm border rounded p-1"
            disabled={readOnly}
          >
            <option value="javascript">JavaScript</option>
            <option value="typescript">TypeScript</option>
            <option value="python">Python</option>
            <option value="java">Java</option>
            <option value="cpp">C++</option>
          </select>
        </div>
        {!readOnly && onExecute && (
          <button
            onClick={handleExecute}
            disabled={isExecuting}
            className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50"
          >
            {isExecuting ? 'Executing...' : 'Execute'}
          </button>
        )}
      </div>

      <div className="flex-1 overflow-auto">
        <SyntaxHighlighter
          language={language}
          style={vscDarkPlus}
          customStyle={{
            margin: 0,
            height: '100%',
            fontSize: '14px',
            fontFamily: 'monospace'
          }}
          showLineNumbers
          wrapLines
        >
          {code}
        </SyntaxHighlighter>
      </div>

      {output && (
        <div className="border-t p-2">
          <div className="text-sm font-medium mb-2">Output:</div>
          <pre className="bg-gray-100 p-2 rounded text-sm overflow-x-auto">
            {output}
          </pre>
        </div>
      )}
    </div>
  );
}; 