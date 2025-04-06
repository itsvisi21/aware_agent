import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface MarkdownEditorProps {
  content: string;
  onChange: (content: string) => void;
  onSave?: () => void;
  readOnly?: boolean;
}

export const MarkdownEditor: React.FC<MarkdownEditorProps> = ({
  content,
  onChange,
  onSave,
  readOnly = false
}) => {
  const [isPreview, setIsPreview] = useState(false);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 's' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      onSave?.();
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between p-2 border-b">
        <div className="flex space-x-2">
          <button
            onClick={() => setIsPreview(false)}
            className={`px-3 py-1 rounded ${
              !isPreview ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'
            }`}
          >
            Edit
          </button>
          <button
            onClick={() => setIsPreview(true)}
            className={`px-3 py-1 rounded ${
              isPreview ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'
            }`}
          >
            Preview
          </button>
        </div>
        {!readOnly && onSave && (
          <button
            onClick={onSave}
            className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600"
          >
            Save
          </button>
        )}
      </div>

      <div className="flex-1 overflow-auto">
        {isPreview ? (
          <div className="p-4 prose max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code({ node, inline, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || '');
                  return !inline && match ? (
                    <SyntaxHighlighter
                      style={vscDarkPlus}
                      language={match[1]}
                      PreTag="div"
                      {...props}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  ) : (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  );
                }
              }}
            >
              {content}
            </ReactMarkdown>
          </div>
        ) : (
          <textarea
            value={content}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            className="w-full h-full p-4 font-mono text-sm resize-none focus:outline-none"
            readOnly={readOnly}
            placeholder="Write your markdown here..."
          />
        )}
      </div>

      <div className="p-2 border-t text-sm text-gray-500">
        {!readOnly && (
          <span>Press Ctrl+S (Cmd+S on Mac) to save</span>
        )}
      </div>
    </div>
  );
}; 