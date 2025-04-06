import React from 'react';

export type InteractionMode = 'research' | 'build' | 'teach' | 'collab';

interface InteractionModeSelectorProps {
    currentMode: InteractionMode;
    onModeChange: (mode: InteractionMode) => void;
}

const modeConfig = {
    research: {
        label: 'Research Mode',
        description: 'Deep exploration and analysis',
        icon: '🔍',
    },
    build: {
        label: 'Build Mode',
        description: 'Idea to execution',
        icon: '🛠️',
    },
    teach: {
        label: 'Teach Mode',
        description: 'Explanation and learning focus',
        icon: '📚',
    },
    collab: {
        label: 'Collaboration Mode',
        description: 'Multi-user with shared memory',
        icon: '👥',
    },
};

export const InteractionModeSelector: React.FC<InteractionModeSelectorProps> = ({
    currentMode,
    onModeChange,
}) => {
    return (
        <div className="flex space-x-2 p-4 bg-white rounded-lg shadow">
            {Object.entries(modeConfig).map(([mode, config]) => (
                <button
                    key={mode}
                    onClick={() => onModeChange(mode as InteractionMode)}
                    className={`flex flex-col items-center p-3 rounded-lg transition-colors ${
                        currentMode === mode
                            ? 'bg-blue-100 text-blue-700'
                            : 'hover:bg-gray-100 text-gray-700'
                    }`}
                >
                    <span className="text-2xl mb-1">{config.icon}</span>
                    <span className="text-sm font-medium">{config.label}</span>
                    <span className="text-xs text-gray-500">{config.description}</span>
                </button>
            ))}
        </div>
    );
}; 