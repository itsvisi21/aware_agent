import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { WebSocketProvider } from '../../contexts/WebSocketContext';
import { InteractionModeSelector, InteractionMode } from '../../components/InteractionModeSelector';
import { Message } from '../../types/chat';

// Mock WebSocket
const mockWebSocket = {
    send: jest.fn(),
    close: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    onopen: jest.fn(),
    onmessage: jest.fn(),
    onerror: jest.fn(),
    onclose: jest.fn(),
    CONNECTING: 0,
    OPEN: 1,
    CLOSING: 2,
    CLOSED: 3,
    readyState: 1,
};

global.WebSocket = jest.fn().mockImplementation(() => mockWebSocket);

const TestChat = () => {
    return (
        <WebSocketProvider url="ws://localhost:8080/ws">
            <div>
                <InteractionModeSelector
                    currentMode="research"
                    onModeChange={jest.fn()}
                />
                <input
                    data-testid="message-input"
                    type="text"
                    placeholder="Type your message..."
                />
                <button data-testid="send-button">Send</button>
                <div data-testid="messages"></div>
            </div>
        </WebSocketProvider>
    );
};

describe('Chat End-to-End', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    it('should handle complete message flow', async () => {
        render(<TestChat />);

        // Simulate WebSocket connection
        const openEvent = new Event('open');
        mockWebSocket.onopen(openEvent);

        // Type and send a message
        const input = screen.getByTestId('message-input');
        const sendButton = screen.getByTestId('send-button');

        fireEvent.change(input, { target: { value: 'Hello, agent!' } });
        fireEvent.click(sendButton);

        // Verify message was sent
        await waitFor(() => {
            expect(mockWebSocket.send).toHaveBeenCalledWith(
                JSON.stringify({
                    type: 'message',
                    payload: expect.objectContaining({
                        content: 'Hello, agent!',
                        sender: 'user',
                    }),
                })
            );
        });

        // Simulate agent response
        const agentResponse: Message = {
            id: '1',
            content: 'Hello! How can I help you today?',
            sender: 'agent',
            timestamp: Date.now(),
            metadata: {
                agent: 'Researcher',
                role: 'Research and Analysis',
                mode: 'research',
            },
        };

        const messageEvent = {
            data: JSON.stringify({
                type: 'message',
                payload: agentResponse,
            }),
        };

        mockWebSocket.onmessage(messageEvent);

        // Verify response is displayed
        await waitFor(() => {
            expect(screen.getByText('Hello! How can I help you today?')).toBeInTheDocument();
        });
    });

    it('should handle mode switching', async () => {
        const handleModeChange = jest.fn();
        render(
            <InteractionModeSelector
                currentMode="research"
                onModeChange={handleModeChange}
            />
        );

        // Click on build mode
        const buildModeButton = screen.getByText('Build Mode');
        fireEvent.click(buildModeButton);

        expect(handleModeChange).toHaveBeenCalledWith('build');
    });

    it('should handle connection errors', async () => {
        render(<TestChat />);

        // Simulate WebSocket error
        const errorEvent = new Event('error');
        mockWebSocket.onerror(errorEvent);

        // Verify error is displayed
        await waitFor(() => {
            expect(screen.getByText('WebSocket error occurred')).toBeInTheDocument();
        });
    });
}); 