import React from 'react';
import { render, screen, act } from '@testing-library/react';
import { WebSocketProvider, useWebSocket } from '../contexts/WebSocketContext';
import { WebSocketService } from '../services/websocket';

// Mock WebSocketService
jest.mock('../services/websocket', () => ({
    WebSocketService: jest.fn().mockImplementation(() => ({
        connect: jest.fn(),
        disconnect: jest.fn(),
        sendMessage: jest.fn(),
        onMessage: jest.fn(),
        onError: jest.fn(),
        onStatus: jest.fn(),
    })),
}));

const TestComponent = () => {
    const { messages, status, error, sendMessage, isConnected } = useWebSocket();
    return (
        <div>
            <div data-testid="status">{status}</div>
            <div data-testid="error">{error}</div>
            <div data-testid="connected">{isConnected.toString()}</div>
            <div data-testid="messages">{JSON.stringify(messages)}</div>
            <button onClick={() => sendMessage('test message')}>Send</button>
        </div>
    );
};

describe('WebSocketContext', () => {
    let mockWsService: WebSocketService;

    beforeEach(() => {
        mockWsService = new WebSocketService('ws://localhost:8080/ws');
    });

    it('should provide WebSocket context to children', () => {
        render(
            <WebSocketProvider url="ws://localhost:8080/ws">
                <TestComponent />
            </WebSocketProvider>
        );

        expect(screen.getByTestId('status')).toHaveTextContent('Disconnected');
        expect(screen.getByTestId('error')).toHaveTextContent('');
        expect(screen.getByTestId('connected')).toHaveTextContent('false');
        expect(screen.getByTestId('messages')).toHaveTextContent('[]');
    });

    it('should handle message events', () => {
        const messageHandler = jest.fn();
        (mockWsService.onMessage as jest.Mock).mockImplementation((handler) => {
            messageHandler(handler);
        });

        render(
            <WebSocketProvider url="ws://localhost:8080/ws">
                <TestComponent />
            </WebSocketProvider>
        );

        const testMessage = {
            id: '1',
            content: 'Test message',
            sender: 'user',
            timestamp: Date.now(),
        };

        act(() => {
            messageHandler(testMessage);
        });

        expect(screen.getByTestId('messages')).toHaveTextContent(
            JSON.stringify([testMessage])
        );
    });

    it('should handle status events', () => {
        const statusHandler = jest.fn();
        (mockWsService.onStatus as jest.Mock).mockImplementation((handler) => {
            statusHandler(handler);
        });

        render(
            <WebSocketProvider url="ws://localhost:8080/ws">
                <TestComponent />
            </WebSocketProvider>
        );

        act(() => {
            statusHandler('Connected to server');
        });

        expect(screen.getByTestId('status')).toHaveTextContent('Connected to server');
        expect(screen.getByTestId('connected')).toHaveTextContent('true');
    });

    it('should handle error events', () => {
        const errorHandler = jest.fn();
        (mockWsService.onError as jest.Mock).mockImplementation((handler) => {
            errorHandler(handler);
        });

        render(
            <WebSocketProvider url="ws://localhost:8080/ws">
                <TestComponent />
            </WebSocketProvider>
        );

        act(() => {
            errorHandler('Connection error');
        });

        expect(screen.getByTestId('error')).toHaveTextContent('Connection error');
    });

    it('should send messages when sendMessage is called', () => {
        render(
            <WebSocketProvider url="ws://localhost:8080/ws">
                <TestComponent />
            </WebSocketProvider>
        );

        const sendButton = screen.getByText('Send');
        act(() => {
            sendButton.click();
        });

        expect(mockWsService.sendMessage).toHaveBeenCalled();
    });
}); 