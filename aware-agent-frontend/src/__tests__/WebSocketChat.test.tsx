/// <reference types="jest" />

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { WebSocketChat } from '../components/WebSocketChat';
import { WebSocketServiceImpl } from '../services/WebSocketService';

jest.mock('../services/WebSocketService', () => ({
    WebSocketServiceImpl: jest.fn().mockImplementation(() => ({
        connect: jest.fn(),
        disconnect: jest.fn(),
        sendMessage: jest.fn(),
        onMessage: jest.fn(),
        onError: jest.fn(),
        onClose: jest.fn(),
        isConnected: jest.fn().mockReturnValue(true),
    })),
}));

describe('WebSocketChat', () => {
    let mockWebSocketService: jest.Mocked<WebSocketServiceImpl>;

    beforeEach(() => {
        jest.clearAllMocks();
        mockWebSocketService = new WebSocketServiceImpl('') as jest.Mocked<WebSocketServiceImpl>;
        (WebSocketServiceImpl as jest.Mock).mockImplementation(() => mockWebSocketService);
    });

    it('renders chat interface', () => {
        render(<WebSocketChat />);
        expect(screen.getByPlaceholderText('Type your message...')).toBeInTheDocument();
        expect(screen.getByText('Send')).toBeInTheDocument();
    });

    it('sends message when send button is clicked', async () => {
        render(<WebSocketChat />);
        const input = screen.getByPlaceholderText('Type your message...');
        const sendButton = screen.getByText('Send');

        fireEvent.change(input, { target: { value: 'Hello' } });
        fireEvent.click(sendButton);

        await waitFor(() => {
            expect(mockWebSocketService.sendMessage).toHaveBeenCalledWith('Hello');
        });
    });

    it('displays connection status', () => {
        render(<WebSocketChat />);
        expect(screen.getByText('Connected')).toBeInTheDocument();
    });

    it('handles errors', async () => {
        const errorMessage = 'Connection failed';
        mockWebSocketService.onError.mockImplementation((callback) => {
            callback(new Error(errorMessage));
        });

        render(<WebSocketChat />);
        expect(await screen.findByText(`Error: ${errorMessage}`)).toBeInTheDocument();
    });
}); 