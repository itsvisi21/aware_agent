import { WebSocketServiceImpl } from '../services/WebSocketService';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';

describe('WebSocketService', () => {
    test('WebSocketService should connect to server', async () => {
        const wsService = new WebSocketServiceImpl('ws://localhost:8000/ws');
        await wsService.connect();
        // Add a small delay to allow the connection to establish
        await new Promise(resolve => setTimeout(resolve, 100));
        expect(wsService.isConnected()).toBe(true);
        await wsService.disconnect();
    });

    test('WebSocketService should handle messages', async () => {
        const wsService = new WebSocketServiceImpl('ws://localhost:8000/ws');
        const mockMessage = '{"text": "Hello"}';
        const mockCallback = jest.fn();
        
        wsService.onMessage(mockCallback);
        await wsService.connect();
        
        // Simulate receiving a message
        const mockEvent = new MessageEvent('message', { data: mockMessage });
        (wsService as any).ws.onmessage(mockEvent);
        
        expect(mockCallback).toHaveBeenCalledWith(mockMessage);
        await wsService.disconnect();
    });

    test('WebSocketService should handle errors', async () => {
        const wsService = new WebSocketServiceImpl('ws://localhost:8000/ws');
        const mockCallback = jest.fn();
        
        wsService.onError(mockCallback);
        await wsService.connect();
        
        // Simulate an error
        const mockError = new Event('error');
        (wsService as any).ws.onerror(mockError);
        
        expect(mockCallback).toHaveBeenCalled();
        await wsService.disconnect();
    });

    test('WebSocketService should handle disconnection', async () => {
        const wsService = new WebSocketServiceImpl('ws://localhost:8000/ws');
        const mockCallback = jest.fn();
        
        wsService.onClose(mockCallback);
        await wsService.connect();
        
        // Add a small delay to allow the connection to establish
        await new Promise(resolve => setTimeout(resolve, 100));
        
        await wsService.disconnect();
        
        // Add a small delay to allow the disconnection to complete
        await new Promise(resolve => setTimeout(resolve, 100));
        
        expect(mockCallback).toHaveBeenCalled();
        expect(wsService.isConnected()).toBe(false);
    });
}); 