import { WebSocketService } from '../services/websocket';
import { Message } from '../types/chat';

describe('WebSocketService', () => {
    let wsService: WebSocketService;
    const mockUrl = 'ws://localhost:8080/ws';
    let mockSocket: WebSocket;

    beforeEach(() => {
        // Mock WebSocket
        global.WebSocket = jest.fn().mockImplementation(() => ({
            send: jest.fn(),
            close: jest.fn(),
            addEventListener: jest.fn(),
            removeEventListener: jest.fn(),
        })) as any;

        wsService = new WebSocketService(mockUrl);
        mockSocket = new WebSocket(mockUrl);
    });

    afterEach(() => {
        jest.clearAllMocks();
    });

    it('should connect to WebSocket server', () => {
        wsService.connect();
        expect(WebSocket).toHaveBeenCalledWith(mockUrl);
    });

    it('should handle message events', () => {
        const mockHandler = jest.fn();
        wsService.onMessage(mockHandler);

        const mockMessage = {
            type: 'message',
            payload: {
                id: '1',
                content: 'Test message',
                sender: 'user',
                timestamp: Date.now(),
            } as Message,
        };

        // Simulate WebSocket message event
        const event = { data: JSON.stringify(mockMessage) } as MessageEvent;
        (mockSocket.onmessage as jest.Mock)(event);

        expect(mockHandler).toHaveBeenCalledWith(mockMessage.payload);
    });

    it('should handle error events', () => {
        const mockHandler = jest.fn();
        wsService.onError(mockHandler);

        // Simulate WebSocket error event
        (mockSocket.onerror as jest.Mock)();

        expect(mockHandler).toHaveBeenCalledWith('WebSocket error occurred');
    });

    it('should handle status events', () => {
        const mockHandler = jest.fn();
        wsService.onStatus(mockHandler);

        const mockStatus = {
            type: 'status',
            payload: 'Connected to server',
        };

        // Simulate WebSocket message event with status
        const event = { data: JSON.stringify(mockStatus) } as MessageEvent;
        (mockSocket.onmessage as jest.Mock)(event);

        expect(mockHandler).toHaveBeenCalledWith(mockStatus.payload);
    });

    it('should send messages when connected', () => {
        wsService.connect();
        const message: Message = {
            id: '1',
            content: 'Test message',
            sender: 'user',
            timestamp: Date.now(),
        };

        wsService.sendMessage(message);
        expect(mockSocket.send).toHaveBeenCalledWith(
            JSON.stringify({
                type: 'message',
                payload: message,
            })
        );
    });

    it('should handle disconnection', () => {
        wsService.connect();
        wsService.disconnect();
        expect(mockSocket.close).toHaveBeenCalled();
    });
}); 