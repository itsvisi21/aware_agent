import { WebSocketService } from '../../services/websocket';
import { waitFor } from '@testing-library/react';
import { Message } from '../../types/chat';

describe('WebSocket Integration', () => {
    let wsService: WebSocketService;
    const testUrl = 'ws://localhost:8080/ws';

    beforeEach(() => {
        wsService = new WebSocketService(testUrl);
    });

    afterEach(() => {
        wsService.disconnect();
    });

    it('should establish connection with backend', async () => {
        const connectPromise = new Promise<void>((resolve) => {
            wsService.onStatus((status) => {
                if (status === 'Connected to server') {
                    resolve();
                }
            });
        });

        wsService.connect();
        await expect(connectPromise).resolves.toBeUndefined();
    });

    it('should send and receive messages', async () => {
        const testMessage: Message = {
            id: '1',
            content: 'Hello, backend!',
            sender: 'user',
            timestamp: Date.now(),
            metadata: {
                agent: 'test',
                role: 'test',
                mode: 'research'
            }
        };

        const receivedMessage = new Promise<Message>((resolve) => {
            wsService.onMessage((message) => resolve(message));
        });

        wsService.connect();
        await waitFor(() => {
            const statusHandler = jest.fn();
            wsService.onStatus(statusHandler);
            expect(statusHandler).toHaveBeenCalledWith('Connected to server');
        });

        wsService.sendMessage(testMessage);
        const response = await receivedMessage;
        expect(response).toBeDefined();
    });

    it('should handle connection errors', async () => {
        const errorPromise = new Promise<string>((resolve) => {
            wsService.onError((error) => resolve(error));
        });

        wsService.connect();
        await expect(errorPromise).resolves.toBeDefined();
    });

    it('should maintain connection state', async () => {
        const statusHandler = jest.fn();
        wsService.onStatus(statusHandler);

        wsService.connect();
        await waitFor(() => {
            expect(statusHandler).toHaveBeenCalledWith('Connected to server');
        });

        wsService.disconnect();
        await waitFor(() => {
            expect(statusHandler).toHaveBeenCalledWith('Disconnected from server');
        });
    });
}); 