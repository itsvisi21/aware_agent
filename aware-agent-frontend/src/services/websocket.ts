import { Message } from '../types/chat';

interface WebSocketMessage {
    type: 'message' | 'error' | 'status' | 'ping' | 'pong';
    payload: any;
}

// Check if we're in a Web3 wallet context
const isWeb3Context = () => {
    return typeof window !== 'undefined' && 
           (window.location.protocol === 'chrome-extension:' || 
            window.ethereum !== undefined);
};

export class WebSocketService {
    private socket: WebSocket | null = null;
    private messageHandlers: ((message: Message) => void)[] = [];
    private errorHandlers: ((error: string) => void)[] = [];
    private statusHandlers: ((status: string) => void)[] = [];
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;
    private reconnectTimeout: NodeJS.Timeout | null = null;
    private isWeb3 = isWeb3Context();
    private connectionState: 'disconnected' | 'connecting' | 'connected' = 'disconnected';
    private messageQueue: Message[] = [];
    private connectionPromise: Promise<void> | null = null;
    private connectionTimeout: NodeJS.Timeout | null = null;
    private pingInterval: NodeJS.Timeout | null = null;
    private lastPingTime: number = 0;
    private readonly CONNECTION_TIMEOUT = 30000; // 30 seconds
    private web3ConnectionDelay = 5000; // 5 seconds base delay
    private readonly PING_INTERVAL = 60000; // 60 seconds
    private readonly PING_TIMEOUT = 30000; // 30 seconds
    private readonly RECONNECT_DELAY = 5000; // 5 seconds base delay
    private readonly MAX_RECONNECT_DELAY = 30000; // 30 seconds max delay

    constructor(private url: string) {
        if (!url.startsWith('ws://') && !url.startsWith('wss://')) {
            throw new Error('Invalid WebSocket URL format');
        }

        // Log context information
        console.log('WebSocket context:', {
            isWeb3: this.isWeb3,
            protocol: typeof window !== 'undefined' ? window.location.protocol : 'undefined',
            hasEthereum: typeof window !== 'undefined' ? !!window.ethereum : false
        });

        // Automatically handle page visibility changes
        if (typeof document !== 'undefined') {
            document.addEventListener('visibilitychange', () => {
                if (document.visibilityState === 'visible') {
                    this.handleVisibilityChange();
                }
            });
        }

        // Handle MetaMask interference
        if (this.isWeb3) {
            this.handleMetaMaskInterference();
        }
    }

    private handleMetaMaskInterference() {
        // Add a small delay to let MetaMask initialize
        setTimeout(() => {
            // Check if MetaMask is still interfering
            if (window.ethereum && window.ethereum.isMetaMask) {
                console.warn('MetaMask detected, applying additional connection delay');
                this.web3ConnectionDelay = 10000; // Increase delay to 10 seconds
            }
        }, 1000);
    }

    private handleVisibilityChange() {
        if (this.connectionState === 'disconnected' || 
            (this.socket && this.socket.readyState > WebSocket.OPEN)) {
            console.log('Page became visible, checking connection...');
            this.connect();
        }
    }

    private clearConnectionTimeout() {
        if (this.connectionTimeout) {
            clearTimeout(this.connectionTimeout);
            this.connectionTimeout = null;
        }
    }

    private setupConnectionTimeout(reject: (error: any) => void) {
        this.clearConnectionTimeout();
        this.connectionTimeout = setTimeout(() => {
            if (this.connectionState === 'connecting') {
                console.error('WebSocket connection timed out');
                this.connectionState = 'disconnected';
                this.disconnect();
                reject(new Error('WebSocket connection timed out'));
            }
        }, this.CONNECTION_TIMEOUT);
    }

    private startPingInterval() {
        this.stopPingInterval();
        this.pingInterval = setInterval(() => {
            if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                this.lastPingTime = Date.now();
                this.socket.send(JSON.stringify({ type: 'ping' }));
            }
        }, this.PING_INTERVAL);
    }

    private stopPingInterval() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    private setupWebSocket(resolve: () => void, reject: (error: any) => void) {
        try {
            // Create a unique URL to prevent caching and interference
            const uniqueUrl = new URL(this.url);
            uniqueUrl.searchParams.set('_t', Date.now().toString());
            uniqueUrl.searchParams.set('_r', Math.random().toString(36).substring(2));
            
            // Add additional parameters for Web3 context
            if (this.isWeb3) {
                uniqueUrl.searchParams.set('_web3', 'true');
                uniqueUrl.searchParams.set('_protocol', window.location.protocol);
                uniqueUrl.searchParams.set('_mm', window.ethereum?.isMetaMask ? 'true' : 'false');
            }

            this.socket = new WebSocket(uniqueUrl.toString());

            // Set up connection timeout
            this.setupConnectionTimeout(reject);

            // Set up WebSocket handlers
            this.setupWebSocketHandlers(resolve, reject);
        } catch (error) {
            console.error('Error in WebSocket setup:', error);
            reject(error);
        }
    }

    async connect(): Promise<void> {
        // If already connecting, return existing promise
        if (this.connectionPromise) {
            return this.connectionPromise;
        }

        // If already connected, resolve immediately
        if (this.connectionState === 'connected' && 
            this.socket && 
            this.socket.readyState === WebSocket.OPEN) {
            return Promise.resolve();
        }

        this.connectionPromise = new Promise<void>((resolve, reject) => {
            try {
                console.log('Attempting to connect to WebSocket at:', this.url);
                this.connectionState = 'connecting';
                
                // Clean up any existing connection
                this.disconnect();

                if (this.isWeb3) {
                    console.warn('Running in Web3 context, using defensive WebSocket setup');
                    // Add delay for Web3 context
                    setTimeout(() => {
                        this.setupWebSocket(resolve, reject);
                    }, this.web3ConnectionDelay);
                } else {
                    this.setupWebSocket(resolve, reject);
                }
            } catch (error) {
                console.error('Error establishing WebSocket connection:', error);
                this.errorHandlers.forEach(handler => handler('Failed to establish WebSocket connection'));
                this.connectionState = 'disconnected';
                this.clearConnectionTimeout();
                reject(error);
            }
        });

        try {
            await this.connectionPromise;
            // Process any queued messages
            while (this.messageQueue.length > 0) {
                const message = this.messageQueue.shift();
                if (message) {
                    await this.sendMessage(message);
                }
            }
            return this.connectionPromise;
        } catch (error) {
            console.error('Connection attempt failed:', error);
            throw error;
        } finally {
            this.connectionPromise = null;
            this.clearConnectionTimeout();
        }
    }

    private setupWebSocketHandlers(resolve: () => void, reject: (error: any) => void) {
        if (!this.socket) {
            reject(new Error('WebSocket is null'));
            return;
        }

        this.socket.onopen = () => {
            console.log('WebSocket connection established');
            this.reconnectAttempts = 0;
            this.connectionState = 'connected';
            this.clearConnectionTimeout();
            this.startPingInterval();
            this.notifyStatus('Connected to server');
            resolve();
        };

        this.socket.onmessage = (event) => {
            try {
                if (!event || !event.data) {
                    console.warn('Received invalid event or empty message from WebSocket');
                    return;
                }

                let data: WebSocketMessage;
                try {
                    data = JSON.parse(event.data);
                } catch (parseError) {
                    console.error('Failed to parse WebSocket message:', parseError);
                    this.errorHandlers.forEach(handler => handler('Invalid message format received'));
                    return;
                }

                if (!data || typeof data !== 'object') {
                    console.warn('Invalid message format received:', data);
                    return;
                }

                if (!data.type || typeof data.type !== 'string') {
                    console.warn('Message missing type field:', data);
                    return;
                }

                switch (data.type) {
                    case 'message':
                        if (data.payload && typeof data.payload === 'object') {
                            this.messageHandlers.forEach(handler => handler(data.payload));
                        } else {
                            console.warn('Invalid message payload:', data.payload);
                        }
                        break;
                    case 'error':
                        const errorMessage = typeof data.payload === 'string' 
                            ? data.payload 
                            : 'Unknown error occurred';
                        this.errorHandlers.forEach(handler => handler(errorMessage));
                        break;
                    case 'status':
                        const statusMessage = typeof data.payload === 'string'
                            ? data.payload
                            : 'Status update';
                        this.statusHandlers.forEach(handler => handler(statusMessage));
                        break;
                    case 'ping':
                        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                            this.socket.send(JSON.stringify({ type: 'pong' }));
                        }
                        break;
                    case 'pong':
                        this.lastPingTime = Date.now();
                        break;
                    default:
                        console.warn('Unknown message type received:', data.type);
                }
            } catch (error) {
                console.error('Error processing WebSocket message:', error);
                this.errorHandlers.forEach(handler => handler('Error processing message from server'));
            }
        };

        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.errorHandlers.forEach(handler => handler('WebSocket error occurred'));
            
            // Check if we've exceeded ping timeout
            if (Date.now() - this.lastPingTime > this.PING_TIMEOUT) {
                console.warn('Ping timeout detected, reconnecting...');
                this.connectionState = 'disconnected';
                this.disconnect();
            }
            
            // Attempt to reconnect if we haven't exceeded max attempts
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                const delay = Math.min(
                    this.RECONNECT_DELAY * Math.pow(2, this.reconnectAttempts - 1),
                    this.MAX_RECONNECT_DELAY
                );
                console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts}) after ${delay}ms...`);
                this.reconnectTimeout = setTimeout(() => this.connect(), delay);
            }
        };

        this.socket.onclose = (event) => {
            console.log('WebSocket connection closed:', event.code, event.reason);
            this.notifyStatus('Disconnected from server');
            this.clearConnectionTimeout();
            this.stopPingInterval();
            
            // Attempt to reconnect if the closure wasn't clean
            if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                const delay = Math.min(
                    this.RECONNECT_DELAY * Math.pow(2, this.reconnectAttempts - 1),
                    this.MAX_RECONNECT_DELAY
                );
                console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts}) after ${delay}ms...`);
                this.reconnectTimeout = setTimeout(() => this.connect(), delay);
            }
        };
    }

    disconnect() {
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }

        this.clearConnectionTimeout();
        this.stopPingInterval();

        if (this.socket) {
            try {
                this.socket.close(1000, 'Client disconnecting');
            } catch (error) {
                console.error('Error closing WebSocket connection:', error);
            } finally {
                this.socket = null;
                this.connectionState = 'disconnected';
            }
        }
    }

    async sendMessage(message: Message) {
        if (!message) {
            console.error('Attempted to send null or undefined message');
            return;
        }

        // If not connected, try to connect first
        if (this.connectionState !== 'connected' || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
            console.log('Connection not ready, queueing message');
            this.messageQueue.push(message);
            await this.connect();
            return;
        }

        try {
            const payload: WebSocketMessage = {
                type: 'message',
                payload: message
            };
            this.socket.send(JSON.stringify(payload));
        } catch (error) {
            console.error('Error sending message:', error);
            this.errorHandlers.forEach(handler => handler('Failed to send message'));
            
            // If send fails, try to reconnect
            this.connectionState = 'disconnected';
            await this.connect();
        }
    }

    onMessage(handler: (message: Message) => void) {
        if (typeof handler === 'function') {
            this.messageHandlers.push(handler);
        } else {
            console.warn('Invalid message handler provided');
        }
    }

    onError(handler: (error: string) => void) {
        if (typeof handler === 'function') {
            this.errorHandlers.push(handler);
        } else {
            console.warn('Invalid error handler provided');
        }
    }

    onStatus(handler: (status: string) => void) {
        if (typeof handler === 'function') {
            this.statusHandlers.push(handler);
        } else {
            console.warn('Invalid status handler provided');
        }
    }

    private notifyStatus(status: string) {
        if (status) {
            this.statusHandlers.forEach(handler => handler(status));
        }
    }
} 