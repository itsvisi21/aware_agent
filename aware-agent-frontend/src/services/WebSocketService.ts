export interface WebSocketService {
    connect(): void;
    disconnect(): void;
    sendMessage(message: string): void;
    onMessage(callback: (message: string) => void): void;
    onError(callback: (error: Error) => void): void;
    onClose(callback: () => void): void;
    onOpen(callback: () => void): void;
    isConnected(): boolean;
}

class RobustWebSocket {
    private ws: WebSocket | null = null;
    private messageCallbacks: ((message: string) => void)[] = [];
    private errorCallbacks: ((error: Error) => void)[] = [];
    private closeCallbacks: (() => void)[] = [];
    private openCallbacks: (() => void)[] = [];
    private isConnected: boolean = false;
    private messageQueue: string[] = [];
    private reconnectAttempts: number = 0;
    private maxReconnectAttempts: number = 3;
    private reconnectDelay: number = 1000;
    private connectionTimeout: NodeJS.Timeout | null = null;
    private isConnecting: boolean = false;
    private pingInterval: NodeJS.Timeout | null = null;
    private lastPingTime: number = 0;
    private readonly PING_INTERVAL = 30000; // 30 seconds
    private readonly PING_TIMEOUT = 10000; // 10 seconds
    private readonly CONNECTION_TIMEOUT = 5000; // 5 seconds

    constructor(private url: string) {}

    private setupWebSocket(): void {
        if (this.isConnecting) {
            console.warn('WebSocket connection already in progress');
            return;
        }

        try {
            this.isConnecting = true;

            // Create a unique URL to prevent caching and interference
            const uniqueUrl = new URL(this.url);
            uniqueUrl.searchParams.set('_t', Date.now().toString());
            uniqueUrl.searchParams.set('_r', Math.random().toString(36).substring(2));

            this.ws = new WebSocket(uniqueUrl.toString());

            // Set up connection timeout
            this.connectionTimeout = setTimeout(() => {
                if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
                    console.warn('WebSocket connection timeout');
                    this.handleError(new Error('Connection timeout'));
                    this.ws.close();
                }
            }, this.CONNECTION_TIMEOUT);

            this.ws.onopen = () => {
                if (this.connectionTimeout) {
                    clearTimeout(this.connectionTimeout);
                    this.connectionTimeout = null;
                }
                this.isConnected = true;
                this.isConnecting = false;
                this.reconnectAttempts = 0;
                this.lastPingTime = Date.now();
                this.startPingInterval();
                this.openCallbacks.forEach(callback => {
                    try {
                        callback();
                    } catch (error) {
                        console.error('Error in onOpen callback:', error);
                    }
                });
                // Send any queued messages
                while (this.messageQueue.length > 0) {
                    const message = this.messageQueue.shift();
                    if (message) this.send(message);
                }
            };

            this.ws.onmessage = (event) => {
                if (event && event.data) {
                    this.lastPingTime = Date.now();
                    this.messageCallbacks.forEach(callback => {
                        try {
                            callback(event.data);
                        } catch (error) {
                            console.error('Error in onMessage callback:', error);
                        }
                    });
                }
            };

            this.ws.onerror = (event) => {
                console.error('WebSocket error:', event);
                this.handleError(new Error('WebSocket error occurred'));
            };

            this.ws.onclose = () => {
                this.cleanup();
                this.closeCallbacks.forEach(callback => {
                    try {
                        callback();
                    } catch (error) {
                        console.error('Error in onClose callback:', error);
                    }
                });

                // Attempt to reconnect if we haven't exceeded max attempts
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    this.reconnectDelay *= 2; // Exponential backoff
                    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts}) after ${this.reconnectDelay}ms`);
                    setTimeout(() => this.connect(), this.reconnectDelay);
                }
            };
        } catch (error) {
            this.isConnecting = false;
            console.error('Failed to setup WebSocket:', error);
            this.handleError(error as Error);
        }
    }

    private startPingInterval(): void {
        this.stopPingInterval();
        this.pingInterval = setInterval(() => {
            if (this.isConnected && this.ws) {
                const now = Date.now();
                if (now - this.lastPingTime > this.PING_TIMEOUT) {
                    console.warn('WebSocket ping timeout');
                    this.handleError(new Error('Ping timeout'));
                    this.ws.close();
                } else {
                    try {
                        this.ws.send('ping');
                    } catch (error) {
                        console.error('Error sending ping:', error);
                        this.handleError(error as Error);
                    }
                }
            }
        }, this.PING_INTERVAL);
    }

    private stopPingInterval(): void {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    private cleanup(): void {
        if (this.connectionTimeout) {
            clearTimeout(this.connectionTimeout);
            this.connectionTimeout = null;
        }
        this.stopPingInterval();
        this.isConnected = false;
        this.isConnecting = false;
    }

    private handleError(error: Error): void {
        this.cleanup();
        this.errorCallbacks.forEach(callback => {
            try {
                callback(error);
            } catch (error) {
                console.error('Error in error callback:', error);
            }
        });
    }

    connect(): void {
        if (this.isConnected || this.isConnecting) {
            console.warn('WebSocket already connected or connecting');
            return;
        }

        try {
            // Disconnect any existing connection
            this.disconnect();

            this.setupWebSocket();
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.handleError(error as Error);
        }
    }

    disconnect(): void {
        this.cleanup();
        if (this.ws) {
            try {
                this.ws.close();
            } catch (error) {
                console.error('Error while disconnecting WebSocket:', error);
            } finally {
                this.ws = null;
            }
        }
    }

    send(message: string): void {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            try {
                this.ws.send(message);
            } catch (error) {
                console.error('Error while sending message:', error);
                this.handleError(error as Error);
            }
        } else {
            this.messageQueue.push(message);
        }
    }

    onMessage(callback: (message: string) => void): void {
        this.messageCallbacks.push(callback);
    }

    onError(callback: (error: Error) => void): void {
        this.errorCallbacks.push(callback);
    }

    onClose(callback: () => void): void {
        this.closeCallbacks.push(callback);
    }

    onOpen(callback: () => void): void {
        this.openCallbacks.push(callback);
    }

    getConnectionStatus(): boolean {
        return this.isConnected;
    }
}

export class WebSocketServiceImpl implements WebSocketService {
    private robustWs: RobustWebSocket | null = null;
    private messageCallbacks: ((message: string) => void)[] = [];
    private errorCallbacks: ((error: Error) => void)[] = [];
    private closeCallbacks: (() => void)[] = [];
    private openCallbacks: (() => void)[] = [];
    private isWeb3Context: boolean = false;
    private connectionTimeout: NodeJS.Timeout | null = null;
    private reconnectAttempts: number = 0;
    private maxReconnectAttempts: number = 3;
    private reconnectDelay: number = 1000;
    private isConnecting: boolean = false;

    constructor(private url: string) {
        // Check if we're in a Web3 context
        this.isWeb3Context = typeof window !== 'undefined' && 
            (window.location.protocol === 'chrome-extension:' || 
             (window as any).ethereum !== undefined);
    }

    private setupWebSocket(): void {
        if (this.isConnecting) {
            console.warn('WebSocket connection already in progress');
            return;
        }

        try {
            this.isConnecting = true;

            // Create a new robust WebSocket instance
            this.robustWs = new RobustWebSocket(this.url);

            // Set up callbacks
            this.robustWs.onMessage((message) => {
                this.messageCallbacks.forEach(callback => {
                    try {
                        callback(message);
                    } catch (error) {
                        console.error('Error in message callback:', error);
                    }
                });
            });

            this.robustWs.onError((error) => {
                this.errorCallbacks.forEach(callback => {
                    try {
                        callback(error);
                    } catch (error) {
                        console.error('Error in error callback:', error);
                    }
                });
            });

            this.robustWs.onClose(() => {
                this.closeCallbacks.forEach(callback => {
                    try {
                        callback();
                    } catch (error) {
                        console.error('Error in close callback:', error);
                    }
                });

                // Attempt to reconnect if we haven't exceeded max attempts
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    this.reconnectDelay *= 2; // Exponential backoff
                    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts}) after ${this.reconnectDelay}ms`);
                    setTimeout(() => this.connect(), this.reconnectDelay);
                }
            });

            this.robustWs.onOpen(() => {
                this.reconnectAttempts = 0;
                this.isConnecting = false;
                this.openCallbacks.forEach(callback => {
                    try {
                        callback();
                    } catch (error) {
                        console.error('Error in open callback:', error);
                    }
                });
            });

            // Connect the WebSocket
            this.robustWs.connect();
        } catch (error) {
            this.isConnecting = false;
            console.error('Failed to setup WebSocket:', error);
            this.errorCallbacks.forEach(callback => {
                try {
                    callback(error as Error);
                } catch (error) {
                    console.error('Error in error callback:', error);
                }
            });
        }
    }

    connect(): void {
        if (this.isConnected() || this.isConnecting) {
            console.warn('WebSocket already connected or connecting');
            return;
        }

        try {
            // Disconnect any existing connection
            this.disconnect();

            // Clear any existing timeout
            if (this.connectionTimeout) {
                clearTimeout(this.connectionTimeout);
                this.connectionTimeout = null;
            }

            // Add a delay before connecting in Web3 contexts
            const connectWithDelay = () => {
                this.setupWebSocket();
            };

            // Add a delay in Web3 contexts to allow extensions to initialize
            if (this.isWeb3Context) {
                console.log('Web3 context detected, adding connection delay');
                this.connectionTimeout = setTimeout(connectWithDelay, 2000);
            } else {
                connectWithDelay();
            }
        } catch (error) {
            this.isConnecting = false;
            console.error('Failed to setup WebSocket connection:', error);
            this.errorCallbacks.forEach(callback => {
                try {
                    callback(error as Error);
                } catch (error) {
                    console.error('Error in error callback:', error);
                }
            });
        }
    }

    disconnect(): void {
        if (this.connectionTimeout) {
            clearTimeout(this.connectionTimeout);
            this.connectionTimeout = null;
        }

        if (this.robustWs) {
            try {
                this.robustWs.disconnect();
            } catch (error) {
                console.error('Error while disconnecting WebSocket:', error);
            } finally {
                this.robustWs = null;
                this.isConnecting = false;
            }
        }
    }

    sendMessage(message: string): void {
        if (!message) {
            console.warn('Attempted to send empty message');
            return;
        }

        if (this.robustWs && this.robustWs.getConnectionStatus()) {
            try {
                this.robustWs.send(message);
            } catch (error) {
                console.error('Error while sending message:', error);
                this.errorCallbacks.forEach(callback => {
                    try {
                        callback(error as Error);
                    } catch (error) {
                        console.error('Error in error callback:', error);
                    }
                });
            }
        } else {
            console.warn('Attempted to send message while WebSocket is not open');
        }
    }

    onMessage(callback: (message: string) => void): void {
        this.messageCallbacks.push(callback);
    }

    onError(callback: (error: Error) => void): void {
        this.errorCallbacks.push(callback);
    }

    onClose(callback: () => void): void {
        this.closeCallbacks.push(callback);
    }

    onOpen(callback: () => void): void {
        this.openCallbacks.push(callback);
    }

    isConnected(): boolean {
        return this.robustWs !== null && this.robustWs.getConnectionStatus();
    }
} 