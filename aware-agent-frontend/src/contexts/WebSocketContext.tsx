import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { WebSocketService } from '../services/websocket';
import { Message } from '../types/chat';

interface WebSocketContextType {
    messages: Message[];
    status: string;
    error: string | null;
    sendMessage: (content: string) => void;
    isConnected: boolean;
    reconnect: () => void;
    isWeb3Context: boolean;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

export const useWebSocket = () => {
    const context = useContext(WebSocketContext);
    if (!context) {
        throw new Error('useWebSocket must be used within a WebSocketProvider');
    }
    return context;
};

interface WebSocketProviderProps {
    children: React.ReactNode;
    url: string;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children, url }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [status, setStatus] = useState<string>('Disconnected');
    const [error, setError] = useState<string | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [isWeb3Context, setIsWeb3Context] = useState(false);
    const [wsService] = useState(() => new WebSocketService(url));

    useEffect(() => {
        // Check if we're in a Web3 context
        const checkWeb3Context = () => {
            const isWeb3 = typeof window !== 'undefined' && 
                (window.location.protocol === 'chrome-extension:' || 
                 window.ethereum !== undefined);
            setIsWeb3Context(isWeb3);
        };

        checkWeb3Context();

        // Set up WebSocket handlers
        wsService.onMessage(handleMessage);
        wsService.onError(handleError);
        wsService.onStatus(handleStatus);

        // Connect to WebSocket
        const connect = async () => {
            try {
                await wsService.connect();
            } catch (error) {
                console.error('Failed to connect to WebSocket:', error);
                setError('Failed to connect to WebSocket');
            }
        };

        connect();

        return () => {
            wsService.disconnect();
        };
    }, [wsService]);

    const handleMessage = useCallback((message: Message) => {
        setMessages(prev => [...prev, message]);
    }, []);

    const handleError = useCallback((error: string) => {
        console.error('WebSocket error:', error);
        setError(error);
        setIsConnected(false);
    }, []);

    const handleStatus = useCallback((status: string) => {
        setStatus(status);
        setIsConnected(status === 'Connected to server');
    }, []);

    const sendMessage = useCallback(async (content: string) => {
        if (!content.trim()) return;

        const message: Message = {
            id: Date.now().toString(),
            content,
            timestamp: new Date().toISOString(),
            sender: 'user'
        };

        try {
            await wsService.sendMessage(message);
        } catch (error) {
            console.error('Failed to send message:', error);
            setError('Failed to send message');
        }
    }, [wsService]);

    const reconnect = useCallback(async () => {
        try {
            await wsService.connect();
        } catch (error) {
            console.error('Failed to reconnect:', error);
            setError('Failed to reconnect');
        }
    }, [wsService]);

    const value = {
        messages,
        status,
        error,
        sendMessage,
        isConnected,
        reconnect,
        isWeb3Context
    };

    return (
        <WebSocketContext.Provider value={value}>
            {children}
        </WebSocketContext.Provider>
    );
}; 