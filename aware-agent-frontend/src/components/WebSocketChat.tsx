import React, { useState, useEffect, useCallback } from 'react';
import { WebSocketServiceImpl } from '../services/WebSocketService';

export const WebSocketChat: React.FC = () => {
    const [message, setMessage] = useState('');
    const [messages, setMessages] = useState<string[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [ws] = useState(() => new WebSocketServiceImpl('ws://localhost:8000/ws'));

    useEffect(() => {
        ws.connect();
        setIsConnected(ws.isConnected());

        ws.onMessage((message: string) => {
            setMessages((prev) => [...prev, message]);
        });

        ws.onError((error: Error) => {
            setError(error.message);
        });

        ws.onClose(() => {
            setIsConnected(false);
        });

        return () => {
            ws.disconnect();
        };
    }, [ws]);

    const handleSendMessage = useCallback(() => {
        if (message.trim()) {
            ws.sendMessage(message);
            setMessage('');
        }
    }, [message, ws]);

    return (
        <div className="chat-container">
            <div className="status-bar">
                {isConnected ? (
                    <span className="status connected">Connected</span>
                ) : (
                    <span className="status disconnected">Disconnected</span>
                )}
                {error && <span className="error">Error: {error}</span>}
            </div>
            <div className="messages">
                {messages.map((msg, index) => (
                    <div key={index} className="message">
                        {msg}
                    </div>
                ))}
            </div>
            <div className="input-container">
                <input
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Type your message..."
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                />
                <button onClick={handleSendMessage}>Send</button>
            </div>
            <style jsx>{`
                .chat-container {
                    display: flex;
                    flex-direction: column;
                    height: 100%;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }

                .status-bar {
                    margin-bottom: 20px;
                }

                .status {
                    padding: 5px 10px;
                    border-radius: 4px;
                    font-size: 14px;
                }

                .connected {
                    background-color: #4caf50;
                    color: white;
                }

                .disconnected {
                    background-color: #f44336;
                    color: white;
                }

                .error {
                    margin-left: 10px;
                    color: #f44336;
                }

                .messages {
                    flex: 1;
                    overflow-y: auto;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    padding: 10px;
                    margin-bottom: 20px;
                }

                .message {
                    margin-bottom: 10px;
                    padding: 10px;
                    background-color: #f5f5f5;
                    border-radius: 4px;
                }

                .input-container {
                    display: flex;
                    gap: 10px;
                }

                input {
                    flex: 1;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    font-size: 16px;
                }

                button {
                    padding: 10px 20px;
                    background-color: #2196f3;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 16px;
                }

                button:hover {
                    background-color: #1976d2;
                }
            `}</style>
        </div>
    );
}; 