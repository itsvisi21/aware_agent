import React, { useState, useEffect, useRef } from 'react';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useConversation } from '../contexts/ConversationContext';
import { Message } from '../types/chat';

export const ChatInterface: React.FC = () => {
    const { currentConversation, addMessage } = useConversation();
    const { sendMessage, isConnected, error, status } = useWebSocket();
    const [inputText, setInputText] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);
    
    useEffect(() => {
        scrollToBottom();
    }, [currentConversation?.messages]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const handleSend = async () => {
        if (!inputText.trim() || !isConnected) return;

        try {
            await sendMessage(inputText);
            setInputText('');
        } catch (error) {
            console.error('Failed to send message:', error);
        }
    };

    return (
        <div className="chat-interface">
            <div className="chat-messages">
                {currentConversation?.messages.map((message) => (
                    <div key={message.id} className={`message ${message.sender}`}>
                        <div className="message-content">{message.content}</div>
                        <div className="message-timestamp">
                            {new Date(message.timestamp).toLocaleTimeString()}
                        </div>
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>
            
            <div className="chat-status">
                {error && <div className="error">{error}</div>}
                <div className="status">{status}</div>
            </div>

            <div className="chat-input">
                <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                    placeholder={isConnected ? "Type a message..." : "Connecting..."}
                    disabled={!isConnected}
                />
                <button 
                    onClick={handleSend}
                    disabled={!isConnected || !inputText.trim()}
                >
                    Send
                </button>
            </div>
        </div>
    );
}; 