import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { ConversationService, Conversation } from '../services/conversation';
import { Message } from '../types/chat';

interface ConversationContextType {
    currentConversation: Conversation | undefined;
    conversations: Conversation[];
    createConversation: (title: string) => Conversation;
    forkConversation: (fromId: string, title: string) => Conversation;
    addMessage: (message: Message) => void;
    setCurrentConversation: (id: string) => void;
    exportConversation: (id: string) => string;
    importConversation: (data: string) => Conversation;
}

const ConversationContext = createContext<ConversationContextType | null>(null);

export const useConversation = () => {
    const context = useContext(ConversationContext);
    if (!context) {
        throw new Error('useConversation must be used within a ConversationProvider');
    }
    return context;
};

interface ConversationProviderProps {
    children: React.ReactNode;
}

export const ConversationProvider: React.FC<ConversationProviderProps> = ({ children }) => {
    const [conversationService] = useState(() => new ConversationService());
    const [conversations, setConversations] = useState<Conversation[]>([]);
    const [isInitialized, setIsInitialized] = useState(false);

    const updateConversations = useCallback(() => {
        setConversations(conversationService.getConversationTree());
    }, [conversationService]);

    useEffect(() => {
        // Initialize with a default conversation if none exists
        if (!isInitialized) {
            const current = conversationService.getCurrentConversation();
            if (!current) {
                conversationService.createConversation('New Conversation');
                updateConversations();
            }
            setIsInitialized(true);
        }
    }, [conversationService, updateConversations, isInitialized]);

    const createConversation = useCallback((title: string) => {
        const conversation = conversationService.createConversation(title);
        updateConversations();
        return conversation;
    }, [conversationService, updateConversations]);

    const forkConversation = useCallback((fromId: string, title: string) => {
        const conversation = conversationService.forkConversation(fromId, title);
        updateConversations();
        return conversation;
    }, [conversationService, updateConversations]);

    const addMessage = useCallback((message: Message) => {
        const current = conversationService.getCurrentConversation();
        if (current) {
            conversationService.addMessage(current.id, message);
            updateConversations();
        }
    }, [conversationService, updateConversations]);

    const setCurrentConversation = useCallback((id: string) => {
        conversationService.setCurrentConversation(id);
        updateConversations();
    }, [conversationService, updateConversations]);

    const exportConversation = useCallback((id: string) => {
        return conversationService.exportConversation(id);
    }, [conversationService]);

    const importConversation = useCallback((data: string) => {
        const conversation = conversationService.importConversation(data);
        updateConversations();
        return conversation;
    }, [conversationService, updateConversations]);

    return (
        <ConversationContext.Provider
            value={{
                currentConversation: conversationService.getCurrentConversation(),
                conversations,
                createConversation,
                forkConversation,
                addMessage,
                setCurrentConversation,
                exportConversation,
                importConversation,
            }}
        >
            {children}
        </ConversationContext.Provider>
    );
}; 