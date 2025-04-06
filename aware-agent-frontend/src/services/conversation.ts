import { Message } from '../types/chat';

export interface Conversation {
    id: string;
    title: string;
    messages: Message[];
    createdAt: number;
    updatedAt: number;
    parentId?: string;
    children: string[];
}

export class ConversationService {
    private conversations: Map<string, Conversation> = new Map();
    private currentConversationId: string | null = null;

    constructor() {
        // Initialize with a default conversation
        this.createConversation('New Conversation');
    }

    createConversation(title: string, parentId?: string): Conversation {
        const conversation: Conversation = {
            id: Date.now().toString(),
            title,
            messages: [],
            createdAt: Date.now(),
            updatedAt: Date.now(),
            parentId,
            children: [],
        };

        this.conversations.set(conversation.id, conversation);

        if (parentId) {
            const parent = this.conversations.get(parentId);
            if (parent) {
                parent.children.push(conversation.id);
                this.conversations.set(parentId, parent);
            }
        }

        this.currentConversationId = conversation.id;
        return conversation;
    }

    forkConversation(fromId: string, title: string): Conversation {
        const sourceConversation = this.conversations.get(fromId);
        if (!sourceConversation) {
            throw new Error('Source conversation not found');
        }

        const forkedConversation = this.createConversation(title, fromId);
        forkedConversation.messages = [...sourceConversation.messages];
        this.conversations.set(forkedConversation.id, forkedConversation);

        return forkedConversation;
    }

    addMessage(conversationId: string, message: Message): void {
        const conversation = this.conversations.get(conversationId);
        if (!conversation) {
            throw new Error('Conversation not found');
        }

        conversation.messages.push(message);
        conversation.updatedAt = Date.now();
        this.conversations.set(conversationId, conversation);
    }

    getConversation(id: string): Conversation | undefined {
        return this.conversations.get(id);
    }

    getCurrentConversation(): Conversation | undefined {
        if (!this.currentConversationId) return undefined;
        return this.conversations.get(this.currentConversationId);
    }

    setCurrentConversation(id: string): void {
        if (!this.conversations.has(id)) {
            throw new Error('Conversation not found');
        }
        this.currentConversationId = id;
    }

    getConversationTree(): Conversation[] {
        const rootConversations: Conversation[] = [];
        
        this.conversations.forEach(conversation => {
            if (!conversation.parentId) {
                rootConversations.push(conversation);
            }
        });

        return rootConversations;
    }

    exportConversation(id: string): string {
        const conversation = this.conversations.get(id);
        if (!conversation) {
            throw new Error('Conversation not found');
        }

        return JSON.stringify(conversation, null, 2);
    }

    importConversation(data: string): Conversation {
        const conversation: Conversation = JSON.parse(data);
        this.conversations.set(conversation.id, conversation);
        return conversation;
    }
} 