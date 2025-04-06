import Head from 'next/head';
import { ChatInterface } from '../components/ChatInterface';
import { WebSocketProvider } from '../contexts/WebSocketContext';
import { ConversationProvider } from '../contexts/ConversationContext';

export default function Home() {
    return (
        <>
            <Head>
                <title>Aware Agent Chat</title>
                <meta name="description" content="Aware Agent Chat Interface" />
                <link rel="icon" href="/favicon.ico" />
            </Head>
            <WebSocketProvider url={process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws'}>
                <ConversationProvider>
                    <ChatInterface />
                </ConversationProvider>
            </WebSocketProvider>
        </>
    );
} 