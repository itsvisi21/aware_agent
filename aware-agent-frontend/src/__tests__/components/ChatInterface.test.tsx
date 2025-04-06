import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import ChatInterface from '../../components/ChatInterface';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { vi, describe, it, expect, beforeEach } from 'vitest';

// Mock the API calls
vi.mock('../../lib/api', () => ({
  sendMessage: vi.fn(() => Promise.resolve({ response: 'Test response' })),
  getConversationHistory: vi.fn(() => Promise.resolve([])),
}));

describe('ChatInterface', () => {
  const queryClient = new QueryClient();
  
  beforeEach(() => {
    // Wrap component with necessary providers
    render(
      <QueryClientProvider client={queryClient}>
        <ChatInterface />
      </QueryClientProvider>
    );
  });

  it('renders the chat interface', () => {
    expect(screen.getByRole('textbox')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
  });

  it('handles user input correctly', () => {
    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: 'Test message' } });
    expect(input).toHaveValue('Test message');
  });

  it('sends message on form submission', async () => {
    const input = screen.getByRole('textbox');
    const sendButton = screen.getByRole('button', { name: /send/i });

    fireEvent.change(input, { target: { value: 'Test message' } });
    fireEvent.click(sendButton);

    await waitFor(() => {
      expect(input).toHaveValue('');
    });
  });

  it('displays error message when API call fails', async () => {
    const { sendMessage } = require('../../lib/api');
    sendMessage.mockRejectedValueOnce(new Error('API Error'));

    const input = screen.getByRole('textbox');
    const sendButton = screen.getByRole('button', { name: /send/i });

    fireEvent.change(input, { target: { value: 'Test message' } });
    fireEvent.click(sendButton);

    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument();
    });
  });

  it('loads conversation history on mount', async () => {
    const { getConversationHistory } = require('../../lib/api');
    const mockHistory = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there!' }
    ];
    getConversationHistory.mockResolvedValueOnce(mockHistory);

    await waitFor(() => {
      expect(screen.getByText('Hello')).toBeInTheDocument();
      expect(screen.getByText('Hi there!')).toBeInTheDocument();
    });
  });

  it('handles markdown content correctly', async () => {
    const { sendMessage } = require('../../lib/api');
    sendMessage.mockResolvedValueOnce({ 
      response: '**Bold** and *italic* text' 
    });

    const input = screen.getByRole('textbox');
    const sendButton = screen.getByRole('button', { name: /send/i });

    fireEvent.change(input, { target: { value: 'Test markdown' } });
    fireEvent.click(sendButton);

    await waitFor(() => {
      expect(screen.getByText('Bold')).toHaveStyle('font-weight: bold');
      expect(screen.getByText('italic')).toHaveStyle('font-style: italic');
    });
  });
}); 