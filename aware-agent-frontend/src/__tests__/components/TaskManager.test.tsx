import { render, screen, fireEvent, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { TaskManager } from '../../components/TaskManager';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { getTasks, createTask, updateTask, deleteTask } from '../../lib/api';
import { Task, TaskStatus, TaskPriority } from '../../types/Task';

jest.mock('../../lib/api');

const mockTasks: Task[] = [
    {
        id: '1',
        title: 'Test Task 1',
        description: 'Description 1',
        status: 'pending' as TaskStatus,
        priority: 'high' as TaskPriority,
        createdAt: '2025-04-05T21:56:58.136Z',
        updatedAt: '2025-04-05T21:56:58.136Z'
    },
    {
        id: '2',
        title: 'Test Task 2',
        description: 'Description 2',
        status: 'in_progress' as TaskStatus,
        priority: 'medium' as TaskPriority,
        createdAt: '2025-04-05T21:56:58.136Z',
        updatedAt: '2025-04-05T21:56:58.136Z'
    }
];

const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            retry: false,
        },
    },
});

describe('TaskManager Component', () => {
    beforeEach(() => {
        (getTasks as jest.Mock).mockResolvedValue(mockTasks);
        (createTask as jest.Mock).mockImplementation((task) => Promise.resolve(task));
        (updateTask as jest.Mock).mockImplementation((id, updates) => Promise.resolve({ ...mockTasks.find(t => t.id === id), ...updates }));
        (deleteTask as jest.Mock).mockResolvedValue(true);
    });

    afterEach(() => {
        jest.clearAllMocks();
    });

    it('renders task list', async () => {
        render(
            <QueryClientProvider client={queryClient}>
                <TaskManager />
            </QueryClientProvider>
        );

        expect(await screen.findByText('Test Task 1')).toBeInTheDocument();
        expect(await screen.findByText('Test Task 2')).toBeInTheDocument();
    });

    it('creates new task', async () => {
        render(
            <QueryClientProvider client={queryClient}>
                <TaskManager />
            </QueryClientProvider>
        );

        await act(async () => {
            fireEvent.click(screen.getByText('New Task'));
        });

        const newTask = {
            title: 'New Task',
            description: 'New Description',
            priority: 'medium' as TaskPriority,
            status: 'pending' as TaskStatus
        };

        await act(async () => {
            fireEvent.change(screen.getByLabelText('Title'), { target: { value: newTask.title } });
            fireEvent.change(screen.getByLabelText('Description'), { target: { value: newTask.description } });
            fireEvent.click(screen.getByText('Save'));
        });

        expect(createTask).toHaveBeenCalledWith(expect.objectContaining(newTask));
    });

    it('updates task status', async () => {
        render(
            <QueryClientProvider client={queryClient}>
                <TaskManager />
            </QueryClientProvider>
        );

        await screen.findByText('Test Task 1');

        await act(async () => {
            const taskElement = screen.getByText('Test Task 1').closest('[role="listitem"]');
            const statusButton = taskElement?.querySelector('button[class*="text-gray-500"]');
            if (statusButton) {
                fireEvent.click(statusButton);
                const completeButton = screen.getByRole('button', { name: 'Complete task' });
                fireEvent.click(completeButton);
            }
        });

        expect(updateTask).toHaveBeenCalledWith('1', { status: 'completed' });
    });

    it('deletes task', async () => {
        render(
            <QueryClientProvider client={queryClient}>
                <TaskManager />
            </QueryClientProvider>
        );

        await screen.findByText('Test Task 1');

        await act(async () => {
            const taskElement = screen.getByText('Test Task 1').closest('[role="listitem"]');
            const deleteButton = taskElement?.querySelector('button[aria-label="Delete task"]');
            if (deleteButton) {
                fireEvent.click(deleteButton);
            }
        });

        expect(deleteTask).toHaveBeenCalledWith('1');
    });
}); 