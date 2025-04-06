import { WebSocketService } from '../services/WebSocketService';
import * as matchers from '@testing-library/jest-dom/matchers';

declare global {
    namespace jest {
        interface Matchers<T> extends matchers.TestingLibraryMatchers<T, void> {
            toBe(expected: T): void;
            toHaveBeenCalled(): void;
            toHaveBeenCalledWith(...args: any[]): void;
            rejects: {
                toThrow(error?: string | Error | RegExp): Promise<void>;
            };
        }
    }
    const describe: (name: string, fn: () => void) => void;
    const test: (name: string, fn: () => void) => void;
    const expect: jest.Expect;
}

export {}; 