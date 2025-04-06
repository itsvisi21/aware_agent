interface Window {
    ethereum?: {
        isMetaMask?: boolean;
        request: (args: { method: string; params?: any[] }) => Promise<any>;
        on: (event: string, handler: (params: any) => void) => void;
        removeListener: (event: string, handler: (params: any) => void) => void;
    };
} 