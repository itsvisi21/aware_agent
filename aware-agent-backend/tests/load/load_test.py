import asyncio
import time
import random
from typing import List, Dict
import websockets
import json
from deploy.monitoring import MonitoringService
from deploy.config import DeploymentConfig

class LoadTester:
    def __init__(self, num_clients: int = 100, messages_per_client: int = 1000):
        self.num_clients = num_clients
        self.messages_per_client = messages_per_client
        self.config = DeploymentConfig()
        self.monitoring = MonitoringService(self.config)
        self.results: List[Dict] = []

    async def simulate_client(self, client_id: str, websocket_url: str):
        try:
            async with websockets.connect(websocket_url) as websocket:
                # Send messages
                for i in range(self.messages_per_client):
                    message = {
                        'type': 'message',
                        'content': f'Load test message {i} from {client_id}',
                        'sender': f'client_{client_id}',
                        'agent': random.choice(['research', 'builder', 'teacher', 'collaborator'])
                    }
                    
                    start_time = time.time()
                    await websocket.send(json.dumps(message))
                    response = await websocket.recv()
                    end_time = time.time()
                    
                    self.results.append({
                        'client_id': client_id,
                        'message_id': i,
                        'response_time': end_time - start_time,
                        'success': True
                    })
                    
        except Exception as e:
            print(f"Client {client_id} error: {str(e)}")
            self.results.append({
                'client_id': client_id,
                'message_id': i,
                'error': str(e),
                'success': False
            })

    async def run_test(self, websocket_url: str):
        start_time = time.time()
        
        # Create client tasks
        tasks = [
            self.simulate_client(str(i), websocket_url)
            for i in range(self.num_clients)
        ]
        
        # Run all clients concurrently
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate statistics
        successful_messages = [r for r in self.results if r['success']]
        failed_messages = [r for r in self.results if not r['success']]
        
        response_times = [r['response_time'] for r in successful_messages]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        messages_per_second = len(successful_messages) / total_time
        
        # Print results
        print(f"\nLoad Test Results:")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Total Messages: {len(self.results)}")
        print(f"Successful Messages: {len(successful_messages)}")
        print(f"Failed Messages: {len(failed_messages)}")
        print(f"Average Response Time: {avg_response_time:.4f} seconds")
        print(f"Max Response Time: {max_response_time:.4f} seconds")
        print(f"Messages per Second: {messages_per_second:.2f}")
        
        # Save results to file
        with open('load_test_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'total_time': total_time,
                    'total_messages': len(self.results),
                    'successful_messages': len(successful_messages),
                    'failed_messages': len(failed_messages),
                    'avg_response_time': avg_response_time,
                    'max_response_time': max_response_time,
                    'messages_per_second': messages_per_second
                },
                'detailed_results': self.results
            }, f, indent=2)

async def main():
    # Configuration
    websocket_url = "ws://localhost:8000/ws"
    num_clients = 100
    messages_per_client = 1000
    
    # Run load test
    tester = LoadTester(num_clients, messages_per_client)
    await tester.run_test(websocket_url)

if __name__ == "__main__":
    asyncio.run(main()) 