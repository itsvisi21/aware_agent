import asyncio
import websockets
import json
import pytest
from typing import Dict, Any
from deploy.config import DeploymentConfig
from deploy.monitoring import MonitoringService

class SecurityTester:
    def __init__(self):
        self.config = DeploymentConfig()
        self.monitoring = MonitoringService(self.config)
        self.results: Dict[str, Any] = {
            'tests': [],
            'vulnerabilities': [],
            'recommendations': []
        }

    async def test_message_validation(self, websocket_url: str):
        """Test message validation and sanitization"""
        test_name = "Message Validation"
        try:
            async with websockets.connect(websocket_url) as websocket:
                # Test SQL injection attempt
                malicious_message = {
                    'type': 'message',
                    'content': "'; DROP TABLE conversations; --",
                    'sender': 'malicious_user',
                    'agent': 'research'
                }
                await websocket.send(json.dumps(malicious_message))
                response = await websocket.recv()
                response_data = json.loads(response)
                
                if 'error' in response_data:
                    self.results['tests'].append({
                        'name': test_name,
                        'status': 'passed',
                        'details': 'SQL injection attempt blocked'
                    })
                else:
                    self.results['vulnerabilities'].append({
                        'type': 'SQL Injection',
                        'severity': 'high',
                        'description': 'System vulnerable to SQL injection'
                    })

                # Test XSS attempt
                xss_message = {
                    'type': 'message',
                    'content': '<script>alert("XSS")</script>',
                    'sender': 'malicious_user',
                    'agent': 'research'
                }
                await websocket.send(json.dumps(xss_message))
                response = await websocket.recv()
                response_data = json.loads(response)
                
                if '<script>' not in response_data.get('content', ''):
                    self.results['tests'].append({
                        'name': test_name,
                        'status': 'passed',
                        'details': 'XSS attempt blocked'
                    })
                else:
                    self.results['vulnerabilities'].append({
                        'type': 'XSS',
                        'severity': 'high',
                        'description': 'System vulnerable to XSS attacks'
                    })

        except Exception as e:
            self.results['tests'].append({
                'name': test_name,
                'status': 'failed',
                'error': str(e)
            })

    async def test_authentication(self, websocket_url: str):
        """Test authentication and authorization"""
        test_name = "Authentication"
        try:
            # Test without authentication
            async with websockets.connect(websocket_url) as websocket:
                message = {
                    'type': 'message',
                    'content': 'Test message',
                    'sender': 'unauthorized_user',
                    'agent': 'research'
                }
                await websocket.send(json.dumps(message))
                response = await websocket.recv()
                response_data = json.loads(response)
                
                if 'error' in response_data and 'unauthorized' in response_data['error'].lower():
                    self.results['tests'].append({
                        'name': test_name,
                        'status': 'passed',
                        'details': 'Unauthorized access blocked'
                    })
                else:
                    self.results['vulnerabilities'].append({
                        'type': 'Authentication',
                        'severity': 'critical',
                        'description': 'System allows unauthorized access'
                    })

        except Exception as e:
            self.results['tests'].append({
                'name': test_name,
                'status': 'failed',
                'error': str(e)
            })

    async def test_rate_limiting(self, websocket_url: str):
        """Test rate limiting and DoS protection"""
        test_name = "Rate Limiting"
        try:
            async with websockets.connect(websocket_url) as websocket:
                # Send rapid messages
                for i in range(100):
                    message = {
                        'type': 'message',
                        'content': f'Rapid message {i}',
                        'sender': 'test_user',
                        'agent': 'research'
                    }
                    await websocket.send(json.dumps(message))
                    try:
                        response = await websocket.recv()
                        response_data = json.loads(response)
                        if 'error' in response_data and 'rate limit' in response_data['error'].lower():
                            self.results['tests'].append({
                                'name': test_name,
                                'status': 'passed',
                                'details': 'Rate limiting working as expected'
                            })
                            break
                    except websockets.exceptions.ConnectionClosed:
                        self.results['tests'].append({
                            'name': test_name,
                            'status': 'passed',
                            'details': 'Connection closed due to rate limiting'
                        })
                        break

        except Exception as e:
            self.results['tests'].append({
                'name': test_name,
                'status': 'failed',
                'error': str(e)
            })

    async def run_security_tests(self, websocket_url: str):
        """Run all security tests"""
        await self.test_message_validation(websocket_url)
        await self.test_authentication(websocket_url)
        await self.test_rate_limiting(websocket_url)
        
        # Generate recommendations
        if self.results['vulnerabilities']:
            self.results['recommendations'] = [
                {
                    'priority': 'high',
                    'action': 'Implement input validation and sanitization',
                    'reason': 'Prevent SQL injection and XSS attacks'
                },
                {
                    'priority': 'critical',
                    'action': 'Implement proper authentication and authorization',
                    'reason': 'Prevent unauthorized access'
                },
                {
                    'priority': 'medium',
                    'action': 'Implement rate limiting',
                    'reason': 'Prevent DoS attacks'
                }
            ]
        
        # Save results
        with open('security_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return self.results

async def main():
    websocket_url = "ws://localhost:8000/ws"
    tester = SecurityTester()
    results = await tester.run_security_tests(websocket_url)
    
    # Print summary
    print("\nSecurity Test Results:")
    print(f"Tests Run: {len(results['tests'])}")
    print(f"Vulnerabilities Found: {len(results['vulnerabilities'])}")
    print(f"Recommendations: {len(results['recommendations'])}")
    
    if results['vulnerabilities']:
        print("\nVulnerabilities:")
        for vuln in results['vulnerabilities']:
            print(f"- {vuln['type']} ({vuln['severity']}): {vuln['description']}")
    
    if results['recommendations']:
        print("\nRecommendations:")
        for rec in results['recommendations']:
            print(f"- [{rec['priority']}] {rec['action']}: {rec['reason']}")

if __name__ == "__main__":
    asyncio.run(main()) 