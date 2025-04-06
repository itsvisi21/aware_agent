from typing import Dict, Any, List
from .base_agent import BaseAgent

class BuilderAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Builder",
            role="Implementation and Development"
        )
        self.project_structure: Dict[str, Any] = {}
        self.code_snippets: List[Dict[str, Any]] = []
        self.dependencies: List[str] = []

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an implementation-related message and return a response.
        """
        content = message.get('content', '')
        self.add_to_history(message)

        # Update project structure and code snippets
        self._update_project_info(content)

        # Generate implementation response
        response = await self._generate_implementation_response(content)

        # Update state with new information
        self.update_state({
            'last_task': content,
            'dependencies': self.dependencies
        })

        return {
            'id': message.get('id', '') + '_response',
            'content': response,
            'sender': 'agent',
            'timestamp': message.get('timestamp', 0),
            'metadata': {
                'agent': self.name,
                'role': self.role,
                'mode': 'build',
                'project_structure': self.project_structure,
                'dependencies': self.dependencies
            }
        }

    def _update_project_info(self, content: str) -> None:
        """
        Update project structure and code snippets based on message content.
        """
        # Simple project structure tracking for demonstration
        if 'file' in content.lower() or 'directory' in content.lower():
            component = content.split()[-1]
            if component not in self.project_structure:
                self.project_structure[component] = {
                    'type': 'file' if 'file' in content.lower() else 'directory',
                    'last_modified': content
                }

        # Track dependencies mentioned in the content
        if 'dependency' in content.lower() or 'package' in content.lower():
            dep = content.split()[-1]
            if dep not in self.dependencies:
                self.dependencies.append(dep)

    async def _generate_implementation_response(self, content: str) -> str:
        """
        Generate an implementation-focused response based on the message content.
        """
        context = self.get_context()
        
        # Simple response generation for demonstration
        response = f"Based on your implementation request about '{content}', "
        if self.project_structure:
            response += f"I'm tracking project components including: {', '.join(self.project_structure.keys())}. "
        if self.dependencies:
            response += f"Required dependencies: {', '.join(self.dependencies)}. "
        response += "Would you like me to help with any specific implementation details?"
        
        return response 