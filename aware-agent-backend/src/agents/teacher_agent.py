from typing import Dict, Any, List
from .base_agent import BaseAgent

class TeacherAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Teacher",
            role="Explanation and Education"
        )
        self.knowledge_base: Dict[str, Any] = {}
        self.learning_paths: Dict[str, List[str]] = {}
        self.concepts_covered: List[str] = []

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an education-related message and return a response.
        """
        content = message.get('content', '')
        self.add_to_history(message)

        # Update knowledge base and learning paths
        self._update_knowledge(content)

        # Generate educational response
        response = await self._generate_educational_response(content)

        # Update state with new information
        self.update_state({
            'last_topic': content,
            'concepts_covered': self.concepts_covered
        })

        return {
            'id': message.get('id', '') + '_response',
            'content': response,
            'sender': 'agent',
            'timestamp': message.get('timestamp', 0),
            'metadata': {
                'agent': self.name,
                'role': self.role,
                'mode': 'teach',
                'knowledge_base': self.knowledge_base,
                'learning_paths': self.learning_paths
            }
        }

    def _update_knowledge(self, content: str) -> None:
        """
        Update knowledge base and learning paths based on message content.
        """
        # Simple concept extraction for demonstration
        if 'explain' in content.lower() or 'teach' in content.lower():
            concept = content.split('explain')[-1].split('teach')[-1].strip()
            if concept and concept not in self.knowledge_base:
                self.knowledge_base[concept] = {
                    'times_explained': 1,
                    'last_explanation': content
                }
                self.concepts_covered.append(concept)

        # Track learning paths
        if 'path' in content.lower() or 'sequence' in content.lower():
            path_name = content.split('path')[-1].split('sequence')[-1].strip()
            if path_name and path_name not in self.learning_paths:
                self.learning_paths[path_name] = []

    async def _generate_educational_response(self, content: str) -> str:
        """
        Generate an education-focused response based on the message content.
        """
        context = self.get_context()
        
        # Simple response generation for demonstration
        response = f"Based on your request to learn about '{content}', "
        if self.concepts_covered:
            response += f"I've covered concepts including: {', '.join(self.concepts_covered)}. "
        if self.learning_paths:
            response += f"Available learning paths: {', '.join(self.learning_paths.keys())}. "
        response += "Would you like me to explain any specific concept in more detail?"
        
        return response 