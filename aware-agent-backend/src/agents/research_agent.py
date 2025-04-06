from typing import Dict, Any
from .base_agent import BaseAgent

class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Researcher",
            role="Research and Analysis"
        )
        self.research_topics: Dict[str, Any] = {}
        self.sources: list[str] = []

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a research-related message and return a response.
        """
        content = message.get('content', '')
        self.add_to_history(message)

        # Update research topics based on message content
        self._update_research_topics(content)

        # Generate response based on current context
        response = await self._generate_research_response(content)

        # Update state with new information
        self.update_state({
            'last_topic': content,
            'sources_used': self.sources
        })

        return {
            'id': message.get('id', '') + '_response',
            'content': response,
            'sender': 'agent',
            'timestamp': message.get('timestamp', 0),
            'metadata': {
                'agent': self.name,
                'role': self.role,
                'mode': 'research',
                'topics': list(self.research_topics.keys()),
                'sources': self.sources
            }
        }

    def _update_research_topics(self, content: str) -> None:
        """
        Update the research topics based on message content.
        """
        # Simple keyword extraction for demonstration
        keywords = content.lower().split()
        for keyword in keywords:
            if keyword not in self.research_topics:
                self.research_topics[keyword] = {
                    'count': 1,
                    'last_mentioned': content
                }
            else:
                self.research_topics[keyword]['count'] += 1
                self.research_topics[keyword]['last_mentioned'] = content

    async def _generate_research_response(self, content: str) -> str:
        """
        Generate a research-focused response based on the message content.
        """
        context = self.get_context()
        topics = list(self.research_topics.keys())
        
        # Simple response generation for demonstration
        response = f"Based on your research request about '{content}', "
        if topics:
            response += f"I've been tracking topics including: {', '.join(topics)}. "
        response += "Would you like me to dive deeper into any specific aspect?"
        
        return response 