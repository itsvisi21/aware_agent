from typing import Optional

from src.common.agent_types import AgentType, AgentResponse
from src.core.services.semantic_abstraction import ContextNode
from src.core.services.base_agent import BaseAgent

class ContextAgent(BaseAgent):
    """Agent responsible for managing and analyzing context."""
    
    def __init__(self, llm=None):
        super().__init__(
            name="context",
            description="Manages and analyzes context",
            agent_type=AgentType.CONTEXT,
            llm=llm
        )
        self.template = """
        As a context agent, analyze the following context and input:
        Context: {context}
        Input: {input}
        
        Provide:
        1. Context analysis
        2. Key insights
        3. Relevant patterns
        4. Contextual relationships
        5. Recommendations
        """

    async def process(self, context: Optional[ContextNode], input_text: str) -> AgentResponse:
        """Process input text and return a context analysis response."""
        response = await self.chain.arun(
            context=context.visualize_context_tree() if context else "",
            input=input_text
        )

        return AgentResponse(
            content=response,
            reasoning="Generated based on semantic context analysis",
            confidence=0.85,
            next_steps=["planning", "validation"],
            agent_type=self.agent_type
        ) 