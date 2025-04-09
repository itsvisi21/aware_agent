from typing import Optional

from src.common.agent_types import AgentType, AgentResponse
from src.core.services.semantic_abstraction import ContextNode
from src.core.services.base_agent import BaseAgent

class PlanningAgent(BaseAgent):
    """Agent responsible for planning and strategy development."""
    
    def __init__(self):
        super().__init__(
            name="planning",
            description="Develops plans and strategies",
            agent_type=AgentType.PLANNING
        )
        self.template = """
        As a planning agent, analyze the following context and input:
        Context: {context}
        Input: {input}
        
        Provide:
        1. Strategic plan
        2. Action items
        3. Timeline
        4. Resource allocation
        5. Risk assessment
        """

    async def process(self, context: Optional[ContextNode], input_text: str) -> AgentResponse:
        """Process input text and return a planning response."""
        response = await self.chain.arun(
            context=context.visualize_context_tree() if context else "",
            input=input_text
        )

        return AgentResponse(
            content=response,
            reasoning="Generated based on semantic context and planning requirements",
            confidence=0.85,
            next_steps=["validation", "execution"],
            agent_type=self.agent_type
        ) 