from typing import Optional

from src.common.agent_types import AgentType, AgentResponse
from src.core.services.semantic_abstraction import ContextNode
from src.core.services.base_agent import BaseAgent

class ValidationAgent(BaseAgent):
    """Agent responsible for validating and verifying outputs."""
    
    def __init__(self, llm=None):
        super().__init__(
            name="validation",
            description="Validates and verifies outputs",
            agent_type=AgentType.VALIDATION,
            llm=llm
        )
        self.template = """
        As a validation agent, analyze the following context and input:
        Context: {context}
        Input: {input}
        
        Provide:
        1. Validation criteria
        2. Verification results
        3. Error analysis
        4. Quality assessment
        5. Recommendations
        """

    async def process(self, context: Optional[ContextNode], input_text: str) -> AgentResponse:
        """Process input text and return a validation response."""
        response = await self.chain.arun(
            context=context.visualize_context_tree() if context else "",
            input=input_text
        )

        return AgentResponse(
            content=response,
            reasoning="Generated based on semantic validation requirements",
            confidence=0.85,
            next_steps=["execution", "feedback"],
            agent_type=self.agent_type
        ) 