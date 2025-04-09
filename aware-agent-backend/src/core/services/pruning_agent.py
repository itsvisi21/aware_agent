from typing import Optional
import asyncio

from src.common.agent_types import AgentType, AgentResponse
from src.core.services.semantic_abstraction import ContextNode
from src.core.services.base_agent import BaseAgent

class PruningAgent(BaseAgent):
    """Agent responsible for pruning and optimizing options."""
    
    def __init__(self, llm=None):
        super().__init__(
            name="Pruning Agent",
            description="Agent responsible for pruning and organizing knowledge",
            agent_type="pruning",
            llm=llm
        )
        self.template = """
        You are a pruning agent responsible for organizing and cleaning up knowledge.
        
        History: {history}
        Input: {input}
        
        Please analyze the input and suggest how to organize or prune this information.
        """
        
        # Initialize the chain synchronously
        self.initialized = False

    async def initialize(self):
        """Initialize the agent asynchronously."""
        if not self.initialized:
            await self.initialize_chain(llm=self.llm)
            self.initialized = True

    async def process(self, context: Optional[ContextNode], input_text: str) -> AgentResponse:
        """Process input text and return a pruning response."""
        if not self.initialized:
            await self.initialize()
            
        if not self.chain:
            raise ValueError("Chain not initialized. Please provide an LLM during initialization.")
            
        response = await self.chain.arun(
            history="",  # Empty history since we're not using it
            input=input_text
        )

        return AgentResponse(
            content=response,
            reasoning="Generated based on semantic context and pruning requirements",
            confidence=0.85,
            next_steps=["validation", "execution"],
            agent_type=self.agent_type
        ) 