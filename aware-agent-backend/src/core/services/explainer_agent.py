import logging
from typing import Dict, Any, Optional

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from src.common.agent_types import AgentType, AgentResponse
from src.core.services.semantic_abstraction import ContextNode
from src.core.services.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ExplainerAgent(BaseAgent):
    def __init__(
        self,
        name: str = "explainer",
        description: str = "Explains and clarifies concepts and solutions",
        agent_type: AgentType = AgentType.EXPLAINER,
    ):
        super().__init__(name, description, agent_type)
        self.chain = None
        self.memory = ConversationBufferMemory()
        self.template = """
        You are an explainer agent that helps clarify concepts and solutions.
        
        Previous conversation:
        {history}
        
        Current input:
        {input}
        
        Please provide a clear and detailed explanation:
        """

    async def initialize_chain(self, template: Optional[str] = None, llm: Optional[Any] = None):
        """Initialize the LLM chain with the given template."""
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template or self.template,
        )
        self.chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=self.memory,
        )
        self.initialized = True
        return self.chain

    async def process(self, context: ContextNode, input_text: str) -> AgentResponse:
        """Process the input text and return a response."""
        if not self.chain:
            raise ValueError("Chain not initialized. Call initialize_chain first.")

        try:
            # Get the response from the chain
            response = await self.chain.arun(
                history=self.memory.buffer,
                input=input_text,
            )

            # Parse the response
            response_dict = {
                "response": response,
                "confidence": 0.9,  # Default confidence
                "next_steps": ["validation", "execution"],
            }

            return AgentResponse(
                response=response_dict,
                confidence=response_dict["confidence"],
                next_steps=response_dict["next_steps"],
                agent_type=self.agent_type,
                status="success",
            )

        except Exception as e:
            logger.error(f"Error in ExplainerAgent: {str(e)}")
            return AgentResponse(
                response={"error": str(e)},
                confidence=0.0,
                next_steps=[],
                agent_type=self.agent_type,
                status="error",
            ) 