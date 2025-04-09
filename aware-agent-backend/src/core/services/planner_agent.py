import logging
from typing import Dict, Any, Optional

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from src.common.agent_types import AgentType, AgentResponse
from .semantic_abstraction import ContextNode
from .base_agent import BaseAgent
from src.common.exceptions import AgentError

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    def __init__(
        self,
        name: str = "planner",
        description: str = "Plans and organizes tasks and solutions",
        agent_type: AgentType = AgentType.PLANNER,
        llm: Any = None,
    ):
        super().__init__(name, description, agent_type, llm=llm)
        self.initialized = False
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.template = """
        You are a planning agent that helps organize tasks and solutions. Your role is to analyze inputs and create detailed, actionable plans.
        
        Previous conversation:
        {history}
        
        Current input:
        {input}
        
        Please provide a comprehensive plan with the following structure:

        1. Objective Analysis:
           - Main goals and objectives
           - Key constraints and requirements
           - Success criteria

        2. Task Breakdown:
           - High-level tasks
           - Subtasks and dependencies
           - Priority levels and timeline estimates

        3. Resource Requirements:
           - Required skills and expertise
           - Tools and technologies needed
           - External dependencies

        4. Risk Assessment:
           - Potential challenges
           - Mitigation strategies
           - Contingency plans

        5. Success Metrics:
           - Key performance indicators
           - Quality assurance measures
           - Validation criteria

        6. Next Steps:
           - Immediate actions
           - Key decision points
           - Review and adjustment points

        Please analyze the input and provide a structured plan following this format:
        """

    async def initialize_chain(self, template: Optional[str] = None, llm: Optional[Any] = None) -> LLMChain:
        """Initialize the LLM chain with the given template."""
        try:
            if template:
                self.template = template

            if llm:
                self.llm = llm

            if not self.llm:
                raise AgentError("No LLM provided for planning chain")

            prompt = PromptTemplate(
                input_variables=["history", "input"],
                template=self.template,
            )

            self.chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                memory=self.memory,
                verbose=True
            )

            self.initialized = True
            logger.info("Planning chain initialized successfully")
            return self.chain
        except Exception as e:
            logger.error(f"Failed to initialize planning chain: {str(e)}")
            raise AgentError(f"Failed to initialize planning chain: {str(e)}")

    async def process(self, context: ContextNode, input_text: str) -> AgentResponse:
        """Process the input text and return a response."""
        if not self.initialized:
            raise AgentError("Planning agent not initialized")

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
            logger.error(f"Error in PlannerAgent: {str(e)}")
            return AgentResponse(
                response={"error": str(e)},
                confidence=0.0,
                next_steps=[],
                agent_type=self.agent_type,
                status="error",
            )

    async def reset(self):
        """Reset the agent's state."""
        try:
            self.memory.clear()
            await super().cleanup()
        except Exception as e:
            logger.error(f"Error resetting planner agent: {str(e)}")
            raise AgentError(f"Error resetting planner agent: {str(e)}") 