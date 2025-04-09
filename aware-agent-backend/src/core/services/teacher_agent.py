import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from src.common.agent_types import AgentType, AgentResponse
from src.core.services.semantic_abstraction import ContextNode
from src.core.services.base_agent import BaseAgent
from src.common.exceptions import AgentError

logger = logging.getLogger(__name__)


class TeacherAgent(BaseAgent):
    """Agent responsible for teaching and explaining concepts."""

    def __init__(self, name: str = None, description: str = None, agent_type: str = None, llm: Any = None):
        """Initialize the TeacherAgent.

        Args:
            name (str, optional): The name of the agent. Defaults to None.
            description (str, optional): The description of the agent. Defaults to None.
            agent_type (str, optional): The type of the agent. Defaults to None.
            llm (Any, optional): The language model to use. Defaults to None.
        """
        super().__init__(name=name, description=description, agent_type=agent_type, llm=llm)
        self.knowledge_base = {}
        self.learning_paths = {}
        self.initialized = False
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.template = """
        You are a teaching assistant responsible for explaining concepts and guiding learning.
        Your task is to:
        1. Analyze the input and identify key concepts
        2. Check prerequisites and foundational knowledge
        3. Structure explanations in a clear, logical manner
        4. Provide examples and analogies
        5. Suggest follow-up topics and exercises
        
        Input: {input}
        """

    async def initialize_chain(self, template: str = None, llm: Any = None) -> LLMChain:
        """Initialize the teaching chain."""
        try:
            if template:
                self.template = template

            if llm:
                self.llm = llm

            if not self.llm:
                raise AgentError("No LLM provided for teaching chain")

            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.template
            )

            self.chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                memory=self.memory,
                verbose=True
            )

            self.initialized = True
            logger.info("Teaching chain initialized successfully")
            return self.chain
        except Exception as e:
            logger.error(f"Failed to initialize teaching chain: {str(e)}")
            raise AgentError(f"Failed to initialize teaching chain: {str(e)}")

    async def process_message(self, message: Union[str, Dict[str, Any]]) -> AgentResponse:
        """Process a message and return a response."""
        if not self.initialized:
            raise AgentError("Teacher agent not initialized")

        try:
            # Extract content from message
            content = message if isinstance(message, str) else message.get('content', '')
            if not content:
                raise AgentError("Message must contain content")

            # Generate teacher response
            response = await self.chain.arun(input=content)

            # Return formatted response
            return AgentResponse(
                content=response,
                reasoning="Explanation generated",
                confidence=0.85,
                next_steps=["validation", "execution"],
                metadata={
                    "topics": [],
                    "prerequisites": []
                },
                agent_type=AgentType.TEACHER
            )
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise AgentError(f"Error processing message: {str(e)}")
            
    async def _update_knowledge(self, content: str, response: str) -> None:
        """Update knowledge base and learning paths based on content and analysis."""
        try:
            # Add content to knowledge base
            topic_id = f"topic_{len(self.knowledge_base)}"
            self.knowledge_base[topic_id] = {
                "content": content,
                "response": response,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Add content to learning paths
            path_id = f"path_{len(self.learning_paths)}"
            self.learning_paths[path_id] = [topic_id]

            logger.info("Knowledge base updated successfully")
        except Exception as e:
            logger.error(f"Error updating knowledge: {str(e)}")
            raise AgentError(f"Error updating knowledge: {str(e)}")
            
    async def reset(self):
        """Reset the agent's state."""
        try:
            self.knowledge_base.clear()
            self.learning_paths.clear()
            self.memory.clear()
            await super().cleanup()
        except Exception as e:
            logger.error(f"Error resetting teacher agent: {str(e)}")
            raise AgentError(f"Error resetting teacher agent: {str(e)}") 