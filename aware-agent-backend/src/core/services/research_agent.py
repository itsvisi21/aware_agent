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


class ResearchAgent(BaseAgent):
    """Agent specialized in conducting research and gathering information."""
    
    def __init__(self, name: str = None, description: str = None, agent_type: str = None, llm: Any = None):
        """Initialize the ResearchAgent.

        Args:
            name (str, optional): The name of the agent. Defaults to None.
            description (str, optional): The description of the agent. Defaults to None.
            agent_type (str, optional): The type of the agent. Defaults to None.
            llm (Any, optional): The language model to use. Defaults to None.
        """
        super().__init__(name=name, description=description, agent_type=agent_type, llm=llm)
        self.research_topics = {}
        self.sources = set()
        self.initialized = False
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.template = """
        You are a research agent tasked with analyzing and gathering information.
        
        Context: {context}
        Current Topics: {topics}
        Available Sources: {sources}
        
        Input: {input}
        
        Based on the above, please:
        1. Analyze the input
        2. Relate it to existing research topics
        3. Identify new topics if relevant
        4. Provide insights and findings
        
        Response should include:
        - Main findings and insights
        - Connections to existing topics
        - Confidence level in findings
        - Suggested next steps for research
        """
        
    async def initialize_chain(self, template: str = None, llm: Any = None) -> LLMChain:
        """Initialize the LLM chain for research."""
        try:
            if template:
                self.template = template

            if llm:
                self.llm = llm

            if not self.llm:
                raise AgentError("No LLM provided for research chain")

            prompt = PromptTemplate(
                input_variables=["input", "context", "topics", "sources"],
                template=self.template
            )

            self.chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                memory=self.memory,
                verbose=True
            )

            self.initialized = True
            logger.info("Research chain initialized successfully")
            return self.chain
        except Exception as e:
            logger.error(f"Failed to initialize research chain: {str(e)}")
            raise AgentError(f"Failed to initialize research chain: {str(e)}")
            
    async def process_message(self, message: Union[str, Dict[str, Any]]) -> AgentResponse:
        """Process a message and return a response."""
        if not self.initialized:
            raise AgentError("Research agent not initialized")

        try:
            # Extract content from message
            content = message if isinstance(message, str) else message.get('content', '')
            if not content:
                raise AgentError("Message must contain content")

            # Prepare context for the chain
            context = {
                "input": content,
                "context": "",
                "topics": list(self.research_topics.keys()),
                "sources": list(self.sources)
            }

            # Generate research response
            response = await self.chain.arun(**context)

            # Update research with new content and analysis
            await self._update_research(content, response)

            # Return formatted response
            return AgentResponse(
                content=response,
                reasoning="Research analysis completed",
                confidence=0.8,
                next_steps=["Continue research", "Analyze findings"],
                metadata={
                    "research_topics": list(self.research_topics.keys()),
                    "sources": list(self.sources)
                },
                agent_type=AgentType.RESEARCH
            )
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise AgentError(f"Error processing message: {str(e)}")
            
    async def _update_research(self, content: str, response: str) -> None:
        """Update research topics and sources based on content and analysis."""
        try:
            # Add content as a source
            source_id = f"source_{len(self.sources)}"
            self.sources.add(source_id)

            # Add content as a research topic
            topic_id = f"topic_{len(self.research_topics)}"
            self.research_topics[topic_id] = {
                "content": content,
                "response": response,
                "timestamp": datetime.utcnow().isoformat()
            }

            logger.info("Research updated successfully")
        except Exception as e:
            logger.error(f"Error updating research: {str(e)}")
            raise AgentError(f"Error updating research: {str(e)}")
            
    async def reset(self):
        """Reset the agent's state."""
        try:
            self.research_topics.clear()
            self.sources.clear()
            self.memory.clear()
            await super().cleanup()
        except Exception as e:
            logger.error(f"Error resetting research agent: {str(e)}")
            raise AgentError(f"Error resetting research agent: {str(e)}") 