import logging
from typing import Dict, Any, Optional, Union, List
import re

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from src.common.agent_types import AgentType, AgentResponse
from src.core.services.semantic_abstraction import ContextNode
from src.core.services.base_agent import BaseAgent
from src.common.exceptions import AgentError

logger = logging.getLogger(__name__)


class BuilderAgent(BaseAgent):
    def __init__(
        self,
        name: str = "builder",
        description: str = "Builds and constructs solutions based on plans",
        agent_type: AgentType = AgentType.BUILDER,
        llm: Optional[Any] = None,
    ):
        super().__init__(name, description, agent_type)
        self.chain = None
        self.memory = ConversationBufferMemory()
        self.project_structure: Dict[str, Dict[str, Any]] = {}
        self.dependencies: List[str] = []
        self.conversation_history: List[Dict[str, Any]] = []
        self.state: Dict[str, Any] = {}
        self.initialized = False
        self.template = """
        You are a builder agent responsible for constructing and implementing solutions based on plans.
        Your role is to take plans and requirements and transform them into concrete, actionable implementations.

        Previous conversation:
        {history}

        Current input:
        {input}

        Please provide a detailed implementation plan with the following structure:

        1. Solution Architecture:
           - Component breakdown
           - Technical specifications
           - Integration points
           - Dependencies and requirements

        2. Implementation Details:
           - Code structure and organization
           - Key algorithms and data structures
           - API definitions and interfaces
           - Database schema (if applicable)

        3. Technical Requirements:
           - Development environment setup
           - Required libraries and frameworks
           - Configuration details
           - Build and deployment specifications

        4. Quality Assurance:
           - Unit test coverage
           - Integration test scenarios
           - Performance considerations
           - Security measures

        5. Implementation Steps:
           - Step-by-step development guide
           - Code examples and snippets
           - Configuration instructions
           - Deployment procedures

        6. Validation Criteria:
           - Functional requirements checklist
           - Performance benchmarks
           - Security compliance
           - Code quality standards

        Please analyze the input and provide a detailed implementation plan following this format:
        """
        if llm:
            self.initialize_chain(llm=llm)

    def initialize_chain(self, template: Optional[str] = None, llm: Optional[Any] = None):
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
            logger.error(f"Error in BuilderAgent: {str(e)}")
            return AgentResponse(
                response={"error": str(e)},
                confidence=0.0,
                next_steps=[],
                agent_type=self.agent_type,
                status="error",
            )

    async def process_message(self, message: Union[str, Dict[str, Any]]) -> AgentResponse:
        """Process a message and return a response."""
        if not self.initialized:
            raise AgentError("Builder agent not initialized")

        try:
            # Extract content from message
            content = message if isinstance(message, str) else message.get('content', '')
            if not content:
                raise AgentError("Message must contain content")

            # Update conversation history
            self.conversation_history.append({
                'content': content,
                'timestamp': message.get('timestamp', 0) if isinstance(message, dict) else 0
            })

            # Process file creation requests
            if 'create a new file' in content.lower():
                file_match = re.search(r'called\s+([^\s]+)', content.lower())
                if file_match:
                    file_name = file_match.group(1)
                    self.project_structure[file_name] = {'type': 'file'}

            # Process directory creation requests
            elif 'create a new directory' in content.lower():
                dir_match = re.search(r'called\s+([^\s]+)', content.lower())
                if dir_match:
                    dir_name = dir_match.group(1)
                    self.project_structure[dir_name] = {'type': 'directory'}

            # Process dependency requests
            elif 'add' in content.lower() and 'as a dependency' in content.lower():
                dep_match = re.search(r'add\s+([^\s]+)\s+as a dependency', content.lower())
                if dep_match:
                    dependency = dep_match.group(1)
                    self.dependencies.append(dependency)

            # Update state
            self.state['last_task'] = content

            # Generate builder response
            response = await self.chain.arun(content)

            # Return formatted response
            return AgentResponse(
                content=response,
                reasoning="Implementation plan generated",
                confidence=0.9,
                next_steps=["validation", "execution"],
                metadata={
                    "components": list(self.project_structure.keys()),
                    "dependencies": self.dependencies
                },
                agent_type=AgentType.BUILDER
            )
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise AgentError(f"Error processing message: {str(e)}")

    def get_context(self) -> Dict[str, Any]:
        """Get the current context of the agent."""
        return {
            'state': self.state,
            'recent_history': self.conversation_history[-5:] if self.conversation_history else [],
            'project_structure': self.project_structure,
            'dependencies': self.dependencies
        }

    def reset(self) -> None:
        """Reset the agent's state to its initial values."""
        self.project_structure.clear()
        self.dependencies.clear()
        self.conversation_history.clear()
        self.state.clear()
        self.initialized = False 