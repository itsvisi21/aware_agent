import logging
from typing import Dict, Any, Optional, Union

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from src.common.agent_types import AgentType, AgentResponse
from src.core.services.semantic_abstraction import ContextNode
from src.core.services.base_agent import BaseAgent
from src.common.exceptions import AgentError

logger = logging.getLogger(__name__)


class CollaboratorAgent(BaseAgent):
    def __init__(
        self,
        name: str = "collaborator",
        description: str = "Facilitates collaboration between agents",
        agent_type: AgentType = AgentType.COLLABORATOR,
        llm: Optional[Any] = None,
    ):
        super().__init__(name, description, agent_type)
        self.chain = None
        self.memory = ConversationBufferMemory()
        self.initialized = False
        self.team_members = {}
        self.tasks = {}
        self.progress_tracking = {}
        self.conversation_history = []
        self.state = {}
        self.template = """
        You are a collaborator agent responsible for facilitating effective communication and coordination between different agents.
        Your role is to ensure smooth information flow, resolve conflicts, and maintain productive collaboration.

        Previous conversation:
        {history}

        Current input:
        {input}

        Please provide a collaboration strategy with the following structure:

        1. Communication Analysis:
           - Key messages and requirements
           - Potential misunderstandings
           - Information gaps
           - Stakeholder perspectives

        2. Coordination Plan:
           - Task dependencies
           - Resource allocation
           - Timeline synchronization
           - Handoff points

        3. Conflict Resolution:
           - Identified conflicts
           - Root cause analysis
           - Resolution strategies
           - Prevention measures

        4. Information Flow:
           - Required information sharing
           - Communication channels
           - Documentation needs
           - Feedback mechanisms

        5. Collaboration Tools:
           - Recommended tools and platforms
           - Integration requirements
           - Access and permissions
           - Usage guidelines

        6. Success Metrics:
           - Collaboration effectiveness
           - Communication quality
           - Conflict resolution success
           - Team satisfaction

        Please analyze the input and provide a collaboration strategy following this format:
        """
        if llm:
            self.initialize_chain(llm=llm)
            self.initialized = True

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

    def reset(self):
        """Reset the agent's state."""
        self.team_members = {}
        self.tasks = {}
        self.progress_tracking = {}
        self.conversation_history = []
        self.state = {}
        self.memory.clear()

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
            logger.error(f"Error in CollaboratorAgent: {str(e)}")
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
            raise AgentError("Collaborator agent not initialized")

        try:
            # Extract content from message
            content = message if isinstance(message, str) else message.get('content', '')
            if not content:
                raise AgentError("Message must contain content")

            # Add message to conversation history
            self.conversation_history.append(message)

            # Handle team member addition
            if 'Add team member' in content:
                member_name = content.split('Add team member ')[1].strip()
                self.team_members[member_name] = {
                    'role': 'team member',
                    'tasks_assigned': []
                }

            # Handle task creation
            elif 'Create task' in content:
                task_name = content.split('Create task ')[1].strip()
                self.tasks[task_name] = {
                    'status': 'pending',
                    'assigned_to': None,
                    'progress': 0
                }
                self.progress_tracking[task_name] = {
                    'last_update': 'created',
                    'status': 'pending'
                }

            # Handle task progress update
            elif 'Update progress on' in content:
                task_name = content.split('Update progress on ')[1].strip()
                if task_name in self.tasks:
                    self.progress_tracking[task_name]['last_update'] = 'updated'
                    self.progress_tracking[task_name]['status'] = 'in_progress'

            # Generate collaborator response
            response = await self.chain.arun(content)

            # Return formatted response
            return AgentResponse(
                content=response,
                reasoning="Collaboration strategy generated",
                confidence=0.9,
                next_steps=["validation", "execution"],
                metadata={
                    "team_members": list(self.team_members.keys()),
                    "tasks": list(self.tasks.keys())
                },
                agent_type=AgentType.COLLABORATOR
            )
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise AgentError(f"Error processing message: {str(e)}")

    def get_context(self) -> Dict[str, Any]:
        """Get the current context of the agent."""
        return {
            'state': {
                'active_tasks': list(self.tasks.keys()),
                'team_members': list(self.team_members.keys())
            },
            'recent_history': self.conversation_history[-5:] if self.conversation_history else []
        } 