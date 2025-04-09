import logging
import asyncio
from typing import Any, Optional, Dict
from datetime import datetime

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from src.common.agent_types import AgentType, AgentResponse
from src.core.services.semantic_abstraction import ContextNode
from src.common.exceptions import AgentError

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all agents in the system."""
    
    def __init__(
        self,
        name: str = None,
        description: str = None,
        agent_type: str = None,
        llm: Any = None,
        database_service: Any = None,
        semantic_service: Any = None,
        cache_service: Any = None,
        monitoring_service: Any = None
    ):
        """Initialize the base agent.

        Args:
            name (str, optional): The name of the agent. Defaults to None.
            description (str, optional): The description of the agent. Defaults to None.
            agent_type (str, optional): The type of the agent. Defaults to None.
            llm (Any, optional): The language model to use. Defaults to None.
            database_service (Any, optional): The database service to use. Defaults to None.
            semantic_service (Any, optional): The semantic service to use. Defaults to None.
            cache_service (Any, optional): The cache service to use. Defaults to None.
            monitoring_service (Any, optional): The monitoring service to use. Defaults to None.
        """
        self.name = name
        self.description = description
        self.agent_type = agent_type
        self.llm = llm
        self.initialized = False
        self.memory = ConversationBufferMemory()
        self.chain = None
        self.template = None
        self._running_tasks = set()
        self._state = {}
        
        # Service dependencies
        self.database_service = database_service
        self.semantic_service = semantic_service
        self.cache_service = cache_service
        self.monitoring_service = monitoring_service

    async def initialize(self):
        """Initialize the agent."""
        try:
            if self.initialized:
                logger.warning(f"Agent {self.name} is already initialized")
                return

            # Ensure agent has a name
            if self.name is None:
                self.name = self.__class__.__name__

            # Import services lazily to avoid circular imports
            from src.data.database import DatabaseService
            from src.core.services.semantic_understanding import SemanticUnderstandingService
            from src.utils.cache import CacheService
            from src.utils.monitoring import MonitoringService
            from src.config.settings import Settings
            from langchain_openai import OpenAI

            # Initialize services if not provided
            if self.database_service is None:
                self.database_service = DatabaseService()
            if self.semantic_service is None:
                self.semantic_service = SemanticUnderstandingService()
            if self.cache_service is None:
                self.cache_service = CacheService()
            if self.monitoring_service is None:
                self.monitoring_service = MonitoringService(Settings())
            if self.llm is None:
                self.llm = OpenAI()

            # Initialize services that have an initialize method
            if hasattr(self.database_service, 'initialize'):
                await self.database_service.initialize()
            if hasattr(self.semantic_service, 'initialize'):
                await self.semantic_service.initialize()
            if hasattr(self.cache_service, 'initialize'):
                await self.cache_service.initialize()
            if hasattr(self.monitoring_service, 'initialize'):
                await self.monitoring_service.initialize()
            
            # Load state if exists
            await self._load_state()
            
            # Initialize chain with LLM
            self.chain = await self.initialize_chain(llm=self.llm)
            if self.chain is None:
                raise AgentError("Failed to initialize chain")
            
            self.initialized = True
            logger.info(f"Agent {self.name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent {self.name}: {str(e)}")
            raise AgentError(f"Failed to initialize agent {self.name}: {str(e)}")

    async def initialize_chain(self, llm: Any = None) -> Any:
        """Initialize the agent's chain with the given LLM.
        
        Args:
            llm (Any, optional): The language model to use. Defaults to None.
            
        Returns:
            Any: The initialized chain.
            
        Raises:
            AgentError: If chain initialization fails.
        """
        try:
            if llm is not None:
                self.llm = llm
                
            if self.llm is None:
                raise AgentError("LLM must be provided for chain initialization")
                
            if self.template is None:
                self.template = self._get_prompt_template()
                
            prompt = PromptTemplate(
                input_variables=["history", "input"],
                template=self.template
            )
            
            self.chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                memory=self.memory
            )
            
            if self.chain is None:
                raise AgentError("Failed to create chain")
                
            return self.chain
        except Exception as e:
            logger.error(f"Failed to initialize chain: {str(e)}")
            raise AgentError(f"Failed to initialize chain: {str(e)}")

    def _get_prompt_template(self) -> str:
        """Get the prompt template for this agent."""
        return """
        History: {history}
        Input: {input}
        Response:
        """

    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input data and return a response.
        
        Args:
            input_data: The input data to process
            context: Optional context for processing
            
        Returns:
            Dict containing the processed response
            
        Raises:
            AgentError: If processing fails
        """
        try:
            if not self.initialized:
                await self.initialize()
                
            # Process the input
            response = await self.process_message(input_data)
            return response
            
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            raise AgentError(f"Failed to process input: {str(e)}")

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a message."""
        if not self.initialized:
            raise AgentError("Agent not initialized")
            
        # Track the task
        task = asyncio.current_task()
        if task:
            self._running_tasks.add(task)
            
        try:
            # Validate message format
            if not isinstance(message, dict):
                raise AgentError("Message must be a dictionary")
            if 'content' not in message:
                raise AgentError("Message must contain 'content' field")
            
            # Analyze the message
            analysis = await self._analyze_input(message.get('content', ''), message.get('context', {}))
            
            # Process the message
            response = await self._generate_response(message)
            
            # Save state
            await self._save_state()
            
            return {
                'analysis': analysis,
                'response': response,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except AgentError:
            raise
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise AgentError(f"Message processing failed: {str(e)}")
            
        finally:
            # Remove the task from tracking
            if task and task in self._running_tasks:
                self._running_tasks.remove(task)

    async def _analyze_input(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze input content and context."""
        try:
            if not self.semantic_service:
                raise AgentError("Semantic service not initialized")
                
            return await self.semantic_service.analyze({
                'content': content,
                'context': context
            })
        except Exception as e:
            logger.error(f"Error analyzing input: {str(e)}")
            raise AgentError(f"Error analyzing input: {str(e)}")

    async def _generate_response(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a response to the message."""
        try:
            if not self.chain:
                raise AgentError("Chain not initialized")
                
            response = await self.chain.agenerate(
                [{
                    "history": message.get('context', ''),
                    "input": message.get('content', '')
                }]
            )
            
            return {
                'content': response.generations[0][0].text,
                'timestamp': datetime.utcnow().isoformat(),
                'agent_type': self.agent_type
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise AgentError(f"Error generating response: {str(e)}")

    async def cleanup(self):
        """Clean up agent resources."""
        try:
            # Cancel all running tasks
            for task in self._running_tasks:
                if not task.done():
                    task.cancel()
            
            # Clear resources
            self.chain = None
            self.memory.clear()
            self._running_tasks.clear()
            
            # Save final state
            await self._save_state()
            
            # Set initialized to False after all cleanup is done
            self.initialized = False
            
            logger.info(f"Agent {self.name} cleaned up successfully")
        except Exception as e:
            # Set initialized to False even if there's an error
            self.initialized = False
            logger.error(f"Error cleaning up agent {self.name}: {str(e)}")
            raise AgentError(f"Error cleaning up agent {self.name}: {str(e)}")

    async def _load_state(self) -> None:
        """Load agent state from database."""
        try:
            if not self.database_service:
                logger.warning("Database service not initialized")
                return
                
            state = await self.database_service.get_agent_state(self.name)
            if state:
                self._state = state
                logger.info(f"Loaded state for agent {self.name}: {state}")
            else:
                logger.info(f"No state found for agent {self.name}")
        except Exception as e:
            logger.warning(f"Failed to load state for agent {self.name}: {str(e)}")
            # Don't raise the error, just log it

    async def _save_state(self) -> None:
        """Save agent state to database."""
        if not self.database_service:
            logger.warning("Database service not initialized")
            return
            
        await self.database_service.save_agent_state(
            agent_id=self.name,
            agent_type=self.agent_type,
            state=self._state
        )

    def __del__(self):
        """Ensure cleanup on object destruction."""
        if self.initialized and self._running_tasks:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create cleanup task
                    cleanup_task = loop.create_task(self.cleanup())
                    # Wait for cleanup to complete
                    loop.run_until_complete(cleanup_task)
            except Exception as e:
                logger.error(f"Error in destructor: {str(e)}") 