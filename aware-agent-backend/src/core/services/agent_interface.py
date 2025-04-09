from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from src.common.agent_types import AgentType, AgentResponse
from src.core.services.semantic_abstraction import ContextNode

class AgentOrchestratorInterface(ABC):
    """Interface for agent orchestration."""
    
    @abstractmethod
    async def process_task(self, agent_type: AgentType, query: str) -> AgentResponse:
        """Process a task using the specified agent type.
        
        Args:
            agent_type: The type of agent to use
            query: The query to process
            
        Returns:
            AgentResponse with the results
        """
        pass
    
    @abstractmethod
    async def create_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a plan based on the given context.
        
        Args:
            context: The context for plan creation
            
        Returns:
            Dict containing the plan details
        """
        pass
    
    @abstractmethod
    async def execute_plan(
        self,
        plan_id: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Execute a plan.
        
        Args:
            plan_id: ID of the plan to execute
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict containing execution results
        """
        pass 