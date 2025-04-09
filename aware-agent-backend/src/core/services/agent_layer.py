import logging
from typing import Dict, Any, Optional, List, Union

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from src.common.agent_types import AgentType, AgentResponse
from src.core.services.base_agent import BaseAgent
from src.core.services.builder_agent import BuilderAgent
from src.core.services.collaborator_agent import CollaboratorAgent
from src.common.exceptions import AgentError
from src.core.services.explainer_agent import ExplainerAgent
from src.core.services.goal_alignment_agent import GoalAlignmentAgent
from src.utils.monitoring import MonitoringService
from src.core.services.planning_agent import PlanningAgent
from src.core.services.planner_agent import PlannerAgent
from src.core.services.research_agent import ResearchAgent
from src.core.services.semantic_abstraction import SemanticDimension, KarakaMapping, ContextNode
from src.core.services.teacher_agent import TeacherAgent
from src.core.services.validator_agent import ValidatorAgent
from src.core.services.pruning_agent import PruningAgent
from src.core.services.context_agent import ContextAgent
from src.core.services.validation_agent import ValidationAgent

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """Orchestrates multiple agents to process queries."""
    
    def __init__(self, llm: Any = None):
        """Initialize the orchestrator.
        
        Args:
            llm (Any, optional): The language model to use. Defaults to None.
        """
        self.llm = llm
        self.agents = {
            AgentType.PLANNER.name.lower(): PlannerAgent(),
            AgentType.RESEARCH.name.lower(): ResearchAgent(),
            AgentType.EXPLAINER.name.lower(): ExplainerAgent(),
            AgentType.VALIDATOR.name.lower(): ValidatorAgent()
        }
        # Set LLM for all agents after initialization
        for agent in self.agents.values():
            agent.llm = llm
        self.monitoring = MonitoringService()
        
    async def process_query(self, context: ContextNode, query: str) -> Dict[str, AgentResponse]:
        """Process a query using all agents.
        
        Args:
            context (ContextNode): The context for the query.
            query (str): The query to process.
            
        Returns:
            Dict[str, AgentResponse]: Dictionary of agent responses.
        """
        responses = {}
        for agent_name, agent in self.agents.items():
            response = await agent.process(context, query)
            responses[agent_name] = response
        return responses
        
    async def process_task(self, agent_type: Union[AgentType, str], query: str) -> AgentResponse:
        """Process a task using a specific agent.
        
        Args:
            agent_type (Union[AgentType, str]): The type of agent to use.
            query (str): The query to process.
            
        Returns:
            AgentResponse: The agent's response.
        """
        if isinstance(agent_type, str):
            agent_type = AgentType[agent_type.upper()]
            
        agent_name = agent_type.name.lower()
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        agent = self.agents[agent_name]
        return await agent.process_message({"content": query})
        
    def get_agent_metrics(self) -> Dict[str, float]:
        """Get metrics for all agents.
        
        Returns:
            Dict[str, float]: Dictionary of agent metrics.
        """
        return {
            "total_tasks": self.monitoring.get_metric("total_tasks", 0),
            "successful_tasks": self.monitoring.get_metric("successful_tasks", 0),
            "failed_tasks": self.monitoring.get_metric("failed_tasks", 0),
            "average_confidence": self.monitoring.get_metric("average_confidence", 0.0)
        }
        
    def reset_metrics(self):
        """Reset all agent metrics."""
        self.monitoring.reset_metrics()


class BuilderAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="builder",
            description="Builds and implements plans",
            agent_type=AgentType.BUILDER
        )
        self.template = """
        As a builder agent, analyze the following context and input:
        Context: {context}
        Input: {input}
        
        Provide:
        1. Implementation plan
        2. Task allocation
        3. Timeline estimates
        4. Resource requirements
        """

    async def process(self, context: ContextNode, input_text: str) -> AgentResponse:
        response = await self.chain.arun(
            context=context.visualize_context_tree(),
            input=input_text
        )

        return AgentResponse(
            content=response,
            reasoning="Generated based on semantic context and implementation requirements",
            confidence=0.85,
            next_steps=["collaboration", "validation"],
            agent_type=self.agent_type
        )


class TeacherAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="teacher",
            description="Teaches and mentors",
            agent_type=AgentType.TEACHER
        )
        self.template = """
        As a teacher agent, analyze the following context and input:
        Context: {context}
        Input: {input}
        
        Provide:
        1. Teaching plan
        2. Learning objectives
        3. Assessment methods
        4. Feedback mechanisms
        """

    async def process(self, context: ContextNode, input_text: str) -> AgentResponse:
        response = await self.chain.arun(
            context=context.visualize_context_tree(),
            input=input_text
        )

        return AgentResponse(
            content=response,
            reasoning="Generated based on semantic context and teaching requirements",
            confidence=0.85,
            next_steps=["validation"],
            agent_type=self.agent_type
        )


class AgentOrchestrationLayer:
    """Layer for orchestrating agents with context management."""
    
    def __init__(self, llm: Any = None):
        """Initialize the orchestration layer.
        
        Args:
            llm (Any, optional): The language model to use. Defaults to None.
        """
        self.llm = llm
        self.agents = {
            AgentType.PLANNER.name.lower(): PlannerAgent(),
            AgentType.RESEARCH.name.lower(): ResearchAgent(),
            AgentType.EXPLAINER.name.lower(): ExplainerAgent(),
            AgentType.VALIDATOR.name.lower(): ValidatorAgent(),
            AgentType.GOAL_ALIGNMENT.name.lower(): GoalAlignmentAgent(),
            AgentType.CONTEXT.name.lower(): ContextAgent(),
            AgentType.PRUNING.name.lower(): PruningAgent(),
            AgentType.VALIDATION.name.lower(): ValidationAgent()
        }
        # Set LLM for all agents after initialization
        for agent in self.agents.values():
            agent.llm = llm
        self.context = {}
        
    async def process_with_all_agents(self, query: str) -> Dict[str, AgentResponse]:
        """Process a query using all agents.
        
        Args:
            query (str): The query to process.
            
        Returns:
            Dict[str, AgentResponse]: Dictionary of agent responses.
        """
        responses = {}
        for agent_name, agent in self.agents.items():
            response = await agent.process_message({"content": query})
            responses[agent_name] = response
        return responses
        
    def update_context(self, dimension: SemanticDimension, content: str, mapping: KarakaMapping):
        """Update the context with new information.
        
        Args:
            dimension (SemanticDimension): The semantic dimension.
            content (str): The content to add.
            mapping (KarakaMapping): The karaka mapping.
        """
        self.context[dimension.domain] = {
            "content": content,
            "mapping": mapping
        }
        
    def clear_context(self):
        """Clear all context information."""
        self.context.clear()
        
    async def align_goals(self, context: ContextNode) -> AgentResponse:
        """Align goals using the goal alignment agent.
        
        Args:
            context (ContextNode): The context for goal alignment.
            
        Returns:
            AgentResponse: The agent's response.
        """
        agent = self.agents[AgentType.GOAL_ALIGNMENT.name.lower()]
        return await agent.process(context, "Align goals")
        
    async def plan_strategy(self, context: ContextNode) -> AgentResponse:
        """Plan strategy using the planner agent.
        
        Args:
            context (ContextNode): The context for planning.
            
        Returns:
            AgentResponse: The agent's response.
        """
        agent = self.agents[AgentType.PLANNER.name.lower()]
        return await agent.process(context, "Plan strategy")
        
    async def prune_options(self, context: ContextNode) -> AgentResponse:
        """Prune options using the pruning agent.
        
        Args:
            context (ContextNode): The context for pruning.
            
        Returns:
            AgentResponse: The agent's response.
        """
        agent = self.agents[AgentType.PRUNING.name.lower()]
        return await agent.process(context, "Prune options")
        
    async def analyze_context(self, context: ContextNode) -> AgentResponse:
        """Analyze context using the context agent.
        
        Args:
            context (ContextNode): The context to analyze.
            
        Returns:
            AgentResponse: The agent's response.
        """
        agent = self.agents[AgentType.CONTEXT.name.lower()]
        return await agent.process(context, "Analyze context")
        
    async def validate_plan(self, context: ContextNode) -> AgentResponse:
        """Validate plan using the validation agent.
        
        Args:
            context (ContextNode): The context for validation.
            
        Returns:
            AgentResponse: The agent's response.
        """
        agent = self.agents[AgentType.VALIDATION.name.lower()]
        return await agent.process(context, "Validate plan")
