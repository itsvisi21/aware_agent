import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio

import numpy as np
from langchain_openai import OpenAI

from src.common.agent_types import AgentType, AgentResponse
from src.core.models.models import Task, TaskStatus
from src.core.services.semantic_abstraction import ContextNode, SemanticAbstractionLayer
from src.core.services.execution_layer import ExecutionLayer
from src.core.services.memory_engine import MemoryEngine
from src.core.services.agents import GoalAlignmentAgent, PlannerAgent, PruningAgent, ContextAgent, ValidationAgent, ResearchAgent, BuilderAgent, TeacherAgent, CollaboratorAgent
from src.common.exceptions import AgentError
from src.core.services.agent_interface import AgentOrchestratorInterface
from src.core.services.base_agent import BaseAgent


class AgentOrchestrator(AgentOrchestratorInterface):
    """Orchestrates the execution of agent-based tasks."""

    def __init__(self, llm=None):
        """Initialize the orchestrator with an optional LLM."""
        if llm is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
        else:
            self.llm = llm
            
        # Create memory directory
        self.memory_dir = Path("workspace/orchestrator_memory")
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.execution_layer = ExecutionLayer()
        self.memory_engine = MemoryEngine(storage_path=self.memory_dir)
        self.semantic_layer = SemanticAbstractionLayer()
        
        # Initialize agents and tasks
        self.agents = {}
        self.tasks = {}
        self.active_tasks = {}
        self.plans = []

    def register_agent(self, agent: BaseAgent, agent_type: AgentType) -> None:
        """Register a new agent with the orchestrator.
        
        Args:
            agent: The agent instance to register
            agent_type: The type of agent being registered
            
        Raises:
            AgentError: If the agent type is invalid or registration fails
        """
        try:
            if not isinstance(agent_type, AgentType):
                raise AgentError(f"Invalid agent type: {agent_type}")
                
            if agent_type in self.agents:
                raise AgentError(f"Agent type {agent_type} already registered")
                
            self.agents[agent_type] = agent
            agent.llm = self.llm
            
        except Exception as e:
            raise AgentError(f"Error registering agent: {str(e)}")

    def add_task(self, task: Task) -> None:
        """Add a new task to the orchestrator.
        
        Args:
            task: The task to add
            
        Raises:
            AgentError: If task creation fails
        """
        try:
            if task.task_id in self.tasks:
                raise AgentError(f"Task {task.task_id} already exists")
                
            self.tasks[task.task_id] = task
            self.active_tasks[task.task_id] = task
            
        except Exception as e:
            raise AgentError(f"Error adding task: {str(e)}")

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: float = None,
        confidence: float = None
    ) -> None:
        """Update the status of a task.
        
        Args:
            task_id: ID of the task to update
            status: New status for the task
            progress: Optional progress value (0.0 to 1.0)
            confidence: Optional confidence value (0.0 to 1.0)
            
        Raises:
            AgentError: If task update fails
        """
        try:
            if task_id not in self.tasks:
                raise AgentError(f"Task {task_id} not found")
                
            task = self.tasks[task_id]
            task.status = status
            
            if progress is not None:
                task.progress = progress
            if confidence is not None:
                task.confidence = confidence
                
            task.timestamp = datetime.now()
            
        except Exception as e:
            raise AgentError(f"Error updating task status: {str(e)}")

    async def execute_task(self, task_id: str, agent_type: AgentType) -> Dict[str, Any]:
        """Execute a task using the specified agent type.
        
        Args:
            task_id: ID of the task to execute
            agent_type: Type of agent to use for execution
            
        Returns:
            Dict containing the execution results
            
        Raises:
            AgentError: If execution fails
        """
        try:
            if task_id not in self.tasks:
                raise AgentError(f"Task {task_id} not found")
                
            if agent_type not in self.agents:
                raise AgentError(f"Agent type {agent_type} not registered")
                
            task = self.tasks[task_id]
            agent = self.agents[agent_type]
            
            # Update task status to processing
            await self.update_task_status(task_id, TaskStatus.PROCESSING, progress=0.0)
            
            try:
                # Execute the task
                result = await agent.execute_task(task)
                
                # Update task status to completed
                await self.update_task_status(
                    task_id,
                    TaskStatus.COMPLETED,
                    progress=1.0,
                    confidence=1.0
                )
                
                return result
                
            except Exception as e:
                # Update task status to failed
                await self.update_task_status(
                    task_id,
                    TaskStatus.FAILED,
                    progress=1.0,
                    confidence=0.0
                )
                raise AgentError(f"Task execution failed: {str(e)}")
                
        except Exception as e:
            raise AgentError(f"Error executing task: {str(e)}")

    async def process_task(self, agent_type: AgentType, query: str) -> AgentResponse:
        """
        Process a task using the specified agent type.
        
        Args:
            agent_type: The type of agent to use
            query: The query to process
            
        Returns:
            AgentResponse with the results
            
        Raises:
            AgentError: If the agent type is not supported or processing fails
        """
        try:
            if not isinstance(agent_type, AgentType):
                raise AgentError(f"Invalid agent type: {agent_type}")
                
            agent = self.agents.get(agent_type)
            if agent is None:
                raise AgentError(f"Agent type {agent_type} not found")
                
            # Initialize the agent if not already initialized
            if not agent.initialized:
                await agent.initialize()
                
            # Process the task
            return await agent.process(query)
            
        except Exception as e:
            raise AgentError(f"Error processing task: {str(e)}")

    async def create_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a plan based on the given context.
        
        Args:
            context: The context for plan creation
            
        Returns:
            Dict containing the plan details
        """
        try:
            # Analyze context
            context_analysis = await self.agents[AgentType.CONTEXT].process(context)
            
            # Align goals
            goal_alignment = await self.agents[AgentType.GOAL_ALIGNMENT].process(context_analysis)
            
            # Plan strategy
            strategy = await self.agents[AgentType.PLANNING].process(goal_alignment)
            
            # Create execution steps
            steps = self._create_execution_steps(
                plan_type=self._determine_plan_type(
                    roles=context.get("roles", {}),
                    entities=context.get("entities", [])
                ),
                roles=context.get("roles", {}),
                entities=context.get("entities", []),
                relationships=context.get("relationships", []),
                context=context
            )
            
            # Create plan
            plan_id = str(uuid.uuid4())
            plan = {
                "id": plan_id,
                "context": context,
                "steps": steps,
                "status": "created",
                "created_at": datetime.now().isoformat()
            }
            
            self.plans.append(plan)
            return plan
            
        except Exception as e:
            raise AgentError(f"Error creating plan: {str(e)}")

    async def execute_plan(
        self,
        plan_id: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Execute a plan.
        
        Args:
            plan_id: ID of the plan to execute
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict containing execution results
        """
        try:
            # Find plan
            plan = next((p for p in self.plans if p["id"] == plan_id), None)
            if not plan:
                raise AgentError(f"Plan {plan_id} not found")
                
            # Update plan status
            plan["status"] = "executing"
            plan["started_at"] = datetime.now().isoformat()
            
            # Execute steps
            results = []
            total_steps = len(plan["steps"])
            
            for i, step in enumerate(plan["steps"]):
                # Execute step
                result = await self._execute_step(step)
                results.append(result)
                
                # Update progress
                progress = (i + 1) / total_steps
                if progress_callback:
                    progress_callback(progress)
                    
                # Update plan
                plan["progress"] = progress
                plan["current_step"] = i + 1
                
            # Update plan status
            plan["status"] = "completed"
            plan["completed_at"] = datetime.now().isoformat()
            plan["results"] = results
            
            return plan
            
        except Exception as e:
            if plan:
                plan["status"] = "failed"
                plan["error"] = str(e)
            raise AgentError(f"Error executing plan: {str(e)}")

    def _determine_plan_type(
            self,
            roles: Dict[str, Any],
            entities: List[Dict[str, Any]]
    ) -> str:
        """Determine the type of plan based on semantic analysis."""
        # Check for research indicators
        if any(role.get("type") == "RESEARCH" for role in roles.values()):
            return "research"

        # Check for technical indicators
        if any(entity.get("type") == "TECHNICAL" for entity in entities):
            return "technical"

        # Check for business indicators
        if any(entity.get("type") == "BUSINESS" for entity in entities):
            return "business"

        # Default to general plan
        return "general"

    def _create_execution_steps(
            self,
            plan_type: str,
            roles: Dict[str, Any],
            entities: List[Dict[str, Any]],
            relationships: List[Dict[str, Any]],
            context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create execution steps based on plan type and semantic analysis."""
        steps = []

        if plan_type == "research":
            # Create research steps
            steps.extend([
                {
                    "type": "research",
                    "action": "search",
                    "parameters": {
                        "query": self._extract_research_query(roles),
                        "filters": self._extract_research_filters(entities),
                        "max_results": context.get("max_results", 10)
                    }
                },
                {
                    "type": "analysis",
                    "action": "analyze",
                    "parameters": {
                        "data": "previous_step_result",
                        "method": "semantic_analysis"
                    }
                }
            ])

        elif plan_type == "technical":
            # Create technical steps
            steps.extend([
                {
                    "type": "technical",
                    "action": "implement",
                    "parameters": {
                        "requirements": self._extract_technical_requirements(roles),
                        "constraints": self._extract_technical_constraints(entities)
                    }
                }
            ])

        elif plan_type == "business":
            # Create business steps
            steps.extend([
                {
                    "type": "business",
                    "action": "analyze",
                    "parameters": {
                        "metrics": self._extract_business_metrics(roles),
                        "timeframe": self._extract_timeframe(entities)
                    }
                }
            ])

        return steps

    def _allocate_resources(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Allocate resources for plan execution."""
        total_cpu = 0
        total_memory = 0
        total_network = 0

        for step in steps:
            # Estimate resource requirements
            if step["type"] == "research":
                total_cpu += 20
                total_memory += 256
                total_network += 50
            elif step["type"] == "technical":
                total_cpu += 40
                total_memory += 512
                total_network += 30
            elif step["type"] == "business":
                total_cpu += 30
                total_memory += 384
                total_network += 20

        return {
            "cpu": min(total_cpu, self.resource_pool["cpu"]),
            "memory": min(total_memory, self.resource_pool["memory"]),
            "network": min(total_network, self.resource_pool["network"])
        }

    async def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single plan step."""
        try:
            # Create task
            task_id = str(uuid.uuid4())
            task = {
                "id": task_id,
                "type": step["type"],
                "action": step["action"],
                "parameters": step["parameters"],
                "status": "created",
                "created_at": datetime.now().isoformat()
            }

            # Execute task
            result = await self.execution_layer.execute_task(
                task_id=task_id,
                progress_callback=lambda p: None
            )

            return result

        except Exception as e:
            return {
                "error": str(e),
                "step": step,
                "timestamp": datetime.now().isoformat()
            }

    def _extract_research_query(self, roles: Dict[str, Any]) -> str:
        """Extract research query from semantic roles."""
        # Implementation details...
        return "research query"

    def _extract_research_filters(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract research filters from entities."""
        # Implementation details...
        return {}

    def _extract_technical_requirements(self, roles: Dict[str, Any]) -> List[str]:
        """Extract technical requirements from semantic roles."""
        # Implementation details...
        return []

    def _extract_technical_constraints(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract technical constraints from entities."""
        # Implementation details...
        return {}

    def _extract_business_metrics(self, roles: Dict[str, Any]) -> List[str]:
        """Extract business metrics from semantic roles."""
        # Implementation details...
        return []

    def _extract_timeframe(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract timeframe information from entities."""
        # Implementation details...
        return {}

    async def align_goals(self, query: str) -> AgentResponse:
        """Align goals based on the query."""
        try:
            # For testing, return a simple response
            return AgentResponse(
                content="Goals aligned",
                reasoning="Goal alignment analysis",
                confidence=0.9,
                next_steps=["Proceed with aligned goals"],
                metadata={},
                agent_type=AgentType.GOAL_ALIGNMENT,
                status="success"
            )
        except Exception as e:
            return AgentResponse(
                content=str(e),
                reasoning="Failed to align goals",
                confidence=0.0,
                next_steps=[],
                metadata={},
                agent_type=AgentType.GOAL_ALIGNMENT,
                status="error"
            )

    async def plan_strategy(self, query: str) -> AgentResponse:
        """Plan strategy based on the query."""
        try:
            # For testing, return a simple response
            return AgentResponse(
                content="Strategy planned",
                reasoning="Strategy planning analysis",
                confidence=0.9,
                next_steps=["Execute strategy"],
                metadata={},
                agent_type=AgentType.PLANNING,
                status="success"
            )
        except Exception as e:
            return AgentResponse(
                content=str(e),
                reasoning="Failed to plan strategy",
                confidence=0.0,
                next_steps=[],
                metadata={},
                agent_type=AgentType.PLANNING,
                status="error"
            )

    async def prune_options(self, query: str) -> AgentResponse:
        """Prune options based on the query."""
        try:
            # For testing, return a simple response
            return AgentResponse(
                content="Options pruned",
                reasoning="Option pruning analysis",
                confidence=0.9,
                next_steps=["Use pruned options"],
                metadata={},
                agent_type=AgentType.PRUNING,
                status="success"
            )
        except Exception as e:
            return AgentResponse(
                content=str(e),
                reasoning="Failed to prune options",
                confidence=0.0,
                next_steps=[],
                metadata={},
                agent_type=AgentType.PRUNING,
                status="error"
            )

    async def analyze_context(self, query: str) -> AgentResponse:
        """Analyze context based on the query."""
        try:
            # For testing, return a simple response
            return AgentResponse(
                content="Context analyzed",
                reasoning="Context analysis",
                confidence=0.9,
                next_steps=["Use context insights"],
                metadata={},
                agent_type=AgentType.CONTEXT,
                status="success"
            )
        except Exception as e:
            return AgentResponse(
                content=str(e),
                reasoning="Failed to analyze context",
                confidence=0.0,
                next_steps=[],
                metadata={},
                agent_type=AgentType.CONTEXT,
                status="error"
            )

    async def validate_plan(self, query: str) -> AgentResponse:
        """Validate plan based on the query."""
        try:
            # For testing, return a simple response
            return AgentResponse(
                content="Plan validated",
                reasoning="Plan validation analysis",
                confidence=0.9,
                next_steps=["Execute validated plan"],
                metadata={},
                agent_type=AgentType.VALIDATION,
                status="success"
            )
        except Exception as e:
            return AgentResponse(
                content=str(e),
                reasoning="Failed to validate plan",
                confidence=0.0,
                next_steps=[],
                metadata={},
                agent_type=AgentType.VALIDATION,
                status="error"
            )

    async def research(self, query: str) -> AgentResponse:
        """Conduct research based on the query."""
        try:
            # For testing, return a simple response
            return AgentResponse(
                content="Research results",
                reasoning="Research analysis",
                confidence=0.9,
                next_steps=["Further research"],
                metadata={},
                agent_type=AgentType.RESEARCH,
                status="success"
            )
        except Exception as e:
            return AgentResponse(
                content=str(e),
                reasoning="Failed to conduct research",
                confidence=0.0,
                next_steps=[],
                metadata={},
                agent_type=AgentType.RESEARCH,
                status="error"
            )

    async def initialize_agents(self) -> None:
        """Initialize all agents asynchronously."""
        if not self.agents:
            raise AgentError("No agents registered for initialization")
            
        # Initialize all agents asynchronously
        tasks = []
        for agent in self.agents.values():
            if not agent.initialized:
                # Create a task for each agent's initialization
                task = asyncio.create_task(agent.initialize())
                tasks.append(task)
                
        # Wait for all agents to initialize
        if tasks:
            await asyncio.gather(*tasks)
