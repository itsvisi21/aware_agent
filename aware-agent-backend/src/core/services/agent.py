import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from langchain_openai import OpenAI

from src.core.services.agent_conversation import AgentConversation
from src.core.services.agent_layer import AgentOrchestrator
from src.core.services.execution_layer import ExecutionLayer
from src.core.services.memory_engine import MemoryEngine
from src.common.agent_types import AgentType, AgentResponse
from src.common.exceptions import AgentError
from src.core.services.semantic_abstraction import ContextNode, SemanticAbstractionLayer

logger = logging.getLogger(__name__)


def create_agent(llm: Optional[Any] = None) -> 'Agent':
    """Create a new agent instance.
    
    Args:
        llm (Optional[Any], optional): The language model to use. Defaults to None.
    
    Returns:
        Agent: A new agent instance.
    """
    return Agent(llm)


class Agent:
    """Main agent class that integrates orchestration and conversation layers with self-awareness."""

    def __init__(self, llm: Optional[Any] = None):
        """Initialize agent components.
        
        Args:
            llm (Optional[Any], optional): The language model to use. Defaults to None.
        """
        self.logger = logging.getLogger(__name__)
        self.conversation_history = []
        self.self_awareness = {}
        
        # Initialize LLM
        self.llm = llm or OpenAI(temperature=0.7)
        
        # Initialize components
        self.orchestrator = AgentOrchestrator(llm=self.llm)
        self.semantic_layer = SemanticAbstractionLayer()
        
        # Create workspace directory
        self.workspace_dir = Path("workspace")
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Initialize memory with workspace storage path
        memory_path = self.workspace_dir / "memory"
        memory_path.mkdir(exist_ok=True)
        self.memory = MemoryEngine(storage_path=memory_path)
        
        # Initialize conversation and execution
        self.conversation = AgentConversation()
        self.execution = ExecutionLayer()
        
        # Initialize state
        self.state = {}
        self.performance_patterns = []

    async def reset(self):
        """Reset the agent's state and conversation history."""
        try:
            self.state = {}
            self.conversation_history = []
            self.performance_patterns = []
            
            # Reset memory and conversation
            if hasattr(self.memory, 'clear'):
                await self.memory.clear()
            if hasattr(self.conversation, 'clear'):
                await self.conversation.clear()
        except Exception as e:
            self.logger.error(f"Error resetting agent: {str(e)}")
            raise AgentError(f"Error resetting agent: {str(e)}")

    async def update_state(self, new_state: Dict[str, Any]):
        """Update the agent's state."""
        self.state.update(new_state)

    async def add_to_history(self, content: str, role: str = "user") -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append({
            "content": content,
            "role": role,
            "timestamp": datetime.now().isoformat()
        })

    async def get_context(self) -> Dict[str, Any]:
        """Get the current context."""
        if not self.state:
            raise AgentError("No context available")
        return self.state

    async def process_input(self, input_text: str) -> dict:
        """Process input text and return a response."""
        try:
            # Get current context
            context = await self.get_context()

            # Execute task
            execution_result = await self._execute_task(input_text, context)

            # Check self-awareness
            await self._check_self_awareness(execution_result)

            return execution_result
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            raise

    async def _check_self_awareness(self, execution_result: Dict[str, Any]):
        """Check and update agent's self-awareness."""
        # Analyze execution results
        performance_metrics = self._analyze_performance(execution_result)

        # Update self-awareness state
        await self._update_self_awareness(performance_metrics)

        # Log self-awareness state
        self._log_self_awareness(performance_metrics)

    def _analyze_performance(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze agent's performance metrics."""
        return {
            "execution_time": execution_result.get("execution_time", 0),
            "success_rate": execution_result.get("success_rate", 0),
            "confidence": execution_result.get("confidence", 0),
            "error_rate": execution_result.get("error_rate", 0)
        }

    async def _update_self_awareness(self, metrics: Dict[str, Any]):
        """Update agent's self-awareness based on performance metrics."""
        # Update internal state based on performance
        self.self_awareness = {
            "performance": metrics["success_rate"],
            "confidence": metrics["confidence"],
            "stability": 1 - metrics["error_rate"]
        }
        
        if metrics["success_rate"] < 0.8:
            logger.warning("Agent performance below threshold, triggering self-improvement")
            await self._trigger_self_improvement()

    async def _trigger_self_improvement(self):
        """Trigger self-improvement mechanisms."""
        # Analyze recent failures
        recent_failures = await self.memory.retrieve_recent_failures()

        # Update strategies based on failures
        await self._update_strategies(recent_failures)

        # Optimize performance
        await self._optimize_performance()

    async def _update_strategies(self, failures: List[Dict[str, Any]]):
        """Update agent strategies based on failure analysis."""
        all_lessons = []
        for failure in failures:
            # Extract lessons learned
            lessons = self._extract_lessons(failure)
            all_lessons.extend(lessons)

        # Update strategies once with all lessons
        await self._apply_lessons(all_lessons)

    async def _optimize_performance(self):
        """Optimize agent performance based on self-analysis."""
        # Analyze performance patterns
        patterns = self._analyze_performance_patterns()

        # Apply optimizations
        await self._apply_optimizations(patterns)

    def _log_self_awareness(self, metrics: Dict[str, Any]):
        """Log self-awareness state and metrics."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "state": {
                "confidence": metrics["confidence"],
                "performance": metrics["success_rate"],
                "stability": 1 - metrics["error_rate"]
            }
        }

        log_file = self.workspace_dir / "self_awareness_logs.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_data) + "\n")

    def _analyze_performance_patterns(self) -> List[str]:
        """Analyze performance patterns."""
        # Basic patterns to monitor
        self.performance_patterns = [
            "category_distribution",
            "success_rate",
            "error_patterns",
            "response_time",
            "confidence_trends"
        ]
        return self.performance_patterns

    def _apply_optimizations(self, patterns: List[str]):
        """Apply optimizations based on performance patterns."""
        for pattern in patterns:
            if pattern == "category_distribution":
                self._optimize_category_distribution()
            elif pattern == "success_rate":
                self._optimize_success_rate()
            elif pattern == "error_patterns":
                self._optimize_error_handling()
            elif pattern == "response_time":
                self._optimize_response_time()
            elif pattern == "confidence_trends":
                self._optimize_confidence()

    def _optimize_category_distribution(self):
        """Optimize category distribution."""
        pass

    def _optimize_success_rate(self):
        """Optimize success rate."""
        pass

    def _optimize_error_handling(self):
        """Optimize error handling."""
        pass

    def _optimize_response_time(self):
        """Optimize response time."""
        pass

    def _optimize_confidence(self):
        """Optimize confidence levels."""
        pass

    async def log_self_awareness(self) -> None:
        """Log self-awareness metrics."""
        try:
            log_file = Path("logs/agent_self_awareness.log")
            log_file.parent.mkdir(exist_ok=True)
            
            with open(log_file, "a") as f:
                f.write(json.dumps(self.self_awareness) + "\n")
        except Exception as e:
            logger.error(f"Error logging self-awareness: {str(e)}")
