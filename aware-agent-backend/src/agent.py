from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from pathlib import Path
import json
import os
from langchain_openai import OpenAI
from src.agent_orchestration import AgentOrchestrator
from src.agent_conversation import AgentConversation, AgentRole, AgentMessage
from src.semantic_abstraction import SemanticAbstractionLayer
from src.memory_engine import MemoryEngine
from src.execution_layer import ExecutionLayer

logger = logging.getLogger(__name__)

class Agent:
    """Main agent class that integrates orchestration and conversation layers with self-awareness."""
    
    def __init__(self):
        # Initialize LLM
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
        
        # Initialize components
        self.orchestration = AgentOrchestrator(llm=llm)
        self.conversation = AgentConversation()
        self.semantic_layer = SemanticAbstractionLayer()
        self.memory = MemoryEngine()
        self.execution = ExecutionLayer()
        self.workspace_dir = Path("workspace")
        self.workspace_dir.mkdir(exist_ok=True)
        
    async def process_input(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process input through the agent's cognitive pipeline."""
        try:
            # 1. Semantic Analysis
            semantic_result = await self.semantic_layer.analyze(input_text)
            
            # 2. Memory Integration
            relevant_memories = await self.memory.retrieve(semantic_result)
            semantic_result["memories"] = relevant_memories
            
            # 3. Create and Execute Plan
            plan = await self.orchestration.create_plan(semantic_result, context)
            execution_result = await self.orchestration.execute_plan(plan["plan_id"])
            
            # 4. Memory Update
            await self.memory.store({
                "input": input_text,
                "semantic_result": semantic_result,
                "execution_result": execution_result,
                "timestamp": datetime.now().isoformat()
            })
            
            # 5. Self-awareness check
            self._check_self_awareness(execution_result)
            
            return {
                "status": "success",
                "result": execution_result,
                "confidence": execution_result.get("confidence", 0.0),
                "semantic_context": semantic_result
            }
            
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _check_self_awareness(self, execution_result: Dict[str, Any]):
        """Check and update agent's self-awareness state."""
        # Analyze execution results
        performance_metrics = self._analyze_performance(execution_result)
        
        # Update self-awareness state
        self._update_self_awareness(performance_metrics)
        
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
    
    def _update_self_awareness(self, metrics: Dict[str, Any]):
        """Update agent's self-awareness based on performance metrics."""
        # Update internal state based on performance
        if metrics["success_rate"] < 0.8:
            logger.warning("Agent performance below threshold, triggering self-improvement")
            self._trigger_self_improvement()
    
    def _trigger_self_improvement(self):
        """Trigger self-improvement mechanisms."""
        # Analyze recent failures
        recent_failures = self.memory.retrieve_recent_failures()
        
        # Update strategies based on failures
        self._update_strategies(recent_failures)
        
        # Optimize performance
        self._optimize_performance()
    
    def _update_strategies(self, failures: List[Dict[str, Any]]):
        """Update agent strategies based on failure analysis."""
        for failure in failures:
            # Extract lessons learned
            lessons = self._extract_lessons(failure)
            
            # Update strategies
            self._apply_lessons(lessons)
    
    def _optimize_performance(self):
        """Optimize agent performance based on self-analysis."""
        # Analyze performance patterns
        patterns = self._analyze_performance_patterns()
        
        # Apply optimizations
        self._apply_optimizations(patterns)
    
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

# Global agent instance
agent = Agent() 