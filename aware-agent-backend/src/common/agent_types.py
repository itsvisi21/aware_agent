from enum import Enum
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, validator, Field


class AgentType(Enum):
    """Enum for different types of agents."""
    BASE = "base"
    RESEARCH = "research"
    TEACHER = "teacher"
    BUILDER = "builder"
    COLLABORATOR = "collaborator"
    EXPLAINER = "explainer"
    VALIDATOR = "validator"
    GOAL_ALIGNMENT = "goal_alignment"
    PLANNER = "planner"
    CONTEXT = "context"
    PLANNING = "planning"
    PRUNING = "pruning"
    VALIDATION = "validation"
    TEST = "test"  # Added for testing purposes


class AgentResponse(BaseModel):
    """Model for agent responses."""
    content: str
    reasoning: str
    confidence: float
    next_steps: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    agent_type: AgentType
    status: str = "success"

    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v

    def model_dump(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "next_steps": self.next_steps,
            "metadata": self.metadata,
            "status": self.status
        }
