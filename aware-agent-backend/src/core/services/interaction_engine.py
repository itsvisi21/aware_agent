from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime

import numpy as np
from langchain.schema import HumanMessage, AIMessage
from src.core.services.semantic_abstraction import ContextNode, SemanticDimension, KarakaMapping


class ConversationState(Enum):
    """States of a conversation."""
    INITIAL = "initial"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class InteractionType(str, Enum):
    GENERAL = "general"
    CONVERSATIONAL = "conversational"
    VALIDATION = "validation"
    CLARIFICATION = "clarification"


@dataclass
class InteractionResponse:
    type: InteractionType
    content: Dict[str, Any]
    actions: List[Dict[str, Any]]
    feedback: Dict[str, Any]
    confidence: float
    semantic_context: Optional[Dict[str, Any]] = None
    status: str = "success"


@dataclass
class ConversationState:
    messages: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    context_tree: Dict[str, Any] = field(default_factory=dict)
    current_goal: str = field(default="")
    current_goals: List[str] = field(default_factory=list)
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_message(self, message: str):
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

    def update_context(self, context: Dict[str, Any]):
        self.context.update(context)
        self.updated_at = datetime.utcnow()

    def update_context_tree(self, context_tree: Dict[str, Any]):
        self.context_tree.update(context_tree)
        self.updated_at = datetime.utcnow()

    def set_current_goal(self, goal: str):
        self.current_goal = goal
        self.updated_at = datetime.utcnow()

    def add_goal(self, goal: str):
        self.current_goals.append(goal)
        self.updated_at = datetime.utcnow()

    def add_feedback(self, feedback: Dict[str, Any]):
        self.feedback_history.append(feedback)
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the conversation state to a dictionary for storage."""
        return {
            "messages": [
                {
                    "type": "human" if isinstance(msg, HumanMessage) else "ai",
                    "content": msg.content
                }
                for msg in self.messages
            ],
            "context": self.context,
            "context_tree": {
                "name": self.context_tree.name,
                "dimension": {
                    "domain": self.context_tree.dimension.domain,
                    "role": self.context_tree.dimension.role,
                    "objective": self.context_tree.dimension.objective,
                    "timeframe": self.context_tree.dimension.timeframe
                },
                "karaka": {
                    "agent": self.context_tree.karaka.agent,
                    "object": self.context_tree.karaka.object,
                    "instrument": self.context_tree.karaka.instrument
                }
            },
            "current_goal": self.current_goal,
            "current_goals": self.current_goals,
            "feedback_history": self.feedback_history,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationState':
        """Create a conversation state from a dictionary."""
        messages = []
        for msg_data in data.get("messages", []):
            if msg_data["type"] == "human":
                messages.append(HumanMessage(content=msg_data["content"]))
            else:
                messages.append(AIMessage(content=msg_data["content"]))

        context_tree_data = data.get("context_tree", {})
        context_tree = ContextNode(
            name=context_tree_data.get("name", "root"),
            dimension=SemanticDimension(
                domain=context_tree_data.get("dimension", {}).get("domain", ""),
                role=context_tree_data.get("dimension", {}).get("role", ""),
                objective=context_tree_data.get("dimension", {}).get("objective", ""),
                timeframe=context_tree_data.get("dimension", {}).get("timeframe", "")
            ),
            karaka=KarakaMapping(
                agent=context_tree_data.get("karaka", {}).get("agent", ""),
                object=context_tree_data.get("karaka", {}).get("object", ""),
                instrument=context_tree_data.get("karaka", {}).get("instrument", "")
            )
        )

        return cls(
            messages=messages,
            context=data.get("context", {}),
            context_tree=context_tree,
            current_goal=data.get("current_goal", ""),
            current_goals=data.get("current_goals", []),
            feedback_history=data.get("feedback_history", []),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat()))
        )


class PromptBuilder:
    """Builds prompts for different interaction types."""

    def build_goal_clarification_prompt(self, query: str, context: Dict) -> str:
        if not query:
            raise ValueError("Query cannot be empty")
            
        if not context:
            context = {}
            
        # Add a random component to ensure different prompts
        random_id = str(uuid.uuid4())[:8]
        
        # Include previous interaction if present
        previous_interaction = ""
        if "previous_interaction" in context:
            previous_interaction = f"\nPrevious Interaction: {context['previous_interaction']}"
        
        return f"""
        Query: {query}
        
        Context:
        Domain: {context.get('domain', 'general')}
        Role: {context.get('role', 'user')}
        Session ID: {random_id}{previous_interaction}
        """

    def build_response_synthesis_prompt(self, responses: List[Dict]) -> str:
        if not responses:
            raise ValueError("Responses cannot be empty")
            
        # Add a random component to ensure different prompts
        random_id = str(uuid.uuid4())[:8]
        
        response_texts = [r.get("content", {}).get("text", "") for r in responses]
        return f"""
        Synthesize the following responses:
        {chr(10).join(response_texts)}
        
        Session ID: {random_id}
        """

    def _format_message_history(self, messages: List[Any]) -> str:
        """Format message history into a readable string."""
        history = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history += f"Human: {msg.content}\n"
            else:
                history += f"Assistant: {msg.content}\n"
        return history


class ResponseTranslator:
    """Translates raw responses into structured formats."""

    def translate(self, text: str, format_type: str = "markdown") -> InteractionResponse:
        if not text:
            raise ValueError("Text cannot be empty")
            
        if format_type == "markdown":
            return InteractionResponse(
                type=InteractionType.GENERAL,
                content={"text": text},
                actions=[],
                feedback={},
                confidence=0.8,  # Set a reasonable default confidence
                semantic_context=None
            )
        elif format_type == "conversational":
            return InteractionResponse(
                type=InteractionType.CONVERSATIONAL,
                content={"text": text},
                actions=[],
                feedback={},
                confidence=0.8,  # Set a reasonable default confidence
                semantic_context=None
            )
        else:
            raise ValueError(f"Unsupported format type: {format_type}")


class FeedbackIntegrator:
    """Integrates user feedback into the interaction context."""

    def integrate(self, feedback: Dict, context: Dict) -> Dict:
        """Integrate feedback into the context."""
        if not feedback or not context:
            raise ValueError("Both feedback and context are required")

        updated_context = context.copy()

        # Add feedback to context
        updated_context["feedback"] = {
            "rating": feedback.get("rating", 0),
            "comments": feedback.get("comments", ""),
            "suggestions": feedback.get("suggestions", []),
            "areas_for_improvement": self._extract_improvement_areas(feedback)
        }

        # Update context based on feedback
        if feedback.get("rating", 0) >= 4:
            updated_context["successful_patterns"] = updated_context.get("successful_patterns", [])
            updated_context["successful_patterns"].append({
                "context": context,
                "feedback": feedback
            })

        return updated_context

    def _extract_improvement_areas(self, feedback: Dict[str, Any]) -> List[str]:
        """Extract areas for improvement from feedback."""
        areas = []

        if feedback.get("rating", 5) < 4:
            areas.append("general_quality")

        if feedback.get("comments"):
            if "technical" in feedback["comments"].lower():
                areas.append("technical_depth")
            if "explain" in feedback["comments"].lower():
                areas.append("clarity")
            if "example" in feedback["comments"].lower():
                areas.append("examples")

        return areas


class InteractionEngine:
    def __init__(self, llm: Any, memory_engine: Any):
        self.llm = llm
        self.memory_engine = memory_engine
        self.conversations: Dict[str, Dict] = {}
        self.prompt_builder = PromptBuilder()
        self.response_translator = ResponseTranslator()
        self.feedback_integrator = FeedbackIntegrator()

    def construct_prompt(self, query: str, context: Optional[Dict] = None) -> str:
        if not query:
            raise ValueError("Query cannot be empty")
            
        if not context:
            context = {}
            
        return self.prompt_builder.build_goal_clarification_prompt(query, context)

    def translate_response(self, response: str) -> InteractionResponse:
        if not response:
            raise ValueError("Response cannot be empty")
            
        return self.response_translator.translate(response)

    def integrate_feedback(self, feedback: Dict, context: Dict) -> Dict:
        if not feedback:
            raise ValueError("Feedback cannot be empty")
            
        if not context:
            raise ValueError("Context cannot be empty")
            
        return self.feedback_integrator.integrate(feedback, context)

    async def create_conversation(self, metadata: Dict) -> str:
        if not metadata:
            raise ValueError("Metadata cannot be empty")
            
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = {
            "metadata": metadata,
            "messages": [],
            "context": {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        return conversation_id

    async def process_message(self, conversation_id: str, message: str) -> InteractionResponse:
        if not conversation_id or conversation_id not in self.conversations:
            raise ValueError("Invalid conversation ID")
            
        if not message:
            raise ValueError("Message cannot be empty")
            
        conversation = self.conversations[conversation_id]
        conversation["messages"].append(message)
        conversation["updated_at"] = datetime.utcnow()
        
        response = await self.llm.agenerate([message])
        return self.translate_response(response.generations[0][0].text)

    async def get_conversation_history(self, conversation_id: str) -> List[str]:
        if not conversation_id or conversation_id not in self.conversations:
            raise ValueError("Invalid conversation ID")
            
        return self.conversations[conversation_id]["messages"]

    async def get_context_tree(self, conversation_id: str) -> Dict:
        if not conversation_id or conversation_id not in self.conversations:
            raise ValueError("Invalid conversation ID")
            
        return self.conversations[conversation_id]["context"]

    async def list_conversations(self) -> List[Dict]:
        return [
            {
                "id": conv_id,
                "metadata": conv["metadata"],
                "created_at": conv["created_at"],
                "updated_at": conv["updated_at"]
            }
            for conv_id, conv in self.conversations.items()
        ]

    async def get_agent_statuses(self, conversation_id: str) -> Dict[str, str]:
        if not conversation_id or conversation_id not in self.conversations:
            raise ValueError("Invalid conversation ID")
            
        return {
            "research": "active",
            "builder": "idle",
            "collaborator": "idle",
            "teacher": "idle",
            "explainer": "idle",
            "validator": "idle",
            "planner": "idle",
            "goal_alignment": "idle",
            "pruning": "idle",
            "context": "idle",
            "validation": "idle"
        }

    async def initialize_conversation(self, context: Dict, initial_goal: str) -> str:
        if not context:
            raise ValueError("Context cannot be empty")
            
        if not initial_goal:
            raise ValueError("Initial goal cannot be empty")
            
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = {
            "metadata": {
                "initial_goal": initial_goal,
                "context": context
            },
            "messages": [],
            "context": context,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        return conversation_id
