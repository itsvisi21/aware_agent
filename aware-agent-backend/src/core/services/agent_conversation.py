import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent roles in the conversation."""
    PLANNER = "planner"
    PRUNER = "pruner"
    ANALYZER = "analyzer"
    SYNTHESIZER = "synthesizer"


@dataclass
class AgentMessage:
    """Message from an agent in the conversation."""
    role: AgentRole
    content: str
    reasoning: str
    confidence: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationState:
    """State of the agent conversation."""
    goal: str
    context: Dict[str, Any]
    messages: List[AgentMessage]
    current_focus: Optional[str] = None
    confidence: float = 1.0
    is_active: bool = True


class AgentConversation:
    """Manages multi-agent conversations with self-awareness and goal anchoring."""

    def __init__(self):
        self.conversations: Dict[str, ConversationState] = {}
        self.log_dir = Path("conversation_logs")
        self.log_dir.mkdir(exist_ok=True)
        self.history = []

    def create_conversation(
            self,
            goal: str,
            initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new agent conversation."""
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.conversations[conversation_id] = ConversationState(
            goal=goal,
            context=initial_context or {},
            messages=[],
            current_focus=goal
        )

        return conversation_id

    def add_message(
            self,
            conversation_id: str,
            role: AgentRole,
            content: str,
            reasoning: str,
            confidence: float = 1.0,
            metadata: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Add a message to the conversation."""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        conversation = self.conversations[conversation_id]
        if not conversation.is_active:
            raise ValueError("Conversation is not active")

        message = AgentMessage(
            role=role,
            content=content,
            reasoning=reasoning,
            confidence=confidence,
            timestamp=datetime.now(),
            metadata=metadata
        )

        conversation.messages.append(message)
        self._update_conversation_state(conversation_id)

        return message

    def _update_conversation_state(self, conversation_id: str):
        """Update conversation state based on recent messages."""
        conversation = self.conversations[conversation_id]

        # Calculate overall confidence
        if conversation.messages:
            conversation.confidence = sum(
                msg.confidence for msg in conversation.messages
            ) / len(conversation.messages)

        # Update current focus based on recent messages
        recent_messages = conversation.messages[-3:]  # Last 3 messages
        if recent_messages:
            # Find most discussed topic
            topics = {}
            for msg in recent_messages:
                if msg.metadata and "topics" in msg.metadata:
                    for topic in msg.metadata["topics"]:
                        topics[topic] = topics.get(topic, 0) + 1

            if topics:
                conversation.current_focus = max(topics.items(), key=lambda x: x[1])[0]

    def get_conversation_state(self, conversation_id: str) -> ConversationState:
        """Get current state of a conversation."""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        return self.conversations[conversation_id]

    def end_conversation(self, conversation_id: str):
        """End a conversation and save logs."""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        conversation = self.conversations[conversation_id]
        conversation.is_active = False

        # Save conversation logs
        log_data = {
            "conversation_id": conversation_id,
            "goal": conversation.goal,
            "start_time": conversation.messages[0].timestamp.isoformat(),
            "end_time": datetime.now().isoformat(),
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "reasoning": msg.reasoning,
                    "confidence": msg.confidence,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in conversation.messages
            ],
            "final_state": {
                "current_focus": conversation.current_focus,
                "confidence": conversation.confidence
            }
        }

        log_file = self.log_dir / f"{conversation_id}.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

    def analyze_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Analyze conversation patterns and effectiveness."""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        conversation = self.conversations[conversation_id]

        # Calculate role distribution
        role_counts = {}
        for msg in conversation.messages:
            role_counts[msg.role] = role_counts.get(msg.role, 0) + 1

        # Calculate average confidence per role
        role_confidence = {}
        for role in AgentRole:
            role_messages = [msg for msg in conversation.messages if msg.role == role]
            if role_messages:
                role_confidence[role] = sum(
                    msg.confidence for msg in role_messages
                ) / len(role_messages)

        # Calculate goal alignment
        goal_alignment = 0.0
        if conversation.messages:
            goal_keywords = set(conversation.goal.lower().split())
            message_keywords = set()
            for msg in conversation.messages:
                message_keywords.update(msg.content.lower().split())

            if goal_keywords:
                goal_alignment = len(goal_keywords.intersection(message_keywords)) / len(goal_keywords)

        return {
            "role_distribution": {
                role.value: count for role, count in role_counts.items()
            },
            "role_confidence": {
                role.value: confidence for role, confidence in role_confidence.items()
            },
            "goal_alignment": goal_alignment,
            "message_count": len(conversation.messages),
            "average_confidence": conversation.confidence,
            "current_focus": conversation.current_focus
        }

    async def clear(self) -> None:
        """Clear the conversation history."""
        self.history = []
        logger.info("Conversation history cleared")


# Global conversation manager instance
agent_conversation = AgentConversation()
