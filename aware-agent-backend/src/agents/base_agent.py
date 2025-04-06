from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from ..services.database import DatabaseService
from ..services.semantic_understanding import SemanticUnderstandingService
from ..services.cache import CacheService

class BaseAgent(ABC):
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.state: Dict[str, Any] = {}
        self.conversation_history: list[Dict[str, Any]] = []
        self.db_service = DatabaseService()
        self.semantic_service = SemanticUnderstandingService()
        self.cache_service = CacheService()

    @abstractmethod
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message and return a response.
        This method should be implemented by specific agent types.
        """
        pass

    async def analyze_message(self, message: str) -> Dict[str, Any]:
        """
        Analyze a message using semantic understanding.
        """
        cache_key = f"semantic_analysis:{message}"
        return await self.cache_service.get_or_set(
            cache_key,
            lambda: self.semantic_service.analyze_message(message),
            ttl=300  # Cache semantic analysis for 5 minutes
        )

    async def update_state(self, new_state: Dict[str, Any]) -> None:
        """
        Update the agent's state with new information and persist it.
        """
        self.state.update(new_state)
        await self.db_service.save_agent_state(self.name, self.state)
        # Invalidate cached state
        await self.cache_service.delete(f"agent_state:{self.name}")

    async def add_to_history(self, message: Dict[str, Any]) -> None:
        """
        Add a message to the conversation history and persist it.
        """
        self.conversation_history.append(message)
        await self.db_service.save_conversation({
            'id': f"{self.name}_{len(self.conversation_history)}",
            'messages': self.conversation_history,
            'agent': self.name,
            'role': self.role
        })
        # Invalidate cached history
        await self.cache_service.delete(f"conversation_history:{self.name}")

    async def get_context(self) -> Dict[str, Any]:
        """
        Get the current context including state and recent history.
        """
        cache_key = f"agent_context:{self.name}"
        return await self.cache_service.get_or_set(
            cache_key,
            self._load_context,
            ttl=60  # Cache context for 1 minute
        )

    async def _load_context(self) -> Dict[str, Any]:
        """
        Load the context from database and current state.
        """
        # Load persisted state if available
        persisted_state = await self.db_service.get_agent_state(self.name)
        if persisted_state:
            self.state.update(persisted_state)

        return {
            'state': self.state,
            'recent_history': self.conversation_history[-5:] if self.conversation_history else [],
            'semantic_context': self.semantic_service.context_cache
        }

    async def reset(self) -> None:
        """
        Reset the agent's state and history.
        """
        self.state = {}
        self.conversation_history = []
        self.semantic_service = SemanticUnderstandingService()
        # Clear persisted data
        await self.db_service.save_agent_state(self.name, {})
        # Clear all cached data
        await self.cache_service.clear() 