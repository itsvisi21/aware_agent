import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional


class CollaborationRole(Enum):
    OWNER = "owner"
    CONTRIBUTOR = "contributor"
    VIEWER = "viewer"


@dataclass
class ResearchSession:
    """Represents a collaborative research session."""
    session_id: str
    owner_id: str
    title: str
    description: str
    created_at: datetime
    participants: Dict[str, CollaborationRole]
    research_tasks: List[str]
    shared_context: Dict[str, Any]
    is_active: bool = True


class CollaborationManager:
    def __init__(self):
        """Initialize collaboration manager."""
        self.sessions: Dict[str, ResearchSession] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    async def create_session(
            self,
            owner_id: str,
            title: str,
            description: str,
            initial_context: Optional[Dict[str, Any]] = None
    ) -> ResearchSession:
        """Create a new research session."""
        session_id = str(uuid.uuid4())
        session = ResearchSession(
            session_id=session_id,
            owner_id=owner_id,
            title=title,
            description=description,
            created_at=datetime.now(),
            participants={owner_id: CollaborationRole.OWNER},
            research_tasks=[],
            shared_context=initial_context or {},
            is_active=True
        )

        self.sessions[session_id] = session
        self._locks[session_id] = asyncio.Lock()

        return session

    async def add_participant(
            self,
            session_id: str,
            user_id: str,
            role: CollaborationRole
    ) -> bool:
        """Add a participant to a research session."""
        if session_id not in self.sessions:
            return False

        async with self._locks[session_id]:
            session = self.sessions[session_id]
            if not session.is_active:
                return False

            session.participants[user_id] = role
            return True

    async def remove_participant(
            self,
            session_id: str,
            user_id: str
    ) -> bool:
        """Remove a participant from a research session."""
        if session_id not in self.sessions:
            return False

        async with self._locks[session_id]:
            session = self.sessions[session_id]
            if not session.is_active:
                return False

            if user_id in session.participants:
                del session.participants[user_id]
                return True
            return False

    async def add_research_task(
            self,
            session_id: str,
            task_id: str,
            user_id: str
    ) -> bool:
        """Add a research task to the session."""
        if session_id not in self.sessions:
            return False

        async with self._locks[session_id]:
            session = self.sessions[session_id]
            if not session.is_active:
                return False

            if user_id not in session.participants:
                return False

            if task_id not in session.research_tasks:
                session.research_tasks.append(task_id)
                return True
            return False

    async def update_shared_context(
            self,
            session_id: str,
            user_id: str,
            context_updates: Dict[str, Any]
    ) -> bool:
        """Update shared context for the session."""
        if session_id not in self.sessions:
            return False

        async with self._locks[session_id]:
            session = self.sessions[session_id]
            if not session.is_active:
                return False

            if user_id not in session.participants:
                return False

            # Only owner and contributors can update context
            if session.participants[user_id] == CollaborationRole.VIEWER:
                return False

            session.shared_context.update(context_updates)
            return True

    async def get_session(
            self,
            session_id: str,
            user_id: str
    ) -> Optional[ResearchSession]:
        """Get session details if user is a participant."""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        if user_id not in session.participants:
            return None

        return session

    async def end_session(
            self,
            session_id: str,
            user_id: str
    ) -> bool:
        """End a research session (only owner can do this)."""
        if session_id not in self.sessions:
            return False

        async with self._locks[session_id]:
            session = self.sessions[session_id]
            if not session.is_active:
                return False

            if user_id != session.owner_id:
                return False

            session.is_active = False
            return True


# Global collaboration manager instance
collaboration_manager = CollaborationManager()
