from typing import Dict, Any, List
import json
import os
from datetime import datetime

class DatabaseService:
    def __init__(self, storage_dir: str = "data"):
        self.storage_dir = storage_dir
        self._ensure_storage_dir()

    def _ensure_storage_dir(self):
        """Ensure the storage directory exists."""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def _get_file_path(self, collection: str, id: str) -> str:
        """Get the file path for a specific record."""
        return os.path.join(self.storage_dir, f"{collection}_{id}.json")

    async def save_conversation(self, conversation: Dict[str, Any]) -> str:
        """Save a conversation to the database."""
        conversation_id = conversation.get('id', str(datetime.now().timestamp()))
        file_path = self._get_file_path('conversations', conversation_id)
        
        with open(file_path, 'w') as f:
            json.dump(conversation, f)
        
        return conversation_id

    async def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Retrieve a conversation from the database."""
        file_path = self._get_file_path('conversations', conversation_id)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Conversation {conversation_id} not found")
        
        with open(file_path, 'r') as f:
            return json.load(f)

    async def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations in the database."""
        conversations = []
        for filename in os.listdir(self.storage_dir):
            if filename.startswith('conversations_'):
                with open(os.path.join(self.storage_dir, filename), 'r') as f:
                    conversations.append(json.load(f))
        return conversations

    async def save_agent_state(self, agent_type: str, state: Dict[str, Any]) -> None:
        """Save an agent's state to the database."""
        file_path = self._get_file_path('agent_states', agent_type)
        
        with open(file_path, 'w') as f:
            json.dump(state, f)

    async def get_agent_state(self, agent_type: str) -> Dict[str, Any]:
        """Retrieve an agent's state from the database."""
        file_path = self._get_file_path('agent_states', agent_type)
        
        if not os.path.exists(file_path):
            return {}
        
        with open(file_path, 'r') as f:
            return json.load(f)

    async def export_conversation(self, conversation_id: str) -> str:
        """Export a conversation to a JSON string."""
        conversation = await self.get_conversation(conversation_id)
        return json.dumps(conversation)

    async def import_conversation(self, conversation_json: str) -> str:
        """Import a conversation from a JSON string."""
        conversation = json.loads(conversation_json)
        return await self.save_conversation(conversation) 