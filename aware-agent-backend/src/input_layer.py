from typing import Dict, Any, Optional
from pydantic import BaseModel

class UserQuery(BaseModel):
    text: str
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class InputLayer:
    def __init__(self):
        self.context_memory = {}
        
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> UserQuery:
        """Process user input and prepare it for the semantic abstraction layer."""
        return UserQuery(
            text=query,
            context=context or self.context_memory,
            metadata={"timestamp": "current_time"}  # TODO: Add actual timestamp
        )
    
    def load_context(self, context_id: str) -> Dict[str, Any]:
        """Load previous context from memory."""
        return self.context_memory.get(context_id, {})
    
    def save_context(self, context_id: str, context: Dict[str, Any]):
        """Save context to memory."""
        self.context_memory[context_id] = context 