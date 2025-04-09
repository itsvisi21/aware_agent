from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from src.core.services.semantic_abstraction import ContextNode, SemanticDimension, KarakaMapping
from src.core.services.memory_engine import MemoryEngine

class SemanticLayer(ABC):
    """Interface for semantic processing layer."""
    
    @abstractmethod
    async def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process text with semantic analysis.
        
        Args:
            text: The text to process
            context: Optional context information
            
        Returns:
            Dict containing semantic analysis results
        """
        pass 