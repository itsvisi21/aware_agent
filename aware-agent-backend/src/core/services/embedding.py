from typing import List
from abc import ABC, abstractmethod
from chromadb.utils.embedding_functions import EmbeddingFunction

class EmbeddingFunctionInterface(ABC):
    """Interface for embedding functions."""
    
    @abstractmethod
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for input texts.
        
        Args:
            input: List of input texts
            
        Returns:
            List of embeddings
        """
        pass 