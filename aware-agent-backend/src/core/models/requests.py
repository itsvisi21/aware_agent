from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ResearchRequest(BaseModel):
    """Request model for research operations."""
    query: str
    context: Optional[Dict[str, Any]] = None
    options: Optional[List[str]] = None
    max_results: Optional[int] = 10
    filters: Optional[Dict[str, Any]] = None 