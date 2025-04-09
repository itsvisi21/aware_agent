from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field, validator

class KarakaRole(Enum):
    AGENT = "Agent"  # The doer of the action
    OBJECT = "Object"  # The thing being acted upon
    INSTRUMENT = "Instrument"  # The means by which the action is performed
    LOCATION = "Location"  # Where the action takes place
    SOURCE = "Source"  # The origin of the action
    DESTINATION = "Destination"  # The target of the action
    BENEFICIARY = "Beneficiary"  # Who benefits from the action
    TIME = "Time"  # When the action takes place
    MANNER = "Manner"  # How the action is performed
    CAUSE = "Cause"  # Why the action is performed
    PURPOSE = "Purpose"  # For what purpose the action is performed

class SemanticDimension(BaseModel):
    """Represents a semantic dimension of a context node."""
    domain: str
    role: str
    objective: str
    timeframe: str
    attributes: List[str] = Field(default_factory=list)
    relationships: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert the dimension to a dictionary representation."""
        return {
            "domain": self.domain,
            "role": self.role,
            "objective": self.objective,
            "timeframe": self.timeframe,
            "attributes": self.attributes,
            "relationships": self.relationships,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticDimension":
        """Create a dimension from a dictionary representation."""
        return cls(
            domain=data["domain"],
            role=data["role"],
            objective=data["objective"],
            timeframe=data["timeframe"],
            attributes=data.get("attributes", []),
            relationships=data.get("relationships", []),
            confidence=data.get("confidence", 0.0)
        )

class KarakaMapping(BaseModel):
    """Represents a mapping of karaka roles in a context node."""
    agent: str
    object: str
    instrument: str
    attributes: List[str] = Field(default_factory=list)
    relationships: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert the mapping to a dictionary representation."""
        return {
            "agent": self.agent,
            "object": self.object,
            "instrument": self.instrument,
            "attributes": self.attributes,
            "relationships": self.relationships,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KarakaMapping":
        """Create a mapping from a dictionary representation."""
        return cls(
            agent=data["agent"],
            object=data["object"],
            instrument=data["instrument"],
            attributes=data.get("attributes", []),
            relationships=data.get("relationships", []),
            confidence=data.get("confidence", 0.0)
        ) 