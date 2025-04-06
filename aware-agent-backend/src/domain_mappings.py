from typing import Dict, Any, List, Optional
from enum import Enum
import json
from pathlib import Path
from dataclasses import dataclass
from src.karaka_mapper import KarakaRole

class DomainType(Enum):
    LEGAL = "legal"
    MEDICAL = "medical"
    PROGRAMMING = "programming"
    ACADEMIC = "academic"
    BUSINESS = "business"
    CUSTOM = "custom"

@dataclass
class DomainMapping:
    """Configuration for domain-specific Karaka mappings."""
    domain: DomainType
    role_patterns: Dict[KarakaRole, List[str]]
    entity_types: List[str]
    relationship_types: List[str]
    context_rules: Dict[str, Any]
    priority_rules: Dict[str, Any]

class DomainMapper:
    def __init__(self, mappings_dir: str = "mappings"):
        """Initialize domain mapper with custom mappings."""
        self.mappings_dir = Path(mappings_dir)
        self.mappings_dir.mkdir(exist_ok=True)
        self.domain_mappings: Dict[DomainType, DomainMapping] = {}
        self._load_default_mappings()
    
    def _load_default_mappings(self):
        """Load default mappings for standard domains."""
        # Legal Domain
        self.domain_mappings[DomainType.LEGAL] = DomainMapping(
            domain=DomainType.LEGAL,
            role_patterns={
                KarakaRole.AGENT: ["plaintiff", "defendant", "attorney", "judge"],
                KarakaRole.OBJECT: ["case", "evidence", "ruling", "motion"],
                KarakaRole.INSTRUMENT: ["statute", "precedent", "testimony"],
                KarakaRole.PURPOSE: ["justice", "compensation", "dismissal"]
            },
            entity_types=["legal_entity", "court", "document"],
            relationship_types=["represents", "challenges", "supports"],
            context_rules={
                "jurisdiction": "required",
                "case_type": "required",
                "filing_date": "optional"
            },
            priority_rules={
                "precedent": "high",
                "statute": "high",
                "testimony": "medium"
            }
        )
        
        # Medical Domain
        self.domain_mappings[DomainType.MEDICAL] = DomainMapping(
            domain=DomainType.MEDICAL,
            role_patterns={
                KarakaRole.AGENT: ["patient", "doctor", "nurse", "specialist"],
                KarakaRole.OBJECT: ["diagnosis", "treatment", "medication"],
                KarakaRole.INSTRUMENT: ["test", "procedure", "equipment"],
                KarakaRole.PURPOSE: ["recovery", "prevention", "management"]
            },
            entity_types=["medical_entity", "facility", "condition"],
            relationship_types=["treats", "diagnoses", "prescribes"],
            context_rules={
                "patient_history": "required",
                "current_condition": "required",
                "allergies": "required"
            },
            priority_rules={
                "critical_condition": "high",
                "routine_checkup": "low"
            }
        )
        
        # Programming Domain
        self.domain_mappings[DomainType.PROGRAMMING] = DomainMapping(
            domain=DomainType.PROGRAMMING,
            role_patterns={
                KarakaRole.AGENT: ["developer", "system", "user"],
                KarakaRole.OBJECT: ["code", "function", "module"],
                KarakaRole.INSTRUMENT: ["language", "framework", "tool"],
                KarakaRole.PURPOSE: ["implementation", "optimization", "debugging"]
            },
            entity_types=["code_entity", "dependency", "interface"],
            relationship_types=["imports", "extends", "implements"],
            context_rules={
                "language": "required",
                "framework": "optional",
                "dependencies": "required"
            },
            priority_rules={
                "core_functionality": "high",
                "edge_cases": "medium"
            }
        )
    
    def load_custom_mapping(self, domain_name: str, mapping_file: str) -> None:
        """Load custom domain mapping from file."""
        filepath = self.mappings_dir / mapping_file
        if not filepath.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
        
        with open(filepath, 'r') as f:
            mapping_data = json.load(f)
        
        domain = DomainType.CUSTOM
        self.domain_mappings[domain] = DomainMapping(
            domain=domain,
            role_patterns={
                KarakaRole[role]: patterns
                for role, patterns in mapping_data.get('role_patterns', {}).items()
            },
            entity_types=mapping_data.get('entity_types', []),
            relationship_types=mapping_data.get('relationship_types', []),
            context_rules=mapping_data.get('context_rules', {}),
            priority_rules=mapping_data.get('priority_rules', {})
        )
    
    def save_custom_mapping(self, domain_name: str, mapping: DomainMapping) -> None:
        """Save custom domain mapping to file."""
        mapping_data = {
            'domain': domain_name,
            'role_patterns': {
                role.name: patterns
                for role, patterns in mapping.role_patterns.items()
            },
            'entity_types': mapping.entity_types,
            'relationship_types': mapping.relationship_types,
            'context_rules': mapping.context_rules,
            'priority_rules': mapping.priority_rules
        }
        
        filepath = self.mappings_dir / f"{domain_name}_mapping.json"
        with open(filepath, 'w') as f:
            json.dump(mapping_data, f, indent=2)
    
    def get_mapping(self, domain: DomainType) -> Optional[DomainMapping]:
        """Get mapping for specified domain."""
        return self.domain_mappings.get(domain)
    
    def create_custom_mapping(
        self,
        domain_name: str,
        role_patterns: Dict[KarakaRole, List[str]],
        entity_types: List[str],
        relationship_types: List[str],
        context_rules: Dict[str, Any],
        priority_rules: Dict[str, Any]
    ) -> DomainMapping:
        """Create a new custom domain mapping."""
        mapping = DomainMapping(
            domain=DomainType.CUSTOM,
            role_patterns=role_patterns,
            entity_types=entity_types,
            relationship_types=relationship_types,
            context_rules=context_rules,
            priority_rules=priority_rules
        )
        
        self.domain_mappings[DomainType.CUSTOM] = mapping
        self.save_custom_mapping(domain_name, mapping)
        
        return mapping

# Global domain mapper instance
domain_mapper = DomainMapper() 