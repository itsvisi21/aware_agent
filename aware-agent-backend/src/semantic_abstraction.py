from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from scipy.spatial.distance import cosine
import spacy
import re
from datetime import datetime
from dateutil import parser
import pytz
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from karaka_mapper import KarakaMapper, KarakaRole
from domain_mappings import DomainMapper
import logging
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Load spaCy model for NLP processing
nlp = spacy.load("en_core_web_sm")

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class SemanticRoleType(str, Enum):
    AGENT = "Agent"
    OBJECT = "Object"
    INSTRUMENT = "Instrument"
    LOCATION = "Location"
    TIME = "Time"
    PURPOSE = "Purpose"
    SOURCE = "Source"
    DESTINATION = "Destination"
    BENEFICIARY = "Beneficiary"
    CAUSE = "Cause"
    MANNER = "Manner"
    CONDITION = "Condition"
    CONCESSION = "Concession"
    COMPARISON = "Comparison"
    EXTENT = "Extent"
    FREQUENCY = "Frequency"
    DEGREE = "Degree"
    MODALITY = "Modality"
    NEGATION = "Negation"
    QUANTITY = "Quantity"
    QUALITY = "Quality"
    STATE = "State"
    TRANSITION = "Transition"
    RESULT = "Result"
    MEANS = "Means"
    ACCOMPANIMENT = "Accompaniment"
    POSSESSION = "Possession"
    REFERENCE = "Reference"
    TOPIC = "Topic"
    FOCUS = "Focus"
    BACKGROUND = "Background"
    EVALUATION = "Evaluation"
    EVIDENCE = "Evidence"
    JUSTIFICATION = "Justification"
    EXPLANATION = "Explanation"
    EXAMPLE = "Example"
    CONTRAST = "Contrast"
    ADDITION = "Addition"
    ALTERNATIVE = "Alternative"
    CONSEQUENCE = "Consequence"
    METHOD = "Method"
    STAKEHOLDER = "Stakeholder"
    GOAL = "Goal"
    STRATEGY = "Strategy"
    PARTY = "Party"
    CLAIM = "Claim"
    PRACTITIONER = "Practitioner"
    PATIENT = "Patient"
    TREATMENT = "Treatment"

class SemanticRole(BaseModel):
    role_type: SemanticRoleType
    entity: str
    confidence: float
    attributes: Dict[str, Any] = {}
    relationships: List[Dict[str, Any]] = []
    sub_roles: List['SemanticRole'] = []
    parent_role: Optional['SemanticRole'] = None
    temporal_scope: Optional[Dict[str, Any]] = None
    spatial_scope: Optional[Dict[str, Any]] = None
    modality_scope: Optional[Dict[str, Any]] = None
    evidence_scope: Optional[Dict[str, Any]] = None
    justification_scope: Optional[Dict[str, Any]] = None
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValidationError("Confidence must be between 0 and 1")
        return v
    
    @validator('role_type')
    def validate_role_type(cls, v):
        if v not in SemanticRoleType:
            raise ValidationError(f"Invalid role type: {v}")
        return v
    
    def add_sub_role(self, sub_role: 'SemanticRole'):
        """Add a sub-role to this role."""
        sub_role.parent_role = self
        self.sub_roles.append(sub_role)
    
    def get_role_hierarchy(self) -> List['SemanticRole']:
        """Get the complete role hierarchy starting from this role."""
        hierarchy = [self]
        for sub_role in self.sub_roles:
            hierarchy.extend(sub_role.get_role_hierarchy())
        return hierarchy
    
    def get_temporal_scope(self) -> Dict[str, Any]:
        """Get the temporal scope of this role."""
        if self.temporal_scope:
            return self.temporal_scope
        if self.parent_role:
            return self.parent_role.get_temporal_scope()
        return {}
    
    def get_spatial_scope(self) -> Dict[str, Any]:
        """Get the spatial scope of this role."""
        if self.spatial_scope:
            return self.spatial_scope
        if self.parent_role:
            return self.parent_role.get_spatial_scope()
        return {}
    
    def get_modality_scope(self) -> Dict[str, Any]:
        """Get the modality scope of this role."""
        if self.modality_scope:
            return self.modality_scope
        if self.parent_role:
            return self.parent_role.get_modality_scope()
        return {}
    
    def get_evidence_scope(self) -> Dict[str, Any]:
        """Get the evidence scope of this role."""
        if self.evidence_scope:
            return self.evidence_scope
        if self.parent_role:
            return self.parent_role.get_evidence_scope()
        return {}
    
    def get_justification_scope(self) -> Dict[str, Any]:
        """Get the justification scope of this role."""
        if self.justification_scope:
            return self.justification_scope
        if self.parent_role:
            return self.parent_role.get_justification_scope()
        return {}
    
    def transform_role(self, new_role_type: SemanticRoleType, 
                      confidence_adjustment: float = 0.0) -> 'SemanticRole':
        """Transform this role into a new role type with adjusted confidence."""
        if not 0 <= confidence_adjustment <= 1:
            raise ValidationError("Confidence adjustment must be between 0 and 1")
        
        new_confidence = max(0.0, min(1.0, self.confidence + confidence_adjustment))
        
        return SemanticRole(
            role_type=new_role_type,
            entity=self.entity,
            confidence=new_confidence,
            attributes=self.attributes.copy(),
            relationships=self.relationships.copy(),
            sub_roles=self.sub_roles.copy(),
            parent_role=self.parent_role,
            temporal_scope=self.temporal_scope.copy() if self.temporal_scope else None,
            spatial_scope=self.spatial_scope.copy() if self.spatial_scope else None,
            modality_scope=self.modality_scope.copy() if self.modality_scope else None,
            evidence_scope=self.evidence_scope.copy() if self.evidence_scope else None,
            justification_scope=self.justification_scope.copy() if self.justification_scope else None
        )
    
    def merge_with(self, other_role: 'SemanticRole', 
                  merge_strategy: str = "weighted") -> 'SemanticRole':
        """Merge this role with another role using specified strategy."""
        if not isinstance(other_role, SemanticRole):
            raise ValidationError("Can only merge with another SemanticRole")
        
        if merge_strategy not in ["weighted", "max", "min", "average"]:
            raise ValidationError("Invalid merge strategy")
        
        # Calculate merged confidence based on strategy
        if merge_strategy == "weighted":
            total_confidence = self.confidence + other_role.confidence
            self_weight = self.confidence / total_confidence
            other_weight = other_role.confidence / total_confidence
            merged_confidence = (self.confidence * self_weight + 
                               other_role.confidence * other_weight)
        elif merge_strategy == "max":
            merged_confidence = max(self.confidence, other_role.confidence)
        elif merge_strategy == "min":
            merged_confidence = min(self.confidence, other_role.confidence)
        else:  # average
            merged_confidence = (self.confidence + other_role.confidence) / 2
        
        # Merge attributes
        merged_attributes = self.attributes.copy()
        for key, value in other_role.attributes.items():
            if key in merged_attributes:
                if isinstance(value, (int, float)):
                    merged_attributes[key] = (merged_attributes[key] + value) / 2
                elif isinstance(value, list):
                    merged_attributes[key] = list(set(merged_attributes[key] + value))
                elif isinstance(value, dict):
                    merged_attributes[key].update(value)
            else:
                merged_attributes[key] = value
        
        # Merge relationships
        merged_relationships = self.relationships.copy()
        for rel in other_role.relationships:
            if not any(r["target_entity"] == rel["target_entity"] 
                      and r["relationship"] == rel["relationship"] 
                      for r in merged_relationships):
                merged_relationships.append(rel)
        
        # Merge sub-roles
        merged_sub_roles = self.sub_roles.copy()
        for sub_role in other_role.sub_roles:
            if not any(sr.entity == sub_role.entity for sr in merged_sub_roles):
                merged_sub_roles.append(sub_role)
        
        # Merge scopes
        def merge_scopes(scope1: Optional[Dict[str, Any]], 
                        scope2: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not scope1 and not scope2:
                return None
            if not scope1:
                return scope2.copy()
            if not scope2:
                return scope1.copy()
            
            merged = scope1.copy()
            for key, value in scope2.items():
                if key in merged:
                    if isinstance(value, (int, float)):
                        merged[key] = (merged[key] + value) / 2
                    elif isinstance(value, list):
                        merged[key] = list(set(merged[key] + value))
                    elif isinstance(value, dict):
                        merged[key].update(value)
                else:
                    merged[key] = value
            return merged
        
        return SemanticRole(
            role_type=self.role_type,  # Keep original role type
            entity=self.entity,  # Keep original entity
            confidence=merged_confidence,
            attributes=merged_attributes,
            relationships=merged_relationships,
            sub_roles=merged_sub_roles,
            parent_role=self.parent_role,  # Keep original parent
            temporal_scope=merge_scopes(self.temporal_scope, other_role.temporal_scope),
            spatial_scope=merge_scopes(self.spatial_scope, other_role.spatial_scope),
            modality_scope=merge_scopes(self.modality_scope, other_role.modality_scope),
            evidence_scope=merge_scopes(self.evidence_scope, other_role.evidence_scope),
            justification_scope=merge_scopes(self.justification_scope, other_role.justification_scope)
        )
    
    def validate_hierarchy(self) -> List[str]:
        """Validate the role hierarchy and return any issues found."""
        issues = []
        
        # Check for circular references
        visited = set()
        current = self
        while current:
            if current.entity in visited:
                issues.append(f"Circular reference detected in hierarchy at {current.entity}")
                break
            visited.add(current.entity)
            current = current.parent_role
        
        # Check for valid role type transitions
        valid_transitions = {
            SemanticRoleType.AGENT: [SemanticRoleType.OBJECT, SemanticRoleType.INSTRUMENT],
            SemanticRoleType.OBJECT: [SemanticRoleType.PURPOSE, SemanticRoleType.DESTINATION],
            SemanticRoleType.INSTRUMENT: [SemanticRoleType.MEANS, SemanticRoleType.METHOD],
            SemanticRoleType.PURPOSE: [SemanticRoleType.GOAL, SemanticRoleType.INTENT],
            SemanticRoleType.TIME: [SemanticRoleType.DURATION, SemanticRoleType.FREQUENCY],
            SemanticRoleType.LOCATION: [SemanticRoleType.SOURCE, SemanticRoleType.DESTINATION]
        }
        
        if self.parent_role:
            if (self.parent_role.role_type in valid_transitions and 
                self.role_type not in valid_transitions[self.parent_role.role_type]):
                issues.append(f"Invalid role type transition: {self.parent_role.role_type} -> {self.role_type}")
        
        # Check for valid scope inheritance
        if self.parent_role:
            for scope_type in ["temporal", "spatial", "modality", "evidence", "justification"]:
                parent_scope = getattr(self.parent_role, f"{scope_type}_scope")
                child_scope = getattr(self, f"{scope_type}_scope")
                if parent_scope and child_scope:
                    for key in child_scope:
                        if key in parent_scope and child_scope[key] != parent_scope[key]:
                            issues.append(f"Inconsistent {scope_type} scope for key {key}")
        
        # Check sub-roles recursively
        for sub_role in self.sub_roles:
            issues.extend(sub_role.validate_hierarchy())
        
        return issues
    
    def split_role(self, split_criteria: Dict[str, Any]) -> List['SemanticRole']:
        """Split this role into multiple roles based on specified criteria."""
        if not split_criteria:
            raise ValidationError("Split criteria must be provided")
        
        split_roles = []
        
        # Split based on attributes
        if "attributes" in split_criteria:
            for attr_name, attr_value in self.attributes.items():
                if isinstance(attr_value, list) and len(attr_value) > 1:
                    for value in attr_value:
                        new_attributes = self.attributes.copy()
                        new_attributes[attr_name] = [value]
                        split_roles.append(
                            SemanticRole(
                                role_type=self.role_type,
                                entity=f"{self.entity}_{value}",
                                confidence=self.confidence * 0.8,  # Reduce confidence for splits
                                attributes=new_attributes,
                                relationships=self.relationships.copy(),
                                sub_roles=self.sub_roles.copy(),
                                parent_role=self.parent_role,
                                temporal_scope=self.temporal_scope.copy() if self.temporal_scope else None,
                                spatial_scope=self.spatial_scope.copy() if self.spatial_scope else None,
                                modality_scope=self.modality_scope.copy() if self.modality_scope else None,
                                evidence_scope=self.evidence_scope.copy() if self.evidence_scope else None,
                                justification_scope=self.justification_scope.copy() if self.justification_scope else None
                            )
                        )
        
        # Split based on relationships
        if "relationships" in split_criteria:
            for rel in self.relationships:
                if rel.get("confidence", 0.0) > split_criteria.get("min_confidence", 0.5):
                    new_relationships = [rel]
                    split_roles.append(
                        SemanticRole(
                            role_type=self.role_type,
                            entity=f"{self.entity}_{rel['target_entity']}",
                            confidence=self.confidence * rel.get("confidence", 0.5),
                            attributes=self.attributes.copy(),
                            relationships=new_relationships,
                            sub_roles=self.sub_roles.copy(),
                            parent_role=self.parent_role,
                            temporal_scope=self.temporal_scope.copy() if self.temporal_scope else None,
                            spatial_scope=self.spatial_scope.copy() if self.spatial_scope else None,
                            modality_scope=self.modality_scope.copy() if self.modality_scope else None,
                            evidence_scope=self.evidence_scope.copy() if self.evidence_scope else None,
                            justification_scope=self.justification_scope.copy() if self.justification_scope else None
                        )
                    )
        
        # Split based on scopes
        if "scopes" in split_criteria:
            for scope_type in ["temporal", "spatial", "modality", "evidence", "justification"]:
                scope = getattr(self, f"{scope_type}_scope")
                if scope and isinstance(scope, dict):
                    for key, value in scope.items():
                        if isinstance(value, list) and len(value) > 1:
                            new_scope = {key: [v] for v in value}
                            split_roles.append(
                                SemanticRole(
                                    role_type=self.role_type,
                                    entity=f"{self.entity}_{key}",
                                    confidence=self.confidence * 0.8,
                                    attributes=self.attributes.copy(),
                                    relationships=self.relationships.copy(),
                                    sub_roles=self.sub_roles.copy(),
                                    parent_role=self.parent_role,
                                    **{f"{scope_type}_scope": new_scope}
                                )
                            )
        
        return split_roles if split_roles else [self]
    
    def validate_domain_rules(self, domain: str) -> List[str]:
        """Validate role against domain-specific rules."""
        issues = []
        
        # Domain-specific validation rules
        domain_rules = {
            "technical": {
                "required_attributes": ["complexity", "dependencies"],
                "valid_role_types": [SemanticRoleType.AGENT, SemanticRoleType.OBJECT, 
                                   SemanticRoleType.INSTRUMENT, SemanticRoleType.PURPOSE],
                "min_confidence": 0.7
            },
            "academic": {
                "required_attributes": ["citation", "field", "methodology"],
                "valid_role_types": [SemanticRoleType.AGENT, SemanticRoleType.OBJECT,
                                   SemanticRoleType.EVIDENCE, SemanticRoleType.JUSTIFICATION],
                "min_confidence": 0.6
            },
            "business": {
                "required_attributes": ["stakeholder", "impact", "timeline"],
                "valid_role_types": [SemanticRoleType.AGENT, SemanticRoleType.OBJECT,
                                   SemanticRoleType.PURPOSE, SemanticRoleType.RESULT],
                "min_confidence": 0.5
            }
        }
        
        if domain not in domain_rules:
            raise ValidationError(f"Unknown domain: {domain}")
        
        rules = domain_rules[domain]
        
        # Check required attributes
        for attr in rules["required_attributes"]:
            if attr not in self.attributes:
                issues.append(f"Missing required attribute for {domain} domain: {attr}")
        
        # Check valid role types
        if self.role_type not in rules["valid_role_types"]:
            issues.append(f"Invalid role type for {domain} domain: {self.role_type}")
        
        # Check minimum confidence
        if self.confidence < rules["min_confidence"]:
            issues.append(f"Confidence below minimum threshold for {domain} domain")
        
        # Domain-specific scope validation
        if domain == "technical":
            if self.temporal_scope and "deadline" not in self.temporal_scope:
                issues.append("Missing deadline in temporal scope for technical domain")
        
        elif domain == "academic":
            if not self.evidence_scope:
                issues.append("Missing evidence scope for academic domain")
            if not self.justification_scope:
                issues.append("Missing justification scope for academic domain")
        
        elif domain == "business":
            if not self.spatial_scope or "location" not in self.spatial_scope:
                issues.append("Missing location in spatial scope for business domain")
        
        return issues
    
    def resolve_conflicts(self, other_role: 'SemanticRole', 
                         resolution_strategy: str = "priority") -> 'SemanticRole':
        """Resolve conflicts between this role and another role."""
        if not isinstance(other_role, SemanticRole):
            raise ValidationError("Can only resolve conflicts with another SemanticRole")
        
        if resolution_strategy not in ["priority", "merge", "split", "context"]:
            raise ValidationError("Invalid resolution strategy")
        
        # Identify conflicts
        conflicts = self._identify_conflicts(other_role)
        if not conflicts:
            return self
        
        # Apply resolution strategy
        if resolution_strategy == "priority":
            return self._resolve_by_priority(other_role, conflicts)
        elif resolution_strategy == "merge":
            return self.merge_with(other_role, "weighted")
        elif resolution_strategy == "split":
            return self._resolve_by_splitting(other_role, conflicts)
        else:  # context
            return self._resolve_by_context(other_role, conflicts)
    
    def _identify_conflicts(self, other_role: 'SemanticRole') -> List[Dict[str, Any]]:
        """Identify conflicts between this role and another role."""
        conflicts = []
        
        # Check role type conflicts
        if self.role_type != other_role.role_type:
            conflicts.append({
                "type": "role_type",
                "this": self.role_type,
                "other": other_role.role_type
            })
        
        # Check attribute conflicts
        for attr, value in self.attributes.items():
            if attr in other_role.attributes:
                other_value = other_role.attributes[attr]
                if value != other_value:
                    conflicts.append({
                        "type": "attribute",
                        "attribute": attr,
                        "this": value,
                        "other": other_value
                    })
        
        # Check relationship conflicts
        for rel in self.relationships:
            for other_rel in other_role.relationships:
                if (rel["target_entity"] == other_rel["target_entity"] and 
                    rel["relationship"] != other_rel["relationship"]):
                    conflicts.append({
                        "type": "relationship",
                        "target": rel["target_entity"],
                        "this": rel["relationship"],
                        "other": other_rel["relationship"]
                    })
        
        # Check scope conflicts
        for scope_type in ["temporal", "spatial", "modality", "evidence", "justification"]:
            this_scope = getattr(self, f"{scope_type}_scope")
            other_scope = getattr(other_role, f"{scope_type}_scope")
            if this_scope and other_scope and this_scope != other_scope:
                conflicts.append({
                    "type": "scope",
                    "scope": scope_type,
                    "this": this_scope,
                    "other": other_scope
                })
        
        return conflicts
    
    def _resolve_by_priority(self, other_role: 'SemanticRole', 
                           conflicts: List[Dict[str, Any]]) -> 'SemanticRole':
        """Resolve conflicts by prioritizing this role over the other."""
        resolved_role = self.copy()
        
        for conflict in conflicts:
            if conflict["type"] == "attribute":
                # Keep this role's attribute if confidence is higher
                if self.confidence > other_role.confidence:
                    resolved_role.attributes[conflict["attribute"]] = conflict["this"]
            
            elif conflict["type"] == "relationship":
                # Keep this role's relationship if confidence is higher
                if self.confidence > other_role.confidence:
                    resolved_role.relationships = [
                        rel for rel in resolved_role.relationships
                        if rel["target_entity"] != conflict["target"] or 
                           rel["relationship"] == conflict["this"]
                    ]
            
            elif conflict["type"] == "scope":
                # Keep this role's scope if confidence is higher
                if self.confidence > other_role.confidence:
                    setattr(resolved_role, f"{conflict['scope']}_scope", conflict["this"])
        
        return resolved_role
    
    def _resolve_by_splitting(self, other_role: 'SemanticRole', 
                            conflicts: List[Dict[str, Any]]) -> 'SemanticRole':
        """Resolve conflicts by splitting the role based on conflicts."""
        split_criteria = {
            "attributes": {},
            "relationships": {"min_confidence": 0.5},
            "scopes": {}
        }
        
        for conflict in conflicts:
            if conflict["type"] == "attribute":
                split_criteria["attributes"][conflict["attribute"]] = [
                    conflict["this"],
                    conflict["other"]
                ]
            
            elif conflict["type"] == "scope":
                split_criteria["scopes"][conflict["scope"]] = [
                    conflict["this"],
                    conflict["other"]
                ]
        
        return self.split_role(split_criteria)[0]  # Return first split role
    
    def _resolve_by_context(self, other_role: 'SemanticRole', 
                          conflicts: List[Dict[str, Any]]) -> 'SemanticRole':
        """Resolve conflicts by considering contextual information."""
        resolved_role = self.copy()
        
        for conflict in conflicts:
            if conflict["type"] == "attribute":
                # Use context to determine which attribute value is more relevant
                this_context = self._get_attribute_context(conflict["attribute"])
                other_context = other_role._get_attribute_context(conflict["attribute"])
                
                if this_context > other_context:
                    resolved_role.attributes[conflict["attribute"]] = conflict["this"]
                else:
                    resolved_role.attributes[conflict["attribute"]] = conflict["other"]
            
            elif conflict["type"] == "relationship":
                # Use context to determine which relationship is more relevant
                this_context = self._get_relationship_context(conflict["target"])
                other_context = other_role._get_relationship_context(conflict["target"])
                
                if this_context > other_context:
                    resolved_role.relationships = [
                        rel for rel in resolved_role.relationships
                        if rel["target_entity"] != conflict["target"] or 
                           rel["relationship"] == conflict["this"]
                    ]
                else:
                    resolved_role.relationships = [
                        rel for rel in resolved_role.relationships
                        if rel["target_entity"] != conflict["target"] or 
                           rel["relationship"] == conflict["other"]
                    ]
        
        return resolved_role
    
    def _get_attribute_context(self, attribute: str) -> float:
        """Get contextual relevance score for an attribute."""
        # Implement context scoring based on role type, relationships, and scopes
        score = 0.0
        
        # Role type relevance
        role_type_weights = {
            SemanticRoleType.AGENT: 1.0,
            SemanticRoleType.OBJECT: 0.8,
            SemanticRoleType.INSTRUMENT: 0.6
        }
        score += role_type_weights.get(self.role_type, 0.5)
        
        # Relationship relevance
        for rel in self.relationships:
            score += rel.get("confidence", 0.0) * 0.5
        
        # Scope relevance
        for scope_type in ["temporal", "spatial", "modality"]:
            scope = getattr(self, f"{scope_type}_scope")
            if scope:
                score += 0.3
        
        return score
    
    def _get_relationship_context(self, target: str) -> float:
        """Get contextual relevance score for a relationship."""
        # Implement context scoring based on relationship strength and role types
        score = 0.0
        
        for rel in self.relationships:
            if rel["target_entity"] == target:
                score += rel.get("confidence", 0.0)
                
                # Role type compatibility
                role_type_weights = {
                    (SemanticRoleType.AGENT, SemanticRoleType.OBJECT): 1.0,
                    (SemanticRoleType.INSTRUMENT, SemanticRoleType.OBJECT): 0.8,
                    (SemanticRoleType.AGENT, SemanticRoleType.INSTRUMENT): 0.6
                }
                score += role_type_weights.get(
                    (self.role_type, rel.get("target_role")), 0.5
                )
        
        return score
    
    def transform_for_domain(self, domain: str, 
                           transformation_rules: Optional[Dict[str, Any]] = None) -> 'SemanticRole':
        """Transform role according to domain-specific rules."""
        if not transformation_rules:
            transformation_rules = self._get_default_transformation_rules(domain)
        
        transformed_role = self.copy()
        
        # Apply domain-specific transformations
        if domain == "technical":
            transformed_role = self._transform_technical_role(transformed_role, transformation_rules)
        elif domain == "academic":
            transformed_role = self._transform_academic_role(transformed_role, transformation_rules)
        elif domain == "business":
            transformed_role = self._transform_business_role(transformed_role, transformation_rules)
        elif domain == "legal":
            transformed_role = self._transform_legal_role(transformed_role, transformation_rules)
        elif domain == "medical":
            transformed_role = self._transform_medical_role(transformed_role, transformation_rules)
        
        return transformed_role
    
    def _get_default_transformation_rules(self, domain: str) -> Dict[str, Any]:
        """Get default transformation rules for a domain."""
        rules = {
            "technical": {
                "role_mappings": {
                    SemanticRoleType.AGENT: SemanticRoleType.INSTRUMENT,
                    SemanticRoleType.OBJECT: SemanticRoleType.PURPOSE,
                    SemanticRoleType.INSTRUMENT: SemanticRoleType.MEANS
                },
                "attribute_rules": {
                    "complexity": lambda x: "high" if x > 0.7 else "medium" if x > 0.3 else "low",
                    "dependencies": lambda x: sorted(x, key=lambda d: d.get("priority", 0), reverse=True)
                },
                "scope_rules": {
                    "temporal": lambda x: {"deadline": x.get("deadline", "ASAP")},
                    "spatial": lambda x: {"environment": x.get("environment", "production")}
                }
            },
            "academic": {
                "role_mappings": {
                    SemanticRoleType.AGENT: SemanticRoleType.SOURCE,
                    SemanticRoleType.OBJECT: SemanticRoleType.TOPIC,
                    SemanticRoleType.INSTRUMENT: SemanticRoleType.METHOD
                },
                "attribute_rules": {
                    "citation": lambda x: {"format": "APA", "year": x.get("year", datetime.now().year)},
                    "methodology": lambda x: x.get("type", "qualitative")
                },
                "scope_rules": {
                    "evidence": lambda x: {"strength": "strong" if x.get("confidence", 0) > 0.7 else "weak"},
                    "justification": lambda x: {"type": "theoretical" if x.get("theory_based", False) else "empirical"}
                }
            },
            "business": {
                "role_mappings": {
                    SemanticRoleType.AGENT: SemanticRoleType.STAKEHOLDER,
                    SemanticRoleType.OBJECT: SemanticRoleType.GOAL,
                    SemanticRoleType.INSTRUMENT: SemanticRoleType.STRATEGY
                },
                "attribute_rules": {
                    "stakeholder": lambda x: {"type": "internal" if x.get("internal", False) else "external"},
                    "impact": lambda x: {"level": "high" if x > 0.7 else "medium" if x > 0.3 else "low"}
                },
                "scope_rules": {
                    "temporal": lambda x: {"timeline": x.get("timeline", "Q1")},
                    "spatial": lambda x: {"location": x.get("location", "global")}
                }
            },
            "legal": {
                "role_mappings": {
                    SemanticRoleType.AGENT: SemanticRoleType.PARTY,
                    SemanticRoleType.OBJECT: SemanticRoleType.CLAIM,
                    SemanticRoleType.INSTRUMENT: SemanticRoleType.EVIDENCE
                },
                "attribute_rules": {
                    "jurisdiction": lambda x: {"level": x.get("level", "federal")},
                    "precedent": lambda x: {"relevance": "binding" if x.get("binding", False) else "persuasive"}
                },
                "scope_rules": {
                    "evidence": lambda x: {"admissibility": "admissible" if x.get("admissible", False) else "inadmissible"},
                    "justification": lambda x: {"basis": "statutory" if x.get("statutory", False) else "common law"}
                }
            },
            "medical": {
                "role_mappings": {
                    SemanticRoleType.AGENT: SemanticRoleType.PRACTITIONER,
                    SemanticRoleType.OBJECT: SemanticRoleType.PATIENT,
                    SemanticRoleType.INSTRUMENT: SemanticRoleType.TREATMENT
                },
                "attribute_rules": {
                    "condition": lambda x: {"severity": "critical" if x > 0.8 else "serious" if x > 0.5 else "mild"},
                    "treatment": lambda x: {"type": x.get("type", "standard")}
                },
                "scope_rules": {
                    "temporal": lambda x: {"urgency": "emergency" if x.get("urgent", False) else "routine"},
                    "evidence": lambda x: {"level": "A" if x.get("confidence", 0) > 0.8 else "B" if x > 0.5 else "C"}
                }
            }
        }
        
        return rules.get(domain, {})
    
    def _transform_technical_role(self, role: 'SemanticRole', rules: Dict[str, Any]) -> 'SemanticRole':
        """Transform role for technical domain."""
        # Apply role type mapping
        if role.role_type in rules.get("role_mappings", {}):
            role.role_type = rules["role_mappings"][role.role_type]
        
        # Transform attributes
        for attr, rule in rules.get("attribute_rules", {}).items():
            if attr in role.attributes:
                role.attributes[attr] = rule(role.attributes[attr])
        
        # Transform scopes
        for scope_type, rule in rules.get("scope_rules", {}).items():
            scope = getattr(role, f"{scope_type}_scope")
            if scope:
                setattr(role, f"{scope_type}_scope", rule(scope))
        
        return role
    
    def _transform_academic_role(self, role: 'SemanticRole', rules: Dict[str, Any]) -> 'SemanticRole':
        """Transform role for academic domain."""
        # Apply role type mapping
        if role.role_type in rules.get("role_mappings", {}):
            role.role_type = rules["role_mappings"][role.role_type]
        
        # Transform attributes
        for attr, rule in rules.get("attribute_rules", {}).items():
            if attr in role.attributes:
                role.attributes[attr] = rule(role.attributes[attr])
        
        # Transform scopes
        for scope_type, rule in rules.get("scope_rules", {}).items():
            scope = getattr(role, f"{scope_type}_scope")
            if scope:
                setattr(role, f"{scope_type}_scope", rule(scope))
        
        return role
    
    def _transform_business_role(self, role: 'SemanticRole', rules: Dict[str, Any]) -> 'SemanticRole':
        """Transform role for business domain."""
        # Apply role type mapping
        if role.role_type in rules.get("role_mappings", {}):
            role.role_type = rules["role_mappings"][role.role_type]
        
        # Transform attributes
        for attr, rule in rules.get("attribute_rules", {}).items():
            if attr in role.attributes:
                role.attributes[attr] = rule(role.attributes[attr])
        
        # Transform scopes
        for scope_type, rule in rules.get("scope_rules", {}).items():
            scope = getattr(role, f"{scope_type}_scope")
            if scope:
                setattr(role, f"{scope_type}_scope", rule(scope))
        
        return role
    
    def _transform_legal_role(self, role: 'SemanticRole', rules: Dict[str, Any]) -> 'SemanticRole':
        """Transform role for legal domain."""
        # Apply role type mapping
        if role.role_type in rules.get("role_mappings", {}):
            role.role_type = rules["role_mappings"][role.role_type]
        
        # Transform attributes
        for attr, rule in rules.get("attribute_rules", {}).items():
            if attr in role.attributes:
                role.attributes[attr] = rule(role.attributes[attr])
        
        # Transform scopes
        for scope_type, rule in rules.get("scope_rules", {}).items():
            scope = getattr(role, f"{scope_type}_scope")
            if scope:
                setattr(role, f"{scope_type}_scope", rule(scope))
        
        return role
    
    def _transform_medical_role(self, role: 'SemanticRole', rules: Dict[str, Any]) -> 'SemanticRole':
        """Transform role for medical domain."""
        # Apply role type mapping
        if role.role_type in rules.get("role_mappings", {}):
            role.role_type = rules["role_mappings"][role.role_type]
        
        # Transform attributes
        for attr, rule in rules.get("attribute_rules", {}).items():
            if attr in role.attributes:
                role.attributes[attr] = rule(role.attributes[attr])
        
        # Transform scopes
        for scope_type, rule in rules.get("scope_rules", {}).items():
            scope = getattr(role, f"{scope_type}_scope")
            if scope:
                setattr(role, f"{scope_type}_scope", rule(scope))
        
        return role

class TokenizedQuery(BaseModel):
    tokens: List[str]
    roles: List[SemanticRole]
    context_tree: Dict[str, Any]
    karaka_mapping: Dict[str, Any]
    cross_references: Dict[str, Any] = {}
    role_hierarchy: Dict[str, List[SemanticRole]] = {}
    scope_mappings: Dict[str, Dict[str, Any]] = {}
    
    @validator('roles')
    def validate_roles(cls, v):
        if not v:
            raise ValidationError("At least one role must be present")
        return v
    
    def build_role_hierarchy(self):
        """Build the complete role hierarchy for all roles."""
        self.role_hierarchy = {}
        for role in self.roles:
            self.role_hierarchy[role.entity] = role.get_role_hierarchy()
    
    def build_scope_mappings(self):
        """Build scope mappings for all roles."""
        self.scope_mappings = {
            "temporal": {},
            "spatial": {},
            "modality": {},
            "evidence": {},
            "justification": {}
        }
        
        for role in self.roles:
            self.scope_mappings["temporal"][role.entity] = role.get_temporal_scope()
            self.scope_mappings["spatial"][role.entity] = role.get_spatial_scope()
            self.scope_mappings["modality"][role.entity] = role.get_modality_scope()
            self.scope_mappings["evidence"][role.entity] = role.get_evidence_scope()
            self.scope_mappings["justification"][role.entity] = role.get_justification_scope()
    
    def transform_roles(self, transformation_rules: Dict[SemanticRoleType, Dict[str, Any]]) -> 'TokenizedQuery':
        """Transform roles according to specified rules."""
        transformed_roles = []
        for role in self.roles:
            if role.role_type in transformation_rules:
                rule = transformation_rules[role.role_type]
                new_role = role.transform_role(
                    rule.get("new_type", role.role_type),
                    rule.get("confidence_adjustment", 0.0)
                )
                transformed_roles.append(new_role)
            else:
                transformed_roles.append(role)
        
        return TokenizedQuery(
            tokens=self.tokens,
            roles=transformed_roles,
            context_tree=self.context_tree,
            karaka_mapping=self.karaka_mapping,
            cross_references=self.cross_references,
            role_hierarchy=self.role_hierarchy,
            scope_mappings=self.scope_mappings
        )
    
    def merge_roles(self, merge_rules: Dict[str, Any]) -> 'TokenizedQuery':
        """Merge roles according to specified rules."""
        merged_roles = []
        processed = set()
        
        for i, role1 in enumerate(self.roles):
            if role1.entity in processed:
                continue
                
            merged_role = role1
            for j, role2 in enumerate(self.roles[i+1:], i+1):
                if (role2.entity not in processed and 
                    self._should_merge(role1, role2, merge_rules)):
                    merged_role = merged_role.merge_with(role2, merge_rules.get("strategy", "weighted"))
                    processed.add(role2.entity)
            
            merged_roles.append(merged_role)
            processed.add(role1.entity)
        
        return TokenizedQuery(
            tokens=self.tokens,
            roles=merged_roles,
            context_tree=self.context_tree,
            karaka_mapping=self.karaka_mapping,
            cross_references=self.cross_references,
            role_hierarchy=self.role_hierarchy,
            scope_mappings=self.scope_mappings
        )
    
    def _should_merge(self, role1: SemanticRole, role2: SemanticRole, 
                     merge_rules: Dict[str, Any]) -> bool:
        """Determine if two roles should be merged based on rules."""
        # Check role type compatibility
        if "role_types" in merge_rules:
            if not (role1.role_type in merge_rules["role_types"] and 
                   role2.role_type in merge_rules["role_types"]):
                return False
        
        # Check entity similarity
        if "similarity_threshold" in merge_rules:
            similarity = self._calculate_entity_similarity(role1.entity, role2.entity)
            if similarity < merge_rules["similarity_threshold"]:
                return False
        
        # Check scope compatibility
        if "scope_compatibility" in merge_rules:
            for scope_type in merge_rules["scope_compatibility"]:
                scope1 = getattr(role1, f"{scope_type}_scope")
                scope2 = getattr(role2, f"{scope_type}_scope")
                if scope1 and scope2 and not self._are_scopes_compatible(scope1, scope2):
                    return False
        
        return True
    
    def _calculate_entity_similarity(self, entity1: str, entity2: str) -> float:
        """Calculate similarity between two entities."""
        # Use spaCy for semantic similarity
        doc1 = nlp(entity1)
        doc2 = nlp(entity2)
        return doc1.similarity(doc2)
    
    def _are_scopes_compatible(self, scope1: Dict[str, Any], 
                             scope2: Dict[str, Any]) -> bool:
        """Check if two scopes are compatible for merging."""
        for key in set(scope1.keys()) & set(scope2.keys()):
            if isinstance(scope1[key], (int, float)) and isinstance(scope2[key], (int, float)):
                if abs(scope1[key] - scope2[key]) > 0.5:  # Threshold for numerical values
                    return False
            elif isinstance(scope1[key], list) and isinstance(scope2[key], list):
                if not set(scope1[key]) & set(scope2[key]):  # No common elements
                    return False
            elif scope1[key] != scope2[key]:
                return False
        return True
    
    def split_roles(self, split_rules: Dict[str, Any]) -> 'TokenizedQuery':
        """Split roles according to specified rules."""
        split_roles = []
        
        for role in self.roles:
            if role.role_type in split_rules.get("role_types", []):
                split_roles.extend(role.split_role(split_rules))
            else:
                split_roles.append(role)
        
        return TokenizedQuery(
            tokens=self.tokens,
            roles=split_roles,
            context_tree=self.context_tree,
            karaka_mapping=self.karaka_mapping,
            cross_references=self.cross_references,
            role_hierarchy=self.role_hierarchy,
            scope_mappings=self.scope_mappings
        )
    
    def validate_domain_rules(self, domain: str) -> Dict[str, List[str]]:
        """Validate all roles against domain-specific rules."""
        validation_results = {
            "issues": [],
            "warnings": [],
            "suggestions": []
        }
        
        for role in self.roles:
            issues = role.validate_domain_rules(domain)
            validation_results["issues"].extend(issues)
        
        # Additional domain-specific validation
        if domain == "technical":
            # Check for circular dependencies
            for role in self.roles:
                if "dependencies" in role.attributes:
                    deps = role.attributes["dependencies"]
                    if isinstance(deps, list):
                        for dep in deps:
                            if dep in [r.entity for r in self.roles]:
                                validation_results["warnings"].append(
                                    f"Potential circular dependency: {role.entity} -> {dep}"
                                )
        
        elif domain == "academic":
            # Check for proper citation format
            for role in self.roles:
                if "citation" in role.attributes:
                    citation = role.attributes["citation"]
                    if not isinstance(citation, dict) or "author" not in citation:
                        validation_results["issues"].append(
                            f"Invalid citation format for {role.entity}"
                        )
        
        elif domain == "business":
            # Check for stakeholder alignment
            stakeholders = set()
            for role in self.roles:
                if "stakeholder" in role.attributes:
                    stakeholders.add(role.attributes["stakeholder"])
            
            if len(stakeholders) > 1:
                validation_results["suggestions"].append(
                    "Consider stakeholder alignment across roles"
                )
        
        return validation_results
    
    def resolve_role_conflicts(self, resolution_strategy: str = "priority") -> 'TokenizedQuery':
        """Resolve conflicts between all roles in the query."""
        resolved_roles = []
        processed = set()
        
        for i, role1 in enumerate(self.roles):
            if role1.entity in processed:
                continue
            
            resolved_role = role1
            for j, role2 in enumerate(self.roles[i+1:], i+1):
                if role2.entity not in processed:
                    conflicts = role1._identify_conflicts(role2)
                    if conflicts:
                        resolved_role = resolved_role.resolve_conflicts(
                            role2, resolution_strategy
                        )
                    processed.add(role2.entity)
            
            resolved_roles.append(resolved_role)
            processed.add(role1.entity)
        
        return TokenizedQuery(
            tokens=self.tokens,
            roles=resolved_roles,
            context_tree=self.context_tree,
            karaka_mapping=self.karaka_mapping,
            cross_references=self.cross_references,
            role_hierarchy=self.role_hierarchy,
            scope_mappings=self.scope_mappings
        )
    
    def transform_for_domain(self, domain: str, 
                           transformation_rules: Optional[Dict[str, Any]] = None) -> 'TokenizedQuery':
        """Transform all roles according to domain-specific rules."""
        transformed_roles = []
        
        for role in self.roles:
            transformed_role = role.transform_for_domain(domain, transformation_rules)
            transformed_roles.append(transformed_role)
        
        return TokenizedQuery(
            tokens=self.tokens,
            roles=transformed_roles,
            context_tree=self.context_tree,
            karaka_mapping=self.karaka_mapping,
            cross_references=self.cross_references,
            role_hierarchy=self.role_hierarchy,
            scope_mappings=self.scope_mappings
        )

class KarakaMapper:
    def __init__(self):
        self.role_types = {
            SemanticRoleType.AGENT: ["performer", "doer", "actor", "subject"],
            SemanticRoleType.OBJECT: ["target", "goal", "result", "object"],
            SemanticRoleType.INSTRUMENT: ["tool", "means", "method", "instrument"],
            SemanticRoleType.LOCATION: ["place", "site", "location", "where"],
            SemanticRoleType.TIME: ["when", "duration", "period", "time"],
            SemanticRoleType.PURPOSE: ["reason", "goal", "purpose", "why"],
            SemanticRoleType.SOURCE: ["origin", "from", "source", "starting_point"],
            SemanticRoleType.DESTINATION: ["end", "to", "destination", "target_location"]
        }
        
        # Load models for semantic similarity
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize patterns and relationships
        self._initialize_patterns()
        self._initialize_relationships()
        
        # Initialize enhanced relationship patterns
        self._initialize_enhanced_relationships()
        
        # Initialize visualization settings
        self._initialize_visualization_settings()
        
        # Initialize timeline visualization settings
        self._initialize_timeline_settings()
        
        # Initialize hierarchical tree settings
        self._initialize_hierarchical_settings()
        
        # Initialize layout algorithms
        self._initialize_layout_algorithms()
        
        # Initialize filtering capabilities
        self._initialize_filtering_capabilities()
        
        # Initialize specialized domain patterns
        self._initialize_domain_patterns()
        
        # Initialize domain-specific validation rules
        self._initialize_domain_validation_rules()
        
        # Initialize domain-specific relationship patterns
        self._initialize_domain_relationship_patterns()
        
        # Initialize domain-specific validation rules
        self._initialize_domain_validation_rules()
        
        # Initialize fuzzy matching settings
        self._initialize_fuzzy_matching()
        
        # Initialize context-aware pattern recognition
        self._initialize_context_aware_patterns()
    
    def _initialize_patterns(self):
        """Initialize all pattern dictionaries."""
        self.temporal_patterns = {
            "absolute": r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}",
            "relative": r"(last|next|this)\s+(week|month|year|day)",
            "duration": r"for\s+(\d+)\s+(hours|days|weeks|months|years)",
            "time_of_day": r"(morning|afternoon|evening|night|noon|midnight)"
        }
        
        self.spatial_patterns = {
            "location": r"in\s+([A-Za-z\s]+)",
            "direction": r"(north|south|east|west|up|down|left|right)",
            "distance": r"(\d+)\s*(miles|kilometers|meters|feet)",
            "area": r"area\s+of\s+([A-Za-z\s]+)"
        }
        
        self.modality_patterns = {
            "possibility": r"(can|could|may|might|would)",
            "necessity": r"(must|should|ought|need|have to)",
            "ability": r"(able to|capable of|can|could)",
            "permission": r"(allowed to|permitted to|may|can)"
        }
    
    def _initialize_relationships(self):
        """Initialize relationship patterns between different semantic types."""
        self.role_relationships = {
            (SemanticRoleType.AGENT, SemanticRoleType.OBJECT): [
                "performs action on",
                "interacts with",
                "affects"
            ],
            (SemanticRoleType.INSTRUMENT, SemanticRoleType.OBJECT): [
                "used for",
                "applied to",
                "helps with"
            ],
            (SemanticRoleType.LOCATION, SemanticRoleType.TIME): [
                "at",
                "during",
                "while in"
            ],
            (SemanticRoleType.PURPOSE, SemanticRoleType.OBJECT): [
                "for",
                "to achieve",
                "aiming for"
            ]
        }
    
    def _initialize_enhanced_relationships(self):
        """Initialize enhanced relationship patterns between semantic types."""
        self.role_relationships.update({
            (SemanticRoleType.AGENT, SemanticRoleType.INSTRUMENT): [
                "uses",
                "employs",
                "applies",
                "leverages"
            ],
            (SemanticRoleType.OBJECT, SemanticRoleType.PURPOSE): [
                "serves",
                "fulfills",
                "achieves",
                "accomplishes"
            ],
            (SemanticRoleType.SOURCE, SemanticRoleType.DESTINATION): [
                "leads to",
                "results in",
                "transforms into",
                "evolves into"
            ],
            (SemanticRoleType.TIME, SemanticRoleType.PURPOSE): [
                "enables",
                "facilitates",
                "supports",
                "allows"
            ],
            (SemanticRoleType.LOCATION, SemanticRoleType.OBJECT): [
                "contains",
                "holds",
                "encompasses",
                "surrounds"
            ]
        })
        
        # Add hierarchical relationships
        self.hierarchical_relationships = {
            SemanticRoleType.AGENT: [SemanticRoleType.INSTRUMENT, SemanticRoleType.OBJECT],
            SemanticRoleType.OBJECT: [SemanticRoleType.PURPOSE, SemanticRoleType.DESTINATION],
            SemanticRoleType.TIME: [SemanticRoleType.LOCATION, SemanticRoleType.PURPOSE],
            SemanticRoleType.SOURCE: [SemanticRoleType.OBJECT, SemanticRoleType.DESTINATION]
        }
    
    def _initialize_visualization_settings(self):
        """Initialize settings for network visualization."""
        self.visualization_settings = {
            "node_colors": {
                SemanticRoleType.AGENT: "#FF6B6B",
                SemanticRoleType.OBJECT: "#4ECDC4",
                SemanticRoleType.INSTRUMENT: "#45B7D1",
                SemanticRoleType.LOCATION: "#96CEB4",
                SemanticRoleType.TIME: "#FFEEAD",
                SemanticRoleType.PURPOSE: "#D4A5A5",
                SemanticRoleType.SOURCE: "#9B59B6",
                SemanticRoleType.DESTINATION: "#3498DB"
            },
            "edge_colors": {
                "direct": "#2C3E50",
                "transitive": "#7F8C8D",
                "hierarchical": "#E74C3C"
            },
            "node_sizes": {
                "primary": 3000,
                "secondary": 2000,
                "tertiary": 1000
            }
        }
    
    def _initialize_timeline_settings(self):
        """Initialize settings for timeline visualization."""
        self.timeline_settings = {
            "time_scale": {
                "year": 365 * 24 * 60 * 60,  # seconds
                "month": 30 * 24 * 60 * 60,
                "week": 7 * 24 * 60 * 60,
                "day": 24 * 60 * 60,
                "hour": 60 * 60
            },
            "event_colors": {
                "start": "#2ECC71",
                "end": "#E74C3C",
                "duration": "#3498DB",
                "milestone": "#F1C40F"
            },
            "timeline_height": 100,
            "event_height": 20,
            "padding": 10
        }
    
    def _initialize_hierarchical_settings(self):
        """Initialize settings for hierarchical tree visualization."""
        self.hierarchical_settings = {
            "node_colors": {
                "root": "#2C3E50",
                "branch": "#34495E",
                "leaf": "#7F8C8D"
            },
            "edge_colors": {
                "strong": "#27AE60",
                "medium": "#F39C12",
                "weak": "#E74C3C"
            },
            "node_sizes": {
                "root": 20,
                "branch": 15,
                "leaf": 10
            }
        }
    
    def _initialize_layout_algorithms(self):
        """Initialize settings for different layout algorithms."""
        self.layout_algorithms = {
            "force_directed": {
                "k": 1.0,
                "iterations": 50,
                "threshold": 0.0001,
                "dim": 2
            },
            "spectral": {
                "dim": 2,
                "weight": "weight",
                "scale": 1.0
            },
            "spring": {
                "k": None,
                "pos": None,
                "fixed": None,
                "iterations": 50,
                "threshold": 0.0001,
                "weight": "weight",
                "scale": 1.0,
                "center": None,
                "dim": 2,
                "seed": None
            },
            "kamada_kawai": {
                "dist": None,
                "pos": None,
                "weight": "weight",
                "scale": 1.0,
                "center": None,
                "dim": 2
            },
            "fruchterman_reingold": {
                "k": None,
                "pos": None,
                "fixed": None,
                "iterations": 50,
                "threshold": 0.0001,
                "weight": "weight",
                "scale": 1.0,
                "center": None,
                "dim": 2,
                "seed": None
            }
        }
    
    def _initialize_filtering_capabilities(self):
        """Initialize filtering capabilities for network visualization."""
        self.filtering_capabilities = {
            "role_types": {
                "enabled": True,
                "threshold": 0.3
            },
            "confidence": {
                "enabled": True,
                "min_confidence": 0.5
            },
            "relationships": {
                "enabled": True,
                "min_strength": 0.5
            },
            "clusters": {
                "enabled": True,
                "min_size": 2
            }
        }
    
    def _initialize_domain_patterns(self):
        """Initialize specialized patterns for different domains."""
        self.domain_patterns = {
            "technical": {
                "dependency": r"(depends on|requires|needs|uses)",
                "implementation": r"(implements|realizes|executes|performs)",
                "configuration": r"(configures|sets up|initializes|prepares)",
                "optimization": r"(optimizes|improves|enhances|refines)"
            },
            "academic": {
                "citation": r"(cites|references|quotes|mentions)",
                "analysis": r"(analyzes|examines|studies|investigates)",
                "comparison": r"(compares|contrasts|relates|correlates)",
                "conclusion": r"(concludes|summarizes|infers|deduces)"
            },
            "business": {
                "strategy": r"(plans|strategizes|organizes|coordinates)",
                "execution": r"(executes|implements|carries out|performs)",
                "evaluation": r"(evaluates|assesses|measures|reviews)",
                "improvement": r"(improves|enhances|optimizes|refines)"
            },
            "legal": {
                "precedent": r"(cites|references|relies on|follows)",
                "argument": r"(argues|contends|asserts|claims)",
                "evidence": r"(presents|submits|introduces|provides)",
                "ruling": r"(rules|decides|determines|finds)"
            },
            "medical": {
                "diagnosis": r"(diagnoses|identifies|detects|recognizes)",
                "treatment": r"(treats|manages|addresses|handles)",
                "monitoring": r"(monitors|tracks|observes|follows)",
                "prevention": r"(prevents|avoids|mitigates|reduces)"
            }
        }
    
    def _initialize_domain_validation_rules(self):
        """Initialize validation rules for different domains."""
        self.domain_validation_rules = {
            "technical": {
                "required_attributes": ["complexity", "dependencies"],
                "valid_role_types": [
                    SemanticRoleType.AGENT,
                    SemanticRoleType.OBJECT,
                    SemanticRoleType.INSTRUMENT,
                    SemanticRoleType.PURPOSE
                ],
                "min_confidence": 0.7,
                "scope_requirements": {
                    "temporal": ["deadline"],
                    "spatial": ["environment"]
                },
                "relationship_rules": {
                    "dependency": {
                        "required": True,
                        "min_strength": 0.8
                    },
                    "implementation": {
                        "required": True,
                        "min_strength": 0.9
                    }
                }
            },
            "academic": {
                "required_attributes": ["citation", "methodology"],
                "valid_role_types": [
                    SemanticRoleType.SOURCE,
                    SemanticRoleType.TOPIC,
                    SemanticRoleType.METHOD,
                    SemanticRoleType.EVIDENCE
                ],
                "min_confidence": 0.6,
                "scope_requirements": {
                    "evidence": ["strength"],
                    "justification": ["type"]
                },
                "relationship_rules": {
                    "citation": {
                        "required": True,
                        "min_strength": 0.9
                    },
                    "analysis": {
                        "required": True,
                        "min_strength": 0.8
                    }
                }
            },
            "business": {
                "required_attributes": ["stakeholder", "impact"],
                "valid_role_types": [
                    SemanticRoleType.STAKEHOLDER,
                    SemanticRoleType.GOAL,
                    SemanticRoleType.STRATEGY,
                    SemanticRoleType.RESULT
                ],
                "min_confidence": 0.5,
                "scope_requirements": {
                    "temporal": ["timeline"],
                    "spatial": ["location"]
                },
                "relationship_rules": {
                    "strategy": {
                        "required": True,
                        "min_strength": 0.8
                    },
                    "execution": {
                        "required": True,
                        "min_strength": 0.9
                    }
                }
            },
            "legal": {
                "required_attributes": ["jurisdiction", "precedent"],
                "valid_role_types": [
                    SemanticRoleType.PARTY,
                    SemanticRoleType.CLAIM,
                    SemanticRoleType.EVIDENCE,
                    SemanticRoleType.JUSTIFICATION
                ],
                "min_confidence": 0.8,
                "scope_requirements": {
                    "evidence": ["admissibility"],
                    "justification": ["basis"]
                },
                "relationship_rules": {
                    "precedent": {
                        "required": True,
                        "min_strength": 0.9
                    },
                    "evidence": {
                        "required": True,
                        "min_strength": 0.9
                    }
                }
            },
            "medical": {
                "required_attributes": ["condition", "treatment"],
                "valid_role_types": [
                    SemanticRoleType.PRACTITIONER,
                    SemanticRoleType.PATIENT,
                    SemanticRoleType.TREATMENT,
                    SemanticRoleType.EVIDENCE
                ],
                "min_confidence": 0.9,
                "scope_requirements": {
                    "temporal": ["urgency"],
                    "evidence": ["level"]
                },
                "relationship_rules": {
                    "diagnosis": {
                        "required": True,
                        "min_strength": 0.9
                    },
                    "treatment": {
                        "required": True,
                        "min_strength": 0.8
                    }
                }
            }
        }
    
    def _initialize_domain_relationship_patterns(self):
        """Initialize specialized relationship patterns for different domains."""
        self.domain_relationship_patterns = {
            "technical": {
                "dependency": {
                    "pattern": r"(depends on|requires|needs|uses|relies on|imports|extends|implements)",
                    "strength": 0.8,
                    "direction": "forward",
                    "validation": {
                        "required": True,
                        "min_strength": 0.7,
                        "max_dependencies": 5,
                        "circular_check": True
                    }
                },
                "implementation": {
                    "pattern": r"(implements|realizes|executes|performs|fulfills|satisfies|completes)",
                    "strength": 0.9,
                    "direction": "forward",
                    "validation": {
                        "required": True,
                        "min_strength": 0.8,
                        "completeness_check": True
                    }
                },
                "configuration": {
                    "pattern": r"(configures|sets up|initializes|prepares|establishes|defines|specifies)",
                    "strength": 0.7,
                    "direction": "forward",
                    "validation": {
                        "required": False,
                        "min_strength": 0.6,
                        "consistency_check": True
                    }
                },
                "optimization": {
                    "pattern": r"(optimizes|improves|enhances|refines|tunes|adjusts|calibrates)",
                    "strength": 0.6,
                    "direction": "forward",
                    "validation": {
                        "required": False,
                        "min_strength": 0.5,
                        "impact_analysis": True
                    }
                }
            },
            "academic": {
                "citation": {
                    "pattern": r"(cites|references|quotes|mentions|acknowledges|credits|attributes)",
                    "strength": 0.9,
                    "direction": "backward",
                    "validation": {
                        "required": True,
                        "min_strength": 0.8,
                        "format_check": True,
                        "recency_check": True
                    }
                },
                "analysis": {
                    "pattern": r"(analyzes|examines|studies|investigates|evaluates|assesses|scrutinizes)",
                    "strength": 0.8,
                    "direction": "forward",
                    "validation": {
                        "required": True,
                        "min_strength": 0.7,
                        "methodology_check": True
                    }
                },
                "comparison": {
                    "pattern": r"(compares|contrasts|relates|correlates|juxtaposes|parallels|aligns)",
                    "strength": 0.7,
                    "direction": "bidirectional",
                    "validation": {
                        "required": False,
                        "min_strength": 0.6,
                        "similarity_threshold": 0.5
                    }
                },
                "conclusion": {
                    "pattern": r"(concludes|summarizes|infers|deduces|derives|determines|establishes)",
                    "strength": 0.9,
                    "direction": "forward",
                    "validation": {
                        "required": True,
                        "min_strength": 0.8,
                        "evidence_check": True
                    }
                }
            },
            "business": {
                "strategy": {
                    "pattern": r"(plans|strategizes|organizes|coordinates|aligns|orchestrates|directs)",
                    "strength": 0.8,
                    "direction": "forward",
                    "validation": {
                        "required": True,
                        "min_strength": 0.7,
                        "alignment_check": True
                    }
                },
                "execution": {
                    "pattern": r"(executes|implements|carries out|performs|delivers|realizes|achieves)",
                    "strength": 0.9,
                    "direction": "forward",
                    "validation": {
                        "required": True,
                        "min_strength": 0.8,
                        "progress_tracking": True
                    }
                },
                "evaluation": {
                    "pattern": r"(evaluates|assesses|measures|reviews|analyzes|examines|scrutinizes)",
                    "strength": 0.7,
                    "direction": "backward",
                    "validation": {
                        "required": True,
                        "min_strength": 0.6,
                        "metrics_check": True
                    }
                },
                "improvement": {
                    "pattern": r"(improves|enhances|optimizes|refines|upgrades|advances|develops)",
                    "strength": 0.6,
                    "direction": "forward",
                    "validation": {
                        "required": False,
                        "min_strength": 0.5,
                        "impact_analysis": True
                    }
                }
            },
            "legal": {
                "precedent": {
                    "pattern": r"(cites|references|relies on|follows|adheres to|applies|invokes)",
                    "strength": 0.9,
                    "direction": "backward",
                    "validation": {
                        "required": True,
                        "min_strength": 0.8,
                        "jurisdiction_check": True
                    }
                },
                "argument": {
                    "pattern": r"(argues|contends|asserts|claims|maintains|posits|proposes)",
                    "strength": 0.8,
                    "direction": "forward",
                    "validation": {
                        "required": True,
                        "min_strength": 0.7,
                        "logical_check": True
                    }
                },
                "evidence": {
                    "pattern": r"(presents|submits|introduces|provides|offers|furnishes|supplies)",
                    "strength": 0.9,
                    "direction": "forward",
                    "validation": {
                        "required": True,
                        "min_strength": 0.8,
                        "admissibility_check": True
                    }
                },
                "ruling": {
                    "pattern": r"(rules|decides|determines|finds|holds|concludes|resolves)",
                    "strength": 0.9,
                    "direction": "forward",
                    "validation": {
                        "required": True,
                        "min_strength": 0.8,
                        "precedent_check": True
                    }
                }
            },
            "medical": {
                "diagnosis": {
                    "pattern": r"(diagnoses|identifies|detects|recognizes|determines|ascertains|establishes)",
                    "strength": 0.9,
                    "direction": "forward",
                    "validation": {
                        "required": True,
                        "min_strength": 0.8,
                        "symptom_check": True
                    }
                },
                "treatment": {
                    "pattern": r"(treats|manages|addresses|handles|administers|prescribes|applies)",
                    "strength": 0.8,
                    "direction": "forward",
                    "validation": {
                        "required": True,
                        "min_strength": 0.7,
                        "protocol_check": True
                    }
                },
                "monitoring": {
                    "pattern": r"(monitors|tracks|observes|follows|watches|checks|assesses)",
                    "strength": 0.7,
                    "direction": "bidirectional",
                    "validation": {
                        "required": True,
                        "min_strength": 0.6,
                        "frequency_check": True
                    }
                },
                "prevention": {
                    "pattern": r"(prevents|avoids|mitigates|reduces|minimizes|deters|forestalls)",
                    "strength": 0.8,
                    "direction": "forward",
                    "validation": {
                        "required": False,
                        "min_strength": 0.7,
                        "risk_assessment": True
                    }
                }
            }
        }
    
    def _initialize_domain_validation_rules(self):
        """Initialize validation rules for different domains."""
        self.domain_validation_rules = {
            "technical": {
                "required_attributes": ["complexity", "dependencies", "version", "documentation"],
                "valid_role_types": [
                    SemanticRoleType.AGENT,
                    SemanticRoleType.OBJECT,
                    SemanticRoleType.INSTRUMENT,
                    SemanticRoleType.PURPOSE
                ],
                "min_confidence": 0.7,
                "scope_requirements": {
                    "temporal": ["deadline", "version"],
                    "spatial": ["environment", "deployment"]
                },
                "relationship_rules": {
                    "dependency": {
                        "required": True,
                        "min_strength": 0.8,
                        "max_dependencies": 5,
                        "circular_check": True
                    },
                    "implementation": {
                        "required": True,
                        "min_strength": 0.9,
                        "completeness_check": True
                    }
                },
                "validation_checks": {
                    "version_compatibility": True,
                    "dependency_resolution": True,
                    "security_audit": True
                }
            },
            "academic": {
                "required_attributes": ["citation", "methodology", "field", "contribution"],
                "valid_role_types": [
                    SemanticRoleType.SOURCE,
                    SemanticRoleType.TOPIC,
                    SemanticRoleType.METHOD,
                    SemanticRoleType.EVIDENCE
                ],
                "min_confidence": 0.6,
                "scope_requirements": {
                    "evidence": ["strength", "source"],
                    "justification": ["type", "basis"]
                },
                "relationship_rules": {
                    "citation": {
                        "required": True,
                        "min_strength": 0.9,
                        "format_check": True,
                        "recency_check": True
                    },
                    "analysis": {
                        "required": True,
                        "min_strength": 0.8,
                        "methodology_check": True
                    }
                },
                "validation_checks": {
                    "plagiarism_check": True,
                    "methodology_validation": True,
                    "citation_consistency": True
                }
            },
            "business": {
                "required_attributes": ["stakeholder", "impact", "timeline", "budget"],
                "valid_role_types": [
                    SemanticRoleType.STAKEHOLDER,
                    SemanticRoleType.GOAL,
                    SemanticRoleType.STRATEGY,
                    SemanticRoleType.RESULT
                ],
                "min_confidence": 0.5,
                "scope_requirements": {
                    "temporal": ["timeline", "milestones"],
                    "spatial": ["location", "market"]
                },
                "relationship_rules": {
                    "strategy": {
                        "required": True,
                        "min_strength": 0.8,
                        "alignment_check": True
                    },
                    "execution": {
                        "required": True,
                        "min_strength": 0.9,
                        "progress_tracking": True
                    }
                },
                "validation_checks": {
                    "roi_analysis": True,
                    "risk_assessment": True,
                    "stakeholder_alignment": True
                }
            },
            "legal": {
                "required_attributes": ["jurisdiction", "precedent", "statute", "case"],
                "valid_role_types": [
                    SemanticRoleType.PARTY,
                    SemanticRoleType.CLAIM,
                    SemanticRoleType.EVIDENCE,
                    SemanticRoleType.JUSTIFICATION
                ],
                "min_confidence": 0.8,
                "scope_requirements": {
                    "evidence": ["admissibility", "weight"],
                    "justification": ["basis", "authority"]
                },
                "relationship_rules": {
                    "precedent": {
                        "required": True,
                        "min_strength": 0.9,
                        "jurisdiction_check": True
                    },
                    "evidence": {
                        "required": True,
                        "min_strength": 0.9,
                        "admissibility_check": True
                    }
                },
                "validation_checks": {
                    "jurisdiction_verification": True,
                    "precedent_analysis": True,
                    "evidence_chain": True
                }
            },
            "medical": {
                "required_attributes": ["condition", "treatment", "patient", "history"],
                "valid_role_types": [
                    SemanticRoleType.PRACTITIONER,
                    SemanticRoleType.PATIENT,
                    SemanticRoleType.TREATMENT,
                    SemanticRoleType.EVIDENCE
                ],
                "min_confidence": 0.9,
                "scope_requirements": {
                    "temporal": ["urgency", "duration"],
                    "evidence": ["level", "source"]
                },
                "relationship_rules": {
                    "diagnosis": {
                        "required": True,
                        "min_strength": 0.9,
                        "symptom_check": True
                    },
                    "treatment": {
                        "required": True,
                        "min_strength": 0.8,
                        "protocol_check": True
                    }
                },
                "validation_checks": {
                    "patient_safety": True,
                    "treatment_effectiveness": True,
                    "risk_mitigation": True
                }
            }
        }
    
    def _initialize_fuzzy_matching(self):
        """Initialize settings for fuzzy pattern matching."""
        self.fuzzy_matching_settings = {
            "similarity_threshold": 0.7,
            "max_distance": 2,
            "context_weight": 0.3,
            "pattern_weights": {
                "exact": 1.0,
                "fuzzy": 0.8,
                "context": 0.6
            }
        }
        
        # Initialize fuzzy matching patterns
        self.fuzzy_patterns = {
            "action": {
                "base_patterns": [
                    r"(performs|executes|carries out|implements|realizes)",
                    r"(creates|generates|produces|develops|builds)",
                    r"(analyzes|examines|studies|investigates|evaluates)",
                    r"(modifies|changes|alters|adjusts|adapts)"
                ],
                "variations": {
                    "tense": ["past", "present", "future"],
                    "aspect": ["simple", "continuous", "perfect"],
                    "voice": ["active", "passive"]
                }
            },
            "relationship": {
                "base_patterns": [
                    r"(connects|links|associates|relates|ties)",
                    r"(depends on|relies on|requires|needs|uses)",
                    r"(influences|affects|impacts|changes|modifies)",
                    r"(causes|triggers|initiates|starts|begins)"
                ],
                "variations": {
                    "direction": ["forward", "backward", "bidirectional"],
                    "strength": ["strong", "medium", "weak"],
                    "type": ["causal", "temporal", "spatial"]
                }
            }
        }

    def _initialize_context_aware_patterns(self):
        """Initialize context-aware pattern recognition settings."""
        self.context_aware_settings = {
            "context_window": 3,  # Number of tokens to consider for context
            "context_weights": {
                "immediate": 1.0,
                "near": 0.8,
                "far": 0.5
            },
            "context_types": {
                "semantic": ["synonyms", "antonyms", "hypernyms", "hyponyms"],
                "syntactic": ["subject", "object", "modifier", "complement"],
                "pragmatic": ["intent", "goal", "purpose", "outcome"]
            }
        }
        
        # Initialize context patterns
        self.context_patterns = {
            "semantic_context": {
                "patterns": {
                    "synonyms": r"(similar to|like|resembling|comparable to)",
                    "antonyms": r"(unlike|different from|opposite to|contrary to)",
                    "hypernyms": r"(type of|kind of|category of|class of)",
                    "hyponyms": r"(example of|instance of|specific type of|particular kind of)"
                },
                "weights": {
                    "synonyms": 0.9,
                    "antonyms": 0.8,
                    "hypernyms": 0.7,
                    "hyponyms": 0.6
                }
            },
            "syntactic_context": {
                "patterns": {
                    "subject": r"(subject|agent|actor|performer)",
                    "object": r"(object|target|goal|recipient)",
                    "modifier": r"(modifier|qualifier|descriptor|attribute)",
                    "complement": r"(complement|completion|finisher|ender)"
                },
                "weights": {
                    "subject": 0.9,
                    "object": 0.8,
                    "modifier": 0.7,
                    "complement": 0.6
                }
            },
            "pragmatic_context": {
                "patterns": {
                    "intent": r"(intends to|aims to|seeks to|wants to)",
                    "goal": r"(goal is|objective is|purpose is|target is)",
                    "purpose": r"(in order to|so that|for the purpose of|to)",
                    "outcome": r"(resulting in|leading to|causing|producing)"
                },
                "weights": {
                    "intent": 0.9,
                    "goal": 0.8,
                    "purpose": 0.7,
                    "outcome": 0.6
                }
            }
        }

    def fuzzy_match_pattern(self, text: str, pattern_type: str) -> Dict[str, Any]:
        """Perform fuzzy pattern matching with context awareness."""
        matches = []
        base_patterns = self.fuzzy_patterns[pattern_type]["base_patterns"]
        variations = self.fuzzy_patterns[pattern_type]["variations"]
        
        for base_pattern in base_patterns:
            # Exact matching
            exact_matches = re.finditer(base_pattern, text, re.IGNORECASE)
            for match in exact_matches:
                matches.append({
                    "pattern": base_pattern,
                    "match": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 1.0,
                    "type": "exact"
                })
            
            # Fuzzy matching
            for variation_type, variation_values in variations.items():
                for value in variation_values:
                    fuzzy_pattern = self._create_fuzzy_pattern(base_pattern, variation_type, value)
                    fuzzy_matches = re.finditer(fuzzy_pattern, text, re.IGNORECASE)
                    for match in fuzzy_matches:
                        confidence = self._calculate_fuzzy_confidence(
                            match.group(),
                            base_pattern,
                            variation_type,
                            value
                        )
                        if confidence >= self.fuzzy_matching_settings["similarity_threshold"]:
                            matches.append({
                                "pattern": fuzzy_pattern,
                                "match": match.group(),
                                "start": match.start(),
                                "end": match.end(),
                                "confidence": confidence,
                                "type": "fuzzy",
                                "variation": {variation_type: value}
                            })
        
        # Add context awareness
        for match in matches:
            context_score = self._calculate_context_score(text, match)
            match["confidence"] = (
                match["confidence"] * (1 - self.fuzzy_matching_settings["context_weight"]) +
                context_score * self.fuzzy_matching_settings["context_weight"]
            )
        
        return {
            "matches": sorted(matches, key=lambda x: x["confidence"], reverse=True),
            "best_match": max(matches, key=lambda x: x["confidence"]) if matches else None
        }

    def _create_fuzzy_pattern(self, base_pattern: str, variation_type: str, value: str) -> str:
        """Create a fuzzy pattern based on base pattern and variation."""
        if variation_type == "tense":
            return self._add_tense_variation(base_pattern, value)
        elif variation_type == "aspect":
            return self._add_aspect_variation(base_pattern, value)
        elif variation_type == "voice":
            return self._add_voice_variation(base_pattern, value)
        elif variation_type == "direction":
            return self._add_direction_variation(base_pattern, value)
        elif variation_type == "strength":
            return self._add_strength_variation(base_pattern, value)
        elif variation_type == "type":
            return self._add_type_variation(base_pattern, value)
        return base_pattern

    def _calculate_fuzzy_confidence(self, match: str, base_pattern: str, 
                                  variation_type: str, value: str) -> float:
        """Calculate confidence score for fuzzy match."""
        base_confidence = 0.8  # Base confidence for fuzzy matches
        
        # Adjust based on variation type
        if variation_type in ["tense", "aspect", "voice"]:
            base_confidence *= 0.9
        elif variation_type in ["direction", "strength", "type"]:
            base_confidence *= 0.8
        
        # Adjust based on string similarity
        similarity = self.calculate_similarity(match, base_pattern)
        base_confidence *= similarity
        
        return min(base_confidence, 1.0)

    def _calculate_context_score(self, text: str, match: Dict[str, Any]) -> float:
        """Calculate context score for a match."""
        context_score = 0.0
        window = self.context_aware_settings["context_window"]
        
        # Get context window
        start = max(0, match["start"] - window)
        end = min(len(text), match["end"] + window)
        context = text[start:end]
        
        # Check semantic context
        for context_type, patterns in self.context_patterns["semantic_context"]["patterns"].items():
            if re.search(patterns, context, re.IGNORECASE):
                context_score += self.context_patterns["semantic_context"]["weights"][context_type]
        
        # Check syntactic context
        for context_type, patterns in self.context_patterns["syntactic_context"]["patterns"].items():
            if re.search(patterns, context, re.IGNORECASE):
                context_score += self.context_patterns["syntactic_context"]["weights"][context_type]
        
        # Check pragmatic context
        for context_type, patterns in self.context_patterns["pragmatic_context"]["patterns"].items():
            if re.search(patterns, context, re.IGNORECASE):
                context_score += self.context_patterns["pragmatic_context"]["weights"][context_type]
        
        return min(context_score, 1.0)

    def _add_tense_variation(self, pattern: str, tense: str) -> str:
        """Add tense variation to pattern."""
        tense_map = {
            "past": r"(ed|d|t|en)",
            "present": r"(s|es|ing)",
            "future": r"(will|shall|going to)"
        }
        return pattern.replace(r"(\w+)", f"\\w+{tense_map[tense]}")

    def _add_aspect_variation(self, pattern: str, aspect: str) -> str:
        """Add aspect variation to pattern."""
        aspect_map = {
            "simple": r"(\w+)",
            "continuous": r"(be \w+ing)",
            "perfect": r"(have \w+ed)"
        }
        return pattern.replace(r"(\w+)", aspect_map[aspect])

    def _add_voice_variation(self, pattern: str, voice: str) -> str:
        """Add voice variation to pattern."""
        if voice == "passive":
            return pattern.replace(r"(\w+)", r"(be \w+ed)")
        return pattern

    def _add_direction_variation(self, pattern: str, direction: str) -> str:
        """Add direction variation to pattern."""
        direction_map = {
            "forward": r"(forward|ahead|onward)",
            "backward": r"(backward|reverse|back)",
            "bidirectional": r"(both ways|two-way|mutual)"
        }
        return f"{pattern} {direction_map[direction]}"

    def _add_strength_variation(self, pattern: str, strength: str) -> str:
        """Add strength variation to pattern."""
        strength_map = {
            "strong": r"(strongly|firmly|powerfully)",
            "medium": r"(moderately|somewhat|partially)",
            "weak": r"(weakly|slightly|minimally)"
        }
        return f"{pattern} {strength_map[strength]}"

    def _add_type_variation(self, pattern: str, type_: str) -> str:
        """Add type variation to pattern."""
        type_map = {
            "causal": r"(because|since|as|due to)",
            "temporal": r"(when|while|during|after)",
            "spatial": r"(where|in which|at which|on which)"
        }
        return f"{pattern} {type_map[type_]}"
    
    def map_roles(self, tokens: List[str], context: Dict[str, Any]) -> List[SemanticRole]:
        """Map tokens to semantic roles using Karaka principles."""
        roles = []
        
        # Combine tokens into meaningful phrases
        phrases = self._create_phrases(tokens)
        
        for phrase in phrases:
            best_role = None
            best_score = 0.0
            
            for role_type, keywords in self.role_types.items():
                # Calculate similarity with role keywords
                scores = [self.calculate_similarity(phrase, kw) for kw in keywords]
                avg_score = sum(scores) / len(scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_role = role_type
            
            if best_role and best_score > 0.3:  # Threshold for role assignment
                role = SemanticRole(
                    role_type=best_role,
                    entity=phrase,
                    confidence=best_score,
                    attributes=self._extract_attributes(phrase, best_role, context)
                )
                roles.append(role)
        
        # Add cross-references between roles
        self._add_cross_references(roles)
        
        return roles
    
    def _add_cross_references(self, roles: List[SemanticRole]):
        """Add cross-references between related roles."""
        for i, role1 in enumerate(roles):
            for j, role2 in enumerate(roles[i+1:], i+1):
                relationship = self._find_relationship(role1.role_type, role2.role_type)
                if relationship:
                    role1.relationships.append({
                        "target_role": role2.role_type,
                        "target_entity": role2.entity,
                        "relationship": relationship,
                        "confidence": min(role1.confidence, role2.confidence)
                    })
                    role2.relationships.append({
                        "target_role": role1.role_type,
                        "target_entity": role1.entity,
                        "relationship": relationship,
                        "confidence": min(role1.confidence, role2.confidence)
                    })
    
    def _find_relationship(self, role1: SemanticRoleType, role2: SemanticRoleType) -> Optional[str]:
        """Find the most appropriate relationship between two roles."""
        # Check direct relationships
        if (role1, role2) in self.role_relationships:
            return self.role_relationships[(role1, role2)][0]
        if (role2, role1) in self.role_relationships:
            return self.role_relationships[(role2, role1)][0]
        
        # Check transitive relationships
        for (r1, r2), rels in self.role_relationships.items():
            if role1 == r1 and role2 != r2:
                # Check if role2 has a relationship with r2
                if (role2, r2) in self.role_relationships:
                    return f"{rels[0]} and {self.role_relationships[(role2, r2)][0]}"
        
        return None
    
    def _validate_extracted_info(self, info: Dict[str, Any], info_type: str) -> bool:
        """Validate extracted information based on type."""
        if not info:
            return True
        
        validation_rules = {
            "temporal": {
                "absolute_time": lambda x: isinstance(parser.parse(x), datetime),
                "relative_time": lambda x: isinstance(x, str),
                "duration": lambda x: isinstance(x, str) and any(unit in x for unit in ["hour", "day", "week", "month", "year"])
            },
            "spatial": {
                "location": lambda x: isinstance(x, list) and all(isinstance(loc, str) for loc in x),
                "direction": lambda x: isinstance(x, list) and all(d in ["north", "south", "east", "west", "up", "down", "left", "right"] for d in x),
                "distance": lambda x: isinstance(x, list) and all(re.match(r"\d+\s*(miles|kilometers|meters|feet)", d) for d in x)
            },
            "modality": {
                "strength": lambda x: x in ["strong", "medium", "weak"],
                "polarity": lambda x: x in ["positive", "negative"]
            }
        }
        
        if info_type in validation_rules:
            for key, validator_func in validation_rules[info_type].items():
                if key in info and not validator_func(info[key]):
                    return False
        
        return True
    
    def _extract_attributes(self, phrase: str, role_type: SemanticRoleType, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate additional attributes for a role."""
        attributes = {
            "temporal": self._extract_temporal_info(phrase),
            "spatial": self._extract_spatial_info(phrase),
            "modality": self._extract_modality(phrase),
            "contextual": self._extract_contextual_info(phrase, context)
        }
        
        # Validate each type of information
        for info_type, info in attributes.items():
            if not self._validate_extracted_info(info, info_type):
                # If validation fails, remove the invalid information
                attributes[info_type] = None
        
        return {k: v for k, v in attributes.items() if v is not None}
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings for text using the model."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        emb1 = self.get_embeddings(text1)
        emb2 = self.get_embeddings(text2)
        return 1 - cosine(emb1, emb2)
    
    def _create_phrases(self, tokens: List[str]) -> List[str]:
        """Combine tokens into meaningful phrases."""
        phrases = []
        current_phrase = []
        
        for token in tokens:
            if token in [".", ",", "!", "?"]:
                if current_phrase:
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
            else:
                current_phrase.append(token)
        
        if current_phrase:
            phrases.append(" ".join(current_phrase))
        
        return phrases
    
    def _extract_temporal_info(self, phrase: str) -> Dict[str, Any]:
        """Extract temporal information from phrase."""
        temporal_info = {}
        
        # Process with spaCy for temporal entities
        doc = nlp(phrase)
        for ent in doc.ents:
            if ent.label_ == "DATE" or ent.label_ == "TIME":
                try:
                    parsed_date = parser.parse(ent.text)
                    temporal_info["absolute_time"] = parsed_date.isoformat()
                except:
                    temporal_info["relative_time"] = ent.text
        
        # Check for temporal patterns
        for pattern_type, pattern in self.temporal_patterns.items():
            matches = re.finditer(pattern, phrase, re.IGNORECASE)
            for match in matches:
                if pattern_type not in temporal_info:
                    temporal_info[pattern_type] = []
                temporal_info[pattern_type].append(match.group())
        
        # Extract temporal relationships
        temporal_relations = {
            "before": ["before", "prior to", "earlier than"],
            "after": ["after", "later than", "following"],
            "during": ["during", "while", "when"],
            "until": ["until", "till", "up to"]
        }
        
        for relation, keywords in temporal_relations.items():
            if any(keyword in phrase.lower() for keyword in keywords):
                temporal_info["relation"] = relation
        
        return temporal_info if temporal_info else None
    
    def _extract_spatial_info(self, phrase: str) -> Dict[str, Any]:
        """Extract spatial information from phrase."""
        spatial_info = {}
        
        # Process with spaCy for location entities
        doc = nlp(phrase)
        for ent in doc.ents:
            if ent.label_ == "GPE" or ent.label_ == "LOC":
                if "location" not in spatial_info:
                    spatial_info["location"] = []
                spatial_info["location"].append(ent.text)
        
        # Check for spatial patterns
        for pattern_type, pattern in self.spatial_patterns.items():
            matches = re.finditer(pattern, phrase, re.IGNORECASE)
            for match in matches:
                if pattern_type not in spatial_info:
                    spatial_info[pattern_type] = []
                spatial_info[pattern_type].append(match.group(1))
        
        # Extract spatial relationships
        spatial_relations = {
            "inside": ["in", "inside", "within"],
            "outside": ["outside", "out of"],
            "near": ["near", "close to", "next to"],
            "far": ["far from", "away from"]
        }
        
        for relation, keywords in spatial_relations.items():
            if any(keyword in phrase.lower() for keyword in keywords):
                spatial_info["relation"] = relation
        
        return spatial_info if spatial_info else None
    
    def _extract_modality(self, phrase: str) -> Dict[str, Any]:
        """Extract modality information from phrase."""
        modality_info = {}
        
        # Check for modality patterns
        for pattern_type, pattern in self.modality_patterns.items():
            matches = re.finditer(pattern, phrase, re.IGNORECASE)
            for match in matches:
                if pattern_type not in modality_info:
                    modality_info[pattern_type] = []
                modality_info[pattern_type].append(match.group())
        
        # Extract modality strength
        strength_indicators = {
            "strong": ["must", "have to", "need to"],
            "medium": ["should", "ought to"],
            "weak": ["can", "may", "might"]
        }
        
        for strength, indicators in strength_indicators.items():
            if any(indicator in phrase.lower() for indicator in indicators):
                modality_info["strength"] = strength
        
        # Extract polarity
        if "not" in phrase.lower() or "n't" in phrase.lower():
            modality_info["polarity"] = "negative"
        else:
            modality_info["polarity"] = "positive"
        
        return modality_info if modality_info else None
    
    def _extract_contextual_info(self, phrase: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract contextual information from phrase and context."""
        contextual_info = {}
        
        # Extract domain-specific information
        domain_keywords = {
            "technical": ["code", "program", "algorithm", "system"],
            "academic": ["research", "study", "paper", "thesis"],
            "business": ["project", "team", "meeting", "report"],
            "personal": ["I", "me", "my", "mine"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in phrase.lower() for keyword in keywords):
                contextual_info["domain"] = domain
        
        # Extract sentiment
        sentiment_indicators = {
            "positive": ["good", "great", "excellent", "successful"],
            "negative": ["bad", "poor", "failed", "unsuccessful"],
            "neutral": ["normal", "standard", "regular", "usual"]
        }
        
        for sentiment, indicators in sentiment_indicators.items():
            if any(indicator in phrase.lower() for indicator in indicators):
                contextual_info["sentiment"] = sentiment
        
        # Extract urgency
        urgency_indicators = {
            "high": ["urgent", "immediate", "asap", "critical"],
            "medium": ["soon", "shortly", "in a while"],
            "low": ["eventually", "sometime", "when possible"]
        }
        
        for urgency, indicators in urgency_indicators.items():
            if any(indicator in phrase.lower() for indicator in indicators):
                contextual_info["urgency"] = urgency
        
        # Merge with existing context
        if context:
            contextual_info["previous_context"] = context
        
        return contextual_info if contextual_info else None
    
    def visualize_semantic_network(self, roles: List[SemanticRole], 
                                 filters: Optional[Dict[str, Any]] = None,
                                 layout_algorithm: str = "force_directed",
                                 output_path: Optional[str] = None) -> go.Figure:
        """Create an interactive visualization of the semantic network with filtering and layout options."""
        # Apply filters if provided
        if filters is None:
            filters = self.filtering_capabilities
        filtered_roles = self.apply_filters(roles, filters)
        
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes
        for role in filtered_roles:
            G.add_node(
                role.entity,
                role_type=role.role_type,
                confidence=role.confidence,
                attributes=role.attributes
            )
        
        # Add edges with relationship information
        for role in filtered_roles:
            for rel in role.relationships:
                G.add_edge(
                    role.entity,
                    rel["target_entity"],
                    relationship=rel["relationship"],
                    confidence=rel["confidence"]
                )
        
        # Get node positions using specified layout algorithm
        pos = self.get_network_layout(G, layout_algorithm)
        
        # Create Plotly figure
        fig = make_subplots(rows=1, cols=1)
        
        # Add nodes to the plot
        for node in G.nodes():
            role_type = G.nodes[node]["role_type"]
            confidence = G.nodes[node]["confidence"]
            
            fig.add_trace(
                go.Scatter(
                    x=[pos[node][0]],
                    y=[pos[node][1]],
                    mode="markers+text",
                    marker=dict(
                        size=self._get_node_size(confidence),
                        color=self.visualization_settings["node_colors"][role_type],
                        line=dict(width=2, color="black")
                    ),
                    text=node,
                    textposition="bottom center",
                    name=role_type.value,
                    customdata=[{
                        "role_type": role_type.value,
                        "confidence": confidence,
                        "attributes": G.nodes[node]["attributes"]
                    }],
                    hovertemplate="<b>%{text}</b><br>" +
                                "Role: %{customdata[0].role_type}<br>" +
                                "Confidence: %{customdata[0].confidence:.2f}<br>" +
                                "<extra></extra>"
                )
            )
        
        # Add edges to the plot
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_type = G.edges[edge]["relationship"]
            confidence = G.edges[edge]["confidence"]
            
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(
                        width=self._get_edge_width(confidence),
                        color=self._get_edge_color(edge_type)
                    ),
                    hoverinfo="text",
                    text=f"{edge[0]} → {edge[1]}<br>Relationship: {edge_type}<br>Confidence: {confidence:.2f}",
                    showlegend=False
                )
            )
        
        # Add filter controls
        fig.update_layout(
            title="Semantic Network Visualization",
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    x=0.57,
                    y=1.2,
                    buttons=[
                        dict(
                            label="Force Directed",
                            method="update",
                            args=[{"layout": {"title": "Force Directed Layout"}}]
                        ),
                        dict(
                            label="Spectral",
                            method="update",
                            args=[{"layout": {"title": "Spectral Layout"}}]
                        ),
                        dict(
                            label="Spring",
                            method="update",
                            args=[{"layout": {"title": "Spring Layout"}}]
                        ),
                        dict(
                            label="Kamada-Kawai",
                            method="update",
                            args=[{"layout": {"title": "Kamada-Kawai Layout"}}]
                        ),
                        dict(
                            label="Fruchterman-Reingold",
                            method="update",
                            args=[{"layout": {"title": "Fruchterman-Reingold Layout"}}]
                        )
                    ]
                )
            ]
        )
        
        if output_path:
            fig.write_html(output_path)
        
        return fig
    
    def _get_node_size(self, confidence: float) -> int:
        """Calculate node size based on confidence."""
        if confidence > 0.8:
            return self.visualization_settings["node_sizes"]["primary"]
        elif confidence > 0.5:
            return self.visualization_settings["node_sizes"]["secondary"]
        return self.visualization_settings["node_sizes"]["tertiary"]
    
    def _get_edge_width(self, confidence: float) -> int:
        """Calculate edge width based on confidence."""
        return int(confidence * 5) + 1
    
    def _get_edge_color(self, relationship: str) -> str:
        """Get edge color based on relationship type."""
        if "and" in relationship:
            return self.visualization_settings["edge_colors"]["transitive"]
        return self.visualization_settings["edge_colors"]["direct"]
    
    def analyze_semantic_clusters(self, roles: List[SemanticRole]) -> Dict[str, Any]:
        """Analyze semantic clusters in the role network."""
        # Create feature vectors for roles
        role_vectors = []
        role_texts = []
        
        for role in roles:
            # Combine role type and entity for embedding
            text = f"{role.role_type.value} {role.entity}"
            role_texts.append(text)
            role_vectors.append(self.get_embeddings(text))
        
        # Convert to numpy array
        X = np.vstack(role_vectors)
        
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        # Apply DBSCAN for clustering
        clustering = DBSCAN(eps=0.5, min_samples=2)
        clusters = clustering.fit_predict(X_tsne)
        
        # Analyze clusters
        cluster_analysis = {
            "n_clusters": len(set(clusters)) - (1 if -1 in clusters else 0),
            "outliers": sum(1 for c in clusters if c == -1),
            "cluster_details": {}
        }
        
        for cluster_id in set(clusters):
            if cluster_id == -1:
                continue
                
            cluster_roles = [roles[i] for i, c in enumerate(clusters) if c == cluster_id]
            cluster_analysis["cluster_details"][cluster_id] = {
                "size": len(cluster_roles),
                "roles": [role.entity for role in cluster_roles],
                "role_types": [role.role_type.value for role in cluster_roles],
                "centroid": X_tsne[clusters == cluster_id].mean(axis=0).tolist()
            }
        
        return cluster_analysis
    
    def visualize_semantic_clusters(self, roles: List[SemanticRole], output_path: Optional[str] = None) -> go.Figure:
        """Create an interactive visualization of semantic clusters."""
        # Get cluster analysis
        cluster_analysis = self.analyze_semantic_clusters(roles)
        
        # Create feature vectors and apply t-SNE
        role_vectors = [self.get_embeddings(f"{role.role_type.value} {role.entity}") for role in roles]
        X = np.vstack(role_vectors)
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add scatter plot for each cluster
        for cluster_id in set(cluster_analysis["cluster_details"].keys()):
            cluster_data = cluster_analysis["cluster_details"][cluster_id]
            mask = [i for i, role in enumerate(roles) if role.entity in cluster_data["roles"]]
            
            fig.add_trace(
                go.Scatter(
                    x=X_tsne[mask, 0],
                    y=X_tsne[mask, 1],
                    mode="markers+text",
                    marker=dict(
                        size=10,
                        color=self.visualization_settings["node_colors"][roles[mask[0]].role_type]
                    ),
                    text=[roles[i].entity for i in mask],
                    textposition="top center",
                    name=f"Cluster {cluster_id}"
                )
            )
        
        # Update layout
        fig.update_layout(
            title="Semantic Role Clusters",
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        if output_path:
            fig.write_html(output_path)
        
        return fig
    
    def visualize_timeline(self, roles: List[SemanticRole], output_path: Optional[str] = None) -> go.Figure:
        """Create an interactive timeline visualization of temporal relationships."""
        # Extract temporal information
        temporal_events = []
        for role in roles:
            if "temporal" in role.attributes:
                temporal_info = role.attributes["temporal"]
                if "absolute_time" in temporal_info:
                    event = {
                        "role": role.entity,
                        "time": parser.parse(temporal_info["absolute_time"]),
                        "type": "milestone",
                        "confidence": role.confidence
                    }
                    temporal_events.append(event)
                elif "duration" in temporal_info:
                    duration = temporal_info["duration"]
                    event = {
                        "role": role.entity,
                        "duration": duration,
                        "type": "duration",
                        "confidence": role.confidence
                    }
                    temporal_events.append(event)
        
        # Sort events by time
        temporal_events.sort(key=lambda x: x.get("time", datetime.min))
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add timeline base
        fig.add_shape(
            type="line",
            x0=0,
            y0=self.timeline_settings["timeline_height"] / 2,
            x1=1,
            y1=self.timeline_settings["timeline_height"] / 2,
            line=dict(color="black", width=2)
        )
        
        # Add events
        for i, event in enumerate(temporal_events):
            y_pos = self.timeline_settings["timeline_height"] / 2 + (
                (i % 2) * 2 - 1
            ) * self.timeline_settings["event_height"]
            
            if event["type"] == "milestone":
                fig.add_trace(
                    go.Scatter(
                        x=[event["time"]],
                        y=[y_pos],
                        mode="markers+text",
                        marker=dict(
                            size=10,
                            color=self.timeline_settings["event_colors"]["milestone"]
                        ),
                        text=event["role"],
                        textposition="top center",
                        name="Milestone"
                    )
                )
            elif event["type"] == "duration":
                fig.add_trace(
                    go.Scatter(
                        x=[event["time"], event["time"] + event["duration"]],
                        y=[y_pos, y_pos],
                        mode="lines+markers+text",
                        line=dict(
                            width=3,
                            color=self.timeline_settings["event_colors"]["duration"]
                        ),
                        marker=dict(
                            size=8,
                            color=self.timeline_settings["event_colors"]["duration"]
                        ),
                        text=[event["role"], ""],
                        textposition="top center",
                        name="Duration"
                    )
                )
        
        # Update layout
        fig.update_layout(
            title="Temporal Relationships Timeline",
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(
                title="Time",
                type="date",
                showgrid=True
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            )
        )
        
        if output_path:
            fig.write_html(output_path)
        
        return fig
    
    def visualize_hierarchical_tree(self, roles: List[SemanticRole], output_path: Optional[str] = None) -> go.Figure:
        """Create an interactive hierarchical tree visualization."""
        # Create hierarchical tree structure
        root = Node("Root")
        nodes = {role.entity: Node(role.entity, parent=root) for role in roles}
        
        # Add relationships
        for role in roles:
            for rel in role.relationships:
                if rel["target_entity"] in nodes:
                    Node(
                        f"{rel['relationship']}",
                        parent=nodes[rel["target_entity"]],
                        relationship=rel
                    )
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add nodes and edges
        for node in RenderTree(root):
            if node.node.name == "Root":
                continue
                
            # Get node properties
            node_type = "root" if node.node.parent.name == "Root" else "branch"
            color = self.hierarchical_settings["node_colors"][node_type]
            size = self.hierarchical_settings["node_sizes"][node_type]
            
            # Add node
            fig.add_trace(
                go.Scatter(
                    x=[node.depth],
                    y=[node.node.name],
                    mode="markers+text",
                    marker=dict(
                        size=size,
                        color=color
                    ),
                    text=node.node.name,
                    textposition="middle right",
                    name=node_type.capitalize()
                )
            )
            
            # Add edge if not root
            if node.node.parent.name != "Root":
                edge_color = self._get_hierarchical_edge_color(node.node.relationship)
                fig.add_trace(
                    go.Scatter(
                        x=[node.depth - 1, node.depth],
                        y=[node.node.parent.name, node.node.name],
                        mode="lines",
                        line=dict(
                            width=2,
                            color=edge_color
                        ),
                        showlegend=False
                    )
                )
        
        # Update layout
        fig.update_layout(
            title="Hierarchical Semantic Tree",
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(
                title="Depth",
                showgrid=True
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            )
        )
        
        if output_path:
            fig.write_html(output_path)
        
        return fig
    
    def _get_hierarchical_edge_color(self, relationship: Dict[str, Any]) -> str:
        """Get edge color based on relationship confidence."""
        confidence = relationship.get("confidence", 0.5)
        if confidence > 0.8:
            return self.hierarchical_settings["edge_colors"]["strong"]
        elif confidence > 0.5:
            return self.hierarchical_settings["edge_colors"]["medium"]
        return self.hierarchical_settings["edge_colors"]["weak"]
    
    def export_tree_dot(self, roles: List[SemanticRole], output_path: str):
        """Export hierarchical tree as DOT file for Graphviz visualization."""
        # Create hierarchical tree structure
        root = Node("Root")
        nodes = {role.entity: Node(role.entity, parent=root) for role in roles}
        
        # Add relationships
        for role in roles:
            for rel in role.relationships:
                if rel["target_entity"] in nodes:
                    Node(
                        f"{rel['relationship']}",
                        parent=nodes[rel["target_entity"]],
                        relationship=rel
                    )
        
        # Export to DOT format
        DotExporter(root).to_dotfile(output_path)
    
    def get_network_layout(self, G: nx.Graph, algorithm: str = "force_directed") -> Dict[str, np.ndarray]:
        """Get node positions using specified layout algorithm."""
        if algorithm not in self.layout_algorithms:
            raise ValueError(f"Unknown layout algorithm: {algorithm}")
        
        params = self.layout_algorithms[algorithm]
        
        if algorithm == "force_directed":
            pos = nx.spring_layout(
                G,
                k=params["k"],
                iterations=params["iterations"],
                threshold=params["threshold"],
                dim=params["dim"]
            )
        elif algorithm == "spectral":
            pos = nx.spectral_layout(
                G,
                dim=params["dim"],
                weight=params["weight"],
                scale=params["scale"]
            )
        elif algorithm == "spring":
            pos = nx.spring_layout(
                G,
                k=params["k"],
                pos=params["pos"],
                fixed=params["fixed"],
                iterations=params["iterations"],
                threshold=params["threshold"],
                weight=params["weight"],
                scale=params["scale"],
                center=params["center"],
                dim=params["dim"],
                seed=params["seed"]
            )
        elif algorithm == "kamada_kawai":
            pos = nx.kamada_kawai_layout(
                G,
                dist=params["dist"],
                pos=params["pos"],
                weight=params["weight"],
                scale=params["scale"],
                center=params["center"],
                dim=params["dim"]
            )
        elif algorithm == "fruchterman_reingold":
            pos = nx.fruchterman_reingold_layout(
                G,
                k=params["k"],
                pos=params["pos"],
                fixed=params["fixed"],
                iterations=params["iterations"],
                threshold=params["threshold"],
                weight=params["weight"],
                scale=params["scale"],
                center=params["center"],
                dim=params["dim"],
                seed=params["seed"]
            )
        
        return pos
    
    def apply_filters(self, roles: List[SemanticRole], filters: Dict[str, Any]) -> List[SemanticRole]:
        """Apply filters to semantic roles."""
        filtered_roles = roles.copy()
        
        if filters.get("role_types", {}).get("enabled", False):
            threshold = filters["role_types"].get("threshold", 0.3)
            filtered_roles = [
                role for role in filtered_roles
                if role.confidence >= threshold
            ]
        
        if filters.get("confidence", {}).get("enabled", False):
            min_confidence = filters["confidence"].get("min_confidence", 0.5)
            filtered_roles = [
                role for role in filtered_roles
                if role.confidence >= min_confidence
            ]
        
        if filters.get("relationships", {}).get("enabled", False):
            min_strength = filters["relationships"].get("min_strength", 0.5)
            for role in filtered_roles:
                role.relationships = [
                    rel for rel in role.relationships
                    if rel.get("confidence", 0.0) >= min_strength
                ]
        
        if filters.get("clusters", {}).get("enabled", False):
            min_size = filters["clusters"].get("min_size", 2)
            cluster_analysis = self.analyze_semantic_clusters(filtered_roles)
            valid_clusters = {
                cluster_id for cluster_id, details in cluster_analysis["cluster_details"].items()
                if details["size"] >= min_size
            }
            filtered_roles = [
                role for i, role in enumerate(filtered_roles)
                if any(
                    role.entity in cluster_analysis["cluster_details"][cluster_id]["roles"]
                    for cluster_id in valid_clusters
                )
            ]
        
        return filtered_roles
    
    def validate_domain_rules(self, role: SemanticRole, domain: str) -> List[str]:
        """Validate a role against domain-specific rules."""
        issues = []
        
        if domain not in self.domain_validation_rules:
            return [f"Unknown domain: {domain}"]
        
        rules = self.domain_validation_rules[domain]
        
        # Check required attributes
        for attr in rules["required_attributes"]:
            if attr not in role.attributes:
                issues.append(f"Missing required attribute for {domain} domain: {attr}")
        
        # Check valid role types
        if role.role_type not in rules["valid_role_types"]:
            issues.append(f"Invalid role type for {domain} domain: {role.role_type}")
        
        # Check minimum confidence
        if role.confidence < rules["min_confidence"]:
            issues.append(f"Confidence below minimum threshold for {domain} domain")
        
        # Check scope requirements
        for scope_type, required_fields in rules["scope_requirements"].items():
            scope = getattr(role, f"{scope_type}_scope")
            if scope:
                for field in required_fields:
                    if field not in scope:
                        issues.append(f"Missing required {scope_type} scope field: {field}")
            else:
                issues.append(f"Missing required {scope_type} scope")
        
        # Check relationship rules
        if "relationship_rules" in rules:
            for rel_type, rel_rules in rules["relationship_rules"].items():
                if rel_rules["required"]:
                    found = False
                    for rel in role.relationships:
                        if rel["type"] == rel_type and rel["strength"] >= rel_rules["min_strength"]:
                            found = True
                            break
                    if not found:
                        issues.append(f"Missing required relationship of type {rel_type} with minimum strength {rel_rules['min_strength']}")
        
        return issues
    
    def extract_domain_relationships(self, text: str, domain: str) -> List[Dict[str, Any]]:
        """Extract domain-specific relationships from text."""
        relationships = []
        
        if domain not in self.domain_relationship_patterns:
            return relationships
        
        patterns = self.domain_relationship_patterns[domain]
        
        for rel_type, rel_config in patterns.items():
            matches = re.finditer(rel_config["pattern"], text, re.IGNORECASE)
            for match in matches:
                relationships.append({
                    "type": rel_type,
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "strength": rel_config["strength"],
                    "direction": rel_config["direction"]
                })
        
        return relationships

class SemanticAbstractionLayer:
    def __init__(self):
        """Initialize the semantic abstraction layer."""
        self.nlp = spacy.load("en_core_web_sm")
        self.karaka_mapper = KarakaMapper()
        self.domain_mapper = DomainMapper()
        
    async def process(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        domain_mapping: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a query through semantic analysis.
        
        Args:
            query: The input query to process
            context: Optional context information
            domain_mapping: Optional domain-specific mapping rules
            
        Returns:
            Dictionary containing semantic analysis results
        """
        try:
            if not query:
                return {"error": "Empty query provided"}
            
            # Basic NLP processing
            doc = self.nlp(query)
            
            # Extract embeddings
            embeddings = self._get_embeddings(doc)
            
            # Extract semantic roles using Karaka grammar
            roles = self.karaka_mapper.map_roles(query, context)
            
            # Apply domain-specific mapping if provided
            if domain_mapping:
                roles = self.domain_mapper.apply_mapping(roles, domain_mapping)
            
            # Extract key entities and relationships
            entities = self._extract_entities(doc)
            relationships = self._extract_relationships(doc)
            
            # Analyze temporal and spatial dimensions
            temporal_info = self._analyze_temporal_dimensions(doc)
            spatial_info = self._analyze_spatial_dimensions(doc)
            
            # Calculate confidence scores
            confidence = self._calculate_confidence(doc, roles, context)
            
            return {
                "query": query,
                "embeddings": embeddings.tolist(),
                "roles": roles,
                "entities": entities,
                "relationships": relationships,
                "temporal_dimensions": temporal_info,
                "spatial_dimensions": spatial_info,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "context_used": bool(context),
                "domain_mapping_used": bool(domain_mapping)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_embeddings(self, doc) -> np.ndarray:
        """Get document embeddings."""
        return doc.vector
    
    def _extract_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract named entities and their properties."""
        return [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "properties": {
                    "is_proper": ent.root.pos_ == "PROPN",
                    "dependencies": [token.dep_ for token in ent]
                }
            }
            for ent in doc.ents
        ]
    
    def _extract_relationships(self, doc) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        relationships = []
        for token in doc:
            if token.dep_ in ["ROOT", "nsubj", "dobj", "pobj"]:
                relationships.append({
                    "source": token.head.text,
                    "target": token.text,
                    "type": token.dep_,
                    "confidence": 0.8
                })
        return relationships
    
    def _analyze_temporal_dimensions(self, doc) -> Dict[str, Any]:
        """Analyze temporal aspects of the text."""
        temporal_info = {
            "time_expressions": [],
            "temporal_relations": [],
            "sequence_indicators": []
        }
        
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME"]:
                temporal_info["time_expressions"].append({
                    "text": ent.text,
                    "type": ent.label_
                })
                
        return temporal_info
    
    def _analyze_spatial_dimensions(self, doc) -> Dict[str, Any]:
        """Analyze spatial aspects of the text."""
        spatial_info = {
            "locations": [],
            "spatial_relations": [],
            "distance_expressions": []
        }
        
        for ent in doc.ents:
            if ent.label_ in ["LOC", "GPE"]:
                spatial_info["locations"].append({
                    "text": ent.text,
                    "type": ent.label_
                })
                
        return spatial_info
    
    def _calculate_confidence(
        self,
        doc,
        roles: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for the semantic analysis."""
        # Base confidence
        confidence = 0.7
        
        # Adjust based on entity recognition
        if doc.ents:
            confidence += 0.1
            
        # Adjust based on role mapping
        if roles and any(roles.values()):
            confidence += 0.1
            
        # Adjust based on context utilization
        if context:
            confidence += 0.1
            
        return min(confidence, 1.0)