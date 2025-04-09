import re
from typing import Dict, List, Any, Optional

import spacy
from src.core.services.types import KarakaRole, SemanticDimension, KarakaMapping


class KarakaMapper:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.role_patterns = {
            KarakaRole.AGENT: ["nsubj", "nsubjpass", "agent"],
            KarakaRole.OBJECT: ["dobj", "pobj", "attr", "acomp"],
            KarakaRole.INSTRUMENT: ["prep_with", "prep_by", "prep_using"],
            KarakaRole.LOCATION: ["prep_in", "prep_on", "prep_at", "prep_under", "prep_over"],
            KarakaRole.SOURCE: ["prep_from", "prep_out_of"],
            KarakaRole.DESTINATION: ["prep_to", "prep_into", "prep_onto"],
            KarakaRole.BENEFICIARY: ["prep_for", "prep_to"],
            KarakaRole.TIME: ["prep_at", "prep_on", "prep_in", "prep_during"],
            KarakaRole.MANNER: ["prep_with", "prep_by", "advmod"],
            KarakaRole.CAUSE: ["prep_because", "prep_due_to", "prep_owing_to"],
            KarakaRole.PURPOSE: ["prep_for", "prep_to", "prep_in_order_to"]
        }
        self.temporal_patterns = {
            "absolute": r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}",
            "relative": r"(today|tomorrow|yesterday|now|later|earlier)",
            "duration": r"(\d+\s+(hours|days|weeks|months|years))",
            "frequency": r"(daily|weekly|monthly|yearly|always|never)"
        }
        self.spatial_patterns = {
            "location": r"(in|on|at|under|over|beside|near|far from)\s+(\w+)",
            "direction": r"(to|from|towards|away from)\s+(\w+)",
            "distance": r"(\d+\s+(meters|kilometers|miles|feet))"
        }

    def map_roles(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[KarakaRole, List[Dict[str, Any]]]:
        """
        Map semantic roles to entities in the text using Karaka grammar principles.
        
        Args:
            text: The input text to analyze
            context: Optional context information for enhanced role mapping
            
        Returns:
            Dictionary mapping Karaka roles to lists of entities with their properties
        """
        doc = self.nlp(text)
        role_mapping = {role: [] for role in KarakaRole}

        # Enhanced role mapping with context awareness
        for token in doc:
            for role, patterns in self.role_patterns.items():
                if any(token.dep_ == pattern for pattern in patterns):
                    entity = {
                        "text": token.text,
                        "lemma": token.lemma_,
                        "pos": token.pos_,
                        "dep": token.dep_,
                        "head": token.head.text,
                        "children": [child.text for child in token.children],
                        "confidence": self._calculate_confidence(token, role, context),
                        "temporal_info": self._extract_temporal_info(token.text),
                        "spatial_info": self._extract_spatial_info(token.text),
                        "context_aware": self._is_context_aware(token, context)
                    }
                    role_mapping[role].append(entity)

        return role_mapping

    def analyze_semantic_roles(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform a comprehensive semantic role analysis of the text.
        
        Args:
            text: The input text to analyze
            context: Optional context information for enhanced analysis
            
        Returns:
            Dictionary containing the semantic analysis results
        """
        role_mapping = self.map_roles(text, context)

        analysis = {
            "text": text,
            "roles": {
                role.name: [
                    {
                        "entity": entity["text"],
                        "properties": {
                            "lemma": entity["lemma"],
                            "pos": entity["pos"],
                            "dependency": entity["dep"],
                            "head": entity["head"],
                            "children": entity["children"],
                            "confidence": entity["confidence"],
                            "temporal_info": entity["temporal_info"],
                            "spatial_info": entity["spatial_info"],
                            "context_aware": entity["context_aware"]
                        }
                    }
                    for entity in entities
                ]
                for role, entities in role_mapping.items()
            },
            "semantic_graph": self._build_semantic_graph(role_mapping),
            "temporal_analysis": self._analyze_temporal_structure(text),
            "spatial_analysis": self._analyze_spatial_structure(text),
            "context_analysis": self._analyze_context(text, context)
        }

        return analysis

    def _build_semantic_graph(self, role_mapping: Dict[KarakaRole, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Build a semantic graph representation of the role mappings.
        
        Args:
            role_mapping: The role mapping dictionary
            
        Returns:
            Dictionary representing the semantic graph
        """
        graph = {
            "nodes": [],
            "edges": [],
            "clusters": []
        }

        # Add nodes for each entity
        for role, entities in role_mapping.items():
            for entity in entities:
                node = {
                    "id": f"{role.name}_{entity['text']}",
                    "label": entity["text"],
                    "role": role.name,
                    "properties": {
                        "lemma": entity["lemma"],
                        "pos": entity["pos"],
                        "confidence": entity["confidence"]
                    }
                }
                graph["nodes"].append(node)

                # Add edges for dependencies
                if entity["head"]:
                    edge = {
                        "source": f"{role.name}_{entity['text']}",
                        "target": f"HEAD_{entity['head']}",
                        "type": entity["dep"],
                        "weight": entity["confidence"]
                    }
                    graph["edges"].append(edge)

        # Add temporal and spatial clusters
        graph["clusters"].extend(self._build_temporal_clusters(role_mapping))
        graph["clusters"].extend(self._build_spatial_clusters(role_mapping))

        return graph

    def _calculate_confidence(self, token: Any, role: KarakaRole, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate confidence score for role assignment."""
        base_confidence = 0.7  # Default confidence

        # Adjust based on dependency pattern match
        if token.dep_ in self.role_patterns[role]:
            base_confidence += 0.2

        # Adjust based on context awareness
        if context and self._is_context_aware(token, context):
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _extract_temporal_info(self, text: str) -> Dict[str, Any]:
        """Extract temporal information from text."""
        temporal_info = {
            "type": None,
            "value": None,
            "confidence": 0.0
        }

        for pattern_type, pattern in self.temporal_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                temporal_info["type"] = pattern_type
                temporal_info["value"] = matches[0]
                temporal_info["confidence"] = 0.8
                break

        return temporal_info

    def _extract_spatial_info(self, text: str) -> Dict[str, Any]:
        """Extract spatial information from text."""
        spatial_info = {
            "type": None,
            "value": None,
            "confidence": 0.0
        }

        for pattern_type, pattern in self.spatial_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                spatial_info["type"] = pattern_type
                spatial_info["value"] = matches[0]
                spatial_info["confidence"] = 0.8
                break

        return spatial_info

    def _is_context_aware(self, token: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if token is context-aware based on provided context."""
        if not context:
            return False

        # Check if token appears in context
        return any(
            token.text.lower() in value.lower() or
            token.lemma_.lower() in value.lower()
            for value in context.values()
            if isinstance(value, str)
        )

    def _analyze_temporal_structure(self, text: str) -> Dict[str, Any]:
        """Analyze temporal structure of the text."""
        temporal_structure = {
            "absolute_time": [],
            "relative_time": [],
            "duration": [],
            "frequency": []
        }

        for pattern_type, pattern in self.temporal_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                temporal_structure[pattern_type].extend(matches)

        return temporal_structure

    def _analyze_spatial_structure(self, text: str) -> Dict[str, Any]:
        """Analyze spatial structure of the text."""
        spatial_structure = {
            "locations": [],
            "directions": [],
            "distances": []
        }

        for pattern_type, pattern in self.spatial_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                spatial_structure[pattern_type].extend(matches)

        return spatial_structure

    def _analyze_context(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze context relevance and relationships."""
        if not context:
            return {"relevance": 0.0, "relationships": []}

        doc = self.nlp(text)
        context_doc = self.nlp(" ".join(str(v) for v in context.values()))

        # Calculate context relevance
        relevance = self._calculate_context_relevance(doc, context_doc)

        # Find relationships between text and context
        relationships = self._find_context_relationships(doc, context_doc)

        return {
            "relevance": relevance,
            "relationships": relationships
        }

    def _calculate_context_relevance(self, doc: Any, context_doc: Any) -> float:
        """Calculate relevance between text and context."""
        # Simple word overlap measure
        text_words = set(token.text.lower() for token in doc)
        context_words = set(token.text.lower() for token in context_doc)

        overlap = len(text_words.intersection(context_words))
        total = len(text_words.union(context_words))

        return overlap / total if total > 0 else 0.0

    def _find_context_relationships(self, doc: Any, context_doc: Any) -> List[Dict[str, Any]]:
        """Find relationships between text and context."""
        relationships = []

        for token1 in doc:
            for token2 in context_doc:
                if token1.text.lower() == token2.text.lower():
                    relationships.append({
                        "text_token": token1.text,
                        "context_token": token2.text,
                        "relationship": "exact_match"
                    })
                elif token1.lemma_ == token2.lemma_:
                    relationships.append({
                        "text_token": token1.text,
                        "context_token": token2.text,
                        "relationship": "lemma_match"
                    })

        return relationships

    def _build_temporal_clusters(self, role_mapping: Dict[KarakaRole, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Build temporal clusters from role mapping."""
        temporal_clusters = []

        for role, entities in role_mapping.items():
            temporal_entities = [
                entity for entity in entities
                if entity["temporal_info"]["type"] is not None
            ]

            if temporal_entities:
                cluster = {
                    "type": "temporal",
                    "entities": [
                        {
                            "id": f"{role.name}_{entity['text']}",
                            "temporal_info": entity["temporal_info"]
                        }
                        for entity in temporal_entities
                    ]
                }
                temporal_clusters.append(cluster)

        return temporal_clusters

    def _build_spatial_clusters(self, role_mapping: Dict[KarakaRole, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Build spatial clusters from role mapping."""
        spatial_clusters = []

        for role, entities in role_mapping.items():
            spatial_entities = [
                entity for entity in entities
                if entity["spatial_info"]["type"] is not None
            ]

            if spatial_entities:
                cluster = {
                    "type": "spatial",
                    "entities": [
                        {
                            "id": f"{role.name}_{entity['text']}",
                            "spatial_info": entity["spatial_info"]
                        }
                        for entity in spatial_entities
                    ]
                }
                spatial_clusters.append(cluster)

        return spatial_clusters

    def map_karaka_roles(self, text: str) -> Dict[str, Any]:
        """Map karaka roles to text."""
        if not text:
            raise ValueError("Empty text")

        # Process text to extract roles
        roles = {
            "agent": [],
            "object": [],
            "instrument": [],
            "attributes": [],
            "relationships": []
        }

        # Split text into tokens
        tokens = text.split()

        # Process each token
        for token in tokens:
            # Check for agent roles
            if any(word in token.lower() for word in self.agent_words):
                roles["agent"].append(token)
            # Check for object roles
            elif any(word in token.lower() for word in self.object_words):
                roles["object"].append(token)
            # Check for instrument roles
            elif any(word in token.lower() for word in self.instrument_words):
                roles["instrument"].append(token)

        # Join role lists into strings
        for role in roles:
            roles[role] = " ".join(roles[role]) if roles[role] else ""

        return roles
