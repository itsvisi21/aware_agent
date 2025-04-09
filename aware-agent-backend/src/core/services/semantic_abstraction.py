# Standard library imports
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple, Union

# Third-party imports
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import spacy
import torch
import torch.nn as nn
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from dateutil import parser
from plotly.subplots import make_subplots
from pydantic import BaseModel, validator, Field
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

# Local imports
from src.core.services.types import SemanticDimension, KarakaMapping, KarakaRole
from src.core.services.domain_mappings import DomainMapper

# Added imports
from transformers import BertTokenizer, BertModel

logger = logging.getLogger(__name__)

# Load spaCy model for NLP processing
nlp = spacy.load("en_core_web_sm")

class ContextNode(BaseModel):
    """Represents a node in the context tree."""
    name: str
    dimension: SemanticDimension
    karaka: KarakaMapping
    children: List["ContextNode"] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_roles: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, name: str, dimension: SemanticDimension, karaka: KarakaMapping, **kwargs):
        super().__init__(name=name, dimension=dimension, karaka=karaka, **kwargs)

    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        return v

    def add_child(self, child: "ContextNode") -> None:
        """Add a child node to this node."""
        if not child:
            raise ValueError("Invalid child node")
        self.children.append(child)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a dictionary representation."""
        return {
            "name": self.name,
            "dimension": self.dimension.to_dict(),
            "karaka": self.karaka.to_dict(),
            "children": [child.to_dict() for child in self.children],
            "confidence": self.confidence,
            "semantic_roles": self.semantic_roles
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextNode":
        """Create a node from a dictionary representation."""
        return cls(
            name=data["name"],
            dimension=SemanticDimension.from_dict(data["dimension"]),
            karaka=KarakaMapping.from_dict(data["karaka"]),
            children=[cls.from_dict(child) for child in data["children"]],
            confidence=data["confidence"],
            semantic_roles=data.get("semantic_roles", {})
        )

    def visualize_context_tree(self) -> str:
        """Visualize the context tree as a string.
        
        Returns:
            str: A string representation of the context tree.
        """
        def _visualize(node: ContextNode, level: int = 0) -> str:
            indent = "  " * level
            result = f"{indent}{node.name} ({node.dimension.domain})\n"
            for child in node.children:
                result += _visualize(child, level + 1)
            return result
            
        return _visualize(self)

class SimpleEncoder(nn.Module):
    def __init__(self, vocab_size=30000, embedding_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, x):
        return self.embedding(x).mean(dim=1)

class KarakaMapper:
    """Maps tokens to karaka roles."""
    
    def __init__(self):
        """Initialize the karaka mapper."""
        self.nlp = spacy.load("en_core_web_sm")
        self.agent_words = {"i", "we", "you", "he", "she", "they", "it", "who", "what", "which"}
        self.object_words = {"me", "us", "you", "him", "her", "them", "it", "whom", "what", "which"}
        self.instrument_words = {"with", "using", "by", "through", "via", "by means of"}
        self.role_mappings = {
            "nsubj": "agent",
            "dobj": "object",
            "pobj": "instrument",
            "attr": "attribute",
            "relcl": "relationship"
        }

    def map_karaka_roles(self, tokens: Union[List[Dict[str, Any]], List[str]]) -> Dict[str, Any]:
        """Map tokens to karaka roles."""
        if not tokens:
            raise ValueError("Tokens cannot be empty")
        
        # Handle both string and dictionary tokens
        text = " ".join(token if isinstance(token, str) else token.get("text", "") for token in tokens)
        doc = self.nlp(text)
        roles = {
            "agent": [],
            "object": [],
            "instrument": [],
            "attributes": [],
            "relationships": []
        }
        
        # Process each token in the document
        for token in doc:
            # Check for agent role
            if token.text.lower() in self.agent_words:
                roles["agent"].append(token.text)
            # Check for object role
            elif token.text.lower() in self.object_words:
                roles["object"].append(token.text)
            # Check for instrument role
            elif token.text.lower() in self.instrument_words:
                roles["instrument"].append(token.text)
            # Add attributes and relationships
            elif token.pos_ in ["ADJ", "ADV"]:
                roles["attributes"].append(token.text)
            elif token.dep_ in ["nsubj", "dobj", "iobj", "pobj"]:
                roles["relationships"].append(token.text)
        
        # Convert lists to strings
        for key in roles:
            roles[key] = " ".join(roles[key]) if roles[key] else ""
        
        return roles

    def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings for text using the model."""
        # Convert text to indices (simplified)
        indices = torch.tensor([hash(word) % 30000 for word in text.split()])
        with torch.no_grad():
            embeddings = self.model(indices).mean(dim=0)
        return embeddings.numpy()

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        emb1 = self.get_embeddings(text1)
        emb2 = self.get_embeddings(text2)
        return 1 - cosine(emb1.flatten(), emb2.flatten())

    # ... existing code ...

class SemanticAbstractionLayer:
    """Layer responsible for semantic analysis and abstraction."""

    def __init__(self):
        """Initialize the semantic abstraction layer."""
        self.nlp = spacy.load("en_core_web_sm")
        self.karaka_mapper = KarakaMapper()

    def tokenize_and_annotate(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Tokenize and annotate the input text."""
        if not text:
            raise ValueError("Input text cannot be empty")

        doc = self.nlp(text)
        tokens = []
        annotations = {
            "domain": context.get("domain", "general") if context else "general",
            "entities": [],
            "noun_phrases": [],
            "verbs": [],
            "adjectives": [],
            "adverbs": []
        }

        for token in doc:
            tokens.append({
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "shape": token.shape_,
                "is_alpha": token.is_alpha,
                "is_stop": token.is_stop
            })

        # Extract named entities
        for ent in doc.ents:
            annotations["entities"].append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

        # Extract noun phrases
        for chunk in doc.noun_chunks:
            annotations["noun_phrases"].append({
                "text": chunk.text,
                "root": chunk.root.text,
                "start": chunk.start_char,
                "end": chunk.end_char
            })

        # Extract verbs, adjectives, and adverbs
        for token in doc:
            if token.pos_ == "VERB":
                annotations["verbs"].append({
                    "text": token.text,
                    "lemma": token.lemma_,
                    "tense": token.morph.get("Tense", [""])[0]
                })
            elif token.pos_ == "ADJ":
                annotations["adjectives"].append({
                    "text": token.text,
                    "lemma": token.lemma_,
                    "degree": token.morph.get("Degree", [""])[0]
                })
            elif token.pos_ == "ADV":
                annotations["adverbs"].append({
                    "text": token.text,
                    "lemma": token.lemma_
                })

        return {
            "tokens": tokens,
            "annotations": annotations,
            "context": context or {}
        }

    def build_context_tree(self, tokenized_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a context tree from tokenized data."""
        if not tokenized_data or "tokens" not in tokenized_data:
            raise ValueError("Invalid tokenized data")

        root_node = ContextNode(
            name="root",
            dimension=SemanticDimension(
                domain=context.get("domain", "research") if context else "research",
                role=context.get("role", "researcher") if context else "researcher",
                objective=context.get("objective", "analyze") if context else "analyze",
                timeframe=context.get("timeframe", "current") if context else "current"
            ),
            karaka=KarakaMapping(
                agent=context.get("agent", "user") if context else "user",
                object=context.get("object", "quantum_entanglement") if context else "quantum_entanglement",
                instrument=context.get("instrument", "research_tools") if context else "research_tools"
            )
        )

        # Add semantic roles to the result
        semantic_roles = self.assign_semantic_roles(tokenized_data)
        root_node.semantic_roles = semantic_roles

        # Process temporal constraints
        temporal_dimensions = self.process_temporal_constraints(tokenized_data, context)

        # Split text into sentences and create sentence nodes
        sentences = self._split_into_sentences(tokenized_data["tokens"])
        for i, sentence_tokens in enumerate(sentences):
            sentence_node = ContextNode(
                name=f"sentence_{i}",
                dimension=SemanticDimension(
                    domain="sentence",
                    role="sentence",
                    objective="sentence",
                    timeframe="sentence"
                ),
                karaka=KarakaMapping(
                    agent="sentence",
                    object="sentence",
                    instrument="sentence"
                )
            )
            root_node.children.append(sentence_node)

        result = root_node.dict()
        result["query"] = tokenized_data.get("query", "")
        result["semantic_roles"] = semantic_roles
        result["temporal_dimensions"] = temporal_dimensions
        result["domain_context"] = {
            "domain": context.get("domain", "research") if context else "research",
            "type": context.get("type", "query") if context else "query",
            "focus": context.get("focus", "methodology") if context else "methodology"
        }
        return result

    def assign_semantic_roles(self, tokenized_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assign semantic roles to tokens using KarakaMapper."""
        if not tokenized_data or "tokens" not in tokenized_data:
            raise ValueError("Invalid tokenized data")

        tokens = tokenized_data["tokens"]
        if not tokens:
            raise ValueError("Empty tokens list")

        # Convert tokens to text
        text = " ".join(token["text"] if isinstance(token, dict) else str(token) for token in tokens)
        
        # Map karaka roles
        roles = self.karaka_mapper.map_karaka_roles(text)
        
        # Return roles at root level
        result = {
            "tokens": tokens,
            "annotations": tokenized_data.get("annotations", {}),
            **roles  # Spread roles at root level
        }
        return result

    def process_temporal_constraints(self, tokenized_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process temporal constraints from tokenized data."""
        if not tokenized_data or "tokens" not in tokenized_data:
            raise ValueError("Invalid tokenized data")

        tokens = tokenized_data["tokens"]
        if not tokens:
            raise ValueError("Empty tokens list")

        # Initialize temporal information
        temporal_info = {
            "timeframes": [],
            "durations": [],
            "frequencies": []
        }

        # Process each token for temporal information
        for token in tokens:
            token_text = token["text"] if isinstance(token, dict) else str(token)
            token_tag = token.get("tag", "") if isinstance(token, dict) else ""
            
            if "TIME" in token_tag or "DATE" in token_tag:
                temporal_info["timeframes"].append(token_text)
            elif "DURATION" in token_tag:
                temporal_info["durations"].append(token_text)
            elif "FREQUENCY" in token_tag:
                temporal_info["frequencies"].append(token_text)

        # Determine urgency based on temporal information
        urgency = "normal"
        urgent_words = {"immediate", "urgent", "asap", "now", "quickly", "soon"}
        text = " ".join(token["text"] if isinstance(token, dict) else str(token) for token in tokens).lower()
        if any(word in text for word in urgent_words):
            urgency = "high"
        elif temporal_info["timeframes"] or temporal_info["durations"]:
            urgency = "medium"

        # Create result with context and temporal information
        result = {
            "temporal_info": temporal_info,
            "context": {
                "domain": context.get("domain", "research") if context else "research",
                "type": context.get("type", "query") if context else "query",
                "focus": context.get("focus", "methodology") if context else "methodology",
                "query": tokenized_data.get("query", ""),
                "role": context.get("role", "researcher") if context else "researcher",
                "objective": context.get("objective", "analyze") if context else "analyze",
                "timeframe": context.get("timeframe", "current") if context else "current",
                "agent": context.get("agent", "user") if context else "user",
                "object": context.get("object", "quantum_entanglement") if context else "quantum_entanglement",
                "instrument": context.get("instrument", "research_tools") if context else "research_tools"
            }
        }

        # Add timeframe and urgency to the root level of the result
        result["timeframe"] = result["context"]["timeframe"]
        result["urgency"] = urgency
        return result

    def create_context_node(self, node_id: str, context: Optional[Dict[str, Any]] = None) -> ContextNode:
        """Create a new context node."""
        if not node_id:
            raise ValueError("Node ID cannot be empty")

        domain = context.get("domain", "general") if context else "general"
        return ContextNode(
            name=node_id,
            dimension=SemanticDimension(
                domain=domain,
                role=context.get("role", "user") if context else "user",
                objective=context.get("objective", "analyze") if context else "analyze",
                timeframe=context.get("timeframe", "current") if context else "current"
            ),
            karaka=KarakaMapping(
                agent=context.get("agent", "user") if context else "user",
                object=context.get("object", "unknown") if context else "unknown",
                instrument=context.get("instrument", "none") if context else "none"
            ),
            confidence=1.0
        )

    def update_context_node(self, node: ContextNode, updates: Dict[str, Any]) -> Optional[ContextNode]:
        """Update a context node with new information."""
        if not node or not updates:
            raise ValueError("Invalid node or updates")

        # Update dimension if provided
        if "dimension" in updates:
            dimension_data = updates["dimension"]
            if isinstance(dimension_data, dict):
                # Create a new dimension with updated values, keeping existing values as defaults
                new_dimension = SemanticDimension(
                    domain=dimension_data.get("domain", node.dimension.domain),
                    role=dimension_data.get("role", node.dimension.role),
                    objective=dimension_data.get("objective", node.dimension.objective),
                    timeframe=dimension_data.get("timeframe", node.dimension.timeframe),
                    attributes=dimension_data.get("attributes", node.dimension.attributes),
                    relationships=dimension_data.get("relationships", node.dimension.relationships),
                    confidence=dimension_data.get("confidence", node.dimension.confidence)
                )
                node.dimension = new_dimension
            elif isinstance(dimension_data, SemanticDimension):
                node.dimension = dimension_data
            else:
                raise ValueError("Invalid dimension update type")

        # Update karaka if provided
        if "karaka" in updates:
            karaka_data = updates["karaka"]
            if isinstance(karaka_data, dict):
                # Create a new karaka with updated values, keeping existing values as defaults
                new_karaka = KarakaMapping(
                    agent=karaka_data.get("agent", node.karaka.agent),
                    object=karaka_data.get("object", node.karaka.object),
                    instrument=karaka_data.get("instrument", node.karaka.instrument),
                    attributes=karaka_data.get("attributes", node.karaka.attributes),
                    relationships=karaka_data.get("relationships", node.karaka.relationships),
                    confidence=karaka_data.get("confidence", node.karaka.confidence)
                )
                node.karaka = new_karaka
            elif isinstance(karaka_data, KarakaMapping):
                node.karaka = karaka_data
            else:
                raise ValueError("Invalid karaka update type")

        # Update other attributes
        if "name" in updates:
            if not isinstance(updates["name"], str):
                raise ValueError("Invalid name update type")
            node.name = updates["name"]
        if "confidence" in updates:
            if not isinstance(updates["confidence"], (int, float)) or not 0 <= updates["confidence"] <= 1:
                raise ValueError("Invalid confidence update value")
            node.confidence = updates["confidence"]
        if "children" in updates:
            if not isinstance(updates["children"], list):
                raise ValueError("Invalid children update type")
            node.children = updates["children"]

        return node

    def merge_context_trees(self, tree1: ContextNode, tree2: ContextNode) -> ContextNode:
        """Merge two context trees."""
        if not tree1 or not tree2:
            raise ValueError("Invalid context trees")
        
        merged = ContextNode(
            name=f"{tree1.name}+{tree2.name}",
            dimension=SemanticDimension(
                domain=f"{tree1.dimension.domain}+{tree2.dimension.domain}",
                role="merged",
                objective="merged",
                timeframe="merged"
            ),
            karaka=KarakaMapping(
                agent="merged",
                object="merged",
                instrument="merged"
            )
        )
        
        # Add children from both trees
        merged.children.extend(tree1.children)
        merged.children.extend(tree2.children)
        
        return merged

    def extract_semantic_graph(self, node: ContextNode) -> Dict[str, Any]:
        """Extract a semantic graph from a context node."""
        if not node:
            raise ValueError("Invalid context node")
        
        graph = {
            "nodes": [],
            "edges": []
        }
        
        # Add root node
        graph["nodes"].append({
            "id": node.name,
            "type": "root",
            "attributes": {
                "domain": node.dimension.domain,
                "role": node.dimension.role,
                "objective": node.dimension.objective,
                "timeframe": node.dimension.timeframe
            }
        })
        
        # Add child nodes and edges
        for child in node.children:
            graph["nodes"].append({
                "id": child.name,
                "type": "sentence",
                "attributes": {
                    "domain": child.dimension.domain,
                    "role": child.dimension.role,
                    "objective": child.dimension.objective,
                    "timeframe": child.dimension.timeframe
                }
            })
            
            graph["edges"].append({
                "source": node.name,
                "target": child.name,
                "type": "contains"
            })
        
        return graph

    def validate_context_tree(self, node: ContextNode) -> List[str]:
        """Validate a context tree."""
        if not node:
            raise ValueError("Node cannot be empty")

        errors = []

        # Check if node has required attributes
        if not all(hasattr(node, attr) for attr in ["name", "dimension", "karaka"]):
            errors.append("Node is missing required attributes")

        # Validate dimension
        if not isinstance(node.dimension, SemanticDimension):
            errors.append("Invalid dimension type")
        else:
            if not node.dimension.domain:
                errors.append("Dimension domain is empty")
            if not node.dimension.role:
                errors.append("Dimension role is empty")
            if not node.dimension.objective:
                errors.append("Dimension objective is empty")
            if not node.dimension.timeframe:
                errors.append("Dimension timeframe is empty")

        # Validate karaka
        if not isinstance(node.karaka, KarakaMapping):
            errors.append("Invalid karaka type")
        else:
            if not node.karaka.agent:
                errors.append("Karaka agent is empty")
            if not node.karaka.object:
                errors.append("Karaka object is empty")
            if not node.karaka.instrument:
                errors.append("Karaka instrument is empty")

        # Validate children recursively
        for child in node.children:
            child_errors = self.validate_context_tree(child)
            errors.extend(child_errors)

        return errors

    def optimize_context_tree(self, node: ContextNode) -> ContextNode:
        """Optimize a context tree by removing redundant nodes."""
        if not node:
            raise ValueError("Invalid node")

        # Create a new optimized node
        optimized = ContextNode(
            name=node.name,
            dimension=node.dimension,
            karaka=node.karaka,
            confidence=node.confidence
        )

        # Process children recursively
        seen_children = set()
        for child in node.children:
            # Skip redundant children
            if self._is_redundant(child) or self._is_duplicate(child, seen_children):
                continue
            
            # Add non-redundant child to seen set and optimized node
            child_key = (child.name, child.dimension.domain, child.karaka.agent)
            seen_children.add(child_key)
            optimized.children.append(self.optimize_context_tree(child))

        return optimized

    def _is_redundant(self, node: ContextNode) -> bool:
        """Check if a node is redundant."""
        # A node is redundant if it has no children and all its attributes are empty or default values
        if node.children:
            return False

        is_empty_dimension = (
            node.dimension.domain == "sentence" and
            node.dimension.role == "sentence" and
            node.dimension.objective == "sentence" and
            node.dimension.timeframe == "sentence" and
            not node.dimension.attributes and
            not node.dimension.relationships
        )

        is_empty_karaka = (
            node.karaka.agent == "sentence" and
            node.karaka.object == "sentence" and
            node.karaka.instrument == "sentence" and
            not node.karaka.attributes and
            not node.karaka.relationships
        )

        return is_empty_dimension and is_empty_karaka

    def _is_duplicate(self, node: ContextNode, seen_children: Set[Tuple[str, str, str]]) -> bool:
        """Check if a node is a duplicate of an already processed node."""
        child_key = (node.name, node.dimension.domain, node.karaka.agent)
        return child_key in seen_children

    def _split_into_sentences(self, tokens: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Split tokens into sentences."""
        sentences = []
        current_sentence = []
        
        for token in tokens:
            current_sentence.append(token)
            if isinstance(token, dict) and token.get("text") in [".", "!", "?"]:
                sentences.append(current_sentence)
                current_sentence = []
        
        if current_sentence:
            sentences.append(current_sentence)
        
        return sentences

    def process(
            self,
            query: str,
            context: Optional[Dict[str, Any]] = None,
            domain_mapping: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a query through semantic abstraction."""
        try:
            # Parse query with spaCy
            doc = nlp(query)

            # Get embeddings
            embeddings = self._get_embeddings(doc)

            # Extract entities and relationships
            entities = self._extract_entities(doc)
            relationships = self._extract_relationships(doc)

            # Analyze temporal and spatial dimensions
            temporal_dimensions = self._analyze_temporal_dimensions(doc)
            spatial_dimensions = self._analyze_spatial_dimensions(doc)

            # Calculate confidence
            confidence = self._calculate_confidence(doc, {
                "entities": entities,
                "relationships": relationships
            }, context)

            return {
                "embeddings": embeddings.tolist(),
                "entities": entities,
                "relationships": relationships,
                "temporal_dimensions": temporal_dimensions,
                "spatial_dimensions": spatial_dimensions,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error in semantic abstraction: {str(e)}")
            raise

    def _get_embeddings(self, doc) -> np.ndarray:
        """Get embeddings for the document."""
        return self.karaka_mapper.get_embeddings(doc.text)

    def _extract_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract entities from the document."""
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        return entities

    def _extract_relationships(self, doc) -> List[Dict[str, Any]]:
        """Extract relationships from the document."""
        relationships = []
        for token in doc:
            if token.dep_ not in ("punct", "det"):
                relationships.append({
                    "source": token.head.text,
                    "type": token.dep_,
                    "target": token.text
                })
        return relationships

    def _analyze_temporal_dimensions(self, doc) -> Dict[str, Any]:
        """Analyze temporal dimensions in the document."""
        temporal_info = {
            "timestamps": [],
            "durations": [],
            "frequencies": []
        }

        for ent in doc.ents:
            if ent.label_ in ("DATE", "TIME"):
                try:
                    parsed_date = parser.parse(ent.text)
                    temporal_info["timestamps"].append({
                        "text": ent.text,
                        "parsed": parsed_date.isoformat()
                    })
                except:
                    temporal_info["durations"].append(ent.text)

        return temporal_info

    def _analyze_spatial_dimensions(self, doc) -> Dict[str, Any]:
        """Analyze spatial dimensions in the document."""
        spatial_info = {
            "locations": [],
            "directions": [],
            "distances": []
        }

        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC"):
                spatial_info["locations"].append(ent.text)

        return spatial_info

    def _calculate_confidence(
            self,
            doc,
            roles: Dict[str, Any],
            context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for the semantic analysis."""
        # Base confidence
        confidence = 0.8

        # Adjust based on entity recognition
        if roles["entities"]:
            confidence += 0.1

        # Adjust based on relationship extraction
        if roles["relationships"]:
            confidence += 0.1

        # Cap at 1.0
        return min(confidence, 1.0)
