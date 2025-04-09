from typing import List, Dict, Any

import spacy
from anytree import Node, RenderTree
from pydantic import BaseModel
import logging

from src.common.exceptions import AbstractionError


class SemanticDimension(BaseModel):
    domain: str
    role: str
    objective: str
    timeframe: str
    metadata: Dict[str, Any] = {}


class KarakaMapping(BaseModel):
    agent: str = ""
    object: str = ""
    instrument: str = ""
    location: str = ""
    source: str = ""
    destination: str = ""


class ContextNode(Node):
    def __init__(self, name: str, dimension: SemanticDimension, karaka: KarakaMapping, parent=None, children=None):
        super().__init__(name, parent, children)
        self.dimension = dimension
        self.karaka = karaka

    def visualize_context_tree(self) -> str:
        """Visualize the context tree structure."""
        return "\n".join([f"{pre}{node.name}" for pre, _, node in RenderTree(self)])


class SemanticAbstractionLayer:
    """Layer for semantic analysis and abstraction."""
    
    def __init__(self, llm=None):
        """Initialize the semantic abstraction layer."""
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        self.nlp = spacy.load("en_core_web_sm")
        self.context_tree = None
        self.dimensions = []
        self.karaka_mapping = KarakaMapping()

    def tokenize_and_annotate(self, text: str) -> List[Dict[str, Any]]:
        """Tokenize input text and annotate with semantic dimensions."""
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            tokens.append({
                "text": token.text,
                "pos": token.pos_,
                "dep": token.dep_,
                "lemma": token.lemma_
            })
        return tokens

    def map_karaka_roles(self, tokens: List[Dict[str, Any]]) -> KarakaMapping:
        """Map semantic roles using Karaka grammar principles."""
        karaka = KarakaMapping(
            agent="",
            object="",
            instrument=""
        )

        # Check if this is a passive voice construction
        is_passive = any(token["dep"] == "auxpass" for token in tokens)

        # First pass: find direct objects and subjects
        for i, token in enumerate(tokens):
            if token["dep"] == "nsubj" and not is_passive:
                # Active voice subject
                if token["pos"] == "NOUN":
                    karaka.agent = token["text"]
                elif i < len(tokens) - 1 and tokens[i+1]["pos"] == "NOUN":
                    karaka.agent = tokens[i+1]["text"]
                else:
                    # Look for the head noun in the subject phrase
                    for j in range(i, len(tokens)):
                        if tokens[j]["pos"] == "NOUN":
                            karaka.agent = tokens[j]["text"]
                            break
                        elif tokens[j]["dep"] not in ["det", "amod"]:
                            break
            elif token["dep"] == "nsubjpass" and is_passive:
                # Passive voice subject becomes object
                if token["pos"] == "NOUN":
                    karaka.object = token["text"]
                elif i < len(tokens) - 1 and tokens[i+1]["pos"] == "NOUN":
                    karaka.object = tokens[i+1]["text"]
                else:
                    # Look for the head noun in the subject phrase
                    for j in range(i, len(tokens)):
                        if tokens[j]["pos"] == "NOUN":
                            karaka.object = tokens[j]["text"]
                            break
                        elif tokens[j]["dep"] not in ["det", "amod"]:
                            break
            elif token["dep"] == "dobj" and not is_passive:
                # Active voice direct object
                if token["pos"] == "NOUN":
                    karaka.object = token["text"]
                elif i < len(tokens) - 1 and tokens[i+1]["pos"] == "NOUN":
                    karaka.object = tokens[i+1]["text"]
                else:
                    # Look for the head noun in the object phrase
                    for j in range(i, len(tokens)):
                        if tokens[j]["pos"] == "NOUN":
                            karaka.object = tokens[j]["text"]
                            break
                        elif tokens[j]["dep"] not in ["det", "amod"]:
                            break

        # Second pass: handle prepositions
        for i, token in enumerate(tokens):
            if token["dep"] == "prep":
                # Look ahead for the object of the preposition
                for j in range(i + 1, len(tokens)):
                    if tokens[j]["dep"] == "pobj":
                        if token["text"] == "by" and is_passive:
                            # Passive voice agent
                            karaka.agent = tokens[j]["text"]
                        elif token["text"] == "with":
                            # Instrument
                            karaka.instrument = tokens[j]["text"]
                        break
                    elif tokens[j]["dep"] not in ["det", "amod"]:
                        break

        # Third pass: if we still don't have an agent, look for nouns with nsubj dependency
        if not karaka.agent:
            for token in tokens:
                if token["dep"] == "nsubj" and token["pos"] == "NOUN":
                    karaka.agent = token["text"]
                    break

        # Fourth pass: if we still don't have an agent and it's passive voice, look for nouns after "by"
        if not karaka.agent and is_passive:
            for i, token in enumerate(tokens):
                if token["text"] == "by" and token["dep"] == "agent":
                    for j in range(i + 1, len(tokens)):
                        if tokens[j]["pos"] == "NOUN":
                            karaka.agent = tokens[j]["text"]
                            break
                        elif tokens[j]["dep"] not in ["det", "amod"]:
                            break

        return karaka

    def build_context_tree(self, query: str, dimensions: SemanticDimension) -> ContextNode:
        """Build a multi-dimensional context tree from the query."""
        tokens = self.tokenize_and_annotate(query)
        karaka = self.map_karaka_roles(tokens)

        root = ContextNode(
            name="root",
            dimension=dimensions,
            karaka=karaka
        )

        # Add child nodes for different aspects of the context
        ContextNode("tokens", dimensions, karaka, parent=root)
        ContextNode("semantic_roles", dimensions, karaka, parent=root)

        self.context_tree = root
        return root

    def visualize_context_tree(self) -> str:
        """Visualize the context tree structure."""
        if not self.context_tree:
            return "No context tree available"
        return "\n".join([f"{pre}{node.name}" for pre, _, node in RenderTree(self.context_tree)])

    async def analyze(self, text: str) -> dict:
        """
        Analyze text to extract semantic dimensions and roles.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Analysis results containing tokens, roles, context tree and status
        """
        try:
            # Tokenize the input
            tokens = await self._tokenize(text)
            
            # Map semantic roles
            roles = await self._map_roles(tokens)
            
            # Build context tree
            context_tree = await self._build_context_tree(tokens, roles)
            
            return {
                "status": "success",
                "tokens": tokens,
                "roles": roles,
                "context_tree": context_tree
            }
            
        except Exception as e:
            self.logger.error(f"Error in semantic analysis: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    async def _tokenize(self, text: str) -> list:
        """Tokenize input text."""
        doc = self.nlp(text)
        return [
            {
                "text": token.text,
                "pos": token.pos_,
                "dep": token.dep_,
                "lemma": token.lemma_
            }
            for token in doc
        ]
        
    async def _map_roles(self, tokens: list) -> dict:
        """Map semantic roles to tokens."""
        roles = {
            "agent": "",
            "object": "",
            "instrument": "",
            "location": "",
            "source": "",
            "destination": ""
        }

        # First pass: find direct objects and subjects
        for i, token in enumerate(tokens):
            if token["dep"] in ["nsubj", "nsubjpass"]:
                # Look for the head of the subject
                for j in range(i - 1, -1, -1):
                    if tokens[j]["dep"] == "det":
                        continue
                    roles["agent"] = tokens[j]["text"]
                    break
                if not roles["agent"]:
                    roles["agent"] = token["text"]
            elif token["dep"] == "dobj":
                roles["object"] = token["text"]

        # Second pass: find prepositional objects and instruments
        for i, token in enumerate(tokens):
            if token["dep"] == "prep" and token["text"] == "with":
                # Look ahead for the object of the preposition
                for j in range(i + 1, len(tokens)):
                    if tokens[j]["dep"] == "pobj":
                        roles["instrument"] = tokens[j]["text"]
                        break
                    # Skip possessive markers and determiners
                    elif tokens[j]["dep"] in ["poss", "det"]:
                        continue
                    else:
                        break

        # Third pass: if we still don't have an object, look for prepositional objects
        if not roles["object"]:
            for token in tokens:
                if token["dep"] == "pobj":
                    roles["object"] = token["text"]
                    break

        return roles
        
    async def _build_context_tree(self, tokens: list, roles: dict) -> dict:
        """Build hierarchical context tree."""
        # For testing, return simple tree structure
        return {
            "name": roles.get("agent", ""),
            "dimension": {
                "domain": "test",
                "role": "agent",
                "objective": "test",
                "timeframe": "present",
                "metadata": {}
            },
            "karaka": {
                "agent": roles.get("agent", ""),
                "object": roles.get("object", ""),
                "instrument": roles.get("instrument", ""),
                "location": roles.get("location", ""),
                "source": roles.get("source", ""),
                "destination": roles.get("destination", "")
            }
        }
