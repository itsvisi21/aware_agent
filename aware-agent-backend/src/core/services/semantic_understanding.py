from typing import Dict, Any, List

import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import os

from src.common.exceptions import ServiceError


class SemanticUnderstandingService:
    def __init__(self):
        """Initialize the service."""
        self._sentiment_analyzer = None
        self._entity_recognizer = None
        self._nlp = None
        self._initialized = False
        self.entity_cache = {}  # Initialize entity cache

    async def initialize(self):
        """Initialize the service by loading required models."""
        if self._initialized:
            return

        try:
            # Load spaCy model first
            self._nlp = spacy.load("en_core_web_sm")
            
            # Initialize sentiment analysis using spaCy
            self._sentiment_analyzer = self._analyze_sentiment
            
            # Initialize entity recognition using spaCy
            self._entity_recognizer = self._extract_entities
            
            self._initialized = True
        except Exception as e:
            raise ServiceError(f"Failed to initialize semantic understanding service: {str(e)}")

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using spaCy."""
        doc = self._nlp(text)
        # Simple sentiment analysis based on word polarity
        positive_words = ["good", "great", "excellent", "positive", "happy", "joy"]
        negative_words = ["bad", "terrible", "awful", "negative", "sad", "angry"]
        
        positive_count = sum(1 for token in doc if token.text.lower() in positive_words)
        negative_count = sum(1 for token in doc if token.text.lower() in negative_words)
        
        total = positive_count + negative_count
        if total == 0:
            return {"label": "NEUTRAL", "score": 0.5}
            
        score = positive_count / total
        label = "POSITIVE" if score > 0.6 else "NEGATIVE" if score < 0.4 else "NEUTRAL"
        
        return {"label": label, "score": score}

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy."""
        doc = self._nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
            
        return entities

    def _get_sentiment(self, text: str) -> Dict[str, Any]:
        if not self._initialized:
            raise ServiceError("Service not initialized")
        
        result = self._sentiment_analyzer(text)
        return {
            "label": result["label"],
            "score": result["score"]
        }

    async def get_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment analysis for text."""
        return self._get_sentiment(text)

    async def get_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from the text."""
        if not self._entity_recognizer or not self._nlp:
            raise ServiceError("Service not initialized")
            
        # Get entities from spaCy
        entities = self._entity_recognizer(text)
        
        # Update entity cache
        for entity in entities:
            if entity["type"] not in self.entity_cache:
                self.entity_cache[entity["type"]] = set()
            self.entity_cache[entity["type"]].add(entity["text"])
        
        return entities

    async def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from the text."""
        if not self._nlp:
            raise ServiceError("Service not initialized")
            
        doc = self._nlp(text)
        concepts = []
        current_concept = []
        
        # Get multi-word entities first
        entities = await self.get_entities(text)
        for entity in entities:
            if entity["type"] in ["ORG", "PERSON", "GPE", "PRODUCT"]:
                concepts.append(entity["text"])  # Preserve original case for proper nouns
        
        # Extract concepts from tokens
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                if token.pos_ == "PROPN":
                    current_concept.append(token.text)  # Preserve case for proper nouns
                else:
                    current_concept.append(token.text.lower())
            elif current_concept:
                concepts.append(" ".join(current_concept))
                current_concept = []
        
        if current_concept:
            concepts.append(" ".join(current_concept))
        
        # Add noun chunks as additional concepts
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text
            if any(token.pos_ == "PROPN" for token in chunk):
                concepts.append(chunk_text)  # Preserve case for proper nouns
            else:
                concepts.append(chunk_text.lower())
        
        # Add compound nouns
        for token in doc:
            if token.dep_ == "compound" and token.head.pos_ == "NOUN":
                compound = f"{token.text} {token.head.text}"
                if any(t.pos_ == "PROPN" for t in [token, token.head]):
                    concepts.append(compound)  # Preserve case for proper nouns
                else:
                    concepts.append(compound.lower())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for concept in concepts:
            if concept not in seen:
                seen.add(concept)
                unique_concepts.append(concept)
        
        return unique_concepts

    async def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text and return comprehensive analysis."""
        if not self._initialized:
            raise ServiceError("Service not initialized")

        doc = self._nlp(text)
        
        # Get basic token analysis
        tokens = [token.text for token in doc]
        lemmas = [token.lemma_ for token in doc]
        pos_tags = [token.pos_ for token in doc]
        
        # Get sentiment
        sentiment = self._get_sentiment(text)
        
        # Get entities
        entities = await self.get_entities(text)
        
        # Get concepts
        concepts = await self.extract_concepts(text)
        
        # Analyze sentence structure
        sentences = []
        for sent in doc.sents:
            root = None
            subject = None
            obj = None
            for token in sent:
                if token.dep_ == "ROOT":
                    root = token.text
                if token.dep_ == "nsubj":
                    subject = token.text
                if token.dep_ == "dobj":
                    obj = token.text
            
            sentences.append({
                "text": sent.text.strip(),
                "root": root,
                "subject": subject,
                "object": obj
            })
        
        # Determine intent
        is_question = any(token.text.lower() in ["what", "why", "how", "when", "where", "who"] for token in doc)
        is_command = (
            doc[0].pos_ == "VERB" and 
            doc[0].text.lower() not in ["is", "are", "was", "were", "have", "has", "had"] and
            not any(token.dep_ == "aux" for token in doc)
        )
        
        intent = {
            "base": "question" if is_question else "command" if is_command else "statement",
            "specific": "action" if is_command else None,
            "confidence": 0.8 if is_question or is_command else 0.7
        }
        
        # Get dependencies
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
        
        return {
            "tokens": tokens,
            "lemmas": lemmas,
            "pos_tags": pos_tags,
            "entities": entities,
            "intent": intent,
            "sentiment": sentiment,
            "concepts": concepts,
            "sentences": sentences,
            "dependencies": dependencies
        }

    async def analyze_message(self, text: str) -> Dict[str, Any]:
        """Alias for analyze method."""
        return await self.analyze(text)
