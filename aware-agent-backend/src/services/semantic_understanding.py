from typing import Dict, Any, List
import re
from collections import defaultdict

class SemanticUnderstandingService:
    def __init__(self):
        self.entity_cache: Dict[str, List[str]] = defaultdict(list)
        self.relationship_cache: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self.context_cache: Dict[str, Any] = {}

    async def analyze_message(self, message: str) -> Dict[str, Any]:
        """
        Analyze a message for semantic understanding.
        """
        # Extract entities
        entities = self._extract_entities(message)
        
        # Identify relationships
        relationships = self._identify_relationships(message, entities)
        
        # Determine context
        context = self._determine_context(message, entities, relationships)
        
        # Update caches
        self._update_caches(entities, relationships, context)
        
        return {
            'entities': entities,
            'relationships': relationships,
            'context': context,
            'intent': self._determine_intent(message)
        }

    def _extract_entities(self, message: str) -> List[str]:
        """
        Extract named entities from the message.
        """
        # Simple entity extraction for demonstration
        # In a real implementation, this would use NLP libraries
        entities = []
        
        # Extract potential entities (words starting with capital letters)
        words = message.split()
        for word in words:
            if word[0].isupper() and len(word) > 1:
                entities.append(word)
        
        return entities

    def _identify_relationships(self, message: str, entities: List[str]) -> List[Dict[str, str]]:
        """
        Identify relationships between entities in the message.
        """
        relationships = []
        
        # Simple relationship extraction for demonstration
        # In a real implementation, this would use dependency parsing
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if entity1 in message and entity2 in message:
                    relationships.append({
                        'source': entity1,
                        'target': entity2,
                        'type': 'related'
                    })
        
        return relationships

    def _determine_context(self, message: str, entities: List[str], relationships: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Determine the context of the message.
        """
        context = {
            'topics': [],
            'actions': [],
            'modifiers': []
        }
        
        # Simple context determination for demonstration
        action_words = ['create', 'update', 'delete', 'explain', 'research', 'build']
        modifier_words = ['urgent', 'important', 'complex', 'simple']
        
        words = message.lower().split()
        for word in words:
            if word in action_words:
                context['actions'].append(word)
            elif word in modifier_words:
                context['modifiers'].append(word)
        
        context['topics'] = entities
        
        return context

    def _determine_intent(self, message: str) -> str:
        """
        Determine the intent of the message.
        """
        # Simple intent classification for demonstration
        intents = {
            'query': ['what', 'how', 'why', 'when', 'where'],
            'command': ['create', 'update', 'delete', 'add', 'remove'],
            'explanation': ['explain', 'describe', 'tell me about'],
            'confirmation': ['is', 'are', 'do', 'does', 'can']
        }
        
        message_lower = message.lower()
        for intent, triggers in intents.items():
            if any(trigger in message_lower for trigger in triggers):
                return intent
        
        return 'statement'

    def _update_caches(self, entities: List[str], relationships: List[Dict[str, str]], context: Dict[str, Any]):
        """
        Update the semantic understanding caches.
        """
        for entity in entities:
            self.entity_cache[entity].append(context)
        
        for relationship in relationships:
            key = f"{relationship['source']}_{relationship['target']}"
            self.relationship_cache[key].append(relationship)
        
        self.context_cache.update(context) 