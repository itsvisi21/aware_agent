from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import numpy as np
from chromadb import Client, Settings
from chromadb.utils import embedding_functions
import sqlite3
import aiosqlite
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Represents a memory entry in the agent's memory."""
    id: str
    content: Dict[str, Any]
    semantic_roles: Dict[str, Any]
    semantic_graph: Dict[str, Any]
    temporal_dimensions: Optional[Dict[str, Any]] = None
    spatial_dimensions: Optional[Dict[str, Any]] = None
    domain_context: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = None
    retention_period: Optional[timedelta] = None
    importance: float = 0.5
    category: str = "general"
    tags: List[str] = None

class MemoryEngine:
    """Handles memory storage, retrieval, and analysis for the agent."""
    
    def __init__(self, storage_path: str = "memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = Client(Settings(
            persist_directory=str(self.storage_path / "chroma"),
            anonymized_telemetry=False
        ))
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create collections
        self.semantic_context = self.client.get_or_create_collection(
            name="semantic_context",
            embedding_function=self.embedding_function
        )
        self.action_history = self.client.get_or_create_collection(
            name="action_history",
            embedding_function=self.embedding_function
        )
        
        # SQLite database path
        self.db_path = str(self.storage_path / "memory.db")
        
        self.memories: List[MemoryEntry] = []
        self.vectorizer = TfidfVectorizer()
        self._load_memories()
        
    def _load_memories(self):
        """Load memories from storage."""
        try:
            memory_file = self.storage_path / "memories.jsonl"
            if memory_file.exists():
                with open(memory_file, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        self.memories.append(MemoryEntry(
                            id=data["id"],
                            content=data["content"],
                            semantic_roles=data["semantic_roles"],
                            semantic_graph=data["semantic_graph"],
                            temporal_dimensions=data.get("temporal_dimensions"),
                            spatial_dimensions=data.get("spatial_dimensions"),
                            domain_context=data.get("domain_context"),
                            timestamp=datetime.fromisoformat(data["timestamp"]),
                            importance=data["importance"],
                            category=data["category"],
                            tags=data["tags"],
                            metadata=data.get("metadata")
                        ))
        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}")
        
    async def initialize_database(self):
        """Initialize SQLite database with required tables."""
        async with aiosqlite.connect(self.db_path) as db:
            # Create conversations table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Create messages table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            
            # Create tasks table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    status TEXT,
                    request TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metrics TEXT,
                    result TEXT
                )
            """)
            
            # Create semantic_logs table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS semantic_logs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT,
                    context_snapshot TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES tasks(id)
                )
            """)
            
            await db.commit()
        
    def store_context(self, task_id: str, context_tree: Dict[str, Any], retention_period: Optional[int] = None) -> str:
        """
        Store semantic context in memory.
        
        Args:
            task_id: The task identifier
            context_tree: The context tree to store
            retention_period: Optional retention period in seconds
            
        Returns:
            Memory entry ID
        """
        # Generate embeddings for semantic context
        semantic_embedding = self._generate_embedding(context_tree.get("semantic_roles", {}))
        temporal_embedding = self._generate_embedding(context_tree.get("temporal_dimensions", {}))
        spatial_embedding = self._generate_embedding(context_tree.get("spatial_dimensions", {}))
        domain_embedding = self._generate_embedding(context_tree.get("domain_context", {}))
        
        # Create memory entry
        entry = MemoryEntry(
            id=task_id,
            content=context_tree,
            semantic_roles=context_tree.get("semantic_roles", {}),
            semantic_graph=context_tree.get("semantic_graph", {}),
            temporal_dimensions=context_tree.get("temporal_dimensions"),
            spatial_dimensions=context_tree.get("spatial_dimensions"),
            domain_context=context_tree.get("domain_context"),
            timestamp=datetime.now(),
            metadata={
                "task_id": task_id,
                "confidence": self._calculate_context_confidence(context_tree)
            },
            retention_period=timedelta(seconds=retention_period) if retention_period else None,
            importance=0.8,
            category="context"
        )
        
        # Store in ChromaDB
        self.semantic_context.add(
            ids=[task_id],
            embeddings=[semantic_embedding],
            metadatas=[entry.metadata],
            documents=[json.dumps(entry.__dict__)]
        )
        
        # Store in JSON for backup
        self._store_entry_json(entry)
        
        return task_id
    
    def store_action(self, task_id: str, action: Dict[str, Any], result: Optional[Dict[str, Any]] = None) -> str:
        """
        Store action and result in memory.
        
        Args:
            task_id: The task identifier
            action: The action performed
            result: Optional action result
            
        Returns:
            Memory entry ID
        """
        # Generate embedding for action
        action_embedding = self._generate_embedding(action)
        
        # Create memory entry
        entry = MemoryEntry(
            id=f"{task_id}_{datetime.now().isoformat()}",
            content=action,
            semantic_roles=result.get("semantic_roles", {}) if result else {},
            semantic_graph=result.get("semantic_graph", {}) if result else {},
            temporal_dimensions=result.get("temporal_dimensions") if result else None,
            spatial_dimensions=result.get("spatial_dimensions") if result else None,
            domain_context=result.get("domain_context") if result else None,
            timestamp=datetime.now(),
            metadata={
                "task_id": task_id,
                "action_type": action.get("type", "unknown"),
                "confidence": result.get("confidence", 0.0) if result else 0.0
            },
            importance=0.7,
            category="action"
        )
        
        # Store in ChromaDB
        self.action_history.add(
            ids=[entry.id],
            embeddings=[action_embedding],
            metadatas=[entry.metadata],
            documents=[json.dumps(entry.__dict__)]
        )
        
        # Store in JSON for backup
        self._store_entry_json(entry)
        
        return entry.id
    
    def retrieve_context(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve semantic context for a task.
        
        Args:
            task_id: The task identifier
            
        Returns:
            Retrieved context or None if not found
        """
        # First try JSON storage
        entry = self._retrieve_entry_json(task_id)
        if entry:
            # Check retention period
            if entry.retention_period:
                if datetime.now() - entry.timestamp > entry.retention_period:
                    self._cleanup_entry(task_id)
                    return None
            return entry.__dict__
            
        # Then try ChromaDB
        results = self.semantic_context.get(
            ids=[task_id],
            include=["metadatas", "documents"]
        )
        
        if results and results["ids"]:
            entry_dict = json.loads(results["documents"][0])
            # Check retention period
            if entry_dict.get("retention_period"):
                retention_period = timedelta(seconds=float(entry_dict["retention_period"]))
                if datetime.now() - datetime.fromisoformat(entry_dict["timestamp"]) > retention_period:
                    self._cleanup_entry(task_id)
                    return None
            return entry_dict
            
        return None
    
    def retrieve_actions(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all actions for a task.
        
        Args:
            task_id: The task identifier
            
        Returns:
            List of action entries
        """
        # Query ChromaDB for actions
        results = self.action_history.query(
            query_embeddings=[self._generate_embedding({"task_id": task_id})],
            n_results=100,
            where={"task_id": task_id},
            include=["metadatas", "documents"]
        )
        
        if not results or not results["ids"]:
            return []
            
        actions = []
        for doc in results["documents"]:
            try:
                action = json.loads(doc)
                actions.append(action)
            except json.JSONDecodeError:
                continue
                
        return actions
    
    def _generate_embedding(self, data: Dict[str, Any]) -> List[float]:
        """Generate embedding for data."""
        # Convert data to text
        text = json.dumps(data)
        
        # Generate embedding
        embedding = self.embedding_function([text])[0]
        
        return embedding.tolist()
    
    def _calculate_context_confidence(self, context_tree: Dict[str, Any]) -> float:
        """Calculate confidence score for context."""
        confidences = []
        
        if "semantic_roles" in context_tree:
            confidences.append(0.8)  # Base confidence for semantic roles
            
        if "temporal_dimensions" in context_tree:
            confidences.append(0.7)  # Base confidence for temporal info
            
        if "spatial_dimensions" in context_tree:
            confidences.append(0.7)  # Base confidence for spatial info
            
        if "domain_context" in context_tree:
            confidences.append(0.9)  # Base confidence for domain context
            
        return np.mean(confidences) if confidences else 0.5
    
    def _store_entry_json(self, entry: MemoryEntry) -> None:
        """Store memory entry as JSON."""
        entry_path = self.storage_path / f"{entry.id}.json"
        with open(entry_path, "w") as f:
            json.dump(entry.__dict__, f)
    
    def _retrieve_entry_json(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve memory entry from JSON."""
        entry_path = self.storage_path / f"{entry_id}.json"
        if not entry_path.exists():
            return None
            
        with open(entry_path, "r") as f:
            data = json.load(f)
            return MemoryEntry(**data)
            
    def _cleanup_entry(self, entry_id: str) -> None:
        """Clean up expired memory entry."""
        # Remove from ChromaDB
        try:
            self.semantic_context.delete(ids=[entry_id])
            self.action_history.delete(where={"task_id": entry_id})
        except:
            pass
            
        # Remove JSON file
        entry_path = self.storage_path / f"{entry_id}.json"
        try:
            entry_path.unlink()
        except:
            pass

    async def store(self, content: Dict[str, Any], importance: float = 0.5, category: str = "general", tags: List[str] = None) -> str:
        """Store a new memory entry."""
        try:
            memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            entry = MemoryEntry(
                id=memory_id,
                content=content,
                semantic_roles={},
                semantic_graph={},
                importance=importance,
                category=category,
                tags=tags or [],
                metadata={"id": memory_id}
            )
            
            self.memories.append(entry)
            await self._save_memory(entry)
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            raise
    
    async def _save_memory(self, entry: MemoryEntry):
        """Save a memory entry to storage."""
        try:
            memory_file = self.storage_path / "memories.jsonl"
            with open(memory_file, "a") as f:
                f.write(json.dumps({
                    "id": entry.id,
                    "content": entry.content,
                    "semantic_roles": entry.semantic_roles,
                    "semantic_graph": entry.semantic_graph,
                    "temporal_dimensions": entry.temporal_dimensions,
                    "spatial_dimensions": entry.spatial_dimensions,
                    "domain_context": entry.domain_context,
                    "timestamp": entry.timestamp.isoformat(),
                    "importance": entry.importance,
                    "category": entry.category,
                    "tags": entry.tags,
                    "metadata": entry.metadata
                }) + "\n")
        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}")
            raise
    
    async def retrieve(self, query: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on a query."""
        try:
            # Convert query to text for similarity search
            query_text = self._dict_to_text(query)
            
            # Get memory texts
            memory_texts = [self._dict_to_text(mem.content) for mem in self.memories]
            
            # Vectorize texts
            if not memory_texts:
                return []
            
            vectors = self.vectorizer.fit_transform(memory_texts + [query_text])
            query_vector = vectors[-1]
            memory_vectors = vectors[:-1]
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, memory_vectors)[0]
            
            # Sort by similarity and importance
            scored_memories = [
                (mem, sim * mem.importance)
                for mem, sim in zip(self.memories, similarities)
            ]
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            
            # Return top memories
            return [
                {
                    "id": mem.id,
                    "content": mem.content,
                    "similarity": sim,
                    "importance": mem.importance,
                    "timestamp": mem.timestamp.isoformat(),
                    "category": mem.category,
                    "tags": mem.tags
                }
                for mem, sim in scored_memories[:limit]
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            return []
    
    def retrieve_recent_failures(self, days: int = 7) -> List[Dict[str, Any]]:
        """Retrieve recent failure memories."""
        cutoff = datetime.now() - timedelta(days=days)
        return [
            mem.content
            for mem in self.memories
            if mem.timestamp > cutoff and mem.category == "failure"
        ]
    
    def _dict_to_text(self, data: Dict[str, Any]) -> str:
        """Convert a dictionary to a text string for similarity search."""
        if isinstance(data, str):
            return data
        return " ".join(str(v) for v in data.values() if v is not None)
    
    def analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in the agent's memory."""
        try:
            # Group memories by category
            category_counts = {}
            for mem in self.memories:
                category_counts[mem.category] = category_counts.get(mem.category, 0) + 1
            
            # Calculate average importance by category
            category_importance = {}
            for mem in self.memories:
                if mem.category not in category_importance:
                    category_importance[mem.category] = []
                category_importance[mem.category].append(mem.importance)
            
            for category in category_importance:
                category_importance[category] = np.mean(category_importance[category])
            
            # Find most common tags
            tag_counts = {}
            for mem in self.memories:
                for tag in mem.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            return {
                "category_distribution": category_counts,
                "category_importance": category_importance,
                "common_tags": sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10],
                "total_memories": len(self.memories),
                "memory_age_range": {
                    "oldest": min(mem.timestamp for mem in self.memories).isoformat(),
                    "newest": max(mem.timestamp for mem in self.memories).isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing memory patterns: {str(e)}")
            return {} 