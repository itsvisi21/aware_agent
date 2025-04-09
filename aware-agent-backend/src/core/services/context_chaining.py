import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional

import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


@dataclass
class ContextNode:
    """Represents a node in the context chain."""
    id: str
    content: str
    timestamp: datetime
    depth: int
    parent_id: Optional[str]
    children_ids: List[str]
    embeddings: List[float]
    metadata: Dict[str, Any]
    confidence: float


@dataclass
class ContextChain:
    """Represents a chain of related contexts."""
    id: str
    root_id: str
    nodes: Dict[str, ContextNode]
    current_depth: int
    metadata: Dict[str, Any]


class ContextChainer:
    """Manages context chains and their relationships."""

    def __init__(self, max_depth: int = 5, similarity_threshold: float = 0.8):
        self.chains: Dict[str, ContextChain] = {}
        self.max_depth = max_depth
        self.similarity_threshold = similarity_threshold
        self.chain_graph = nx.DiGraph()

    async def add_context(
            self,
            content: str,
            embeddings: List[float],
            parent_id: Optional[str] = None,
            chain_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> ContextNode:
        """Add a new context to the chain."""
        try:
            # Generate unique IDs
            node_id = f"context_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if not chain_id:
                chain_id = f"chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Create new node
            node = ContextNode(
                id=node_id,
                content=content,
                timestamp=datetime.now(),
                depth=0,
                parent_id=parent_id,
                children_ids=[],
                embeddings=embeddings,
                metadata=metadata or {},
                confidence=1.0
            )

            if parent_id:
                # Find parent node and chain
                parent_chain = self._find_chain_with_node(parent_id)
                if parent_chain:
                    parent_node = parent_chain.nodes[parent_id]
                    node.depth = parent_node.depth + 1

                    # Check depth limit
                    if node.depth > self.max_depth:
                        # Create new chain branching from parent
                        new_chain = ContextChain(
                            id=chain_id,
                            root_id=node_id,
                            nodes={node_id: node},
                            current_depth=0,
                            metadata={"branch_from": parent_chain.id}
                        )
                        self.chains[chain_id] = new_chain
                        self.chain_graph.add_edge(parent_chain.id, chain_id)
                    else:
                        # Add to existing chain
                        parent_chain.nodes[node_id] = node
                        parent_node.children_ids.append(node_id)
                        parent_chain.current_depth = max(
                            parent_chain.current_depth,
                            node.depth
                        )
                else:
                    # Parent not found, create new chain
                    new_chain = ContextChain(
                        id=chain_id,
                        root_id=node_id,
                        nodes={node_id: node},
                        current_depth=0,
                        metadata={}
                    )
                    self.chains[chain_id] = new_chain
            else:
                # Create new chain
                new_chain = ContextChain(
                    id=chain_id,
                    root_id=node_id,
                    nodes={node_id: node},
                    current_depth=0,
                    metadata={}
                )
                self.chains[chain_id] = new_chain

            return node
        except Exception as e:
            logger.error(f"Failed to add context: {str(e)}")
            raise

    async def find_relevant_contexts(
            self,
            query: str,
            query_embeddings: List[float],
            max_results: int = 5
    ) -> List[ContextNode]:
        """Find relevant contexts based on semantic similarity."""
        relevant_nodes = []

        for chain in self.chains.values():
            for node in chain.nodes.values():
                similarity = 1 - cosine(query_embeddings, node.embeddings)
                if similarity >= self.similarity_threshold:
                    relevant_nodes.append((node, similarity))

        # Sort by similarity and return top results
        relevant_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in relevant_nodes[:max_results]]

    async def get_context_chain(
            self,
            node_id: str,
            include_siblings: bool = False
    ) -> List[ContextNode]:
        """Get the full context chain for a node."""
        chain = self._find_chain_with_node(node_id)
        if not chain:
            return []

        nodes = []
        current_node = chain.nodes[node_id]

        # Get ancestors
        while current_node.parent_id:
            nodes.append(chain.nodes[current_node.parent_id])
            current_node = chain.nodes[current_node.parent_id]

        # Reverse to get chronological order
        nodes.reverse()

        # Add current node
        nodes.append(chain.nodes[node_id])

        # Add descendants
        queue = [node_id]
        while queue:
            current_id = queue.pop(0)
            current_node = chain.nodes[current_id]
            for child_id in current_node.children_ids:
                nodes.append(chain.nodes[child_id])
                queue.append(child_id)

        if include_siblings:
            # Add siblings
            parent_id = chain.nodes[node_id].parent_id
            if parent_id:
                parent_node = chain.nodes[parent_id]
                for sibling_id in parent_node.children_ids:
                    if sibling_id != node_id:
                        nodes.append(chain.nodes[sibling_id])

        return nodes

    async def merge_chains(
            self,
            chain_id1: str,
            chain_id2: str,
            similarity_threshold: float = 0.9
    ) -> Optional[str]:
        """Merge two context chains if they are semantically similar."""
        if chain_id1 not in self.chains or chain_id2 not in self.chains:
            return None

        chain1 = self.chains[chain_id1]
        chain2 = self.chains[chain_id2]

        # Check if chains are already connected
        if nx.has_path(self.chain_graph, chain_id1, chain_id2) or \
                nx.has_path(self.chain_graph, chain_id2, chain_id1):
            return None

        # Calculate similarity between root nodes
        root1 = chain1.nodes[chain1.root_id]
        root2 = chain2.nodes[chain2.root_id]
        similarity = 1 - cosine(root1.embeddings, root2.embeddings)

        if similarity >= similarity_threshold:
            # Merge chains
            merged_id = f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            merged_chain = ContextChain(
                id=merged_id,
                root_id=chain1.root_id,
                nodes={**chain1.nodes, **chain2.nodes},
                current_depth=max(chain1.current_depth, chain2.current_depth),
                metadata={
                    "merged_from": [chain_id1, chain_id2],
                    "similarity": similarity
                }
            )

            # Update chain graph
            self.chain_graph.add_edge(chain_id1, merged_id)
            self.chain_graph.add_edge(chain_id2, merged_id)

            # Store merged chain
            self.chains[merged_id] = merged_chain

            return merged_id

        return None

    def _find_chain_with_node(self, node_id: str) -> Optional[ContextChain]:
        """Find the chain containing a specific node."""
        for chain in self.chains.values():
            if node_id in chain.nodes:
                return chain
        return None

    async def analyze_chain_structure(self, chain_id: str) -> Dict[str, Any]:
        """Analyze the structure of a context chain."""
        if chain_id not in self.chains:
            return {}

        chain = self.chains[chain_id]
        analysis = {
            "depth": chain.current_depth,
            "node_count": len(chain.nodes),
            "branching_factor": self._calculate_branching_factor(chain),
            "semantic_coherence": self._calculate_semantic_coherence(chain),
            "temporal_distribution": self._analyze_temporal_distribution(chain)
        }

        return analysis

    def _calculate_branching_factor(self, chain: ContextChain) -> float:
        """Calculate the average branching factor of the chain."""
        if not chain.nodes:
            return 0.0

        total_children = sum(len(node.children_ids) for node in chain.nodes.values())
        return total_children / len(chain.nodes)

    def _calculate_semantic_coherence(self, chain: ContextChain) -> float:
        """Calculate the semantic coherence of the chain."""
        if not chain.nodes:
            return 0.0

        similarities = []
        for node in chain.nodes.values():
            if node.parent_id:
                parent = chain.nodes[node.parent_id]
                similarity = 1 - cosine(node.embeddings, parent.embeddings)
                similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def _analyze_temporal_distribution(self, chain: ContextChain) -> Dict[str, Any]:
        """Analyze the temporal distribution of nodes in the chain."""
        if not chain.nodes:
            return {}

        timestamps = [node.timestamp for node in chain.nodes.values()]
        time_diffs = np.diff([ts.timestamp() for ts in sorted(timestamps)])

        return {
            "total_duration": (max(timestamps) - min(timestamps)).total_seconds(),
            "average_interval": np.mean(time_diffs) if len(time_diffs) > 0 else 0.0,
            "interval_std": np.std(time_diffs) if len(time_diffs) > 0 else 0.0
        }


# Global chainer instance
context_chainer = ContextChainer()
