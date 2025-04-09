from src.core.services.semantic_abstraction import ContextNode
import logging
from typing import Dict, List, Any, Optional
import os
import json
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from datetime import datetime

logger = logging.getLogger(__name__)

class VisualizationService:
    """Service for creating visualizations of data and context."""
    
    def __init__(self):
        """Initialize the visualization service."""
        self.output_dir = "visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_context_graph(self, context: ContextNode) -> str:
        """Create a graph visualization of a context node and its relationships.
        
        Args:
            context: The context node to visualize
            
        Returns:
            Path to the saved visualization file
        """
        G = nx.DiGraph()
        self._add_node_to_graph(G, context)
        
        pos = nx.spring_layout(G)
        
        # Create the plotly figure
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers+text',
            hoverinfo='text',
            text=[],
            textposition="bottom center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
            ))

        # Add edges to the visualization
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        # Add nodes to the visualization
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([node])

        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Context Graph',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"context_graph_{timestamp}.html")
        fig.write_html(output_file)
        
        return output_file
    
    def _add_node_to_graph(self, G: nx.DiGraph, node: ContextNode, parent: Optional[str] = None) -> None:
        """Recursively add nodes and edges to the graph.
        
        Args:
            G: The NetworkX graph
            node: The current context node
            parent: ID of the parent node (if any)
        """
        G.add_node(node.id, label=node.type)
        if parent:
            G.add_edge(parent, node.id)
        
        for child in node.children:
            self._add_node_to_graph(G, child, node.id)
    
    def create_performance_chart(self, metrics: Dict[str, Any]) -> str:
        """Create a performance visualization from metrics data.
        
        Args:
            metrics: Dictionary containing performance metrics
            
        Returns:
            Path to the saved visualization file
        """
        operations = list(metrics.keys())
        avg_times = [metrics[op]["total_time"] / metrics[op]["count"] if metrics[op]["count"] > 0 else 0 
                    for op in operations]
        
        fig = px.bar(
            x=operations,
            y=avg_times,
            title="Average Operation Times",
            labels={"x": "Operation", "y": "Average Time (seconds)"}
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"performance_chart_{timestamp}.html")
        fig.write_html(output_file)
        
        return output_file
    
    def create_memory_usage_chart(self, memory_data: List[Dict[str, Any]]) -> str:
        """Create a visualization of memory usage over time.
        
        Args:
            memory_data: List of dictionaries containing memory usage data
            
        Returns:
            Path to the saved visualization file
        """
        timestamps = [entry["timestamp"] for entry in memory_data]
        usage = [entry["usage"] for entry in memory_data]
        
        fig = px.line(
            x=timestamps,
            y=usage,
            title="Memory Usage Over Time",
            labels={"x": "Time", "y": "Memory Usage (MB)"}
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"memory_usage_{timestamp}.html")
        fig.write_html(output_file)
        
        return output_file 