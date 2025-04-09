import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class SemanticVisualizer:
    """Handles visualization of semantic analysis results."""

    def __init__(self):
        self.output_dir = Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)

    def create_semantic_graph(
            self,
            semantic_result: Dict[str, Any],
            title: str = "Semantic Analysis Graph"
    ) -> str:
        """Create a graph visualization of semantic relationships."""
        try:
            # Create directed graph
            G = nx.DiGraph()

            # Add nodes for entities
            for entity, data in semantic_result.get("entities", {}).items():
                G.add_node(entity, **data)

            # Add edges for relationships
            for rel in semantic_result.get("relationships", []):
                G.add_edge(rel["source"], rel["target"], **rel)

            # Create visualization
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue',
                    node_size=2000, font_size=10, font_weight='bold')

            # Save plot
            filename = f"semantic_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath)
            plt.close()

            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to create semantic graph: {str(e)}")
            raise

    def create_semantic_embedding_plot(
            self,
            embeddings: Dict[str, List[float]],
            title: str = "Semantic Embeddings"
    ) -> str:
        """Create a 2D plot of semantic embeddings."""
        try:
            # Convert embeddings to numpy array
            entities = list(embeddings.keys())
            X = np.array(list(embeddings.values()))

            # Use PCA for dimensionality reduction
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)

            # Create plot
            plt.figure(figsize=(10, 8))
            plt.scatter(X_2d[:, 0], X_2d[:, 1])

            # Add labels
            for i, entity in enumerate(entities):
                plt.annotate(entity, (X_2d[i, 0], X_2d[i, 1]))

            plt.title(title)

            # Save plot
            filename = f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath)
            plt.close()

            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to create embedding plot: {str(e)}")
            raise

    def create_role_distribution_chart(
            self,
            semantic_result: Dict[str, Any],
            title: str = "Role Distribution"
    ) -> str:
        """Create a chart showing the distribution of semantic roles."""
        try:
            # Count roles
            role_counts = {}
            for entity, data in semantic_result.get("entities", {}).items():
                role = data.get("role", "unknown")
                role_counts[role] = role_counts.get(role, 0) + 1

            # Create bar chart
            plt.figure(figsize=(10, 6))
            plt.bar(role_counts.keys(), role_counts.values())
            plt.title(title)
            plt.xticks(rotation=45)

            # Save plot
            filename = f"role_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath)
            plt.close()

            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to create role distribution chart: {str(e)}")
            raise

    def create_context_timeline(
            self,
            context_history: List[Dict[str, Any]],
            title: str = "Context Timeline"
    ) -> str:
        """Create a timeline visualization of context history."""
        try:
            # Create timeline data
            events = []
            for context in context_history:
                events.append({
                    "timestamp": context["timestamp"],
                    "content": context["content"],
                    "confidence": context.get("confidence", 1.0)
                })

            # Create plotly timeline
            fig = go.Figure()

            for event in events:
                fig.add_trace(go.Scatter(
                    x=[event["timestamp"]],
                    y=[1],  # Simple y-axis for now
                    text=[event["content"]],
                    mode="markers+text",
                    marker=dict(
                        size=10,
                        color=event["confidence"],
                        colorscale="Viridis"
                    ),
                    textposition="top center"
                ))

            fig.update_layout(
                title=title,
                xaxis_title="Time",
                showlegend=False
            )

            # Save plot
            filename = f"timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to create context timeline: {str(e)}")
            raise

    def create_combined_report(
            self,
            semantic_result: Dict[str, Any],
            title: str = "Semantic Analysis Report"
    ) -> str:
        """Create a combined report with multiple visualizations."""
        try:
            # Create individual visualizations
            graph_path = self.create_semantic_graph(semantic_result, f"{title} - Graph")
            embeddings_path = self.create_semantic_embedding_plot(
                semantic_result.get("embeddings", {}),
                f"{title} - Embeddings"
            )
            roles_path = self.create_role_distribution_chart(
                semantic_result,
                f"{title} - Role Distribution"
            )

            # Create HTML report
            report = f"""
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .visualization {{ margin: 20px; }}
                    img {{ max-width: 100%; }}
                </style>
            </head>
            <body>
                <h1>{title}</h1>
                <div class="visualization">
                    <h2>Semantic Graph</h2>
                    <img src="{graph_path}" alt="Semantic Graph">
                </div>
                <div class="visualization">
                    <h2>Semantic Embeddings</h2>
                    <img src="{embeddings_path}" alt="Semantic Embeddings">
                </div>
                <div class="visualization">
                    <h2>Role Distribution</h2>
                    <img src="{roles_path}" alt="Role Distribution">
                </div>
            </body>
            </html>
            """

            # Save report
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                f.write(report)

            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to create combined report: {str(e)}")
            raise


# Global visualizer instance
semantic_visualizer = SemanticVisualizer()
