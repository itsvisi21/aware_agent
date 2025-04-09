import base64
import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


class WorkspaceType(Enum):
    """Supported workspace types."""
    NOTION = "notion"
    OBSIDIAN = "obsidian"
    GITHUB = "github"
    MARKDOWN = "markdown"


class WorkspaceExporter:
    """Handles exporting research results to various workspace platforms."""

    def __init__(self):
        self.output_dir = Path("exports")
        self.output_dir.mkdir(exist_ok=True)

    async def export_to_notion(
            self,
            content: Dict[str, Any],
            title: str,
            api_key: str,
            database_id: str,
            parent_page_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export research results to Notion."""
        try:
            # Prepare Notion page content
            notion_content = {
                "parent": {
                    "database_id": database_id
                },
                "properties": {
                    "Title": {
                        "title": [
                            {
                                "text": {
                                    "content": title
                                }
                            }
                        ]
                    },
                    "Status": {
                        "select": {
                            "name": "Research"
                        }
                    },
                    "Last Updated": {
                        "date": {
                            "start": datetime.now().isoformat()
                        }
                    }
                },
                "children": [
                    {
                        "object": "block",
                        "type": "heading_1",
                        "heading_1": {
                            "rich_text": [
                                {
                                    "text": {
                                        "content": "Research Summary"
                                    }
                                }
                            ]
                        }
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {
                                    "text": {
                                        "content": json.dumps(content, indent=2)
                                    }
                                }
                            ]
                        }
                    }
                ]
            }

            # Add parent page if specified
            if parent_page_id:
                notion_content["parent"] = {
                    "page_id": parent_page_id
                }

            # Create page in Notion
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Notion-Version": "2022-06-28",
                    "Content-Type": "application/json"
                }

                async with session.post(
                        "https://api.notion.com/v1/pages",
                        headers=headers,
                        json=notion_content
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        raise Exception(f"Notion API error: {error}")

                    result = await response.json()
                    return {
                        "page_id": result["id"],
                        "url": result["url"]
                    }
        except Exception as e:
            logger.error(f"Failed to export to Notion: {str(e)}")
            raise

    async def export_to_obsidian(
            self,
            content: Dict[str, Any],
            title: str,
            vault_path: str,
            folder: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export research results to Obsidian vault."""
        try:
            # Create output directory if needed
            output_dir = Path(vault_path)
            if folder:
                output_dir = output_dir / folder
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create markdown content
            markdown = f"# {title}\n\n"
            markdown += f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"

            # Add semantic analysis results
            if "semantic_result" in content:
                markdown += "## Semantic Analysis\n\n"
                markdown += "### Entities\n\n"
                for entity, data in content["semantic_result"].get("entities", {}).items():
                    markdown += f"- **{entity}** ({data.get('role', 'unknown')})\n"

                markdown += "\n### Relationships\n\n"
                for rel in content["semantic_result"].get("relationships", []):
                    markdown += f"- {rel['source']} → {rel['target']} ({rel.get('type', 'unknown')})\n"

            # Add temporal analysis results
            if "temporal_results" in content:
                markdown += "\n## Temporal Analysis\n\n"
                for event in content["temporal_results"].get("events", []):
                    markdown += f"- {event['content']} ({event.get('start_time', 'unknown')})\n"

            # Add abstraction results
            if "abstraction_results" in content:
                markdown += "\n## Abstraction Analysis\n\n"
                for type_, results in content["abstraction_results"].items():
                    markdown += f"### {type_.title()}\n\n"
                    markdown += f"{json.dumps(results, indent=2)}\n\n"

            # Save to file
            filename = f"{title.lower().replace(' ', '_')}.md"
            filepath = output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown)

            return {
                "filepath": str(filepath),
                "vault_path": vault_path
            }
        except Exception as e:
            logger.error(f"Failed to export to Obsidian: {str(e)}")
            raise

    async def export_to_github(
            self,
            content: Dict[str, Any],
            title: str,
            repo: str,
            path: str,
            token: str,
            branch: str = "main"
    ) -> Dict[str, Any]:
        """Export research results to GitHub repository."""
        try:
            # Prepare GitHub API request
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }

            # Create or update file
            filename = f"{title.lower().replace(' ', '_')}.json"
            filepath = f"{path}/{filename}" if path else filename

            # Get file SHA if it exists
            sha = None
            async with aiohttp.ClientSession() as session:
                url = f"https://api.github.com/repos/{repo}/contents/{filepath}"
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        sha = data["sha"]

                # Create or update file
                content_base64 = base64.b64encode(
                    json.dumps(content, indent=2).encode('utf-8')
                ).decode('utf-8')

                payload = {
                    "message": f"Update research: {title}",
                    "content": content_base64,
                    "branch": branch
                }

                if sha:
                    payload["sha"] = sha

                async with session.put(url, headers=headers, json=payload) as response:
                    if response.status not in (200, 201):
                        error = await response.text()
                        raise Exception(f"GitHub API error: {error}")

                    result = await response.json()
                    return {
                        "sha": result["content"]["sha"],
                        "url": result["content"]["html_url"]
                    }
        except Exception as e:
            logger.error(f"Failed to export to GitHub: {str(e)}")
            raise

    async def export_to_markdown(
            self,
            content: Dict[str, Any],
            title: str,
            format: str = "detailed"
    ) -> Dict[str, Any]:
        """Export research results to markdown file."""
        try:
            # Create markdown content
            markdown = f"# {title}\n\n"
            markdown += f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"

            if format == "detailed":
                markdown += json.dumps(content, indent=2)
            else:  # summary
                if "semantic_result" in content:
                    markdown += "## Key Findings\n\n"
                    for entity, data in content["semantic_result"].get("entities", {}).items():
                        markdown += f"- {entity} ({data.get('role', 'unknown')})\n"

            # Save to file
            filename = f"{title.lower().replace(' ', '_')}.md"
            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown)

            return {
                "filepath": str(filepath)
            }
        except Exception as e:
            logger.error(f"Failed to export to markdown: {str(e)}")
            raise


# Global exporter instance
workspace_exporter = WorkspaceExporter()
