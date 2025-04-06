from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

class InteractionType(Enum):
    RESEARCH = "research"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    GENERATION = "generation"
    VALIDATION = "validation"
    CLARIFICATION = "clarification"

@dataclass
class InteractionResponse:
    type: InteractionType
    content: Dict[str, Any]
    actions: List[Dict[str, Any]]
    feedback: Dict[str, Any]
    confidence: float
    semantic_context: Optional[Dict[str, Any]] = None

class InteractionEngine:
    def __init__(self):
        self.prompt_templates = {
            InteractionType.RESEARCH: """
            Based on the semantic analysis:
            - Primary agent: {agent}
            - Primary object: {object}
            - Domain context: {domain}
            - Temporal constraints: {temporal}
            - Spatial constraints: {spatial}
            
            Please provide research guidance for: {query}
            """,
            InteractionType.ANALYSIS: """
            Analyze the following with semantic context:
            - Semantic roles: {roles}
            - Role hierarchy: {hierarchy}
            - Temporal structure: {temporal}
            - Spatial structure: {spatial}
            
            Analysis target: {query}
            """,
            InteractionType.COMPARISON: """
            Compare the following with semantic context:
            - Primary entities: {entities}
            - Role relationships: {relationships}
            - Temporal aspects: {temporal}
            - Spatial aspects: {spatial}
            
            Comparison target: {query}
            """,
            InteractionType.GENERATION: """
            Generate content with semantic context:
            - Target role: {role}
            - Supporting roles: {supporting}
            - Temporal constraints: {temporal}
            - Spatial constraints: {spatial}
            
            Generation target: {query}
            """,
            InteractionType.VALIDATION: """
            Validate with semantic context:
            - Role validation: {roles}
            - Temporal validation: {temporal}
            - Spatial validation: {spatial}
            - Domain validation: {domain}
            
            Validation target: {query}
            """,
            InteractionType.CLARIFICATION: """
            Clarify with semantic context:
            - Ambiguous roles: {roles}
            - Temporal ambiguity: {temporal}
            - Spatial ambiguity: {spatial}
            - Domain ambiguity: {domain}
            
            Clarification target: {query}
            """
        }
        
    def construct_prompt(self, interaction_type: InteractionType, query: str, context: Dict[str, Any]) -> str:
        """
        Construct a prompt based on interaction type and semantic context.
        
        Args:
            interaction_type: The type of interaction
            query: The user query
            context: The semantic context
            
        Returns:
            Constructed prompt string
        """
        template = self.prompt_templates[interaction_type]
        
        # Extract semantic context
        semantic_roles = context.get("semantic_roles", {})
        temporal_dims = context.get("temporal_dimensions", {})
        spatial_dims = context.get("spatial_dimensions", {})
        domain_ctx = context.get("domain_context", {})
        
        # Build prompt parameters
        params = {
            "query": query,
            "agent": self._get_primary_agent(semantic_roles),
            "object": self._get_primary_object(semantic_roles),
            "domain": domain_ctx.get("domain", "general"),
            "temporal": self._format_temporal(temporal_dims),
            "spatial": self._format_spatial(spatial_dims),
            "roles": self._format_roles(semantic_roles),
            "hierarchy": self._format_hierarchy(context.get("role_hierarchy", {})),
            "entities": self._format_entities(semantic_roles),
            "relationships": self._format_relationships(context.get("semantic_graph", {})),
            "supporting": self._format_supporting_roles(semantic_roles)
        }
        
        return template.format(**params)
    
    def translate_response(self, agent_response: Dict[str, Any], context: Dict[str, Any]) -> InteractionResponse:
        """
        Translate agent response into structured interaction response.
        
        Args:
            agent_response: The agent's response
            context: The semantic context
            
        Returns:
            Structured interaction response
        """
        # Determine interaction type based on semantic roles
        interaction_type = self._determine_interaction_type(agent_response, context)
        
        # Extract content and actions
        content = self._extract_content(agent_response, context)
        actions = self._extract_actions(agent_response, context)
        feedback = self._extract_feedback(agent_response, context)
        
        # Calculate confidence
        confidence = self._calculate_confidence(agent_response, context)
        
        return InteractionResponse(
            type=interaction_type,
            content=content,
            actions=actions,
            feedback=feedback,
            confidence=confidence,
            semantic_context=context
        )
    
    def _get_primary_agent(self, semantic_roles: Dict[str, Any]) -> str:
        """Get primary agent from semantic roles."""
        agents = semantic_roles.get("AGENT", [])
        if not agents:
            return "unknown"
            
        # Get highest confidence agent
        return max(agents, key=lambda x: x["properties"]["confidence"])["entity"]
    
    def _get_primary_object(self, semantic_roles: Dict[str, Any]) -> str:
        """Get primary object from semantic roles."""
        objects = semantic_roles.get("OBJECT", [])
        if not objects:
            return "unknown"
            
        # Get highest confidence object
        return max(objects, key=lambda x: x["properties"]["confidence"])["entity"]
    
    def _format_temporal(self, temporal_dims: Dict[str, Any]) -> str:
        """Format temporal dimensions for prompt."""
        if not temporal_dims:
            return "none"
            
        return ", ".join([
            f"{k}: {v}" for k, v in temporal_dims.items()
        ])
    
    def _format_spatial(self, spatial_dims: Dict[str, Any]) -> str:
        """Format spatial dimensions for prompt."""
        if not spatial_dims:
            return "none"
            
        return ", ".join([
            f"{k}: {v}" for k, v in spatial_dims.items()
        ])
    
    def _format_roles(self, semantic_roles: Dict[str, Any]) -> str:
        """Format semantic roles for prompt."""
        if not semantic_roles:
            return "none"
            
        return ", ".join([
            f"{role}: {entity['entity']}" 
            for role, entities in semantic_roles.items()
            for entity in entities
            if entity["properties"]["confidence"] > 0.7
        ])
    
    def _format_hierarchy(self, hierarchy: Dict[str, Any]) -> str:
        """Format role hierarchy for prompt."""
        if not hierarchy:
            return "none"
            
        return ", ".join([
            f"{role}: {', '.join(subroles)}"
            for role, subroles in hierarchy.items()
        ])
    
    def _format_entities(self, semantic_roles: Dict[str, Any]) -> str:
        """Format entities for prompt."""
        if not semantic_roles:
            return "none"
            
        return ", ".join([
            entity["entity"]
            for entities in semantic_roles.values()
            for entity in entities
            if entity["properties"]["confidence"] > 0.7
        ])
    
    def _format_relationships(self, semantic_graph: Dict[str, Any]) -> str:
        """Format relationships for prompt."""
        if not semantic_graph:
            return "none"
            
        return ", ".join([
            f"{edge['source']} -> {edge['target']} ({edge['type']})"
            for edge in semantic_graph.get("edges", [])
        ])
    
    def _format_supporting_roles(self, semantic_roles: Dict[str, Any]) -> str:
        """Format supporting roles for prompt."""
        supporting_roles = ["INSTRUMENT", "BENEFICIARY", "LOCATION", "TIME"]
        if not semantic_roles:
            return "none"
            
        return ", ".join([
            f"{role}: {entity['entity']}"
            for role in supporting_roles
            for entity in semantic_roles.get(role, [])
            if entity["properties"]["confidence"] > 0.7
        ])
    
    def _determine_interaction_type(self, agent_response: Dict[str, Any], context: Dict[str, Any]) -> InteractionType:
        """Determine interaction type based on semantic roles and response."""
        semantic_roles = context.get("semantic_roles", {})
        
        # Check for validation needs
        if any(role["properties"].get("needs_validation", False) for role in semantic_roles.values()):
            return InteractionType.VALIDATION
            
        # Check for ambiguity
        if any(role["properties"].get("is_ambiguous", False) for role in semantic_roles.values()):
            return InteractionType.CLARIFICATION
            
        # Check response type
        response_type = agent_response.get("type", "")
        if "research" in response_type.lower():
            return InteractionType.RESEARCH
        elif "analysis" in response_type.lower():
            return InteractionType.ANALYSIS
        elif "comparison" in response_type.lower():
            return InteractionType.COMPARISON
        elif "generation" in response_type.lower():
            return InteractionType.GENERATION
            
        return InteractionType.RESEARCH
    
    def _extract_content(self, agent_response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from agent response."""
        content = {
            "primary": agent_response.get("response", {}),
            "supporting": self._extract_supporting_elements(agent_response, context),
            "temporal": context.get("temporal_dimensions", {}),
            "spatial": context.get("spatial_dimensions", {}),
            "domain": context.get("domain_context", {})
        }
        
        return content
    
    def _extract_actions(self, agent_response: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract actions from agent response."""
        actions = []
        
        # Extract primary action
        if "primary_action" in agent_response.get("response", {}):
            actions.append({
                "type": "primary",
                "action": agent_response["response"]["primary_action"],
                "confidence": agent_response.get("confidence", 0.0)
            })
            
        # Extract supporting actions
        if "supporting_actions" in agent_response.get("response", {}):
            actions.extend([
                {
                    "type": "supporting",
                    "action": action,
                    "confidence": agent_response.get("confidence", 0.0) * 0.8
                }
                for action in agent_response["response"]["supporting_actions"]
            ])
            
        return actions
    
    def _extract_feedback(self, agent_response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract feedback from agent response."""
        feedback = {
            "validation": self._extract_validation_feedback(agent_response, context),
            "clarification": self._extract_clarification_feedback(agent_response, context),
            "improvement": self._extract_improvement_feedback(agent_response, context)
        }
        
        return feedback
    
    def _calculate_confidence(self, agent_response: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate confidence score for interaction."""
        confidences = []
        
        # Response confidence
        confidences.append(agent_response.get("confidence", 0.0))
        
        # Role confidence
        semantic_roles = context.get("semantic_roles", {})
        if semantic_roles:
            role_confidences = [
                role["properties"]["confidence"]
                for roles in semantic_roles.values()
                for role in roles
            ]
            confidences.append(np.mean(role_confidences) if role_confidences else 0.0)
            
        # Temporal confidence
        temporal_dims = context.get("temporal_dimensions", {})
        if temporal_dims:
            confidences.append(0.8)
            
        # Spatial confidence
        spatial_dims = context.get("spatial_dimensions", {})
        if spatial_dims:
            confidences.append(0.8)
            
        # Domain confidence
        domain_ctx = context.get("domain_context", {})
        if domain_ctx:
            confidences.append(domain_ctx.get("relevance", 0.0))
            
        return np.mean(confidences) if confidences else 0.0
    
    def _extract_supporting_elements(self, agent_response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract supporting elements from agent response."""
        elements = {
            "instruments": [],
            "beneficiaries": [],
            "locations": [],
            "times": []
        }
        
        semantic_roles = context.get("semantic_roles", {})
        
        # Extract instruments
        if "INSTRUMENT" in semantic_roles:
            elements["instruments"] = [
                role["entity"]
                for role in semantic_roles["INSTRUMENT"]
                if role["properties"]["confidence"] > 0.7
            ]
            
        # Extract beneficiaries
        if "BENEFICIARY" in semantic_roles:
            elements["beneficiaries"] = [
                role["entity"]
                for role in semantic_roles["BENEFICIARY"]
                if role["properties"]["confidence"] > 0.7
            ]
            
        # Extract locations
        if "LOCATION" in semantic_roles:
            elements["locations"] = [
                role["entity"]
                for role in semantic_roles["LOCATION"]
                if role["properties"]["confidence"] > 0.7
            ]
            
        # Extract times
        if "TIME" in semantic_roles:
            elements["times"] = [
                role["entity"]
                for role in semantic_roles["TIME"]
                if role["properties"]["confidence"] > 0.7
            ]
            
        return elements
    
    def _extract_validation_feedback(self, agent_response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract validation feedback from agent response."""
        feedback = {
            "role_validation": [],
            "temporal_validation": [],
            "spatial_validation": [],
            "domain_validation": []
        }
        
        # Extract role validation
        semantic_roles = context.get("semantic_roles", {})
        for role, entities in semantic_roles.items():
            for entity in entities:
                if entity["properties"].get("needs_validation", False):
                    feedback["role_validation"].append({
                        "role": role,
                        "entity": entity["entity"],
                        "reason": entity["properties"].get("validation_reason", "")
                    })
                    
        # Extract temporal validation
        temporal_dims = context.get("temporal_dimensions", {})
        if temporal_dims.get("needs_validation", False):
            feedback["temporal_validation"].append({
                "aspect": temporal_dims.get("aspect", ""),
                "reason": temporal_dims.get("validation_reason", "")
            })
            
        # Extract spatial validation
        spatial_dims = context.get("spatial_dimensions", {})
        if spatial_dims.get("needs_validation", False):
            feedback["spatial_validation"].append({
                "aspect": spatial_dims.get("aspect", ""),
                "reason": spatial_dims.get("validation_reason", "")
            })
            
        # Extract domain validation
        domain_ctx = context.get("domain_context", {})
        if domain_ctx.get("needs_validation", False):
            feedback["domain_validation"].append({
                "aspect": domain_ctx.get("aspect", ""),
                "reason": domain_ctx.get("validation_reason", "")
            })
            
        return feedback
    
    def _extract_clarification_feedback(self, agent_response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract clarification feedback from agent response."""
        feedback = {
            "role_clarification": [],
            "temporal_clarification": [],
            "spatial_clarification": [],
            "domain_clarification": []
        }
        
        # Extract role clarification
        semantic_roles = context.get("semantic_roles", {})
        for role, entities in semantic_roles.items():
            for entity in entities:
                if entity["properties"].get("is_ambiguous", False):
                    feedback["role_clarification"].append({
                        "role": role,
                        "entity": entity["entity"],
                        "reason": entity["properties"].get("ambiguity_reason", "")
                    })
                    
        # Extract temporal clarification
        temporal_dims = context.get("temporal_dimensions", {})
        if temporal_dims.get("is_ambiguous", False):
            feedback["temporal_clarification"].append({
                "aspect": temporal_dims.get("aspect", ""),
                "reason": temporal_dims.get("ambiguity_reason", "")
            })
            
        # Extract spatial clarification
        spatial_dims = context.get("spatial_dimensions", {})
        if spatial_dims.get("is_ambiguous", False):
            feedback["spatial_clarification"].append({
                "aspect": spatial_dims.get("aspect", ""),
                "reason": spatial_dims.get("ambiguity_reason", "")
            })
            
        # Extract domain clarification
        domain_ctx = context.get("domain_context", {})
        if domain_ctx.get("is_ambiguous", False):
            feedback["domain_clarification"].append({
                "aspect": domain_ctx.get("aspect", ""),
                "reason": domain_ctx.get("ambiguity_reason", "")
            })
            
        return feedback
    
    def _extract_improvement_feedback(self, agent_response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract improvement feedback from agent response."""
        feedback = {
            "role_improvement": [],
            "temporal_improvement": [],
            "spatial_improvement": [],
            "domain_improvement": []
        }
        
        # Extract role improvement
        semantic_roles = context.get("semantic_roles", {})
        for role, entities in semantic_roles.items():
            for entity in entities:
                if entity["properties"].get("needs_improvement", False):
                    feedback["role_improvement"].append({
                        "role": role,
                        "entity": entity["entity"],
                        "suggestion": entity["properties"].get("improvement_suggestion", "")
                    })
                    
        # Extract temporal improvement
        temporal_dims = context.get("temporal_dimensions", {})
        if temporal_dims.get("needs_improvement", False):
            feedback["temporal_improvement"].append({
                "aspect": temporal_dims.get("aspect", ""),
                "suggestion": temporal_dims.get("improvement_suggestion", "")
            })
            
        # Extract spatial improvement
        spatial_dims = context.get("spatial_dimensions", {})
        if spatial_dims.get("needs_improvement", False):
            feedback["spatial_improvement"].append({
                "aspect": spatial_dims.get("aspect", ""),
                "suggestion": spatial_dims.get("improvement_suggestion", "")
            })
            
        # Extract domain improvement
        domain_ctx = context.get("domain_context", {})
        if domain_ctx.get("needs_improvement", False):
            feedback["domain_improvement"].append({
                "aspect": domain_ctx.get("aspect", ""),
                "suggestion": domain_ctx.get("improvement_suggestion", "")
            })
            
        return feedback 