from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from ..semantic_abstraction import ContextNode
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

class AgentResponse(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

class AgentOrchestrator:
    def __init__(self):
        self.llm = Ollama(model="llama2")
        self.agents = {
            "planner": self._create_planner_agent(),
            "researcher": self._create_researcher_agent(),
            "explainer": self._create_explainer_agent(),
            "validator": self._create_validator_agent()
        }
        self.context: Optional[ContextNode] = None

    def _create_planner_agent(self):
        return PromptTemplate(
            template="""You are a Planning Agent. Your role is to:
            1. Analyze the user's goal and break it down into actionable steps
            2. Create a structured plan with clear objectives
            3. Consider dependencies and potential roadblocks
            
            Current Context: {context}
            User Input: {input}
            
            Provide a clear, structured plan:""",
            input_variables=["context", "input"]
        )

    def _create_researcher_agent(self):
        return PromptTemplate(
            template="""You are a Research Agent. Your role is to:
            1. Gather relevant information and data
            2. Analyze patterns and relationships
            3. Provide evidence-based insights
            
            Current Context: {context}
            User Input: {input}
            
            Share your research findings:""",
            input_variables=["context", "input"]
        )

    def _create_explainer_agent(self):
        return PromptTemplate(
            template="""You are an Explainer Agent. Your role is to:
            1. Break down complex concepts
            2. Provide clear, accessible explanations
            3. Use analogies and examples when helpful
            
            Current Context: {context}
            User Input: {input}
            
            Explain the concept(s):""",
            input_variables=["context", "input"]
        )

    def _create_validator_agent(self):
        return PromptTemplate(
            template="""You are a Validator Agent. Your role is to:
            1. Verify the accuracy of information
            2. Check for logical consistency
            3. Identify potential issues or gaps
            
            Current Context: {context}
            User Input: {input}
            Previous Agent Responses: {previous_responses}
            
            Provide your validation:""",
            input_variables=["context", "input", "previous_responses"]
        )

    async def process_query(self, query: str, context: Optional[ContextNode] = None) -> Dict[str, AgentResponse]:
        self.context = context or self.context
        context_str = self.context.to_string() if self.context else ""
        
        responses: Dict[str, AgentResponse] = {}
        
        # Execute planner
        planner_response = await self.llm.agenerate(
            [self.agents["planner"].format(context=context_str, input=query)]
        )
        responses["planner"] = AgentResponse(
            content=planner_response.generations[0][0].text,
            metadata={"agentType": "planner"}
        )
        
        # Execute researcher
        researcher_response = await self.llm.agenerate(
            [self.agents["researcher"].format(context=context_str, input=query)]
        )
        responses["researcher"] = AgentResponse(
            content=researcher_response.generations[0][0].text,
            metadata={"agentType": "researcher"}
        )
        
        # Execute explainer
        explainer_response = await self.llm.agenerate(
            [self.agents["explainer"].format(context=context_str, input=query)]
        )
        responses["explainer"] = AgentResponse(
            content=explainer_response.generations[0][0].text,
            metadata={"agentType": "explainer"}
        )
        
        # Execute validator with all previous responses
        previous_responses = {
            "planner": responses["planner"].content,
            "researcher": responses["researcher"].content,
            "explainer": responses["explainer"].content
        }
        validator_response = await self.llm.agenerate([
            self.agents["validator"].format(
                context=context_str,
                input=query,
                previous_responses=str(previous_responses)
            )
        ])
        responses["validator"] = AgentResponse(
            content=validator_response.generations[0][0].text,
            metadata={"agentType": "validator"}
        )
        
        return responses

    def update_context(self, context: ContextNode):
        self.context = context 