from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from ..agent_orchestration import AgentResponse
from ..semantic_abstraction import ContextNode
from datetime import datetime
import uuid
from ..memory.persistence import MemoryStore, ConversationRecord, MessageRecord
from ..agent_orchestration import AgentOrchestrator

class ConversationState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    messages: List[Dict[str, Any]]
    context_tree: ContextNode
    current_goal: str
    feedback_history: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}

class PromptBuilder:
    def __init__(self):
        self.templates = {
            "goal_clarification": """
            Based on the following context and conversation history:
            Context: {context}
            History: {history}
            
            Please help clarify the user's goal by:
            1. Identifying any ambiguities
            2. Suggesting specific questions to ask
            3. Proposing potential goal refinements
            """,
            
            "response_synthesis": """
            Synthesize the following agent responses into a coherent, engaging reply:
            Planner: {planner_response}
            Research: {research_response}
            Explainer: {explainer_response}
            Validator: {validator_response}
            
            Consider:
            1. Maintaining conversational flow
            2. Highlighting key insights
            3. Addressing potential concerns
            4. Suggesting next steps
            """,
            
            "feedback_integration": """
            Integrate the following feedback into the conversation:
            Feedback: {feedback}
            Current State: {current_state}
            
            Generate:
            1. Acknowledgment of feedback
            2. Adjustments to approach
            3. Next steps based on feedback
            """
        }

    def build_goal_clarification_prompt(self, state: ConversationState) -> str:
        return self.templates["goal_clarification"].format(
            context=state.context_tree.visualize_context_tree(),
            history=self._format_message_history(state.messages)
        )

    def build_response_synthesis_prompt(self, agent_responses: Dict[str, AgentResponse]) -> str:
        return self.templates["response_synthesis"].format(
            planner_response=agent_responses["planner"].content,
            research_response=agent_responses["research"].content,
            explainer_response=agent_responses["explainer"].content,
            validator_response=agent_responses["validator"].content
        )

    def build_feedback_integration_prompt(self, state: ConversationState, feedback: str) -> str:
        return self.templates["feedback_integration"].format(
            feedback=feedback,
            current_state=state.json()
        )

    def _format_message_history(self, messages: List[Dict[str, Any]]) -> str:
        return "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg['content']}"
            for msg in messages
        ])

class ResponseTranslator:
    def __init__(self):
        self.formats = {
            "markdown": self._format_markdown,
            "structured": self._format_structured,
            "conversational": self._format_conversational
        }

    def translate(self, response: str, format_type: str = "conversational") -> Dict[str, Any]:
        if format_type not in self.formats:
            raise ValueError(f"Unsupported format type: {format_type}")
        return self.formats[format_type](response)

    def _format_markdown(self, response: str) -> Dict[str, Any]:
        return {
            "type": "markdown",
            "content": response,
            "metadata": {"format": "markdown"}
        }

    def _format_structured(self, response: str) -> Dict[str, Any]:
        # Split response into sections based on common patterns
        sections = {
            "main_points": [],
            "supporting_evidence": [],
            "next_steps": []
        }
        # Add parsing logic here
        return {
            "type": "structured",
            "content": sections,
            "metadata": {"format": "structured"}
        }

    def _format_conversational(self, response: str) -> Dict[str, Any]:
        return {
            "type": "conversational",
            "content": response,
            "metadata": {"format": "conversational"}
        }

class FeedbackIntegrator:
    def __init__(self):
        self.feedback_types = {
            "clarification": self._handle_clarification,
            "correction": self._handle_correction,
            "expansion": self._handle_expansion,
            "preference": self._handle_preference
        }

    def integrate(self, feedback: Dict[str, Any], state: ConversationState) -> ConversationState:
        feedback_type = feedback.get("type", "clarification")
        if feedback_type not in self.feedback_types:
            raise ValueError(f"Unsupported feedback type: {feedback_type}")
        
        return self.feedback_types[feedback_type](feedback, state)

    def _handle_clarification(self, feedback: Dict[str, Any], state: ConversationState) -> ConversationState:
        # Update context tree with clarification
        state.context_tree.dimension.metadata["clarifications"] = feedback.get("content", [])
        return state

    def _handle_correction(self, feedback: Dict[str, Any], state: ConversationState) -> ConversationState:
        # Update context tree with corrections
        state.context_tree.dimension.metadata["corrections"] = feedback.get("content", [])
        return state

    def _handle_expansion(self, feedback: Dict[str, Any], state: ConversationState) -> ConversationState:
        # Add new branches to context tree
        state.context_tree.dimension.metadata["expansions"] = feedback.get("content", [])
        return state

    def _handle_preference(self, feedback: Dict[str, Any], state: ConversationState) -> ConversationState:
        # Update conversation preferences
        state.metadata["preferences"] = feedback.get("content", {})
        return state

class InteractionEngine:
    def __init__(self):
        self.memory_store = MemoryStore()
        self.agent_orchestrator = AgentOrchestrator()
        self.active_conversations: Dict[str, str] = {}  # conversation_id -> current_goal
        self.prompt_builder = PromptBuilder()
        self.response_translator = ResponseTranslator()
        self.feedback_integrator = FeedbackIntegrator()
        self.conversation_state = None

    async def create_conversation(self, title: str, initial_goal: str) -> str:
        conversation_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        conversation = ConversationRecord(
            id=conversation_id,
            title=title,
            created_at=now,
            updated_at=now,
            metadata={"current_goal": initial_goal}
        )
        
        await self.memory_store.create_conversation(conversation)
        self.active_conversations[conversation_id] = initial_goal
        
        # Initialize context tree
        context_tree = ContextNode(
            id=str(uuid.uuid4()),
            type="root",
            content={"goal": initial_goal},
            metadata={}
        )
        await self.memory_store.update_context_tree(conversation_id, context_tree)
        
        return conversation_id

    async def process_message(self, conversation_id: str, content: str) -> Dict[str, Any]:
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found or inactive")
        
        # Get current context
        context_tree = await self.memory_store.get_context_tree(conversation_id)
        if not context_tree:
            raise ValueError(f"Context tree not found for conversation {conversation_id}")
        
        # Process through agent orchestrator
        agent_responses = await self.agent_orchestrator.process_query(content, context_tree)
        
        # Store user message
        user_message = MessageRecord(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            content=content,
            role="user",
            timestamp=datetime.utcnow(),
            metadata={}
        )
        await self.memory_store.add_message(user_message)
        
        # Store agent responses
        for agent_type, response in agent_responses.items():
            agent_message = MessageRecord(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                content=response.content,
                role="assistant",
                timestamp=datetime.utcnow(),
                metadata={
                    "agentType": agent_type,
                    **response.metadata
                }
            )
            await self.memory_store.add_message(agent_message)
        
        # Update context tree with new information
        context_tree.add_interaction(content, agent_responses)
        await self.memory_store.update_context_tree(conversation_id, context_tree)
        
        return {
            "responses": agent_responses,
            "context": context_tree.to_dict()
        }

    async def get_conversation_history(self, conversation_id: str) -> List[MessageRecord]:
        return await self.memory_store.get_conversation_history(conversation_id)

    async def get_context_tree(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        context_tree = await self.memory_store.get_context_tree(conversation_id)
        return context_tree.to_dict() if context_tree else None

    async def list_conversations(self, limit: int = 10, offset: int = 0) -> List[ConversationRecord]:
        return await self.memory_store.list_conversations(limit, offset)

    async def get_agent_statuses(self, conversation_id: str) -> Dict[str, Any]:
        # Get latest agent states from the conversation's context
        context_tree = await self.memory_store.get_context_tree(conversation_id)
        if not context_tree:
            return {}
        
        return {
            agent_type: {
                "status": "active",
                "last_update": datetime.utcnow().isoformat()
            }
            for agent_type in ["planner", "researcher", "explainer", "validator"]
        }

    def integrate_feedback(self, feedback: Dict[str, Any]) -> None:
        if not self.conversation_state:
            raise ValueError("Conversation not initialized")

        # Integrate feedback into conversation state
        self.conversation_state = self.feedback_integrator.integrate(feedback, self.conversation_state)
        self.conversation_state.feedback_history.append(feedback) 