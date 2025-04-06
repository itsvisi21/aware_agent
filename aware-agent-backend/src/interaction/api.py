from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from .interaction_engine import InteractionEngine, ConversationState
from ..semantic_abstraction import ContextNode

app = FastAPI()
interaction_engine = InteractionEngine()

class UserInput(BaseModel):
    content: str
    format_type: str = "conversational"

class Feedback(BaseModel):
    type: str
    content: Dict[str, Any]

class ConversationInit(BaseModel):
    initial_context: Dict[str, Any]
    initial_goal: str

@app.post("/conversation/init")
async def initialize_conversation(init: ConversationInit):
    try:
        context_node = ContextNode.from_dict(init.initial_context)
        interaction_engine.initialize_conversation(context_node, init.initial_goal)
        return {"status": "success", "message": "Conversation initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversation/process")
async def process_interaction(input: UserInput):
    try:
        if not interaction_engine.conversation_state:
            raise HTTPException(status_code=400, detail="Conversation not initialized")
        
        # Get agent responses (this would come from your agent orchestration system)
        agent_responses = {
            "planner": {"content": "Planner response"},
            "research": {"content": "Research response"},
            "explainer": {"content": "Explainer response"},
            "validator": {"content": "Validator response"}
        }
        
        response = await interaction_engine.process_interaction(
            input.content,
            agent_responses,
            input.format_type
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversation/feedback")
async def submit_feedback(feedback: Feedback):
    try:
        if not interaction_engine.conversation_state:
            raise HTTPException(status_code=400, detail="Conversation not initialized")
        
        interaction_engine.integrate_feedback(feedback.dict())
        return {"status": "success", "message": "Feedback integrated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/state")
async def get_conversation_state():
    try:
        if not interaction_engine.conversation_state:
            raise HTTPException(status_code=400, detail="Conversation not initialized")
        
        return interaction_engine.conversation_state.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 