"""API module for the aware-agent-backend."""

from typing import Dict, Any
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import OpenAI

from src.core.services.interaction_engine import InteractionEngine
from src.core.services.semantic_abstraction import ContextNode
from src.core.services.memory_engine import MemoryEngine
from src.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
llm = OpenAI(temperature=0.7)
config = Config()
memory_engine = MemoryEngine(storage_path=config.memory_storage_path)
interaction_engine = InteractionEngine(llm=llm, memory_engine=memory_engine)


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
        logger.info("Initializing conversation with context: %s", init.initial_context)
        context_node = ContextNode.from_dict(init.initial_context)
        interaction_engine.initialize_conversation(context_node, init.initial_goal)
        return {"status": "success", "message": "Conversation initialized"}
    except Exception as e:
        logger.error("Error initializing conversation: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversation/process")
async def process_interaction(input: UserInput):
    try:
        if not interaction_engine.conversation_state:
            logger.error("Conversation not initialized")
            raise HTTPException(status_code=400, detail="Conversation not initialized")

        logger.info("Processing interaction: %s", input.content)
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
    except HTTPException as e:
        logger.error("Error processing interaction: %s", str(e))
        raise e
    except Exception as e:
        logger.error("Error processing interaction: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversation/feedback")
async def submit_feedback(feedback: Feedback):
    try:
        if not interaction_engine.conversation_state:
            logger.error("Conversation not initialized")
            raise HTTPException(status_code=400, detail="Conversation not initialized")

        logger.info("Submitting feedback: %s", feedback.model_dump())
        interaction_engine.integrate_feedback(feedback.model_dump())
        return {"status": "success", "message": "Feedback integrated"}
    except HTTPException as e:
        logger.error("Error submitting feedback: %s", str(e))
        raise e
    except Exception as e:
        logger.error("Error submitting feedback: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/state")
async def get_conversation_state():
    try:
        if not interaction_engine.conversation_state:
            logger.error("Conversation not initialized")
            raise HTTPException(status_code=400, detail="Conversation not initialized")

        logger.info("Getting conversation state")
        return interaction_engine.conversation_state.model_dump()
    except HTTPException as e:
        logger.error("Error getting conversation state: %s", str(e))
        raise e
    except Exception as e:
        logger.error("Error getting conversation state: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
