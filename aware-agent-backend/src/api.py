from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
from src.agent import agent
import logging
import json
import asyncio

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aware Agent API",
    description="API for the self-aware agent system",
    version="1.0.0"
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

class AgentRequest(BaseModel):
    """Request model for agent processing."""
    input_text: str
    context: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    """Response model for agent processing."""
    status: str
    result: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    semantic_context: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@app.post("/process", response_model=AgentResponse)
async def process_input(request: AgentRequest) -> AgentResponse:
    """Process input through the agent."""
    try:
        result = await agent.process_input(
            input_text=request.input_text,
            context=request.context
        )
        return AgentResponse(**result)
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get agent metrics."""
    try:
        # Get memory patterns
        memory_patterns = agent.memory.analyze_memory_patterns()
        
        # Get self-awareness logs
        log_file = agent.workspace_dir / "self_awareness_logs.jsonl"
        if log_file.exists():
            with open(log_file, "r") as f:
                logs = [json.loads(line) for line in f]
                latest_log = logs[-1] if logs else {}
        else:
            latest_log = {}
        
        return {
            "memory_metrics": memory_patterns,
            "self_awareness": latest_log,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                request = json.loads(data)
                result = await agent.process_input(
                    input_text=request.get("input_text", ""),
                    context=request.get("context", {})
                )
                await websocket.send_json(result)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                await websocket.send_json({"error": str(e)})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 