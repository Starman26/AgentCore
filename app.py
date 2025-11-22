from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uuid
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
import uvicorn
import os
from pathlib import Path

from agent.graph import graph, State

checkpointer = MemorySaver()
compiled_graph = graph.compile(checkpointer=checkpointer)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_email: Optional[str] = None
    timezone: Optional[str] = "America/Monterrey"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    user_identified: bool
    timestamp: str

class UploadResponse(BaseModel):
    filename: str
    file_path: str
    size: int
    upload_time: str
    message: str

@app.get("/")
async def root():
    """Main Endpoint to check API status"""
    return {
        "message": "AgentCore API is running",
        "version": "1.0.0",
        "endpoints": {
            "message": "/message?mensaje=tu_mensaje (GET - Simple)",
            "chat": "/chat (POST - Completo)",
            "upload": "/upload (POST)",
            "health": "/health"
        },
        "examples": {
            "simple_message": "http://localhost:8000/message?mensaje=Hola como estas"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint to verify server status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/message")
async def simple_message(mensaje: str):
    """
    Simple endpoint to send messages via query parameters.

    Example: http://localhost:8000/message?mensaje=Hello%20how%20are%20you

    Args:
        mensaje: The message to send

    Returns:
        Agent's response in a simple format
    """
    try:
        session_id = str(uuid.uuid4())
        timezone = "America/Monterrey"
        
        config = {"configurable": {"thread_id": session_id}}
        
        initial_state: State = {
            "messages": [HumanMessage(content=mensaje)],
            "tz": timezone,
            "session_id": session_id
        }
        
        result = compiled_graph.invoke(initial_state, config)
        
        messages = result.get("messages", [])
        agent_response = ""
        
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'ai':
                agent_response = getattr(msg, 'content', '')
                break
        
        if not agent_response:
            agent_response = "Sorry, I couldn't process your message. Please try again."
        
        return {
            "response": agent_response,
            "session_id": session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the message: {str(e)}")
    
@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
):
    """
    Endpoint to upload files.
    
    Args:
        file: File to upload
    
    Returns:
        Information about the uploaded file
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Nombre de archivo inválido")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.filename).suffix
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        content = await file.read()
        file_size = len(content)
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        return UploadResponse(
            filename=file.filename,
            file_path=str(file_path),
            size=file_size,
            upload_time=datetime.now().isoformat(),
            message=f"File '{file.filename}' uploaded successfully")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading the file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)