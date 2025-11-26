from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
import uvicorn
from pathlib import Path

from agent.graph import graph, State

checkpointer = MemorySaver()
compiled_graph = graph.compile(checkpointer=checkpointer)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, cambiar a dominios específicos
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
            "health": "/health",
        },
        "examples": {
            "simple_message": "http://localhost:8000/message?mensaje=Hola como estas"
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint to verify server status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/message")
async def simple_message(
    mensaje: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,   # auth.user.id (UUID)
    user_email: Optional[str] = None,
):
    """
    Endpoint simple para enviar mensajes usando query params.
    - session_id: UUID de la sesión de chat
    - user_id: UUID de Supabase (auth.user.id)
    - user_email: solo para contexto, NO se usa como user_id
    """
    try:
        timezone = "America/Monterrey"

        # 1) Resolver session_id (UUID para la sesión)
        real_session_id = session_id or str(uuid.uuid4())

        # 2) Validar que user_id, si viene, parezca un UUID
        valid_user_id: Optional[str] = None
        if user_id:
            try:
                _ = uuid.UUID(user_id)
                valid_user_id = user_id
            except ValueError:
                print(f"[simple_message] WARNING: user_id inválido: {user_id}")

        # 3) Usuario confiable si tenemos un UUID válido
        trusted_user = valid_user_id is not None

        # 4) Config para el grafo (para LangGraph + nuestros tools)
        config = {
            "configurable": {
                "thread_id": real_session_id,
                "session_id": real_session_id,
            }
        }
        if valid_user_id:
            config["configurable"]["user_id"] = valid_user_id
        if user_email:
            config["configurable"]["user_email"] = user_email

        # 5) Estado inicial del grafo
        initial_state: State = {
            "messages": [HumanMessage(content=mensaje)],
            "tz": timezone,
            "session_id": real_session_id,
        }

        if valid_user_id:
            initial_state["user_id"] = valid_user_id
        if user_email:
            initial_state["user_email"] = user_email

        if trusted_user:
            initial_state["user_identified"] = True

        # 6) Invocar grafo
        result: State = compiled_graph.invoke(initial_state, config)

        messages = result.get("messages", [])
        agent_response = ""

        # Último mensaje del agente
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "ai":
                agent_response = getattr(msg, "content", "")
                break

        if not agent_response:
            agent_response = (
                "Lo siento, no pude procesar tu mensaje. Inténtalo de nuevo."
            )

        # ===== EXTRAER INFO DE TOOLS PARA EL PANEL DE DEBUG =====
        tool_events = []
        for msg in messages:
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                tool_events.append(
                    {
                        "from": getattr(msg, "type", "unknown"),
                        "tool_calls": tool_calls,
                    }
                )

        debug_payload = {"tool_events": tool_events}

        return {
            "response": agent_response,
            "session_id": real_session_id,
            "debug": debug_payload,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the message: {str(e)}",
        )


@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
):
    """
    Endpoint to upload files.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Nombre de archivo inválido")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            message=f"File '{file.filename}' uploaded successfully",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading the file: {str(e)}",
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
