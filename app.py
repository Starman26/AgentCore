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

# ================== LANGGRAPH & MEMORIA ==================

checkpointer = MemorySaver()
compiled_graph = graph.compile(checkpointer=checkpointer)

# ================== FASTAPI APP ==================

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
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    timezone: Optional[str] = "America/Monterrey"

    # 👇 NUEVOS CAMPOS PARA EL WIDGET/AVATAR
    avatar_id: Optional[str] = None          # "cat" | "robot" | "duck" | "lab" | "astro" | "cora"
    widget_mode: Optional[str] = None        # "default" | "custom"
    widget_personality: Optional[str] = None
    widget_notes: Optional[str] = None


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


# ================== ENDPOINTS BÁSICOS ==================


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


# ================== ENDPOINT SIMPLE /message ==================


@app.get("/message")
async def simple_message(
    mensaje: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,   # auth.user.id (UUID)
    user_email: Optional[str] = None,

    # 👇 overrides opcionales para probar rápido desde URL
    avatar_id: Optional[str] = None,
    widget_mode: Optional[str] = None,
    widget_personality: Optional[str] = None,
    widget_notes: Optional[str] = None,
):
    """
    Endpoint simple para enviar mensajes usando query params.

    - session_id: UUID de la sesión de chat (se usa como thread_id en LangGraph)
    - user_id: UUID de Supabase (auth.user.id)
    - user_email: solo para contexto / tools
    """
    try:
        timezone = "America/Monterrey"

        # 1) Resolver session_id (UUID para la sesión / thread_id)
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

        # 4) Config para el grafo (LangGraph usa thread_id para la memoria)
        config = {
            "configurable": {
                "thread_id": real_session_id,  # 👈 memoria por conversación
                "session_id": real_session_id,
            }
        }
        if valid_user_id:
            config["configurable"]["user_id"] = valid_user_id
        if user_email:
            config["configurable"]["user_email"] = user_email

        # 👇 pasar overrides de avatar al configurable
        if avatar_id:
            config["configurable"]["avatar_id"] = avatar_id
        if widget_mode:
            config["configurable"]["widget_mode"] = widget_mode
        if widget_personality:
            config["configurable"]["widget_personality"] = widget_personality
        if widget_notes:
            config["configurable"]["widget_notes"] = widget_notes

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

        # ⚠ Fallback para user_name para que los prompts no truene
        if "user_name" not in initial_state:
            initial_state["user_name"] = user_email or "Usuario"

        # 👇 también guardamos los overrides en el State
        if avatar_id:
            initial_state["widget_avatar_id"] = avatar_id
        if widget_mode:
            initial_state["widget_mode"] = widget_mode
        if widget_personality:
            initial_state["widget_personality"] = widget_personality
        if widget_notes:
            initial_state["widget_notes"] = widget_notes

        if trusted_user:
            initial_state["user_identified"] = True

        # Pasar también user_name al configurable (por si algún nodo lo usa desde ahí)
        config["configurable"]["user_name"] = initial_state["user_name"]

        # 6) Invocar grafo (async, con memoria)
        result: State = await compiled_graph.ainvoke(initial_state, config)

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

        # ===== TÍTULO DE SESIÓN, SI VIENE DEL GRAFO =====
        session_title = result.get("session_title")
        if isinstance(session_title, str):
            session_title = session_title.strip() or None
        else:
            session_title = None

        # ===== EXTRA: TOOL CALLS PARA DEBUG =====
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
            "session_title": session_title,
            "user_identified": trusted_user,
            "debug": debug_payload,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the message: {str(e)}",
        )


# ================== ENDPOINT COMPLETO /chat (POST) ==================


@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    """
    Endpoint principal que usa el modelo ChatRequest.
    Aquí se usa desde tu Chat.tsx (POST /chat).
    """
    try:
        timezone = payload.timezone or "America/Monterrey"

        # 1) Resolver session_id
        real_session_id = payload.session_id or str(uuid.uuid4())

        # 2) Validar user_id como en /message
        valid_user_id: Optional[str] = None
        if payload.user_id:
            try:
                _ = uuid.UUID(payload.user_id)
                valid_user_id = payload.user_id
            except ValueError:
                print(f"[chat_endpoint] WARNING: user_id inválido: {payload.user_id}")

        # usuario confiable si tenemos UUID válido
        trusted_user = valid_user_id is not None

        # 3) Config para el grafo
        config = {
            "configurable": {
                "thread_id": real_session_id,
                "session_id": real_session_id,
            }
        }
        if valid_user_id:
            config["configurable"]["user_id"] = valid_user_id
        if payload.user_email:
            config["configurable"]["user_email"] = payload.user_email

        # 👇 pasar configuración del widget/avatar
        if payload.avatar_id:
            config["configurable"]["avatar_id"] = payload.avatar_id
        if payload.widget_mode:
            config["configurable"]["widget_mode"] = payload.widget_mode
        if payload.widget_personality:
            config["configurable"]["widget_personality"] = payload.widget_personality
        if payload.widget_notes:
            config["configurable"]["widget_notes"] = payload.widget_notes

        # 4) Estado inicial
        initial_state: State = {
            "messages": [HumanMessage(content=payload.message)],
            "tz": timezone,
            "session_id": real_session_id,
        }

        if valid_user_id:
            initial_state["user_id"] = valid_user_id
        if payload.user_email:
            initial_state["user_email"] = payload.user_email

        # 👈 MUY IMPORTANTE: marcar que el usuario ya está identificado
        if trusted_user:
            initial_state["user_identified"] = True

        # también guardamos los widget_* en el State
        if payload.avatar_id:
            initial_state["widget_avatar_id"] = payload.avatar_id
        if payload.widget_mode:
            initial_state["widget_mode"] = payload.widget_mode
        if payload.widget_personality:
            initial_state["widget_personality"] = payload.widget_personality
        if payload.widget_notes:
            initial_state["widget_notes"] = payload.widget_notes

        # 5) Invocar grafo
        result: State = await compiled_graph.ainvoke(initial_state, config)

        messages = result.get("messages", [])
        agent_response = ""

        for msg in reversed(messages):
            if getattr(msg, "type", None) == "ai":
                agent_response = getattr(msg, "content", "")
                break

        if not agent_response:
            agent_response = (
                "Lo siento, no pude procesar tu mensaje. Inténtalo de nuevo."
            )

        session_title = result.get("session_title")
        if isinstance(session_title, str):
            session_title = session_title.strip() or None
        else:
            session_title = None

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
            "session_title": session_title,
            "user_identified": trusted_user or bool(payload.user_email),
            "timestamp": datetime.now().isoformat(),
            "debug": debug_payload,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the chat message: {str(e)}",
        )


# ================== ENDPOINT UPLOAD ==================


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


# ================== MAIN ==================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
