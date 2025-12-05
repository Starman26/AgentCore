from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
import uvicorn
from pathlib import Path
from Settings.tools import SB
from agent.graph import graph, State

# ==========================================================
# LANGGRAPH & MEMORIA
# ==========================================================

checkpointer = MemorySaver()
compiled_graph = graph.compile(checkpointer=checkpointer)

# ==========================================================
# FASTAPI APP
# ==========================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ==========================================================
# MODELOS
# ==========================================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_email: Optional[str] = None
    timezone: Optional[str] = "America/Monterrey"

    avatar_id: Optional[str] = None
    widget_mode: Optional[str] = None
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


# ==========================================================
# UTILIDADES
# ==========================================================

def _load_session_metadata(session_id: str) -> Dict[str, Any]:
    try:
        res = (
            SB.table("chat_session")
            .select("metadata")
            .eq("id", session_id)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return {}
        meta = rows[0].get("metadata") or {}
        if not isinstance(meta, dict):
            return {}
        return meta
    except Exception as e:
        print(f"[app._load_session_metadata] Error leyendo metadata para {session_id}: {e}")
        return {}


# ==========================================================
# ENDPOINTS BÁSICOS
# ==========================================================

@app.get("/")
async def root():
    return {
        "message": "AgentCore API is running",
        "version": "1.0.0",
        "endpoints": {
            "message": "/message?mensaje=...",
            "chat": "/chat (POST)",
            "upload": "/upload",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


# ==========================================================
# GET /message  (Usado por Chat.tsx)
# ==========================================================

@app.get("/message")
async def simple_message(
    mensaje: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,

    avatar_id: Optional[str] = None,
    widget_mode: Optional[str] = None,
    widget_personality: Optional[str] = None,
    widget_notes: Optional[str] = None,
):
    try:
        timezone = "America/Monterrey"
        real_session_id = session_id or str(uuid.uuid4())

        # Cargar metadata persistida
        meta = _load_session_metadata(real_session_id)
        chat_type = (meta.get("chat_type") or "default").lower()
        project_id = meta.get("project_id")

        # Validación de UUID del usuario
        valid_user_id = None
        if user_id:
            try:
                _ = uuid.UUID(user_id)
                valid_user_id = user_id
            except ValueError:
                print(f"[simple_message] WARNING: user_id inválido: {user_id}")

        trusted_user = valid_user_id is not None

        # ---------- CONFIGURABLE PARA LANGGRAPH ----------
        config: Dict[str, Any] = {
            "configurable": {
                "thread_id": real_session_id,
                "session_id": real_session_id,
                "chat_type": chat_type,
            }
        }

        if project_id:
            config["configurable"]["project_id"] = project_id
        if valid_user_id:
            config["configurable"]["user_id"] = valid_user_id
        if user_email:
            config["configurable"]["user_email"] = user_email

        # Avatar y configuración del widget SOLO en config
        if avatar_id:
            config["configurable"]["avatar_id"] = avatar_id
        if widget_mode:
            config["configurable"]["widget_mode"] = widget_mode
        if widget_personality:
            config["configurable"]["widget_personality"] = widget_personality
        if widget_notes:
            config["configurable"]["widget_notes"] = widget_notes

        # ---------- ESTADO INICIAL (DEBE RESPETAR State) ----------
        initial_state: State = {
            "messages": [HumanMessage(content=mensaje)],
            "tz": timezone,
            "session_id": real_session_id,
            "chat_type": chat_type,
        }

        if project_id:
            initial_state["project_id"] = project_id
        if valid_user_id:
            initial_state["user_id"] = valid_user_id
        if user_email:
            initial_state["user_email"] = user_email
        if trusted_user:
            initial_state["user_identified"] = True

        # ⚠️ IMPORTANTE: NO meter widget_avatar_id ni widget_* en initial_state

        # Ejecutar grafo
        result: State = await compiled_graph.ainvoke(initial_state, config)

        # Obtener última respuesta del agente
        messages = result.get("messages", [])
        agent_response = ""
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "ai":
                agent_response = getattr(msg, "content", "")
                break

        if not agent_response:
            agent_response = "No pude procesar tu mensaje."

        # Debug tools
        tool_events = []
        for msg in messages:
            tc = getattr(msg, "tool_calls", None)
            if tc:
                tool_events.append({"from": getattr(msg, "type", None), "tool_calls": tc})

        return {
            "response": agent_response,
            "session_id": real_session_id,
            "user_identified": trusted_user,
            "debug": {"tool_events": tool_events},
        }

    except Exception as e:
        # opcional: print stacktrace para depurar
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


# ==========================================================
# POST /chat
# ==========================================================

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    try:
        timezone = payload.timezone or "America/Monterrey"
        real_session_id = payload.session_id or str(uuid.uuid4())

        meta = _load_session_metadata(real_session_id)
        chat_type = (meta.get("chat_type") or "default").lower()
        project_id = meta.get("project_id")

        # ---------- CONFIG ----------
        config: Dict[str, Any] = {
            "configurable": {
                "thread_id": real_session_id,
                "session_id": real_session_id,
                "chat_type": chat_type,
            }
        }

        if project_id:
            config["configurable"]["project_id"] = project_id
        if payload.user_email:
            config["configurable"]["user_email"] = payload.user_email

        if payload.avatar_id:
            config["configurable"]["avatar_id"] = payload.avatar_id
        if payload.widget_mode:
            config["configurable"]["widget_mode"] = payload.widget_mode
        if payload.widget_personality:
            config["configurable"]["widget_personality"] = payload.widget_personality
        if payload.widget_notes:
            config["configurable"]["widget_notes"] = payload.widget_notes

        # ---------- ESTADO INICIAL ----------
        initial_state: State = {
            "messages": [HumanMessage(content=payload.message)],
            "tz": timezone,
            "session_id": real_session_id,
            "chat_type": chat_type,
        }

        if project_id:
            initial_state["project_id"] = project_id
        if payload.user_email:
            initial_state["user_email"] = payload.user_email

        # De nuevo: nada de widget_avatar_id ni widget_* aquí

        result: State = await compiled_graph.ainvoke(initial_state, config)

        messages = result.get("messages", [])
        agent_response = ""
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "ai":
                agent_response = getattr(msg, "content", "")
                break

        if not agent_response:
            agent_response = "No pude procesar tu mensaje."

        tool_events = []
        for msg in messages:
            tc = getattr(msg, "tool_calls", None)
            if tc:
                tool_events.append({"from": getattr(msg, "type", None), "tool_calls": tc})

        return {
            "response": agent_response,
            "session_id": real_session_id,
            "user_identified": bool(payload.user_email),
            "timestamp": datetime.now().isoformat(),
            "debug": {"tool_events": tool_events},
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing the chat message: {str(e)}")


# ==========================================================
# ENDPOINT UPLOAD
# ==========================================================

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
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
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error uploading the file: {str(e)}")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
