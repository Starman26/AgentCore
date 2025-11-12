# -*- coding: utf-8 -*-
# Tools unificados

import os
from typing import List, Optional, Literal, Union
from uuid import UUID
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field, conint
from supabase import create_client, Client
from tavily import TavilyClient

from rag.rag_logic import create_or_update_vectorstore
from Settings.state import State  # solo para tipado opcional

# Constants
now_utc = datetime.now(tz=timezone.utc)
timestamp_ms = int(now_utc.timestamp() * 1000)

# ------------------- CONFIGURACIÓN -------------------
load_dotenv()

SB: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

_TAVILY_KEY = os.getenv("TAVILY_API_KEY")
_tavily: Optional[TavilyClient] = TavilyClient(api_key=_TAVILY_KEY) if _TAVILY_KEY else None

# =================== HELPERS ===================
def _fetch_student(name_or_email: str):
    """Busca un estudiante por email o nombre parcial en Supabase. Regresa dict o None."""
    q = (name_or_email or "").strip()
    if not q:
        return None
    if "@" in q:
        res = SB.table("students").select("*").eq("email", q).limit(1).execute()
    else:
        res = SB.table("students").select("*").ilike("full_name", f"%{q}%").limit(1).execute()
    rows = res.data or []
    return rows[0] if rows else None

def _submit_chat_history(
    session_id: Union[int, str, UUID],
    role: Literal["student", "agent"],
    content: str,
    created_at: Optional[str] = None,
    user_id: Optional[str] = None,
):
    """Guarda un mensaje en Supabase (app_user, chat_session, chat_message)."""
    if isinstance(session_id, UUID):
        session_id = str(session_id)

    created_at = created_at or datetime.now(tz=timezone.utc).isoformat()
    user_id = user_id or "00000000-0000-0000-0000-000000000000"

    try:
        # upsert usuario y sesión
        try:
            SB.table("app_user").upsert({"id": user_id, "created_at": created_at}, on_conflict="id").execute()
        except Exception:
            pass

        SB.table("chat_session").upsert(
            {"id": session_id, "user_id": user_id, "started_at": created_at},
            on_conflict="id",
        ).execute()

        # insertar mensaje
        return SB.table("chat_message").insert(
            {"session_id": session_id, "role": role, "content": content, "created_at": created_at}
        ).execute()
    except Exception as e:
        print(f"Error saving chat: {e}")
        raise

def _submit_student(full_name: str, email: str, major: str, skills: List[str], goals: List[str], interests: str, last_seen: Union[int, str] = timestamp_ms, learning_style: dict = None):
    """Helper to insert student profile to Supabase if the agent doesnt know who the student is."""
    if learning_style is None:
        learning_style = {}
    try:
        SB.table("students").insert({
            "full_name": full_name,
            "email": email,
            "major": major,
            "skills": skills,
            "goals": goals,
            "interests": interests,
            "last_seen": last_seen,
            "learning_style": learning_style
        }, on_conflict="email").execute()
    except Exception as e:
        print(f"Error saving student profile: {e}")
        raise

# =================== TOOLS ===================

# ---- Tool: Investigación Web (Tavily) ----
class WebResearchInput(BaseModel):
    query: str = Field(..., description="Pregunta o tema a investigar")
    depth: Literal["basic", "advanced"] = Field("advanced", description="Profundidad de búsqueda")
    max_results: conint(ge=1, le=10) = 5
    time_filter: Optional[Literal["d", "w", "m", "y"]] = Field(
        None, description="Ventana temporal: d=día, w=semana, m=mes, y=año"
    )

def _summarize(snippets: List[dict], limit: int = 5) -> str:
    parts = []
    for i, s in enumerate(snippets[:limit], 1):
        title = (s.get("title") or s.get("url", ""))[:120]
        url = s.get("url", "")
        content = (s.get("content") or "").replace("\n", " ").strip()
        if len(content) > 400:
            content = content[:397] + "..."
        parts.append(f"[{i}] {title}\n{content}\nFuente: {url}")
    return "\n\n".join(parts)

@tool("web_research", args_schema=WebResearchInput)
def web_research(query: str, depth: str = "advanced", max_results: int = 5,
                 time_filter: Optional[str] = None) -> str:
    """
    Investiga en la web usando Tavily y devuelve un resumen con fuentes.
    Úsala cuando el agente no esté 100% seguro o requiera verificación.
    """
    if _tavily is None:
        return "ERROR_TAVILY::Falta TAVILY_API_KEY en el entorno."
    try:
        max_results = max(1, min(10, int(max_results)))
        kwargs = dict(query=query, search_depth=depth, max_results=max_results)
        if time_filter:
            kwargs["time_range"] = time_filter
        res = _tavily.search(**kwargs)
        results = res.get("results") or []
        answer = res.get("answer") or ""
        summary = _summarize(results, limit=max_results)
        head = f"Respuesta síntesis: {answer}\n\n" if answer else ""
        return head + (summary if summary else "SIN_RESULTADOS")
    except Exception as e:
        return f"ERROR_TAVILY::{type(e).__name__}::{e}"

# ---- Tools de perfil/chat ----
@tool
def get_student_profile(name_or_email: str) -> str:
    """Resumen: carrera, skills, metas, intereses y estilo de aprendizaje."""
    row = _fetch_student(name_or_email)
    if not row:
        return "PERFIL_NO_ENCONTRADO"
    skills = ", ".join(row.get("skills", []) or [])
    goals  = ", ".join(row.get("goals", []) or [])
    intr   = ", ".join(row.get("interests", []) or [])
    major  = row.get("major") or "N/D"
    full   = row.get("full_name") or name_or_email

    ls = row.get("learning_style") or {}
    prefs = []
    if ls.get("prefers_examples"):     prefs.append("con ejemplos")
    if ls.get("prefers_visual"):       prefs.append("de forma visual")
    if ls.get("prefers_step_by_step"): prefs.append("paso a paso")
    if ls.get("prefers_theory"):       prefs.append("con teoría")
    if ls.get("prefers_practice"):     prefs.append("con práctica")
    notes = ls.get("notes", "")
    learning_desc = (f"Prefiere aprender {' y '.join(prefs)}. {notes}"
                     if prefs else "No se ha definido su estilo de aprendizaje.")

    return (f"Perfil de {full} — Carrera: {major}. "
            f"Skills: {skills or 'N/D'}. Metas: {goals or 'N/D'}. "
            f"Intereses: {intr or 'N/D'}. {learning_desc}")

@tool
def submit_chat_history(session_id: int, role: Literal["student", "agent"], content: str,
                        created_at: str = date.today().isoformat()) -> str:
    """Guarda un mensaje en la base de datos."""
    _submit_chat_history(session_id, role, content, created_at)
    return "OK"

@tool
def submit_student_profile(full_name: str, email: str, major: str, skills: List[str], goals: List[str], interests: str, learning_style: dict = None) -> str:
    """Insert or update the student profile."""
    _submit_student(full_name, email, major, skills, goals, interests, learning_style=learning_style)
    return "OK"

@tool
def identify_user_from_message(message: str) -> str:
    """Attempts to identify the user by searching for an email or a name in the message.
    
    Returns a string with the format:
    - 'FOUND:email:name' if a user is found
    - 'NOT_FOUND' if no match is found
    """
    words = message.split()
    
    for word in words:
        if "@" in word:
            clean_email = word.strip(".,;:!?")
            row = _fetch_student(clean_email)
            if row:
                return f"FOUND:{row.get('email')}:{row.get('full_name')}"
    
    for i in range(len(words) - 1):
        if words[i] and words[i][0].isupper() and words[i+1] and words[i+1][0].isupper():
            potential_name = f"{words[i]} {words[i+1]}"
            row = _fetch_student(potential_name)
            if row:
                return f"FOUND:{row.get('email')}:{row.get('full_name')}"
    
    return "NOT_FOUND"

# ---- Tools de actualización de perfil ----
@tool
def update_student_goals(name_or_email: str, new_goal: str) -> str:
    """Agrega una meta al perfil (JSONB)."""
    row = _fetch_student(name_or_email)
    if not row:
        return "PERFIL_NO_ENCONTRADO"
    goals = row.get("goals") or []
    if new_goal and new_goal not in goals:
        goals.append(new_goal)
        SB.table("students").update({"goals": goals}).eq("id", row["id"]).execute()
    return f"OK: objetivos ahora = {goals}"

@tool
def update_learning_style(name_or_email: str, style: str) -> str:
    """Actualiza preferencias de aprendizaje a partir de texto libre."""
    row = _fetch_student(name_or_email)
    if not row:
        return "PERFIL_NO_ENCONTRADO"
    style_l = (style or "").lower()
    ls = row.get("learning_style") or {}
    if "ejemplo" in style_l: ls["prefers_examples"] = True
    if "visual"  in style_l: ls["prefers_visual"]   = True
    if "paso"    in style_l: ls["prefers_step_by_step"] = True
    if "teor"    in style_l: ls["prefers_theory"]   = True
    if "práct" in style_l or "practic" in style_l: ls["prefers_practice"] = True
    ls["notes"] = style
    SB.table("students").update({"learning_style": ls}).eq("id", row["id"]).execute()
    return f"Estilo actualizado para {row['full_name']}: {ls}"

# ---- Tool RAG (incidentes/robots) ----
@tool
def retrieve_context(query: str) -> str:
    """Busca en la base vectorial de robots/incidentes y devuelve pasajes."""
    vectorstore = create_or_update_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    out = []
    for d in docs:
        m = d.metadata
        out.append(
            f"{m.get('created_at')} | {m.get('robot_type')} | {m.get('problem_title')} | {m.get('author')}\n"
            f"{d.page_content}\n"
        )
    return "\n".join(out) if out else "RAG_EMPTY"

# ---- Tool de ruteo interno entre agentes ----
@tool
def route_to(target: str) -> str:
    """Traspaso interno entre agentes. Valores: EDUCATION|LAB|INDUSTRIAL|GENERAL."""
    return f"ROUTE::{(target or '').upper()}"

# ---- Tool de fecha/hora actual (con state opcional) ----
@tool
def current_datetime(state: Optional[State] = None, tz: Optional[str] = None) -> str:
    """
    Devuelve la fecha/hora actual en la zona 'tz' (o state.tz o America/Monterrey)
    en tres formatos: ISO local, ISO UTC y humano en español.
    """
    tz_name = tz or (state.get("tz") if isinstance(state, dict) else None) or "America/Monterrey"
    now_loc = datetime.now(ZoneInfo(tz_name))
    out = {
        "tz": tz_name,
        "now_local": now_loc.isoformat(),
        "now_utc": datetime.now(tz=timezone.utc).isoformat(),
        "now_human": now_loc.strftime("%A, %d %b %Y, %H:%M"),
    }
    return str(out)

# =================== TOOL SETS ===================
LAB_TOOLS     = [retrieve_context, web_research, route_to]
GENERAL_TOOLS = [get_student_profile, update_student_goals, update_learning_style, web_research, route_to]
EDU_TOOLS     = [get_student_profile, update_learning_style, web_research, route_to]
