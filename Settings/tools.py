from datetime import date, datetime, timezone
import os
from uuid import UUID
from langchain_core.tools import tool
from supabase import create_client, Client
from rag.rag_logic import create_or_update_vectorstore
from typing import List, Literal, Union

# Constants
now_utc = datetime.now(tz=timezone.utc)
timestamp_ms = int(now_utc.timestamp() * 1000)

# -------- Supabase client (una sola instancia) --------
SB: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

def _fetch_student(name_or_email: str):
    """Helper: busca un estudiante por email o nombre parcial en Supabase.

    Returns the first matching row dict or None.
    """
    q = name_or_email.strip()
    if "@" in q:
        res = SB.table("students").select("*").eq("email", q).limit(1).execute()
    else:
        res = SB.table("students").select("*").ilike("full_name", f"%{q}%").limit(1).execute()
    rows = res.data or []
    return rows[0] if rows else None

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

def _submit_chat_history(session_id: Union[int, str, UUID], role: Literal["student", "agent"], content: str, created_at: str = None, user_id: str = None):
    """Helper to save chat message to Supabase. Accepts int, string, or UUID for session_id."""
    if isinstance(session_id, UUID):
        session_id = str(session_id)
    
    if created_at is None:
        created_at = datetime.now(tz=timezone.utc).isoformat()
    
    if user_id is None:
        user_id = "00000000-0000-0000-0000-000000000000"  
    try:
        try:
            SB.table("app_user").upsert({
                "id": user_id,
                "created_at": created_at
            }, on_conflict="id").execute()
        except Exception:
            pass

        SB.table("chat_session").upsert({
            "id": session_id,
            "user_id": user_id,
            "started_at": created_at 
        }, on_conflict="id").execute()
        
        return SB.table("chat_message").insert({
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": created_at
        }).execute()
    except Exception as e:
        print(f"Error saving chat: {e}")
        raise

# -------- Tools de perfil de estudiante --------
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
    if ls.get("prefers_examples"):    prefs.append("con ejemplos")
    if ls.get("prefers_visual"):      prefs.append("de forma visual")
    if ls.get("prefers_step_by_step"):prefs.append("paso a paso")
    if ls.get("prefers_theory"):      prefs.append("con teoría")
    if ls.get("prefers_practice"):    prefs.append("con práctica")
    notes = ls.get("notes", "")
    learning_desc = (f"Prefiere aprender {' y '.join(prefs)}. {notes}"
                     if prefs else "No se ha definido su estilo de aprendizaje.")

    return (f"Perfil de {full} — Carrera: {major}. "
            f"Skills: {skills or 'N/D'}. Metas: {goals or 'N/D'}. "
            f"Intereses: {intr or 'N/D'}. {learning_desc}")

@tool
def submit_chat_history(session_id: int, role: Literal["student", "agent"], content: str, created_at: str = date.today().isoformat()) -> str:
    """Save a message to the database."""
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
    style_l = style.lower()
    ls = row.get("learning_style") or {}
    if "ejemplo" in style_l: ls["prefers_examples"] = True
    if "visual"  in style_l: ls["prefers_visual"]   = True
    if "paso"    in style_l: ls["prefers_step_by_step"] = True
    if "teor"    in style_l: ls["prefers_theory"]   = True
    if "práct" in style_l or "practic" in style_l: ls["prefers_practice"] = True
    ls["notes"] = style
    SB.table("students").update({"learning_style": ls}).eq("id", row["id"]).execute()
    return f"Estilo actualizado para {row['full_name']}: {ls}"

# -------- Tool RAG (incidentes/robots) --------
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
            f" {m.get('created_at')} |  {m.get('robot_type')} |  {m.get('problem_title')} |  {m.get('author')}\n"
            f"{d.page_content}\n"
        )
    return "\n".join(out)

@tool
def route_to(target: str) -> str:
    """Pide traspaso interno entre agentes. Valores: EDUCATION|LAB|INDUSTRIAL|GENERAL."""
    target = target.upper()
    return f"ROUTE::{target}"

LAB_TOOLS       = [retrieve_context, route_to]
GENERAL_TOOLS   = [get_student_profile, update_student_goals, update_learning_style, route_to, submit_student_profile, identify_user_from_message]
EDU_TOOLS       = [get_student_profile, update_learning_style, route_to]