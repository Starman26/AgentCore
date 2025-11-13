import os
from langchain_core.tools import tool
from supabase import create_client, Client
from rag.rag_logic import general_chat_db_use, general_student_db_use
from rag.db_access import _fetch_student

# -------- Supabase client (una sola instancia) --------
SB: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

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
def retrieve_context(name_or_email : str, chat_id : int, query: str) -> str:
    """Busca en la base vectorial de robots/incidentes y devuelve pasajes."""
    student_vectorstore = general_student_db_use(name_or_email)
    student_retriever = student_vectorstore.as_retriever(search_kwargs={"k": 2})
    student_docs = student_retriever.invoke(query)
    
    chat_vectorstore = general_chat_db_use(chat_id)
    chat_retriever = chat_vectorstore.as_retriever(search_kwargs={"k": 2})
    chat_docs = chat_retriever.invoke(query)

    out = []
    for d in student_docs:
        m = d.metadata
        out.append(
            f"[STUDENT] {m.get('full_name')} | {m.get('email')}\n{d.page_content}\n"
        )
    for d in chat_docs:
        m = d.metadata
        out.append(
            f"[CHAT] {m.get('session_id')} | {m.get('updated_at')}\n{d.page_content}\n"
        )
    return "\n".join(out)

LAB_TOOLS       = [retrieve_context]
GENERAL_TOOLS   = [get_student_profile, update_student_goals, update_learning_style]
EDU_TOOLS       = [get_student_profile, update_learning_style]

@tool
def route_to(target: str) -> str:
    """Pide traspaso interno entre agentes. Valores: EDUCATION|LAB|INDUSTRIAL|GENERAL."""
    target = target.upper()
    return f"ROUTE::{target}"