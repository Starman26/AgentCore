import os
from langchain_core.tools import tool
from supabase import create_client, Client
from rag.rag_logic import create_or_update_vectorstore
from typing import Literal



SB: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

def _fetch_student(name_or_email: str):
    q = name_or_email.strip()
    if "@" in q:
        res = SB.table("students").select("*").eq("email", q).limit(1).execute()
    else:
        res = SB.table("students").select("*").ilike("full_name", f"%{q}%").limit(1).execute()
    rows = res.data or []
    return rows[0] if rows else None

# -------- Tools de perfil de estudiante --------

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