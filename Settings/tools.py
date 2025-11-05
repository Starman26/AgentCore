import os
from langchain_core.tools import tool
from supabase import create_client, Client
from rag.rag_logic import create_or_update_vectorstore
from typing import Literal
from Settings.state import State
# -------- Supabase client (una sola instancia) --------
SB: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


@tool
def update_student_goals(state: State, new_goal: str) -> str:
    """Agrega una meta al perfil (JSONB)."""
    row = ""
    if not row:
        return "PERFIL_NO_ENCONTRADO"
    goals = row.get("goals") or []
    if new_goal and new_goal not in goals:
        goals.append(new_goal)
        SB.table("students").update({"goals": goals}).eq("id", row["id"]).execute()
    return f"OK: objetivos ahora = {goals}"

@tool
def update_learning_style(state: State, style: str) -> str:
    """Actualiza preferencias de aprendizaje a partir de texto libre."""
    row = ""
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


LAB_TOOLS       = [retrieve_context]
GENERAL_TOOLS   = [update_student_goals, update_learning_style]
EDU_TOOLS       = [update_learning_style]
@tool
def route_to(target: str) -> str:
    """Pide traspaso interno entre agentes. Valores: EDUCATION|LAB|INDUSTRIAL|GENERAL."""
    target = target.upper()
    return f"ROUTE::{target}"