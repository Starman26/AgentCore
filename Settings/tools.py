# -*- coding: utf-8 -*-
# Tools unificados

import os
from typing import Dict,List, Optional, Literal, Union
from uuid import UUID, uuid4, uuid5, NAMESPACE_DNS, NAMESPACE_URL
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
import json

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.documents import Document
from pydantic.v1 import BaseModel, Field, conint
from supabase import create_client, Client
from tavily import TavilyClient
from langchain_openai import ChatOpenAI

from rag.rag_logic import (
    create_or_update_vectorstore,
    general_chat_db_use,
    general_student_db_use,
)
from Settings.state import State  # solo para tipado opcional


# ====================================================
# Constantes
# ====================================================
now_mty = datetime.now(ZoneInfo("America/Monterrey"))
timestamp_iso = now_mty.isoformat()

# ------------------- CONFIGURACIÓN -------------------
load_dotenv()

SB: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

_TAVILY_KEY = os.getenv("TAVILY_API_KEY")
_tavily: Optional[TavilyClient] = TavilyClient(api_key=_TAVILY_KEY) if _TAVILY_KEY else None
_QT_LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

class WebResearchInput(BaseModel):
    query: str = Field(..., description="Pregunta o tema a investigar")
    depth: Literal["basic", "advanced"] = Field(
        "advanced", description="Profundidad de búsqueda"
    )
    max_results: conint(ge=1, le=10) = 5
    time_filter: Optional[Literal["d", "w", "m", "y"]] = Field(
        None, description="Ventana temporal: d=día, w=semana, m=mes, y=año"
    )


# ====================================================
# HELPERS
# ====================================================
def _fetch_student(name_or_email: str):
    """Busca un estudiante por email o nombre parcial en Supabase. Regresa dict o None."""
    q = name_or_email.strip()
    if "@" in q:
        res = (
            SB.table("students")
            .select("*")
            .eq("email", q)
            .limit(1)
            .execute()
        )
    else:
        res = (
            SB.table("students")
            .select("*")
            .ilike("full_name", f"%{q}%")
            .limit(1)
            .execute()
        )
    rows = res.data or []
    return rows[0] if rows else None

def _transform_query_for_rag(raw_query: str, student_row: Optional[Dict] = None) -> str:
    """
    Reescribe la consulta para RAG usando info del estudiante.
    Si algo falla, regresa la consulta original.
    """
    try:
        perfil = ""
        if student_row:
            skills = ", ".join(student_row.get("skills") or [])
            goals = ", ".join(student_row.get("goals") or [])
            interests = ", ".join(student_row.get("interests") or [])
            carrera = student_row.get("career") or ""
            perfil = (
                f"Carrera: {carrera}\n"
                f"Skills: {skills}\n"
                f"Metas: {goals}\n"
                f"Intereses: {interests}\n"
            )

        prompt = f"""
Eres un asistente que reescribe consultas para un sistema RAG.
Usa el perfil para hacer la pregunta más específica y técnica,
pero SIN cambiar la intención.

Perfil del estudiante:
{perfil}

Consulta original:
\"\"\"{raw_query}\"\"\"


Devuelve UNA sola consulta mejorada en una línea, sin explicaciones extra.
"""
        resp = _QT_LLM.invoke(prompt)
        cleaned = (resp.content or "").strip()
        return cleaned or raw_query
    except Exception as e:
        print("[_transform_query_for_rag] error:", e)
        return raw_query


def _semantic_search(vs, query: str, k: int = 3):
    """
    Búsqueda semántica explícita con MMR (diversidad).
    Internamente Chroma embebe el query y compara contra el índice.
    """
    retriever = vs.as_retriever(
        search_type="mmr",           # en lugar de similarity simple
        search_kwargs={
            "k": k,                  # docs finales
            "fetch_k": max(8, 2 * k) # docs candidatos
        },
    )
    return retriever.invoke(query)

MAX_CHARS = 900 
def _clip(text: str, max_chars: int = MAX_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."
 
def _normalize_session_id(session_id: Union[int, str, UUID]) -> str:
    """
    Normaliza cualquier session_id recibido a un UUID string válido.

    - Si ya es UUID válido, lo devuelve tal cual.
    - Si viene vacío, genera uno nuevo (uuid4).
    - Si es una cadena arbitraria (p.ej. "session-123"), genera un UUID
      determinístico basado en esa cadena (uuid5).
    """
    if isinstance(session_id, UUID):
        return str(session_id)

    if session_id is None or str(session_id).strip() == "":
        return str(uuid4())

    s = str(session_id)
    try:
        UUID(s)
        return s  # ya era un UUID válido
    except ValueError:
        # Lo convertimos a un UUID determinístico basado en la cadena
        return str(uuid5(NAMESPACE_URL, s))


def _submit_chat_history(
    session_id: Union[int, str, UUID],
    role: Literal["student", "agent"],
    content: str,
    created_at: Optional[str] = None,
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
):
    """
    Guarda un mensaje de chat asociado a una sesión.
    Relaciona la sesión con el usuario principalmente por email.
    - NO obliga a que user_id sea UUID válido.
    - NO toca app_user ni la tabla users de Supabase.
    """

    # Normalizar session_id a string simple
    if isinstance(session_id, UUID):
        session_id = str(session_id)

    created_at = created_at or datetime.now(tz=timezone.utc).isoformat()

    try:
        # --- 1) Upsert de la sesión ---
        session_payload = {
            "id": session_id,
            "started_at": created_at,
        }

        # Si tenemos email, lo usamos como identificador de usuario
        if user_email and "@" in user_email:
            session_payload["user_email"] = user_email.strip()

        # Si en un futuro quieres volver a usar user_id (UUID), aquí se puede setear
        if user_id:
            session_payload["user_id"] = user_id

        try:
            SB.table("chat_session").upsert(
                session_payload,
                on_conflict="id",
            ).execute()
        except Exception as e:
            print("[_submit_chat_history] ERROR upsert chat_session:", e)

        # --- 2) Insertar el mensaje ---
        try:
            return (
                SB.table("chat_message")
                .insert(
                    {
                        "session_id": session_id,
                        "role": role,
                        "content": content,
                        "created_at": created_at,
                    }
                )
                .execute()
            )
        except Exception as e:
            print("[_submit_chat_history] ERROR insert chat_message:", e)

    except Exception as e:
        print("Error saving chat (fatal):", e)
        return None

def _submit_student(
    full_name: str,
    email: str,
    career: str,
    semester: int,
    skills: List[str],
    goals: List[str],
    interests: Union[str, List[str]],
    last_seen: str = None,
    learning_style: dict = None,
):
    """Inserta/actualiza perfil de estudiante en Supabase."""
    if learning_style is None:
        learning_style = {}

    if isinstance(interests, str):
        interests = [interests] if interests else []

    if last_seen is None:
        last_seen = datetime.now(ZoneInfo("America/Monterrey")).isoformat()

    try:
        SB.table("students").upsert(
            {
                "full_name": full_name,
                "email": email,
                "career": career,
                "semester": semester,
                "skills": skills,
                "goals": goals,
                "interests": interests,
                "last_seen": last_seen,
                "learning_style": learning_style,
            },
            on_conflict="email",
        ).execute()
    except Exception as e:
        print("Error saving student profile:", e)
        raise


def _summarize_all_chats() -> dict:
    """
    Proceso batch: resume todas las sesiones y guarda en chat_summary.
    """
    stats = {
        "total_sessions": 0,
        "successful": 0,
        "failed": 0,
        "session_ids": [],
    }

    try:
        response = (
            SB.table("chat_message")
            .select("*")
            .order("session_id")
            .order("created_at")
            .execute()
        )

        all_messages = response.data or []
        if not all_messages:
            print("No messages to process")
            return stats

        sessions_messages = {}
        for msg in all_messages:
            session_id = msg.get("session_id")
            if session_id:
                sessions_messages.setdefault(session_id, []).append(msg)

        stats["total_sessions"] = len(sessions_messages)
        print(f"Processing {stats['total_sessions']} sessions...")

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        for session_id, messages in sessions_messages.items():
            try:
                conversation_text = []
                for msg in messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    conversation_text.append(f"{role.upper()}: {content}")

                full_conversation = "\n\n".join(conversation_text)

                summary_prompt = f"""Genera un resumen conciso de la siguiente conversación entre un estudiante y un agente educativo.
El resumen debe capturar:
- Los temas principales discutidos
- Las preguntas clave del estudiante
- Las soluciones o respuestas proporcionadas
- Cualquier acción o tarea acordada

Conversación:
{full_conversation}

Resumen en JSON:"""

                summary_response = llm.invoke(summary_prompt)
                summary_json = summary_response.content

                SB.table("chat_summary").upsert(
                    {
                        "session_id": session_id,
                        "summary_json": summary_json,
                        "updated_at": now_mty,
                    },
                    on_conflict="session_id",
                ).execute()

                stats["successful"] += 1
                stats["session_ids"].append(session_id)
                print(f"Summary created for session {session_id}")

            except Exception as e:
                stats["failed"] += 1
                print(f"Error summarizing session {session_id}:", e)

        print(
            f"Completed: {stats['successful']}/{stats['total_sessions']}."
        )

        if stats["successful"] > 0:
            try:
                print(
                    f"Deleting messages for {stats['successful']} sessions..."
                )
                for session_id in stats["session_ids"]:
                    (
                        SB.table("chat_message")
                        .delete()
                        .eq("session_id", session_id)
                        .execute()
                    )
                    print(f"Deleted messages for session {session_id}")
                print("All messages deleted for summarized sessions.")
            except Exception as e:
                print("Error deleting messages:", e)

        return stats

    except Exception as e:
        print("Fatal error in _summarize_all_chats:", e)
        return stats


def _summarize(snippets: List[dict], limit: int = 5) -> str:
    """Compacta resultados de Tavily para contexto."""
    parts = []
    for i, s in enumerate(snippets[:limit], 1):
        title = (s.get("title") or s.get("url", ""))[:120]
        url = s.get("url", "")
        content = (s.get("content") or "").replace("\n", " ").strip()
        if len(content) > 400:
            content = content[:397] + "..."
        parts.append(f"[{i}] {title}\n{content}\nFuente: {url}")
    return "\n\n".join(parts)


def _build_robot_support_docs() -> List[Document]:
    res = (
        SB.table("RoboSupportDB")
        .select(
            "id, created_at, category, robot_type, problem_title, "
            "problem_description, solution_steps, author"
        )
        .execute()
    )
    rows = res.data or []
    docs: List[Document] = []

    for r in rows:
        case_id = r.get("id")
        robot = r.get("robot_type") or r.get("category") or "robot"
        title = r.get("problem_title") or "Problema sin título"
        desc = r.get("problem_description") or "Sin descripción detallada."
        steps = r.get("solution_steps") or "Sin pasos registrados."
        author = r.get("author") or "otro integrante del laboratorio"

        content = (
            f"[ROBOT_SUPPORT_CASE]\n"
            f"ID_CASO: {case_id}\n"
            f"ROBOT: {robot}\n"
            f"TÍTULO: {title}\n"
            f"DESCRIPCIÓN_DEL_PROBLEMA: {desc}\n"
            f"PASOS_DE_SOLUCIÓN_OFICIALES: {steps}\n"
            f"AUTOR: {author}\n"
        )

        metadata = {
            "case_id": case_id,
            "robot_type": robot,
            "problem_title": title,
            "author": author,
        }
        docs.append(Document(page_content=content, metadata=metadata))

    return docs

def _get_agent_tables(
    project_id: Optional[str] = None,
    team_id: Optional[str] = None,
    table_names: Optional[List[str]] = None,
) -> List[dict]:
    """
    Lee la tabla agent_tables y devuelve las tablas a las que el agente
    puede acceder.

    🔹 Regla:
    - Si hay team_id -> se filtra por team_id (clave principal).
    - Si NO hay team_id pero sí project_id -> se filtra por project_id.
    - Si no se encontró nada en el primer intento, hace un fallback con el otro.
    """
    try:
        base = SB.table("agent_tables").select(
            "id, project_id, team_id, table_name, display_name, description, created_at"
        )

        # ---- 1er intento: priorizar team_id ----
        q = base
        if team_id:
            q = q.eq("team_id", team_id)
        elif project_id:
            q = q.eq("project_id", project_id)

        if table_names:
            q = q.in_("table_name", table_names)

        res = q.order("created_at", desc=False).execute()
        data = res.data or []

        # ---- Fallback: si no encontró nada y tengo el otro ID ----
        if not data and project_id and not team_id:
            # Reintenta usando sólo project_id
            q2 = base.eq("project_id", project_id)
            if table_names:
                q2 = q2.in_("table_name", table_names)
            res2 = q2.order("created_at", desc=False).execute()
            data = res2.data or []

        elif not data and team_id and project_id:
            # Si primero filtró por team_id y falló, prueba con project_id
            q2 = base.eq("project_id", project_id)
            if table_names:
                q2 = q2.in_("table_name", table_names)
            res2 = q2.order("created_at", desc=False).execute()
            data = res2.data or []

        return data

    except Exception as e:
        print("[_get_agent_tables] error:", e)
        return []


@tool
def gather_rag_context(
    query: str,
    user_email: Optional[str] = None,
    project_id: Optional[str] = None,
    team_id: Optional[str] = None,
    robot_type: Optional[str] = None,
    image_limit: int = 5,
    web_depth: str = "advanced",
) -> str:
    """
    Orquesta el RAG general del sistema.

    Flujo SIEMPRE:
    1) Buscar en BD (RAG sobre agent_tables) → DB_CONTEXT::
    2) Si hay algo en BD, buscar imágenes relacionadas → IMAGE_CONTEXT::
    3) En todos los casos, complementar con web_research → WEB_CONTEXT::

    El agente debe usar primero DB_CONTEXT, luego IMAGE_CONTEXT como apoyo
    visual y WEB_CONTEXT solo como complemento / actualización.
    """

    # -------- 1) RAG en base de datos (tablas registradas en agent_tables) --------
    db_context = _search_in_db_impl(
        project_id=project_id,
        team_id=team_id,
        query=query,
        max_tables=3,
        k_per_table=4,
        only_tables=None,
    )

    has_db_results = not str(db_context).startswith("DB_CONTEXT::RAG_EMPTY")

    # -------- 2) Imágenes relacionadas (solo si hubo algo en BD) --------
    image_context = "IMAGE_CONTEXT::RAG_EMPTY"
    if has_db_results:
        try:
            imgs = search_manual_images.invoke(
                {
                    "query": query,
                    "project_id": project_id,
                    "robot_type": robot_type,
                    "limit": image_limit,
                }
            )
        except Exception as e:
            print("[gather_rag_context] error search_manual_images:", e)
            imgs = []

        if imgs:
            parts = []
            for im in imgs:
                url = im.get("image_url") or im.get("storage_path") or ""
                title = im.get("title") or ""
                desc = im.get("description") or ""
                tags = im.get("tags") or ""
                parts.append(
                    f"- TITLE={title} | DESC={desc} | TAGS={tags} | URL={url}"
                )
            image_context = "IMAGE_CONTEXT::\n" + "\n".join(parts)

    # -------- 3) Web search SIEMPRE (aunque BD esté vacía) --------
    try:
        web_ctx = web_research.invoke(
            {
                "query": query,
                "depth": web_depth,
                "max_results": 5,
                "time_filter": None,
            }
        )
        web_context = str(web_ctx)  # ya trae "WEB_CONTEXT::..."
    except Exception as e:
        print("[gather_rag_context] error web_research:", e)
        web_context = f"WEB_CONTEXT::ERROR::{type(e).__name__}::{e}"

    # -------- 4) Empaquetar todo --------
    return (
        f"{db_context}\n\n"
        f"{image_context}\n\n"
        f"{web_context}"
    )

@tool
def list_agent_tables(
    project_id: Optional[str] = None,
    team_id: Optional[str] = None,
) -> List[dict]:
    """
    Devuelve las tablas registradas en agent_tables a las que el agente
    tiene acceso para un proyecto/equipo dado.

    Úsalo cuando quieras saber qué tablas puedes consultar antes
    de llamar a search_in_db.
    """
    tables = _get_agent_tables(project_id=project_id, team_id=team_id)
    out: List[dict] = []

    for t in tables:
        out.append(
            {
                "table_name": t.get("table_name"),
                "display_name": t.get("display_name"),
                "description": t.get("description"),
                "project_id": t.get("project_id"),
                "team_id": t.get("team_id"),
            }
        )

    return out
@tool
def describe_agent_table(
    table_name: str,
    sample_rows: int = 3,
) -> dict:
    """
    Devuelve un pequeño resumen del esquema de una tabla real
    (no agent_tables), con nombres de columnas y algunas filas de ejemplo.

    Úsalo DESPUÉS de list_agent_tables para entender qué campos hay
    en cada tabla antes de construir la query de search_in_db.
    """
    try:
        sample_rows = max(1, min(10, int(sample_rows)))
        res = SB.table(table_name).select("*").limit(sample_rows).execute()
        rows = res.data or []

        if not rows:
            return {
                "table_name": table_name,
                "columns": [],
                "sample_rows": [],
                "note": "La tabla existe pero no tiene filas (o no se pudieron leer).",
            }

        # Tomamos las columnas de la primera fila
        columns = list(rows[0].keys())

        return {
            "table_name": table_name,
            "columns": columns,
            "sample_rows": rows,
        }

    except Exception as e:
        return {
            "table_name": table_name,
            "error": str(e),
        }

def _select_relevant_agent_tables(
    tables: List[dict],
    query: str,
    max_tables: int = 3,
) -> List[dict]:
    """
    Usa el LLM para elegir hasta max_tables tablas relevantes
    en función de la descripción de agent_tables y la pregunta.
    """
    if not tables:
        return []

    if len(tables) <= max_tables:
        return tables

    listado = []
    for t in tables:
        listado.append(
            f"- table_name: {t.get('table_name')} | display_name: {t.get('display_name')} | "
            f"descripcion: {t.get('description')}"
        )
    listado_txt = "\n".join(listado)

    prompt = f"""
Eres un asistente que selecciona tablas de base de datos para un sistema RAG.

Pregunta del usuario:
\"\"\"{query}\"\"\"

Tablas disponibles (table_name, display_name, descripcion):
{listado_txt}

Elige HASTA {max_tables} tablas que realmente aporten contexto para responder la pregunta.
Devuelve SOLO un JSON con este formato exacto:

{{"tables": ["table_name_1", "table_name_2"]}}
"""

    try:
        resp = _QT_LLM.invoke(prompt)
        text = (resp.content or "").strip()
        data = json.loads(text)
        names = set(data.get("tables", []))
    except Exception as e:
        print("[_select_relevant_agent_tables] error parse JSON:", e)
        text = (resp.content or "") if "resp" in locals() else ""
        names = {t["table_name"] for t in tables if t["table_name"] in text}

    selected = [t for t in tables if t.get("table_name") in names]
    if not selected:
        selected = tables[:max_tables]

    return selected


def _build_generic_table_docs(agent_table_row: dict) -> List[Document]:
    """
    Construye Documents genéricos a partir de una tabla arbitraria registrada en agent_tables.
    No asume un schema fijo: concatena pares campo:valor en texto legible.
    """
    table_name = agent_table_row.get("table_name")
    display_name = agent_table_row.get("display_name") or table_name
    description = agent_table_row.get("description") or "Sin descripción"

    if not table_name:
        return []

    try:
        # Ajusta el límite si alguna tabla puede ser muy grande
        res = SB.table(table_name).select("*").limit(1000).execute()
    except Exception as e:
        print(f"[_build_generic_table_docs] error leyendo {table_name}:", e)
        return []

    rows = res.data or []
    docs: List[Document] = []

    for r in rows:
        # Construir texto de campos clave:valor
        fields = []
        for k, v in r.items():
            if v is None or v == "":
                continue
            text_val = str(v)
            if len(text_val) > 200:
                text_val = text_val[:197] + "..."
            fields.append(f"{k}: {text_val}")

        if not fields:
            continue

        row_text = "; ".join(fields)
        content = (
            f"Tabla: {display_name} ({table_name}).\n"
            f"Descripción de la tabla: {description}.\n"
            f"Registro:\n{row_text}"
        )
        metadata = {
            "table_name": table_name,
            "display_name": display_name,
            "agent_table_id": agent_table_row.get("id"),
            "row_id": r.get("id"),
        }
        docs.append(Document(page_content=content, metadata=metadata))

    return docs

# ====================================================
# TOOLS
# ====================================================

# ---- Tool: Investigación Web (Tavily) como RAG web ----
@tool("web_research", args_schema=WebResearchInput)
def web_research(
    query: str,
    depth: str = "advanced",
    max_results: int = 5,
    time_filter: Optional[str] = None,
) -> str:
    """
    Consulta la web usando Tavily y devuelve CONTEXTO para el agente, no una respuesta directa.
    El agente debe usar este contexto para responder de forma humana.
    """
    if _tavily is None:
        return "WEB_CONTEXT::ERROR::Falta TAVILY_API_KEY en el entorno."

    try:
        max_results = max(1, min(10, int(max_results)))
        kwargs = dict(
            query=query,
            search_depth=depth,
            max_results=max_results,
        )
        if time_filter:
            kwargs["time_range"] = time_filter

        res = _tavily.search(**kwargs)
        results = res.get("results") or []
        answer = (res.get("answer") or "").replace("\n", " ").strip()

        bullets = []
        for r in results[:max_results]:
            title = (r.get("title") or "")[:120]
            url = r.get("url") or ""
            snippet = (r.get("content") or "").replace("\n", " ").strip()
            if len(snippet) > 350:
                snippet = snippet[:347] + "..."
            bullets.append(f"- {title}: {snippet} (fuente: {url})")

        ctx_body = "\n".join(bullets) if bullets else "SIN_RESULTADOS_DETALLADOS"

        return (
            "WEB_CONTEXT::\n"
            f"RESPUESTA_SINTESIS: {answer or 'Sin síntesis directa.'}\n"
            "DETALLES:\n"
            f"{ctx_body}"
        )

    except Exception as e:
        return f"WEB_CONTEXT::ERROR::{type(e).__name__}::{e}"


# ---- Tools de perfil/chat ----
@tool
def get_student_profile(name_or_email: str) -> str:
    """Resumen: carrera, skills, metas, intereses y estilo de aprendizaje."""
    row = _fetch_student(name_or_email)
    if not row:
        return "PERFIL_NO_ENCONTRADO"
    skills = ", ".join(row.get("skills", []) or [])
    goals = ", ".join(row.get("goals", []) or [])
    intr = ", ".join(row.get("interests", []) or [])
    career = row.get("career") or "N/D"
    semester = row.get("semester") or "N/D"
    full = row.get("full_name") or name_or_email

    ls = row.get("learning_style") or {}
    prefs = []
    if ls.get("prefers_examples"):
        prefs.append("con ejemplos")
    if ls.get("prefers_visual"):
        prefs.append("de forma visual")
    if ls.get("prefers_step_by_step"):
        prefs.append("paso a paso")
    if ls.get("prefers_theory"):
        prefs.append("con teoría")
    if ls.get("prefers_practice"):
        prefs.append("con práctica")
    notes = ls.get("notes", "")
    learning_desc = (
        f"Prefiere aprender {' y '.join(prefs)}. {notes}"
        if prefs
        else "No se ha definido su estilo de aprendizaje."
    )

    return (
        f"Perfil de {full} — Carrera: {career}, Semestre: {semester}. "
        f"Skills: {skills or 'N/D'}. Metas: {goals or 'N/D'}. "
        f"Intereses: {intr or 'N/D'}. {learning_desc}"
    )


@tool
def submit_student_profile(
    full_name: str,
    email: str,
    career: str,
    semester: int,
    skills: List[str],
    goals: List[str],
    interests: str,
    learning_style: dict = None,
) -> str:
    """Inserta o actualiza el perfil de estudiante."""
    _submit_student(
        full_name,
        email,
        career,
        semester,
        skills,
        goals,
        interests,
        learning_style=learning_style,
    )
    return "OK"


@tool
def identify_user_from_message(message: str) -> str:
    """
    Intenta identificar al usuario buscando un email o nombre en el mensaje.

    Retorna:
    - 'FOUND:email:name' si lo encuentra
    - 'NOT_FOUND' si no hay match
    """
    words = message.split()

    for word in words:
        if "@" in word:
            clean_email = word.strip(".,;:!?")
            row = _fetch_student(clean_email)
            if row:
                return f"FOUND:{row.get('email')}:{row.get('full_name')}"

    for i in range(len(words) - 1):
        if (
            words[i]
            and words[i][0].isupper()
            and words[i + 1]
            and words[i + 1][0].isupper()
        ):
            potential_name = f"{words[i]} {words[i+1]}"
            row = _fetch_student(potential_name)
            if row:
                return f"FOUND:{row.get('email')}:{row.get('full_name')}"

    return "NOT_FOUND"


# ---- Tools de actualización de perfil ----
@tool
def summarize_all_chats() -> str:
    """
    Genera resúmenes para TODAS las sesiones de chat y devuelve estadísticas breves.
    """
    stats = _summarize_all_chats()
    return (
        f"Processed {stats['successful']}/{stats['total_sessions']} sessions. "
        f"Failed: {stats['failed']}"
    )


@tool
def update_student_goals(name_or_email: str, new_goal: str) -> str:
    """Agrega una meta al perfil (JSONB)."""
    row = _fetch_student(name_or_email)
    if not row:
        return "PERFIL_NO_ENCONTRADO"
    goals = row.get("goals") or []
    if new_goal and new_goal not in goals:
        goals.append(new_goal)
        (
            SB.table("students")
            .update({"goals": goals})
            .eq("id", row["id"])
            .execute()
        )
    return f"OK: objetivos ahora = {goals}"


@tool
def update_learning_style(name_or_email: str, style: str) -> str:
    """Actualiza preferencias de aprendizaje a partir de texto libre."""
    row = _fetch_student(name_or_email)
    if not row:
        return "PERFIL_NO_ENCONTRADO"
    style_l = (style or "").lower()
    ls = row.get("learning_style") or {}
    if "ejemplo" in style_l:
        ls["prefers_examples"] = True
    if "visual" in style_l:
        ls["prefers_visual"] = True
    if "paso" in style_l:
        ls["prefers_step_by_step"] = True
    if "teor" in style_l:
        ls["prefers_theory"] = True
    if "práct" in style_l or "practic" in style_l:
        ls["prefers_practice"] = True
    ls["notes"] = style
    (
        SB.table("students")
        .update({"learning_style": ls})
        .eq("id", row["id"])
        .execute()
    )
    return f"Estilo actualizado para {row['full_name']}: {ls}"


# ---- Tool RAG (contexto por estudiante + chat) ----
@tool
def retrieve_context(name_or_email: str, chat_id: int, query: str) -> str:
    """
    Busca contexto relevante en la base vectorial asociada al ESTUDIANTE y al HISTORIAL DE CHAT,
    y devuelve pasajes útiles para responder una consulta técnica.
    Ahora:
    - transforma la query según el perfil
    - usa búsqueda semántica avanzada (MMR)
    """
    print(f"RETRIEVE_CONTEXT: name={name_or_email}, chat_id={chat_id}, query={query}")

    student_row = _fetch_student(name_or_email)
    if not student_row:
        print(f"Estudiante '{name_or_email}' no encontrado en DB")
        return "RAG_EMPTY"

    print(
        f"Estudiante encontrado: {student_row.get('full_name')} | "
        f"{student_row.get('email')}"
    )

    # 1) Transformar query para RAG
    transformed_query = _transform_query_for_rag(query, student_row)
    print(f"RAG transformed_query = {transformed_query}")

    # 2) Vectorstores (perfil + historial) y búsqueda semántica avanzada
    try:
        student_vectorstore = general_student_db_use(name_or_email)
        student_docs = _semantic_search(student_vectorstore, transformed_query, k=4) if student_vectorstore else []
    except Exception as e:
        print("[retrieve_context] error student_vectorstore:", e)
        student_docs = []

    try:
        chat_vectorstore = general_chat_db_use(chat_id)
        chat_docs = _semantic_search(chat_vectorstore, transformed_query, k=4) if chat_vectorstore else []
    except Exception as e:
        print("[retrieve_context] error chat_vectorstore:", e)
        chat_docs = []


    out: List[str] = []
    search_name = name_or_email.lower()

    # Estilo de aprendizaje
    ls = student_row.get("learning_style") or {}
    prefs = []
    if ls.get("prefers_examples"):
        prefs.append("con ejemplos")
    if ls.get("prefers_visual"):
        prefs.append("de forma visual")
    if ls.get("prefers_step_by_step"):
        prefs.append("paso a paso")
    if ls.get("prefers_theory"):
        prefs.append("con teoría")
    if ls.get("prefers_practice"):
        prefs.append("con práctica")
    notes = ls.get("notes", "")
    if prefs or notes:
        out.append(
            f"[ESTILO_APRENDIZAJE] {student_row.get('full_name', name_or_email)} "
            f"prefiere aprender {' y '.join(prefs)}. {notes}"
        )

    # Contexto del estudiante (perfil)
    for d in student_docs:
            m = d.metadata or {}
            doc_name = (m.get("full_name") or "").lower()
            doc_email = (m.get("email") or "").lower()

            if search_name in doc_name or search_name in doc_email or (
                doc_email and doc_email in search_name
            ):
                out.append(
                    f"[STUDENT_DOC] {m.get('full_name')} | {m.get('email')}\n"
                    f"{_clip(d.page_content)}\n"
                )
            else:
                print(f"Documento pertenece a {doc_name}, buscando {search_name}")
    # Contexto desde vectorstore de chat
    for d in chat_docs:
        m = d.metadata or {}
        out.append(
            f"{m.get('created_at')} | {m.get('robot_type')} | "
            f"{m.get('problem_title')} | {m.get('author')}\n"
            f"[CHAT] {m.get('session_id')} | {m.get('updated_at')}\n"
            f"{_clip(d.page_content)}\n"
        )

    result = "\n".join(out) if out else "RAG_EMPTY"
    print(f"{len(out)} documentos encontrados para {name_or_email}")
    return result



# ---- Tool RAG específico de RoboSupport ----
@tool
def retrieve_robot_support(
    query: str,
    project_id: Optional[str] = None,
    team_id: Optional[str] = None,
) -> str:
    """
    Wrapper legado que usa la lógica genérica de search_in_db,
    restringida a la tabla 'RoboSupportDB'.

    Debe existir un registro en agent_tables con table_name='RoboSupportDB'.
    """
    return _search_in_db_impl(
        project_id=project_id,
        query=query,
        team_id=team_id,
        max_tables=1,
        k_per_table=4,
        only_tables=["RoboSupportDB"],
    )

def _search_in_db_impl(
    project_id: Optional[str],
    query: str,
    team_id: Optional[str] = None,
    max_tables: int = 3,
    k_per_table: int = 4,
    only_tables: Optional[List[str]] = None,
    max_attempts: int = 3,
) -> str:
    """
    Implementación interna de la búsqueda RAG genérica sobre las tablas
    registradas en agent_tables, con reintentos progresivamente más relajados.
    """
    try:
        # 0) Transformar query una vez
        transformed_query = _transform_query_for_rag(query)
        print(
            f"[search_in_db] original_query='{query}' | "
            f"transformed_query='{transformed_query}'"
        )

        attempt = 1
        last_error = None

        while attempt <= max_attempts:
            print(f"[search_in_db] intento {attempt}/{max_attempts}")

            # --- 1) Elegir configuración según intento ---
            if attempt == 1:
                cur_project_id = project_id
                cur_team_id = team_id
                cur_only_tables = only_tables
            elif attempt == 2:
                # Relajar team_id
                cur_project_id = project_id
                cur_team_id = None
                cur_only_tables = only_tables
            else:
                # Último intento: sin filtros de proyecto/tabla
                cur_project_id = None
                cur_team_id = None
                cur_only_tables = None

            # --- 2) Obtener tablas candidatas ---
            tables = _get_agent_tables(
                project_id=cur_project_id,
                team_id=cur_team_id,
                table_names=cur_only_tables,
            )
            if not tables:
                last_error = (
                    "No hay tablas configuradas en agent_tables para este filtro."
                )
                print(f"[search_in_db] intento {attempt}: sin tablas.")
                attempt += 1
                continue

            # --- 3) Seleccionar tablas relevantes ---
            selected = _select_relevant_agent_tables(
                tables=tables,
                query=transformed_query,
                max_tables=max_tables,
            )

            all_snippets: List[str] = []

            # --- 4) Construir vectorstore por tabla y buscar ---
            for t in selected:
                docs = _build_generic_table_docs(t)
                if not docs:
                    continue

                vs_name = f"agent_{t.get('table_name')}"
                vs = create_or_update_vectorstore(vs_name, docs, len(docs))

                hits = _semantic_search(vs, transformed_query, k=k_per_table)
                for d in hits:
                    m = d.metadata or {}
                    header = (
                        f"[TABLE={m.get('table_name')}] {m.get('display_name')} | "
                        f"row_id={m.get('row_id')}"
                    )
                    all_snippets.append(f"{header}\n{d.page_content}\n")

            # Si hubo resultados, regresamos de inmediato
            if all_snippets:
                body = "\n---\n".join(all_snippets)
                return f"DB_CONTEXT::\n{body}"

            print(f"[search_in_db] intento {attempt}: sin resultados relevantes.")
            attempt += 1

        # Si llegamos aquí, los intentos se agotaron
        msg = last_error or "No encontré registros relevantes tras varios intentos."
        return f"DB_CONTEXT::RAG_EMPTY::{msg} Consulta: {query}"

    except Exception as e:
        print("[_search_in_db_impl] error:", e)
        return f"DB_CONTEXT::ERROR::{e}"


@tool
def search_in_db(
    project_id: Optional[str],
    query: str,
    team_id: Optional[str] = None,
    max_tables: int = 3,
    k_per_table: int = 4,
    only_tables: Optional[List[str]] = None,
) -> str:
    """
    Tool RAG genérica para buscar en las tablas configuradas en agent_tables.

    - Usa agent_tables para saber QUÉ tablas puede usar el agente.
    - Usa LLM para seleccionar las tablas más relevantes para la query.
    - Vectoriza filas de esas tablas y hace búsqueda semántica (MMR).
    - Devuelve contexto legible para que el agente responda.

    Convención:
    - El contexto devuelto empieza con 'DB_CONTEXT::'.
    """
    return _search_in_db_impl(
        project_id=project_id,
        query=query,
        team_id=team_id,
        max_tables=max_tables,
        k_per_table=k_per_table,
        only_tables=only_tables,
    )


# ---- Tool de ruteo interno entre agentes ----
@tool
def route_to(target: str) -> str:
    """Handoff interno entre agentes. Valores: EDUCATION|LAB|INDUSTRIAL|GENERAL."""
    return f"ROUTE::{(target or '').upper()}"


# ---- Current date/time tool ----
@tool
def current_datetime(state: Optional[State] = None, tz: Optional[str] = None) -> str:
    """
    Regresa fecha/hora actual en la zona tz (o state.tz o America/Monterrey)
    en ISO local, ISO UTC y formato legible en español.
    """
    tz_name = (
        tz
        or (state.get("tz") if isinstance(state, dict) else None)
        or "America/Monterrey"
    )
    now_loc = datetime.now(ZoneInfo(tz_name))
    out = {
        "tz": tz_name,
        "now_local": now_loc.isoformat(),
        "now_utc": datetime.now(tz=timezone.utc).isoformat(),
        "now_human": now_loc.strftime("%A, %d %b %Y, %H:%M"),
    }
    return str(out)


# ---- User identification tools ----
@tool
def check_user_exists(email: str) -> str:
    """
    Verifica si un usuario existe en la BD por email.
    Retorna: 'EXISTS:full_name' o 'NOT_FOUND'.
    """
    try:
        row = _fetch_student(email)
        if row:
            return f"EXISTS:{row.get('full_name', 'Usuario')}"
        return "NOT_FOUND"
    except Exception as e:
        return f"ERROR:{str(e)}"


@tool
def register_new_student(
    full_name: str,
    email: str,
    career: str = "",
    semester: int = 1,
    skills: List[str] = None,
    goals: List[str] = None,
    interests: Union[str, List[str]] = None,
    learning_style: dict = None,
) -> str:
    """
    Registra un nuevo estudiante en la BD.
    Retorna: 'OK' o 'ERROR:mensaje'.
    """
    try:
        if skills is None:
            skills = []
        if goals is None:
            goals = []
        if interests is None:
            interests = []
        if learning_style is None:
            learning_style = {}

        _submit_student(
            full_name=full_name,
            email=email,
            career=career,
            semester=semester,
            skills=skills,
            goals=goals,
            interests=interests,
            learning_style=learning_style,
        )
        return "OK"
    except Exception as e:
        return f"ERROR:{str(e)}"


@tool
def update_student_info(
    email: str,
    career: str = None,
    semester: int = None,
    skills: List[str] = None,
    goals: List[str] = None,
    interests: Union[str, List[str]] = None,
) -> str:
    """
    Actualiza información de un estudiante existente.
    Solo actualiza los campos no None.
    Retorna: 'OK' o 'ERROR:mensaje'.
    """
    try:
        row = _fetch_student(email)
        if not row:
            return "ERROR:Usuario no encontrado"

        update_data = {}
        if career is not None:
            update_data["career"] = career
        if semester is not None:
            update_data["semester"] = semester
        if skills is not None:
            update_data["skills"] = skills
        if goals is not None:
            update_data["goals"] = goals
        if interests is not None:
            update_data["interests"] = interests

        if update_data:
            (
                SB.table("students")
                .update(update_data)
                .eq("email", email)
                .execute()
            )

        return "OK"
    except Exception as e:
        return f"ERROR:{str(e)}"


# ====================================================
# TOOLS PARA PRÁCTICAS (PROJECT_TASKS + TASK_STEPS)
# ====================================================

@tool
def get_project_tasks(project_id: str) -> List[dict]:
    """
    Devuelve la lista de tasks (project_tasks) asociadas a un proyecto,
    ordenadas por created_at (o por id si no existe created_at).
    Úsalo para que el agente vea el mapa general de prácticas del proyecto.
    """
    try:
        # Intentar ordenar por created_at; si no existe, Supabase marcará error en logs
        res = (
            SB.table("project_tasks")
            .select("id, project_id, title, description, created_at")
            .eq("project_id", project_id)
            .order("created_at", desc=False)
            .execute()
        )
        return res.data or []
    except Exception as e:
        print("[get_project_tasks] error:", e)
        # Fallback simple: sin orden explícito
        try:
            res = (
                SB.table("project_tasks")
                .select("id, project_id, title, description")
                .eq("project_id", project_id)
                .execute()
            )
            return res.data or []
        except Exception as e2:
            print("[get_project_tasks fallback] error:", e2)
            return []


@tool
def get_task_steps(task_id: str) -> List[dict]:
    """
    Devuelve los pasos (task_steps) de una task de práctica,
    ordenados por step_number.
    
    Espera que la tabla task_steps tenga al menos:
    - id
    - task_id
    - step_number (int)
    - title
    - description
    - is_completed (bool, opcional)
    """
    try:
        res = (
            SB.table("task_steps")
            .select("id, task_id, step_number, title, description, is_completed")
            .eq("task_id", task_id)
            .order("step_number", desc=False)
            .execute()
        )
        return res.data or []
    except Exception as e:
        print("[get_task_steps] error:", e)
        return []


@tool
def complete_task_step(step_id: str) -> str:
    """
    Marca un paso de práctica (task_steps) como completado.
    Útil cuando el estudiante termina un paso y el agente quiere registrar el avance.
    """
    try:
        (
            SB.table("task_steps")
            .update(
                {
                    "is_completed": True,
                    "completed_at": datetime.now(ZoneInfo("America/Monterrey")).isoformat(),
                }
            )
            .eq("id", step_id)
            .execute()
        )
        return "OK"
    except Exception as e:
        print("[complete_task_step] error:", e)
        return f"ERROR:{e}"
# ====================================================
# TOOLS PARA IMÁGENES DE MANUALES / PRÁCTICAS
# ====================================================

@tool
def get_task_step_images(
    task_id: str,
    step_number: Optional[int] = None,
    limit: int = 5,
) -> List[dict]:
    """
    Devuelve imágenes asociadas a una task de práctica (y opcionalmente a un paso concreto),
    a partir de la tabla manual_images.

    Se asume que manual_images tiene algunos de estos campos:
    - id
    - project_id (opcional)
    - project_task_id o task_id
    - step_number (opcional)
    - title
    - description
    - tags
    - image_url  (o storage_path)  -> URL o ruta pública
    """
    try:
        q = (
            SB.table("manual_images")
            .select("*")
            .eq("project_task_id", task_id)
        )
    except Exception:
        # Fallback si el FK se llama task_id en vez de project_task_id
        q = SB.table("manual_images").select("*").eq("task_id", task_id)

    if step_number is not None:
        try:
            q = q.eq("step_number", step_number)
        except Exception as e:
            print("[get_task_step_images] step_number filter error:", e)

    try:
        res = q.limit(max(1, min(20, int(limit)))).execute()
        return res.data or []
    except Exception as e:
        print("[get_task_step_images] error final:", e)
        return []


@tool
def search_manual_images(
    query: str,
    robot_type: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 5,
) -> List[dict]:
    """
    Busca imágenes en la tabla manual_images por texto (título/descripcion/tags)
    y filtros opcionales (robot_type, project_id).

    Úsalo cuando el agente quiera adjuntar una imagen de apoyo en una explicación
    normal (no necesariamente en modo práctica).

    Convención sugerida para el agente:
    - Cuando quieras que el frontend muestre una imagen, incluye en tu respuesta
      algo como: IMAGE::image_url::título o descripción corta.
    """
    try:
        # Hacemos un select amplio para no depender de columnas exactas
        q = SB.table("manual_images").select("*")

        # Filtros por proyecto / tipo de robot si se proporcionan
        if project_id:
            try:
                q = q.eq("project_id", project_id)
            except Exception as e:
                print("[search_manual_images] project_id filter error:", e)

        if robot_type:
            try:
                q = q.ilike("robot_type", f"%{robot_type}%")
            except Exception as e:
                print("[search_manual_images] robot_type filter error:", e)

        # Búsqueda textual básica sobre columnas típicas
        # (ajusta si tus nombres reales cambian)
        try:
            q = q.or_(
                "title.ilike.%{q}%,description.ilike.%{q}%,tags.ilike.%{q}%".format(
                    q=query
                )
            )
        except Exception as e:
            print("[search_manual_images] OR ilike error:", e)

        res = q.limit(max(1, min(20, int(limit)))).execute()
        return res.data or []
    except Exception as e:
        print("[search_manual_images] error:", e)
        return []


