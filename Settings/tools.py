# -*- coding: utf-8 -*-
# Tools unificados

import os
from typing import List, Optional, Literal, Union
from uuid import UUID, uuid4, uuid5, NAMESPACE_DNS, NAMESPACE_URL
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

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
):
    """
    Helper interno: guarda en la BD un mensaje de chat con sesión y usuario.
    Se usa desde graph.py (save_user_input / save_agent_output).
    NO es un tool.

    Respeta las relaciones:
      auth.users.id -> app_user.id -> chat_session.user_id -> chat_message.session_id
    """
    # Normalizar session_id a UUID string
    session_id = _normalize_session_id(session_id)

    created_at = created_at or datetime.now(tz=timezone.utc).isoformat()

    # Si user_id parece un email, avisamos y no guardamos (rompería el FK).
    if user_id and "@" in user_id:
        print(
            "[_submit_chat_history] WARNING: user_id parece email "
            f"('{user_id}'). Se espera auth.user.id (UUID). No se guarda el mensaje."
        )
        return None

    # Si no hay user_id, no podemos cumplir el FK: salimos.
    if not user_id:
        print(
            "[_submit_chat_history] WARNING: llamado sin user_id, "
            "no se guardará nada en chat_session / chat_message."
        )
        return None

    try:
        # 1) Asegurar usuario en app_user (FK -> auth.users.id)
        app_user_payload = {
            "id": user_id,
            "created_at": created_at,
            "role": "laboratorista",  # ⚠️ Debe existir en enum user_role
            "team_id": None,
        }
        try:
            SB.table("app_user").upsert(
                app_user_payload,
                on_conflict="id",
            ).execute()
        except Exception as e:
            print("[app_user upsert] ERROR:", e)
            # Si truena por FK (auth.users.id inexistente), no seguimos
            return None

        # 2) Crear/actualizar chat_session
        session_payload = {
            "id": session_id,
            "user_id": user_id,
            "started_at": created_at,
        }

        SB.table("chat_session").upsert(
            session_payload,
            on_conflict="id",
        ).execute()

        # 3) Insertar mensaje en chat_message
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
        print("Error saving chat:", e)
        raise


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
    """
    Construye Document(s) a partir de la tabla RoboSupportDB para vectorizarla
    (RAG de problemas de robots) con un estilo narrativo/humano.
    """
    res = (
        SB.table("RoboSupportDB")
        .select(
            "created_at, robot_type, problem_title, problem_description, solution_steps, author"
        )
        .execute()
    )
    rows = res.data or []
    docs: List[Document] = []

    for r in rows:
        robot = r.get("robot_type") or "el robot"
        title = r.get("problem_title") or "problema sin título"
        desc = r.get("problem_description") or "Sin descripción detallada."
        steps = r.get("solution_steps") or "Sin pasos registrados."
        author = r.get("author") or "otro integrante del laboratorio"

        content = (
            f"Problema registrado para el robot {robot}: {title}.\n"
            f"Descripción del problema: {desc}\n\n"
            f"Según {author}, los pasos recomendados para resolverlo fueron:\n"
            f"{steps}"
        )
        metadata = {
            "created_at": r.get("created_at"),
            "robot_type": robot,
            "problem_title": title,
            "author": author,
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

    student_vectorstore = general_student_db_use(name_or_email)
    student_retriever = student_vectorstore.as_retriever(search_kwargs={"k": 2})
    student_docs = student_retriever.invoke(query)

    chat_vectorstore = general_chat_db_use(chat_id)
    chat_retriever = chat_vectorstore.as_retriever(search_kwargs={"k": 2})
    chat_docs = chat_retriever.invoke(query)

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

    # Contexto del estudiante
    for d in student_docs:
        m = d.metadata or {}
        doc_name = (m.get("full_name") or "").lower()
        doc_email = (m.get("email") or "").lower()

        if search_name in doc_name or search_name in doc_email or (
            doc_email and doc_email in search_name
        ):
            out.append(
                f"[STUDENT_DOC] {m.get('full_name')} | {m.get('email')}\n"
                f"{d.page_content}\n"
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
            f"{d.page_content}\n"
        )

    result = "\n".join(out) if out else "RAG_EMPTY"
    print(f"{len(out)} documentos encontrados para {name_or_email}")
    return result


# ---- Tool RAG específico de RoboSupport ----
@tool
def retrieve_robot_support(query: str) -> str:
    """
    Busca problemas y soluciones en la base de datos de RoboSupportDB usando RAG.
    Devuelve contexto técnico en lenguaje natural para que el agente genere una respuesta humana.
    """
    docs = _build_robot_support_docs()
    if not docs:
        return "RAG_EMPTY::No hay registros en RoboSupportDB."

    vs = create_or_update_vectorstore("robot_support", docs, len(docs))
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    hits = retriever.invoke(query)

    if not hits:
        return (
            f"RAG_EMPTY::No encontré casos en RoboSupportDB relacionados con: {query}"
        )

    out_parts: List[str] = []
    for i, d in enumerate(hits, 1):
        m = d.metadata or {}
        robot = m.get("robot_type") or "robot"
        title = m.get("problem_title") or "problema sin título"
        author = m.get("author") or "otro integrante del laboratorio"
        out_parts.append(
            f"CASO_{i}:: Robot: {robot} | Problema: {title} | Registrado por: {author}\n"
            f"{d.page_content}\n"
        )

    return "\n\n".join(out_parts)


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
# TOOL SETS
# ====================================================
LAB_TOOLS = [
    retrieve_context,        # RAG estudiante + chat
    retrieve_robot_support,  # RAG específico de RoboSupport
    web_research,            # RAG web
    route_to,
]

GENERAL_TOOLS = [
    get_student_profile,     # para construir profile_summary y estilo
    update_student_goals,
    update_learning_style,
    web_research,
    retrieve_context,        # acceso a RAG global si lo necesita
    route_to,
    summarize_all_chats,
]

EDU_TOOLS = [
    get_student_profile,
    update_learning_style,
    web_research,
    retrieve_context,        # usar RAG cuando explique temas ligados a proyectos previos
    route_to,
]

IDENTIFICATION_TOOLS = [
    check_user_exists,
    register_new_student,
    update_student_info,
]
