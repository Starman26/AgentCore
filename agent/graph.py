"""
===========================================================
                CORE GRAPH – ORGANIZED VERSION
        Assistant orchestration for multi-agent workflow
===========================================================

Este archivo concentra:
- Definición del estado global
- Tools por agente
- Prompts por agente
- Armado del grafo LangGraph
- Router avanzado (con soporte para prácticas)
- Mecanismos de identificación de usuario
- Guardado de historial
- Entrada/salida de agentes
===========================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================

from typing import List, Optional, Literal, Callable
from typing_extensions import TypedDict
from pydantic.v1 import BaseModel, Field
from datetime import datetime
from zoneinfo import ZoneInfo
import locale, os, re

from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv; load_dotenv()

# Prompts y tools internos
from Settings.prompts import (
    general_prompt,
    education_prompt,
    lab_prompt,
    industrial_prompt,
    identification_prompt,
    agent_route_prompt,
)

from Settings.tools import (
    web_research,
    retrieve_context,
    retrieve_robot_support,
    update_student_goals,
    update_learning_style,
    register_new_student,
    get_student_profile,
    check_user_exists,
    update_student_info,
    _fetch_student,
    route_to,
    current_datetime,
    summarize_all_chats,
    _submit_chat_history,
    get_project_tasks,
    get_task_steps,
    get_task_step_images,
    search_manual_images,
    complete_task_step,
)

# ==========================================================
# STATE DEFINICIÓN PRINCIPAL
# ==========================================================

class State(TypedDict, total=False):

    # --- Historial ---
    messages: List[AnyMessage]

    # --- Perfil / Identidad ---
    profile_summary: Optional[str]
    user_identified: Optional[bool]
    user_email: Optional[str]
    user_name: Optional[str]

    # --- Sesión ---
    session_id: Optional[str]
    session_title: Optional[str]

    # --- Tiempo ---
    tz: str
    now_utc: str
    now_local: str
    now_human: str

    # --- Avatar / Widget ---
    avatar_style: Optional[str]
    widget_avatar_id: Optional[str]
    widget_mode: Optional[str]
    widget_personality: Optional[str]
    widget_notes: Optional[str]

    # --- Seguimiento del agente actual ---
    current_agent: List[str]

    # --- Identificación ---
    awaiting_user_info: Optional[str]

    # --- Prácticas / Proyectos ---
    chat_type: Optional[str]           # "practice", etc.
    project_id: Optional[str]
    current_task_id: Optional[str]
    current_step_number: Optional[int]
    practice_completed: Optional[bool]


# ==========================================================
# HELPERS DE STATE
# ==========================================================

def update_current_agent_stack(stack: List[str], new: Optional[str]):
    """Push/pop del stack de agentes."""
    if new == "pop":
        return stack[:-1]
    if isinstance(new, str):
        return stack + [new]
    return stack


def _flatten(content):
    """Convierte mensajes estructurados en texto plano."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join([_flatten(c) for c in content])
    return str(content)


# ==========================================================
# ESTILO POR AVATAR
# ==========================================================

def build_avatar_style(student=None, override_avatar_id=None,
                       override_mode=None, override_personality=None,
                       override_notes=None) -> str:

    student = student or {}

    avatar_id = override_avatar_id or student.get("widget_avatar_id") or "cora"
    mode = override_mode or student.get("widget_mode") or "default"
    custom_personality = override_personality or student.get("widget_personality") or ""
    custom_notes = override_notes or student.get("widget_notes") or ""

    base_styles = {
        "cat": (
            "Modo Gato Analítico:\n"
            "- Tono cálido, claro y ordenado.\n"
            "- Puedes usar analogías suaves con gatos si aportan claridad."
        ),
        "robot": (
            "Modo Robot Industrial:\n"
            "- Tono técnico, preciso y directo.\n"
            "- Prefiere pasos cuando es útil."
        ),
        "duck": (
            "Modo Pato Creativo:\n"
            "- Tono optimista y creativo.\n"
            "- Puedes referenciar patos con moderación."
        ),
        "lab": (
            "Modo Asistente de Laboratorio:\n"
            "- Tono metódico y técnico.\n"
            "- Prefiere procesos paso a paso."
        ),
        "astro": (
            "Modo Explorador XR:\n"
            "- Tono futurista y curioso.\n"
            "- Analiza usando metáforas espaciales cuando ayuden."
        ),
        "cora": (
            "Modo Cora Estándar:\n"
            "- Tono profesional, amable y neutro."
        ),
    }

    style = base_styles.get(avatar_id, base_styles["cora"])

    if mode == "custom":
        if custom_personality:
            style += f"\n\nInstrucciones personalizadas:\n{custom_personality}"
        if custom_notes:
            style += f"\n\nNotas adicionales:\n{custom_notes}"

    return style


# ==========================================================
# LLM BASE
# ==========================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY no está configurada.")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0,
    request_timeout=30,
)


# ==========================================================
# DEFINICIÓN DE TOOLS POR AGENTE
# ==========================================================

GENERAL_TOOLS = [
    web_research, retrieve_context, summarize_all_chats,
    update_student_goals, update_learning_style,
    get_student_profile, route_to, current_datetime,
]

EDU_TOOLS = [
    web_research, retrieve_context, update_learning_style,
    get_student_profile, get_project_tasks, get_task_steps,
    get_task_step_images, search_manual_images, complete_task_step,
    route_to, current_datetime,
]

LAB_TOOLS = [
    web_research, retrieve_context, retrieve_robot_support,
    route_to, current_datetime,
]

IND_TOOLS = LAB_TOOLS[:]  # similar


IDENT_TOOLS = [
    check_user_exists, register_new_student, update_student_info
]


# ==========================================================
# RUNNABLES (prompt + tools)
# ==========================================================

general_llm = llm.bind_tools(GENERAL_TOOLS)
education_llm = llm.bind_tools(EDU_TOOLS)
lab_llm = llm.bind_tools(LAB_TOOLS)
industrial_llm = llm.bind_tools(IND_TOOLS)
identification_llm = llm.bind_tools(IDENT_TOOLS)

general_runnable = general_prompt | general_llm
education_runnable = education_prompt | education_llm
lab_runnable = lab_prompt | lab_llm
industrial_runnable = industrial_prompt | industrial_llm
identification_runnable = identification_prompt | identification_llm


# ==========================================================
# NODOS DE AGENTES
# ==========================================================

def _invoke(runnable, state: State):
    out = runnable.invoke(state)
    return {"messages": out if isinstance(out, list) else [out]}

def general_agent_node(state): return _invoke(general_runnable, state)
def education_agent_node(state): return _invoke(education_runnable, state)
def lab_agent_node(state): return _invoke(lab_runnable, state)
def industrial_agent_node(state): return _invoke(industrial_runnable, state)


# ==========================================================
# IDENTIFICACIÓN DE USUARIO
# ==========================================================

def identify_user_node(state: State):
    if state.get("user_identified"):
        return {}

    messages = state.get("messages", [])
    already_asked = any(
        "nombre" in getattr(m, "content", "").lower() and "correo" in getattr(m, "content", "").lower()
        for m in messages if getattr(m, "type", "") == "ai"
    )

    if not already_asked:
        from langchain_core.messages import AIMessage
        return {
            "messages": [
                AIMessage(content="Para ayudarte mejor, ¿puedes darme tu nombre y correo?")
            ],
            "awaiting_user_info": "name_email",
        }

    result = identification_runnable.invoke(state)
    return {"messages": result if isinstance(result, list) else [result]}


def process_identification_tools(state: State):
    msgs = state.get("messages", [])
    if not msgs or not getattr(msgs[-1], "tool_calls", None):
        return {}

    call = msgs[-1].tool_calls[0]
    name, args, call_id = call["name"], call.get("args"), call["id"]

    # ejecutar tool real
    result = globals()[name].invoke(args)

    tool_msg = ToolMessage(content=result, tool_call_id=call_id)

    # Cuando la tool dice que el estudiante ya existe
    if "EXISTS" in result or result == "OK":
        email = args.get("email")
        student = _fetch_student(email)
        if student:
            summary = get_student_profile.invoke({"name_or_email": email})
            from langchain_core.messages import AIMessage
            confirm = AIMessage(
                content=f"Perfecto, {student.get('full_name')}. Ya te tengo identificado. ¿En qué te ayudo?"
            )
            return {
                "messages": [tool_msg, confirm],
                "user_identified": True,
                "user_email": email,
                "user_name": student.get("full_name"),
                "profile_summary": summary,
                "awaiting_user_info": None,
            }

    return {"messages": [tool_msg]}


def check_identification_status(state: State):
    if not state.get("user_identified"):
        msgs = state.get("messages", [])
        last = msgs[-1] if msgs else None
        if last and getattr(last, "tool_calls", None):
            return "tools"
        return "await_user"
    return "identified"


def check_after_identification_tools(state: State):
    if state.get("user_identified"):
        return "identified"
    msgs = state.get("messages", [])
    last = msgs[-1] if msgs else None
    if last and getattr(last, "tool_calls", None):
        return "continue_identifying"
    return "await_user"


# ==========================================================
# NODO INICIAL
# ==========================================================

def _inject_time_fields(state: State):
    tz = state.get("tz") or "America/Monterrey"
    state["tz"] = tz

    try:
        locale.setlocale(locale.LC_TIME, "es_MX.UTF-8")
    except:
        pass

    now_local = datetime.now(ZoneInfo(tz))
    state["now_local"] = now_local.isoformat()
    state["now_utc"] = datetime.utcnow().isoformat() + "Z"
    state["now_human"] = now_local.strftime("%A, %d %b %Y, %H:%M")


def initial_node(state: State, config: RunnableConfig):
    state = dict(state)
    _inject_time_fields(state)

    if not state.get("profile_summary"):
        state["profile_summary"] = "Perfil aún no registrado."

    if not state.get("session_id"):
        state["session_id"] = config.get("configurable", {}).get("thread_id")

    # cargar estilo de avatar
    student = None
    if state.get("user_identified") and state.get("user_email"):
        summary = get_student_profile.invoke({"name_or_email": state["user_email"]})
        state["profile_summary"] = summary
        student = _fetch_student(state["user_email"])

    conf = config.get("configurable", {})
    state["avatar_style"] = build_avatar_style(
        student,
        override_avatar_id=state.get("widget_avatar_id") or conf.get("avatar_id"),
        override_mode=state.get("widget_mode") or conf.get("widget_mode"),
        override_personality=state.get("widget_personality") or conf.get("widget_personality"),
        override_notes=state.get("widget_notes") or conf.get("widget_notes"),
    )

    return state


# ==========================================================
# GUARDADO DE HISTORIAL
# ==========================================================

def save_user_input(state: State):
    sid = state.get("session_id")
    if not sid:
        return {}

    msgs = state.get("messages", [])
    last = msgs[-1] if msgs else None

    role = "student" if getattr(last, "type", "") == "human" else "agent"
    content = _flatten(getattr(last, "content", ""))

    try:
        _submit_chat_history(sid, role=role, content=content, user_email=state.get("user_email"))
    except Exception as e:
        print(f"[save_user_input] Error: {e}")

    return {}

def generate_session_title_from_history(messages):
    for m in messages:
        if getattr(m, "type", "") in ("human", "student", "user"):
            text = _flatten(getattr(m, "content", ""))
            return (text[:60] + "…") if len(text) > 60 else text
    return "Sesión sin título"

def save_agent_output(state: State):
    sid = state.get("session_id")
    if not sid:
        return {}

    msgs = state.get("messages", [])
    last = msgs[-1]
    content = _flatten(getattr(last, "content", ""))

    try:
        _submit_chat_history(sid, role="agent", content=content, user_email=state.get("user_email"))
    except:
        pass

    try:
        title = generate_session_title_from_history(msgs)
        return {"session_title": title}
    except:
        return {}


# ==========================================================
# ROUTER PRINCIPAL
# ==========================================================

def _fallback_pick_agent(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(plc|robot|scada|hmi|opc)\b", t):
        return "ToAgentIndustrial"
    if re.search(r"\b(laboratorio|muestra|sensor)\b", t):
        return "ToAgentLab"
    if re.search(r"\b(estudio|clase|tarea|proyecto escolar)\b", t):
        return "ToAgentEducation"
    return "ToAgentGeneral"


def intitial_route_function(state: State):
    # forzar prácticas → Education agent
    if (state.get("chat_type") or "").lower() == "practice":
        return "ToAgentEducation"

    # tool condition
    if tools_condition(state) == END:
        return END

    msgs = state["messages"]
    last = msgs[-1]
    tool_calls = getattr(last, "tool_calls", [])
    if tool_calls:
        name = tool_calls[0]["name"]
        if name in {"ToAgentEducation", "ToAgentIndustrial", "ToAgentGeneral", "ToAgentLab"}:
            return name

    return _fallback_pick_agent(getattr(last, "content", ""))


# ==========================================================
# GRAFO PRINCIPAL
# ==========================================================

graph = StateGraph(State)

graph.set_entry_point("initial_node")
graph.add_node("initial_node", initial_node)

graph.add_node("identify_user", identify_user_node)
graph.add_node("identification_tools", process_identification_tools)

graph.add_node("save_user_input", save_user_input)
graph.add_node("save_agent_output", save_agent_output)

graph.add_edge("initial_node", "identify_user")

graph.add_conditional_edges(
    "identify_user",
    check_identification_status,
    {
        "identified": "save_user_input",
        "tools": "identification_tools",
        "await_user": END,
    },
)

graph.add_conditional_edges(
    "identification_tools",
    check_after_identification_tools,
    {
        "identified": "save_user_input",
        "continue_identifying": "identify_user",
        "await_user": END,
    },
)

graph.add_edge("save_user_input", "router")


# Router Node
router_runnable = agent_route_prompt | llm.bind_tools(
    ["ToAgentEducation", "ToAgentGeneral", "ToAgentLab", "ToAgentIndustrial"],
    tool_choice="any",
)

class Assistant:
    def __init__(self, runnable): self.r = runnable
    def __call__(self, state: State, config):
        out = self.r.invoke(state)
        return {"messages": out if isinstance(out, list) else [out]}

graph.add_node("router", Assistant(router_runnable))
graph.add_conditional_edges("router", intitial_route_function)


# ==========================================================
# AGENTE NODES
# ==========================================================

graph.add_node("general_agent_node", general_agent_node)
graph.add_node("education_agent_node", education_agent_node)
graph.add_node("lab_agent_node", lab_agent_node)
graph.add_node("industrial_agent_node", industrial_agent_node)


# Entradas por tool-call (bridge nodes)
def create_entry_node(name, target):
    def entry(state: State):
        call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    tool_call_id=call_id,
                    content=f"Ahora eres {name}. Continúa con la intención del usuario."
                )
            ],
            "current_agent": [target],
        }
    return entry

graph.add_node("ToAgentEducation", create_entry_node("Agente Educativo", "education_agent_node"))
graph.add_node("ToAgentGeneral", create_entry_node("Agente General", "general_agent_node"))
graph.add_node("ToAgentLab", create_entry_node("Agente de Laboratorio", "lab_agent_node"))
graph.add_node("ToAgentIndustrial", create_entry_node("Agente Industrial", "industrial_agent_node"))

graph.add_edge("ToAgentEducation", "education_agent_node")
graph.add_edge("ToAgentGeneral", "general_agent_node")
graph.add_edge("ToAgentLab", "lab_agent_node")
graph.add_edge("ToAgentIndustrial", "industrial_agent_node")


# ==========================================================
# TOOLS NODE
# ==========================================================

tools_node = ToolNode(tools=[
    web_research, retrieve_context, retrieve_robot_support,
    get_student_profile, update_student_goals, update_learning_style,
    summarize_all_chats, route_to, current_datetime,
    get_project_tasks, get_task_steps, get_task_step_images,
    search_manual_images, complete_task_step,
])

graph.add_node("tools", tools_node)


# Return to active agent
def return_to_current_agent(state: State):
    stack = state.get("current_agent") or []
    return stack[-1] if stack else "general_agent_node"

for agent in [
    "general_agent_node", "education_agent_node",
    "lab_agent_node", "industrial_agent_node",
]:
    graph.add_conditional_edges(
        agent,
        tools_condition,
        {"tools": "tools", "__end__": "save_agent_output"},
    )

graph.add_conditional_edges("tools", return_to_current_agent)
graph.add_edge("save_agent_output", END)

