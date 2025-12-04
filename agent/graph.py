"""
===========================================================
                    CORE GRAPH – FINAL VERSION
===========================================================
    Multi-Agent Orchestration for FrEDie
    - User identification
    - Multi-agent routing
    - Practice-guided workflows
    - Avatar-style adaptation
    - Full chat history persistence
===========================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================
from __future__ import annotations

import os
import re
import locale
from datetime import datetime
from typing import List, Optional, Literal, Callable

from zoneinfo import ZoneInfo
from typing_extensions import TypedDict
from pydantic.v1 import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import ToolMessage

# Prompts
from Settings.prompts import (
    general_prompt,
    education_prompt,
    lab_prompt,
    industrial_prompt,
    identification_prompt,
    agent_route_prompt,
)

# Tools
from Settings.tools import (
    web_research, retrieve_context, retrieve_robot_support,
    update_student_goals, update_learning_style,
    register_new_student, update_student_info,
    check_user_exists, get_student_profile, _fetch_student,
    summarize_all_chats, route_to, current_datetime,
    _submit_chat_history,
    get_project_tasks, get_task_steps,
    get_task_step_images, search_manual_images,
    complete_task_step,
)

# ==========================================================
# STATE
# ==========================================================

class State(TypedDict, total=False):

    # Histórico de mensajes
    messages: List[AnyMessage]

    # Identidad
    profile_summary: Optional[str]
    user_identified: Optional[bool]
    user_email: Optional[str]
    user_name: Optional[str]

    # Sesión
    session_id: Optional[str]
    session_title: Optional[str]

    # Tiempo
    tz: str
    now_utc: str
    now_local: str
    now_human: str

    # Avatar
    avatar_style: Optional[str]
    widget_avatar_id: Optional[str]
    widget_mode: Optional[str]
    widget_personality: Optional[str]
    widget_notes: Optional[str]

    # Stack de agentes
    current_agent: List[str]

    # Identificación
    awaiting_user_info: Optional[str]

    # Prácticas / proyectos
    chat_type: Optional[str]
    project_id: Optional[str]
    current_task_id: Optional[str]
    current_step_number: Optional[int]
    practice_completed: Optional[bool]


# ==========================================================
# HELPERS
# ==========================================================

def update_current_agent_stack(stack: List[str], new: Optional[str]):
    if new == "pop":
        return stack[:-1]
    if isinstance(new, str):
        return stack + [new]
    return stack


def _flatten(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join([_flatten(c) for c in content])
    return str(content)


# ==========================================================
# AVATAR STYLE GENERATOR
# ==========================================================

def build_avatar_style(student=None,
                       override_avatar_id=None,
                       override_mode=None,
                       override_personality=None,
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
            "- Analogías suaves con gatos SOLO cuando aporten claridad."
        ),
        "robot": (
            "Modo Robot Industrial:\n"
            "- Tono preciso y directo.\n"
            "- Prefiere explicación en pasos cuando aporta valor."
        ),
        "duck": (
            "Modo Pato Creativo:\n"
            "- Tono optimista y creativo.\n"
            "- Puedes referenciar patos con moderación."
        ),
        "lab": (
            "Modo Asistente de Laboratorio:\n"
            "- Tono metódico y técnico.\n"
            "- Prioriza claridad experimental."
        ),
        "astro": (
            "Modo Explorador XR:\n"
            "- Tono futurista y curioso.\n"
            "- Usa metáforas espaciales solo cuando ayudan."
        ),
        "cora": (
            "Modo Cora Estándar:\n"
            "- Tono profesional y amable."
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
    raise RuntimeError("OPENAI_API_KEY no configurada.")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0,
    request_timeout=30,
)

# ==========================================================
# TOOLS POR AGENTE
# ==========================================================

GENERAL_TOOLS = [
    web_research, retrieve_context, summarize_all_chats,
    update_student_goals, update_learning_style,
    get_student_profile, route_to, current_datetime,
]

EDU_TOOLS = [
    web_research, retrieve_context, update_learning_style,
    get_student_profile, get_project_tasks, get_task_steps,
    get_task_step_images, search_manual_images,
    complete_task_step, route_to, current_datetime,
]

LAB_TOOLS = [
    web_research, retrieve_context, retrieve_robot_support,
    route_to, current_datetime,
]

IND_TOOLS = LAB_TOOLS[:]

IDENT_TOOLS = [
    check_user_exists, register_new_student, update_student_info
]

# ==========================================================
# RUNNABLES (PROMPT + TOOLS)
# ==========================================================

general_runnable = general_prompt | llm.bind_tools(GENERAL_TOOLS)
education_runnable = education_prompt | llm.bind_tools(EDU_TOOLS)
lab_runnable = lab_prompt | llm.bind_tools(LAB_TOOLS)
industrial_runnable = industrial_prompt | llm.bind_tools(IND_TOOLS)
identification_runnable = identification_prompt | llm.bind_tools(IDENT_TOOLS)

# ==========================================================
# AGENT NODES
# ==========================================================

def _invoke(runnable, state: State):
    out = runnable.invoke(state)
    return {"messages": out if isinstance(out, list) else [out]}

def general_agent_node(state):     return _invoke(general_runnable, state)
def education_agent_node(state):   return _invoke(education_runnable, state)
def lab_agent_node(state):         return _invoke(lab_runnable, state)
def industrial_agent_node(state):  return _invoke(industrial_runnable, state)

# ==========================================================
# IDENTIFICATION FLOW
# ==========================================================

def identify_user_node(state: State):
    if state.get("user_identified"):
        return {}

    messages = state.get("messages") or []
    already_asked = any(
        ("nombre" in getattr(m, "content", "").lower() and
         "correo" in getattr(m, "content", "").lower())
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

    result = globals()[name].invoke(args)
    tool_msg = ToolMessage(content=result, tool_call_id=call_id)

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
# INITIAL NODE
# ==========================================================

def _inject_time(state: State):
    tz = state.get("tz") or "America/Monterrey"
    state["tz"] = tz
    try:
        locale.setlocale(locale.LC_TIME, "es_MX.UTF-8")
    except:
        pass
    now = datetime.now(ZoneInfo(tz))
    state["now_local"] = now.isoformat()
    state["now_utc"] = datetime.utcnow().isoformat() + "Z"
    state["now_human"] = now.strftime("%A, %d %b %Y, %H:%M")


def initial_node(state: State, config: RunnableConfig):
    state = dict(state)
    _inject_time(state)

    if not state.get("profile_summary"):
        state["profile_summary"] = "Perfil aún no registrado."

    if not state.get("session_id"):
        state["session_id"] = config.get("configurable", {}).get("thread_id")

    student = None
    if state.get("user_identified") and state.get("user_email"):
        summary = get_student_profile.invoke({"name_or_email": state["user_email"]})
        state["profile_summary"] = summary
        student = _fetch_student(state["user_email"])

    conf = config.get("configurable", {})
    state["avatar_style"] = build_avatar_style(
        student,
        override_avatar_id     = state.get("widget_avatar_id") or conf.get("avatar_id"),
        override_mode          = state.get("widget_mode") or conf.get("widget_mode"),
        override_personality   = state.get("widget_personality") or conf.get("widget_personality"),
        override_notes         = state.get("widget_notes") or conf.get("widget_notes"),
    )

    return state


# ==========================================================
# HISTORY SAVING
# ==========================================================

def save_user_input(state: State):
    sid = state.get("session_id")
    msgs = state.get("messages") or []
    last = msgs[-1] if msgs else None
    if not sid or not last:
        return {}

    role = "student" if getattr(last, "type", "") == "human" else "agent"
    content = _flatten(getattr(last, "content", ""))

    try:
        _submit_chat_history(sid, role, content, state.get("user_email"))
    except:
        pass

    return {}


def generate_session_title(messages: List[AnyMessage]):
    for m in messages:
        if getattr(m, "type", "") in ("human", "student", "user"):
            text = _flatten(getattr(m, "content", ""))
            return text[:60] + "…" if len(text) > 60 else text
    return "Sesión sin título"


def save_agent_output(state: State):
    sid = state.get("session_id")
    msgs = state.get("messages") or []
    last = msgs[-1] if msgs else None
    if not sid or not last:
        return {}

    content = _flatten(getattr(last, "content", ""))

    try:
        _submit_chat_history(sid, "agent", content, state.get("user_email"))
    except:
        pass

    try:
        title = generate_session_title(msgs)
        return {"session_title": title}
    except:
        return {}


# ==========================================================
# ROUTER
# ==========================================================

def _fallback_pick_agent(text: str):
    t = text.lower()
    if re.search(r"\b(plc|robot|scada|hmi|opc)\b", t):
        return "ToAgentIndustrial"
    if re.search(r"\b(laboratorio|muestra|sensor|nda|confidencial)\b", t):
        return "ToAgentLab"
    if re.search(r"\b(estudio|tarea|examen|proyecto escolar)\b", t):
        return "ToAgentEducation"
    return "ToAgentGeneral"


def intitial_route_function(state: State):

    if (state.get("chat_type") or "").lower() == "practice":
        return "ToAgentEducation"

    if tools_condition(state) == END:
        return END

    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", [])
    if tool_calls:
        name = tool_calls[0]["name"]
        if name in {"ToAgentEducation", "ToAgentIndustrial", "ToAgentGeneral", "ToAgentLab"}:
            return name

    return _fallback_pick_agent(getattr(last, "content", ""))



router_runnable = agent_route_prompt | llm.bind_tools(
    [route_to],         # <--- herramienta REAL
    tool_choice="required",
)

class Assistant:
    def __init__(self, runnable): self.r = runnable
    def __call__(self, state: State, config):
        out = self.r.invoke(state)
        return {"messages": out if isinstance(out, list) else [out]}


# ==========================================================
# GRAPH ASSEMBLY
# ==========================================================

graph = StateGraph(State)

graph.set_entry_point("initial_node")
graph.add_node("initial_node", initial_node)
graph.add_node("identify_user", identify_user_node)
graph.add_node("identification_tools", process_identification_tools)
graph.add_node("save_user_input", save_user_input)
graph.add_node("save_agent_output", save_agent_output)

graph.add_node("router", Assistant(router_runnable))

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


# ==========================================================
# AGENT NODES & BRIDGES
# ==========================================================

graph.add_node("general_agent_node", general_agent_node)
graph.add_node("education_agent_node", education_agent_node)
graph.add_node("lab_agent_node", lab_agent_node)
graph.add_node("industrial_agent_node", industrial_agent_node)

def create_entry(name, target):
    def entry(state):
        call_id = state["messages"][-1].tool_calls[0]["id"]
        msg = ToolMessage(
            tool_call_id=call_id,
            content=f"Ahora eres {name}. Continúa con la intención del usuario."
        )
        return {"messages": [msg], "current_agent": [target]}
    return entry

graph.add_node("ToAgentEducation",  create_entry("Agente Educativo", "education_agent_node"))
graph.add_node("ToAgentGeneral",    create_entry("Agente General", "general_agent_node"))
graph.add_node("ToAgentLab",        create_entry("Agente de Laboratorio", "lab_agent_node"))
graph.add_node("ToAgentIndustrial", create_entry("Agente Industrial", "industrial_agent_node"))

graph.add_edge("ToAgentEducation", "education_agent_node")
graph.add_edge("ToAgentGeneral", "general_agent_node")
graph.add_edge("ToAgentLab", "lab_agent_node")
graph.add_edge("ToAgentIndustrial", "industrial_agent_node")


# ==========================================================
# TOOLS NODE + RETURN TO CURRENT AGENT
# ==========================================================

tools_node = ToolNode(tools=[
    web_research, retrieve_context, retrieve_robot_support,
    get_student_profile, update_student_goals, update_learning_style,
    summarize_all_chats, route_to, current_datetime,
    get_project_tasks, get_task_steps, get_task_step_images,
    search_manual_images, complete_task_step,
])

graph.add_node("tools", tools_node)

def return_to_active(state: State):
    stack = state.get("current_agent") or []
    return stack[-1] if stack else "general_agent_node"

for agent in ["general_agent_node", "education_agent_node", "lab_agent_node", "industrial_agent_node"]:
    graph.add_conditional_edges(
        agent, tools_condition,
        {"tools": "tools", "__end__": "save_agent_output"},
    )

graph.add_conditional_edges("tools", return_to_active)
graph.add_edge("save_agent_output", END)

# ==========================================================
# POP AGENT
# ==========================================================

def pop_current_agent(state: State):
    msgs = []
    if state.get("messages") and getattr(state["messages"][-1], "tool_calls", None):
        call_id = state["messages"][-1].tool_calls[0]["id"]
        msgs.append(
            ToolMessage(
                tool_call_id=call_id,
                content="Regresando al asistente principal."
            )
        )
    return {"current_agent": "pop", "messages": msgs}

graph.add_node("leave_agent", pop_current_agent)

# ==========================================================
# END OF FILE
# ==========================================================
