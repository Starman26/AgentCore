# -*- coding: utf-8 -*-
from typing import Annotated, Literal, Optional, List, Callable
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from pydantic.v1 import BaseModel, Field
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv; load_dotenv()
import os, re, locale
from datetime import datetime
from zoneinfo import ZoneInfo

# ===== Prompts y Tools =====
from Settings.prompts import (
    general_prompt, education_prompt, lab_prompt, industrial_prompt
)
from Settings.tools import (
    web_research, retrieve_context, get_student_profile,  # get_student_profile (tool para prompts si lo ocupas)
    update_student_goals, update_learning_style, route_to,
    current_datetime, _submit_chat_history                # helper de guardado
)

# =========================
# Estado del grafo (unificado)
# =========================
def update_current_agent_stack(left: list[str], right: Optional[str]) -> list[str]:
    if right is None:
        return left
    if isinstance(right, list):
        right_list = [r for r in right if isinstance(r, str)]
        return left + right_list if right_list else left
    if right == "pop":
        return left[:-1]
    if not isinstance(right, str):
        return left + [str(right)]
    return left + [right]

class State(TypedDict, total=False):
    # Mensajes
    messages: Annotated[List[AnyMessage], add_messages]

    # Contexto de usuario
    profile_summary: Optional[str]
    user_email: Optional[str]   # (inyectado por el frontend)

    # Reloj / zona horaria (se rellenan en initial_node)
    tz: str
    now_utc: str
    now_local: str
    now_human: str

    # Pila de agentes activos (router moderno)
    current_agent: Annotated[
        List[Literal["education_agent_node","general_agent_node","lab_agent_node","industrial_agent_node","router"]],
        update_current_agent_stack
    ]

    # Compat / persistencia
    session_id: Optional[int]

# (opcional) herramienta de cierre/escala
class CompleteOrEscalate(BaseModel):
    reason: str = Field(description="Motivo para finalizar o escalar.")
    cancel: bool = Field(default=False, description="True=cierra; False=continúa/escalado.")

# =========================
# LLM base
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY en .env")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0, request_timeout=30)

# =========================
# Tools por agente (route_to sólo en ToolNode)
# =========================
GENERAL_TOOLS = [CompleteOrEscalate, web_research, update_student_goals, update_learning_style, current_datetime]
EDU_TOOLS     = [CompleteOrEscalate, web_research, update_learning_style, current_datetime]
LAB_TOOLS     = [CompleteOrEscalate, web_research, retrieve_context, current_datetime]
IND_TOOLS     = [CompleteOrEscalate, web_research, current_datetime]

# =========================
# Runnables por agente
# =========================
general_llm      = llm.bind_tools(GENERAL_TOOLS)
education_llm    = llm.bind_tools(EDU_TOOLS)
lab_llm          = llm.bind_tools(LAB_TOOLS)
industrial_llm   = llm.bind_tools(IND_TOOLS)

general_runnable     = general_prompt    | general_llm
education_runnable   = education_prompt  | education_llm
lab_runnable         = lab_prompt        | lab_llm
industrial_runnable  = industrial_prompt | industrial_llm

def general_agent_node(state: State):     return {"messages": general_runnable.invoke(state)}
def education_agent_node(state: State):   return {"messages": education_runnable.invoke(state)}
def lab_agent_node(state: State):         return {"messages": lab_runnable.invoke(state)}
def industrial_agent_node(state: State):  return {"messages": industrial_runnable.invoke(state)}

# =========================
# Nodo inicial: perfil + fecha/hora
# =========================
def _inject_time_fields(state: State) -> None:
    tz = state.get("tz") or "America/Monterrey"
    state["tz"] = tz
    try:
        locale.setlocale(locale.LC_TIME, "es_MX.UTF-8")
    except Exception:
        pass
    now_local_dt = datetime.now(ZoneInfo(tz))
    state["now_local"] = now_local_dt.isoformat()
    state["now_utc"]   = datetime.utcnow().isoformat() + "Z"
    state["now_human"] = now_local_dt.strftime("%A, %d %b %Y, %H:%M")

def initial_node(state: State) -> State:
    state = dict(state)
    _inject_time_fields(state)
    if state.get("profile_summary"):
        return state

    # Si tienes tool get_student_profile(name/email), aquí lee de tu DB/servicio
    user = state.get("user_email")
    if not user:
        state["profile_summary"] = "ERROR: no se proporcionó user_info"
        return state
    try:
        summary = get_student_profile.invoke({"name_or_email": user}) if hasattr(get_student_profile, "invoke") else get_student_profile(user)  # compat
    except Exception:
        summary = f"Perfil de {user}"
    state["profile_summary"] = summary
    return state

def initial_routing(_: State) -> Literal["router"]:
    return "router"

# =========================
# Router por tool-call
# =========================
def _fallback_pick_agent(text: str) -> str:
    t = (text or "").lower()
    if re.search(r'\b(plc|robot|hmi|scada|opc|ladder|siemens|allen-bradley|automatización)\b', t): return "ToAgentIndustrial"
    if re.search(r'\bnda|confidencial|alcance|laboratorio|experimento|sensor|muestra\b', t):        return "ToAgentLab"
    if re.search(r'\bplan de estudios|tarea|examen|aprender|clase|curso|proyecto escolar|estudio\b', t): return "ToAgentEducation"
    if re.search(r'\bpartes|rfc|domicilio|contrato|datos de contacto|coordinador|registro\b', t):  return "ToAgentGeneral"
    return "ToAgentGeneral"

def intitial_route_function(state: State) -> Literal["ToAgentEducation","ToAgentIndustrial","ToAgentGeneral","ToAgentLab","__end__"]:
    # si el mensaje fue sólo tools y no hay más que hacer
    if tools_condition(state) == END:
        return END
    tool_calls = getattr(state["messages"][-1], "tool_calls", []) or []
    if tool_calls:
        name = tool_calls[0]["name"]
        if name in {"ToAgentEducation","ToAgentIndustrial","ToAgentGeneral","ToAgentLab"}:
            return name
    last_message = getattr(state["messages"][-1], "content", "")
    return _fallback_pick_agent(last_message)

class ToAgentEducation(BaseModel):  reason: str = Field(description="Motivo de transferencia al agente educativo.")
class ToAgentGeneral(BaseModel):    reason: str = Field(description="Motivo de transferencia al agente general.")
class ToAgentLab(BaseModel):        reason: str = Field(description="Motivo de transferencia al agente de laboratorio.")
class ToAgentIndustrial(BaseModel): reason: str = Field(description="Motivo de transferencia al agente industrial.")

agent_route_prompt = ChatPromptTemplate.from_messages([
    ("system", """#MAIN GOAL
Eres el ROUTER. Debes ELEGIR **EXACTAMENTE UN** agente mediante una **llamada de herramienta**
(ToAgentEducation, ToAgentGeneral, ToAgentLab, ToAgentIndustrial). No respondas texto normal.
Perfil: {profile_summary} | {now_human} | ISO: {now_local} | TZ: {tz}"""),
    ("system", """#BEHAVIOUR
EDUCATION: estudiar/tareas/exámenes/estilo de aprendizaje.
LAB: laboratorio/robótica/instrumentación/RAG/NDA.
INDUSTRIAL: PLC/SCADA/OPC/HMI/robots/OT.
GENERAL: coordinación/agenda/datos de partes.
Devuelve sólo una tool call.
"""),
    ("placeholder", "{messages}")
])

class Assistant:
    def __init__(self, runnable): self.runnable = runnable
    def __call__(self, state: State, _config):
        return {"messages": (agent_route_prompt | llm.bind_tools(
            [ToAgentEducation, ToAgentGeneral, ToAgentLab, ToAgentIndustrial],
            tool_choice="any"
        )).invoke(state)}

# =========================
# Persistencia de mensajes
# =========================
def save_user_input(state: State, config: RunnableConfig):
    """Guarda el input del usuario en la base de datos (usa session_id si viene en config/state)."""
    session_id = state.get("session_id") or config.get("configurable", {}).get("session_id") or config.get("configurable", {}).get("thread_id")
    if not session_id: return {}
    msgs = state.get("messages") or []
    if not msgs: return {}
    last = msgs[-1]

    if isinstance(last, BaseMessage):
        role = "student" if last.type == "human" else last.type
        content = last.content
        if isinstance(content, list):
            text_parts = [item.get("text", "") if isinstance(item, dict) else str(item) for item in content]
            content = " ".join(text_parts).strip()
    else:
        role = "student"
        content = last.get("content") if isinstance(last, dict) else str(last)

    try:
        _submit_chat_history(session_id, role, content)
    except Exception as e:
        print(f"[save_user_input] Error: {e}")
    return {}

def save_agent_output(state: State, config: RunnableConfig):
    """Guarda la salida del agente en la base de datos."""
    session_id = state.get("session_id") or config.get("configurable", {}).get("session_id") or config.get("configurable", {}).get("thread_id")
    if not session_id: return {}
    msgs = state.get("messages") or []
    if not msgs: return {}
    last = msgs[-1]

    if isinstance(last, BaseMessage):
        role = "agent"
        content = last.content
        if isinstance(content, list):
            text_parts = [item.get("text", "") if isinstance(item, dict) else str(item) for item in content]
            content = " ".join(text_parts).strip()
    else:
        role = "agent"
        content = last.get("content") if isinstance(last, dict) else str(last)

    try:
        _submit_chat_history(session_id, role, content)
    except Exception as e:
        print(f"[save_agent_output] Error: {e}")
    return {}

# =========================
# Grafo
# =========================
graph = StateGraph(State)

# 1) Inicialización
graph.add_node("initial_node", initial_node)
graph.add_conditional_edges("initial_node", initial_routing)
graph.set_entry_point("initial_node")

# 2) Guardar input y router
graph.add_node("save_user_input", save_user_input)
graph.add_edge("initial_node", "save_user_input")

graph.add_node("router", Assistant(agent_route_prompt))
graph.add_edge("save_user_input", "router")
graph.add_conditional_edges("router", intitial_route_function)

# 3) Entry nodes (tool-calls) -> agentes
def create_entry_node(assistant_name: str, current_agent: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        msg = ToolMessage(tool_call_id=tool_call_id, content=f"Ahora eres {assistant_name}. Continúa con la intención del usuario.")
        return {"messages": [msg], "current_agent": current_agent}
    return entry_node

graph.add_node("ToAgentEducation",  create_entry_node("Agente Educativo",      "education_agent_node"))
graph.add_node("ToAgentGeneral",    create_entry_node("Agente General",        "general_agent_node"))
graph.add_node("ToAgentLab",        create_entry_node("Agente de Laboratorio", "lab_agent_node"))
graph.add_node("ToAgentIndustrial", create_entry_node("Agente Industrial",     "industrial_agent_node"))

graph.add_edge("ToAgentEducation",  "education_agent_node")
graph.add_edge("ToAgentGeneral",    "general_agent_node")
graph.add_edge("ToAgentLab",        "lab_agent_node")
graph.add_edge("ToAgentIndustrial", "industrial_agent_node")

# 4) Nodos de agentes
graph.add_node("general_agent_node",     general_agent_node)
graph.add_node("education_agent_node",   education_agent_node)
graph.add_node("lab_agent_node",         lab_agent_node)
graph.add_node("industrial_agent_node",  industrial_agent_node)

# 5) ToolNode único + ruteo de vuelta al agente activo
tools_node = ToolNode(tools=[web_research, retrieve_context, update_student_goals, update_learning_style, route_to, current_datetime])
graph.add_node("tools", tools_node)

for agent in ["general_agent_node","education_agent_node","lab_agent_node","industrial_agent_node"]:
    graph.add_conditional_edges(agent, tools_condition, {"tools": "tools", "__end__": "save_agent_output"})

def return_to_current_agent(state: State) -> str:
    stack = state.get("current_agent") or []
    return stack[-1] if stack else "general_agent_node"
graph.add_conditional_edges("tools", return_to_current_agent)

# 6) Guardar salida y terminar
graph.add_node("save_agent_output", save_agent_output)
graph.add_edge("save_agent_output", END)

# (Opcional) pop del agente (si usas CompleteOrEscalate para cerrar)
def pop_current_agent(state: State) -> dict:
    messages = []
    if state.get("messages") and getattr(state["messages"][-1], "tool_calls", None):
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        messages.append(ToolMessage(tool_call_id=tool_call_id, content="Reanuda con el asistente principal."))
    return {"current_agent": "pop", "messages": messages}

graph.add_node("leave_agent", pop_current_agent)
# Fin
