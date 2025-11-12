from typing_extensions import TypedDict
from typing import Annotated, Literal, Optional, List, Callable
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import AnyMessage, add_messages
from pydantic.v1 import BaseModel, Field
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv; load_dotenv()
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import locale

from Settings.prompts import general_prompt, education_prompt, lab_prompt, industrial_prompt
from Settings.tools import (
    web_research, retrieve_context, update_student_goals, update_learning_style, route_to,
    current_datetime
)
from helpers.fetch_user import get_student_profile

# =========================
# Estado del grafo
# =========================
def update_current_agent_stack(left: list[str], right: Optional[str]) -> list[str]:
  if right is None:
    return left
  if isinstance(right, list):
    right_list = [r for r in right if isinstance(r, str)]
    if not right_list:
      print("update_current_agent_stack: received empty/non-string list as right; ignoring")
      return left
    return left + right_list
  if right == "pop":
    return left[:-1]
  if not isinstance(right, str):
    print(f"update_current_agent_stack: unexpected type for right: {type(right)}; coercing to str")
    return left + [str(right)]
  return left + [right]

class State(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    profile_summary: Optional[str]
    # Reloj / zona horaria (inyectados en initial_node)
    tz: str
    now_utc: str
    now_local: str
    now_human: str
    current_agent: Annotated[
        List[
            Literal[
                "education_agent_node",
                "general_agent_node",
                "lab_agent_node",
                "industrial_agent_node",
                "router"
            ]
        ],
        update_current_agent_stack
    ]
    # Identidad de usuario (inyectada por el frontend)
    user_email: Optional[str]

# (opcional) herramienta estructurada para cierre/escala
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
# Tools por agente (incluye route_to para ruteo silencioso)
# =========================
GENERAL_TOOLS   = [CompleteOrEscalate, web_research, update_student_goals, update_learning_style, current_datetime]
EDU_TOOLS       = [CompleteOrEscalate, web_research, update_learning_style, current_datetime]
LAB_TOOLS       = [CompleteOrEscalate, web_research, retrieve_context, current_datetime]
IND_TOOLS       = [CompleteOrEscalate, web_research, current_datetime]


# =========================
# Runnables por agente
# =========================
general_llm      = llm.bind_tools(GENERAL_TOOLS)
education_llm    = llm.bind_tools(EDU_TOOLS)
lab_llm          = llm.bind_tools(LAB_TOOLS)
industrial_llm   = llm.bind_tools(IND_TOOLS)

general_runnable     = general_prompt   | general_llm
education_runnable   = education_prompt | education_llm
lab_runnable         = lab_prompt       | lab_llm
industrial_runnable  = industrial_prompt| industrial_llm

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
        # Si no existe el locale en el SO, seguimos con nombres en inglés
        pass
    now_local_dt = datetime.now(ZoneInfo(tz))
    state["now_local"] = now_local_dt.isoformat()
    state["now_utc"] = datetime.utcnow().isoformat() + "Z"
    state["now_human"] = now_local_dt.strftime("%A, %d %b %Y, %H:%M")

def initial_node(state: State) -> State:
    state = dict(state)  # copia defensiva
    _inject_time_fields(state)

    if state.get("profile_summary"):
        return state

    user_info = state.get("user_email")  # el frontend debe setearlo
    if not user_info:
        state["profile_summary"] = "ERROR: no se proporcionó user_info"
        return state

    summary = get_student_profile(user_info)
    state["profile_summary"] = summary
    return state

def initial_routing(state: State) -> Literal["router"]:
    return "router"

# =========================
# Router
# =========================
def _fallback_pick_agent(text: str) -> str:
    t = text.lower()
    if re.search(r'\b(plc|robot|hmi|scada|opc|ladder|siemens|allen-bradley|automatización)\b', t):
        return "ToAgentIndustrial"
    if re.search(r'\bnda|confidencial|alcance|categorías|clasificar info|laboratorio|experimento|sensor|muestra\b', t):
        return "ToAgentLab"
    if re.search(r'\bplan de estudios|tarea|examen|aprender|clase|curso|proyecto escolar|estudio\b', t):
        return "ToAgentEducation"
    if re.search(r'\bpartes|rfc|domicilio|contrato|datos de contacto|coordinador|registro\b', t):
        return "ToAgentGeneral"
    return "ToAgentGeneral"

def intitial_route_function(state: State) -> Literal[
    "ToAgentEducation", "ToAgentIndustrial", "ToAgentGeneral", "ToAgentLab", "__end__"
]:
    from langgraph.prebuilt import tools_condition
    tools = tools_condition(state)
    if tools == END:
        return END

    tool_calls = getattr(state["messages"][-1], "tool_calls", []) or []
    if tool_calls:
        name = tool_calls[0]["name"]
        if name in {"ToAgentEducation", "ToAgentIndustrial", "ToAgentGeneral", "ToAgentLab"}:
            return name

    last_message = getattr(state["messages"][-1], "content", "")
    forced = _fallback_pick_agent(last_message)
    print(f"[Router fallback] No tool call detectada → Dirigiendo a {forced}")
    return forced

class ToAgentEducation(BaseModel):
    reason: str = Field(description="Motivo de transferencia al agente educativo.")

class ToAgentGeneral(BaseModel):
    reason: str = Field(description="Motivo de transferencia al agente general.")

class ToAgentLab(BaseModel):
    reason: str = Field(description="Motivo de transferencia al agente de laboratorio.")

class ToAgentIndustrial(BaseModel):
    reason: str = Field(description="Motivo de transferencia al agente industrial.")

agent_route_prompt = ChatPromptTemplate.from_messages([
    ("system", """#MAIN GOAL
Eres el ROUTER. Debes ELEGIR **EXACTAMENTE UN** agente mediante una **llamada de herramienta**
( ToAgentEducation, ToAgentGeneral, ToAgentLab, ToAgentIndustrial ).
**PROHIBIDO** responder texto normal al usuario.
Perfil del usuario (contexto): {profile_summary}
Fecha/hora: {now_human} | ISO: {now_local} | TZ: {tz}"""),
    ("system", """#BEHAVIOUR
Analiza el último mensaje del usuario y enruta según el contenido:

- **EDUCATION → ToAgentEducation**: aprender/estudiar/explicar; tareas, exámenes, clases; estilo de aprendizaje; material didáctico.
- **LAB → ToAgentLab**: laboratorio/robótica/instrumentación; sensores/cámaras/experimentos; RAG técnico; **NDA/confidencialidad/alcance de información**; integración técnica de proyectos.
- **INDUSTRIAL → ToAgentIndustrial**: PLC/SCADA/OPC UA/HMI; robots; procesos/maquinaria industrial; ladder; Siemens/Allen-Bradley; integraciones OT.
- **GENERAL → ToAgentGeneral**: coordinación/agenda; datos de partes (nombres, RFC, domicilios); saludos/small talk; soporte administrativo.

## REGLAS
1) Emite **solo una** tool call. Si detectas múltiples categorías, aplica **desempate**:
   - Industrial vs Lab → **INDUSTRIAL** si hay PLC/SCADA/robots/OT; si hay **NDA/confidencialidad**, prioriza **LAB**.
   - Lab vs Education → **LAB** si hay hardware/experimentos/RAG técnico o NDA; de lo contrario **EDUCATION**.
   - Cualquier duda menor → elige la opción **más específica**; si es solo saludo/agenda → **GENERAL**.
2) No formules preguntas ni des texto al usuario desde el router.
3) Si el contenido es ruido o vacío, selecciona **GENERAL**.

Devuelve únicamente la tool call apropiada."""),
    ("placeholder", "{messages}")
])

# =========================
# Grafo
# =========================
graph = StateGraph(State)

graph.add_node("initial_node", initial_node)
graph.add_conditional_edges("initial_node", initial_routing)
graph.set_entry_point("initial_node")

router_runnable = agent_route_prompt | llm.bind_tools(
    [ToAgentEducation, ToAgentGeneral, ToAgentLab, ToAgentIndustrial],
    tool_choice="any"
)

class Assistant:
    def __init__(self, runnable):
        self.runnable = runnable
    def __call__(self, state: State, config):
        state = {**state}
        result = self.runnable.invoke(state)
        return {"messages": result}

graph.add_node("router", Assistant(router_runnable))
graph.add_conditional_edges("router", intitial_route_function)

graph.add_node("general_agent_node",     general_agent_node)
graph.add_node("education_agent_node",   education_agent_node)
graph.add_node("lab_agent_node",         lab_agent_node)
graph.add_node("industrial_agent_node",  industrial_agent_node)

# ===== Nodos de entrada por tool-call del router =====
def create_entry_node(assistant_name: str, current_agent: str) -> Callable:
  def entry_node(state: State) -> dict:
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    ca = current_agent
    if isinstance(ca, list):
      ca_list = [x for x in ca if isinstance(x, str)]
      ca = ca_list[-1] if ca_list else None

    msg = ToolMessage(
        tool_call_id=tool_call_id,
        content=f'Ahora eres {assistant_name}. Revisa el contexto y continúa con la intención del usuario.'
    )
    return {"messages": [msg]} if ca is None else {"messages": [msg], "current_agent": ca}
  return entry_node

graph.add_node("ToAgentEducation",  create_entry_node("Agente Educativo",   "education_agent_node"))
graph.add_node("ToAgentGeneral",    create_entry_node("Agente General",     "general_agent_node"))
graph.add_node("ToAgentLab",        create_entry_node("Agente de Laboratorio", "lab_agent_node"))
graph.add_node("ToAgentIndustrial", create_entry_node("Agente Industrial",  "industrial_agent_node"))

graph.add_edge("ToAgentEducation",  "education_agent_node")
graph.add_edge("ToAgentGeneral",    "general_agent_node")
graph.add_edge("ToAgentLab",        "lab_agent_node")
graph.add_edge("ToAgentIndustrial", "industrial_agent_node")

# =========================
# ToolNode + ruteo de vuelta al agente activo
# =========================
from langgraph.prebuilt import ToolNode, tools_condition

tools_node = ToolNode(
    tools=[web_research, retrieve_context, update_student_goals, update_learning_style, route_to,current_datetime]
)
graph.add_node("tools", tools_node)

# Después de cada agente: si hay tool_calls → ejecutar tools; si no, terminar
for agent in ["general_agent_node", "education_agent_node", "lab_agent_node", "industrial_agent_node"]:
    graph.add_conditional_edges(
        agent,
        tools_condition,
        {"tools": "tools", "__end__": END}
    )

# Volver desde "tools" al agente que está en la cima del stack
def return_to_current_agent(state: State) -> str:
    stack = state.get("current_agent") or []
    return stack[-1] if stack else "general_agent_node"

graph.add_conditional_edges("tools", return_to_current_agent)

# =========================
# Pop del agente (si usas una tool de cierre)
# =========================
def pop_current_agent(state: State) -> dict:
  messages = []
  if state.get("messages") and getattr(state["messages"][-1], "tool_calls", None):
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    messages.append(
        ToolMessage(
            tool_call_id=tool_call_id,
            content="Reanuda con el asistente principal. Continúa ayudando al usuario."
        )
    )
  return {"current_agent": "pop", "messages": messages}

graph.add_node("leave_agent", pop_current_agent)
# Fin del archivo
