from typing_extensions import TypedDict
from typing import Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import AnyMessage, add_messages
from pydantic.v1 import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition, ToolNode
from dotenv import load_dotenv; load_dotenv()
import os
from Settings.prompts import general_prompt, education_prompt, lab_prompt, industrial_prompt
from Settings.tools import retrieve_context, update_student_goals, update_learning_style, route_to
from helpers.fetch_user import get_student_profile
from langchain_core.runnables import Runnable, RunnableConfig
from typing import Callable
from langchain_core.messages import ToolMessage
from typing import Annotated, Literal, Optional, TypedDict, List

# Estado del grafo
def update_current_agent_stack(left: list[str], right: Optional[str]) -> list[str]:
  """Push or pop the state.

  This function is defensive: sometimes the graph runtime may pass a list
  (e.g. an empty list) as the "right" value which would produce a nested
  list inside the `current_agent` stack. Handle those cases gracefully:
  - If right is None: return left unchanged
  - If right is the string "pop": pop the stack
  - If right is a list: filter string elements and append them (ignore
    empty/non-string lists)
  - If right is an unexpected type: coerce to string and append
  """
  if right is None:
    return left

  # Defensive: if a list is returned as the 'right' value, flatten/append
  if isinstance(right, list):
    # keep only string items
    right_list = [r for r in right if isinstance(r, str)]
    if not right_list:
      # nothing useful to add; ignore
      print("update_current_agent_stack: received empty/non-string list as right; ignoring")
      return left
    return left + right_list

  if right == "pop":
    return left[:-1]

  if not isinstance(right, str):
    # unexpected type: coerce to string to avoid nested lists
    print(f"update_current_agent_stack: unexpected type for right: {type(right)}; coercing to str")
    return left + [str(right)]

  return left + [right]

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    profile_summary: Optional[str]  
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
    update_current_agent_stack]

class CompleteOrEscalate(BaseModel):
    """
    Herramienta de finalización o escalamiento del agente actual.

    Esta herramienta permite al agente especializado cerrar su intervención actual o escalar el control al asistente principal (por ejemplo, otro agente del flujo o el sistema general) en función del estado del flujo conversacional.
    Campos:
    - `reason`: Explicación textual del motivo por el cual se termina o escala el flujo.
    - `cancel`: Si se debe cancelar completamente el proceso (True) o solo escalarlo para que otro agente o componente lo continúe (False).
    """

    reason: str = Field(
        description="Motivo por el cual el agente finaliza o escala la conversación. "
    )

    cancel: bool = Field(
        default=False,
        description="Indica si se cancela el flujo por completo (`True`) o si se delega a otro agente para continuar (`False`)."
    )

GENERAL_TOOLS = [CompleteOrEscalate]
LAB_TOOLS     = [CompleteOrEscalate]
EDU_TOOLS       = [CompleteOrEscalate]

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        state = {**state}
        result = self.runnable.invoke(state)
        return {"messages": result}
    
# ---------- LLM base ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY en .env")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0, request_timeout=30)


# ---------- Runnables por agente ----------
general_llm      = llm.bind_tools(GENERAL_TOOLS)
education_llm    = llm.bind_tools(EDU_TOOLS)     # para leer/actualizar perfil
lab_llm          = llm.bind_tools(LAB_TOOLS)
industrial_llm   = llm 

class ToAgentEducation(BaseModel):
    """Redirige al agente educativo, encargado de gestionar el plan de estudios, comprender el perfil del estudiante y ofrecer contenido académico personalizado."""
    reason: str = Field(description="Motivo por el cual se transfiere el control a este agente.")

class ToAgentGeneral(BaseModel):
    """Redirige al agente general, responsable de mantener una visión completa del sistema, con acceso a internet y monitoreo global de los demás agentes."""
    reason: str = Field(description="Motivo por el cual se transfiere el control a este agente.")


class ToAgentLab(BaseModel):
    """Redirige al agente de laboratorio, encargado de gestionar sensores, cámaras, bases de datos científicas y la creación de agentes especializados para cada proyecto."""
    reason: str = Field(description="Motivo por el cual se transfiere el control a este agente.") 
    
class ToAgentIndustrial(BaseModel):
    """Redirige al agente industrial, con dominio de ingeniería, PLCs, automatización, robots y acceso a bases de datos técnicas e industriales avanzadas."""
    reason: str = Field(description="Motivo por el cual se transfiere el control a este agente.")
                           # sin tools por ahora

general_runnable     = general_prompt   | general_llm
education_runnable   = education_prompt | education_llm
lab_runnable         = lab_prompt       | lab_llm
industrial_runnable  = industrial_prompt| industrial_llm

def general_agent_node(state: State):     return {"messages": general_runnable.invoke(state)}
def education_agent_node(state: State):   return {"messages": education_runnable.invoke(state)}
def lab_agent_node(state: State):         return {"messages": lab_runnable.invoke(state)}
def industrial_agent_node(state: State):  return {"messages": industrial_runnable.invoke(state)}

#Agregar el perfil del usuario como intial node
def initial_node(state: State) -> State:
    # Inicializar current_agent si no existe
        
    if state.get("profile_summary"):
        return state
    user_info = "luis.torres@tec.mx"   # viene del initial_node
    if not user_info:
        state["profile_summary"] = "ERROR: no se proporcionó user_info"
        return state
    
    summary = get_student_profile(user_info)
    state["profile_summary"] = summary
    return state

def initial_routing(state: State) -> Literal["router", "lab_agent_node", "education_agent_node", "general_agent_node", "industrial_agent_node"]:
    
    current_agent = state.get("current_agent")

    if not isinstance(current_agent, list) or not current_agent:
        print("No current agent, routing to router")
        return "router"
    else:
      print(f"Current agent found: {len(current_agent)} items")
      print(f"Type of last agent: {type(current_agent[-1])}, value: {current_agent[-1]}")
    return current_agent[-1]  # continue with current agent


def intitial_route_function(state: State) -> Literal["ToAgentEducation", "ToAgentIndustrial", "ToAgentGeneral", "ToAgentLab", "__end__"]:
  tools = tools_condition(state)

  if tools == END:
    return END

  tool_calls = state["messages"][-1].tool_calls
  print(tool_calls)

  if tool_calls:
    if tool_calls[0]["name"] == ToAgentEducation.__name__:
      return "ToAgentEducation"
    if tool_calls[0]["name"] == ToAgentIndustrial.__name__:
      return "ToAgentIndustrial"
    if tool_calls[0]["name"] == ToAgentGeneral.__name__:
       return "ToAgentGeneral"
    if tool_calls[0]["name"] == ToAgentLab.__name__:
       return "ToAgentLab"
  return "__end__"

def create_entry_node(assistant_name: str, current_agent: str) -> Callable:
  def entry_node(state: State) -> dict:
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    # Defensive: ensure we return a string (the graph expects a single
    # agent id as the 'right' value). If a list was passed accidentally,
    # take the last string element.
    ca = current_agent
    if isinstance(ca, list):
      ca_list = [x for x in ca if isinstance(x, str)]
      if not ca_list:
        ca = None
      else:
        ca = ca_list[-1]

    if ca is None:
      # nothing sensible to set; don't set the key and let the graph keep
      # the current stack unchanged
      return {
        "messages": [
            ToolMessage(
              tool_call_id=tool_call_id,
              content=f'You are now the {assistant_name}. Reflect on the above conversation and continue with the users intent'
          )
        ]
      }

    return {
      "messages": [
          ToolMessage(
            tool_call_id=tool_call_id,
            content=f'You are now the {assistant_name}. Reflect on the above conversation and continue with the users intent'
        )
      ],
      "current_agent": ca
    }

  return entry_node



def pop_current_agent(state: State) -> dict:
  """This function pops the current agent value and returns to the previous one or main assistant."""
  messages = []
  if state["messages"][-1].tool_calls:
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    messages.append(
        ToolMessage(
            tool_call_id=tool_call_id,
            content="Resume conversation with the main assistant. Reflect on the above conversation and assist the user with their needs."
        )
    )

  return {
      "current_agent": "pop",
      "messages": messages
  }

def agent_tool_handling(state: State) -> Literal["leave_agent", "__end__"]:
  tools = tools_condition(state)

  if tools == END:
    return END

  tool_calls = state["messages"][-1].tool_calls

  did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)

  if did_cancel:
    return "leave_agent"

  return "__end__"

agent_route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """#MAIN GOAL
            Eres un agente especializado en acuerdos de confidencialidad (NDA).
            Tu función principal es identificar el tipo de información que el usuario está proporcionando y enrutarlo al agente más adecuado para completar el proceso del NDA."""),

        ("system", """#BEHAVIOUR
            Analiza cuidadosamente los mensajes del usuario. A partir del contenido, determina si el usuario está:

        - Describiendo el **contexto, propósito o tipo de NDA** → Enruta a **Agent_A**
        - Proporcionando **información de las partes involucradas** (nombres, RFCs, domicilios, etc.) → Enruta a **Agent_B**
        - Indicando el **alcance de la información confidencial** (amplio vs específico, categorías, etc.) → Enruta a **Agent_C**

        Haz preguntas de seguimiento si la información es ambigua, pero si es clara, enruta directamente.

        Este es el usuario actual: {profile_summary}

        """),

        ("placeholder", "{messages}")
    ]
)
  
graph = StateGraph(State)


graph.add_node("initial_node", initial_node)
graph.add_conditional_edges("initial_node", initial_routing)
graph.set_entry_point("initial_node")

# we define the route agent with tools for routing
agent_route_runnable = agent_route_prompt | llm.bind_tools([ToAgentEducation, ToAgentGeneral, ToAgentLab, ToAgentIndustrial])
graph.add_node("router", Assistant(agent_route_runnable))
graph.add_conditional_edges(
    "router", #nodo de inicio
    intitial_route_function,
)

graph.add_conditional_edges("general_agent_node", agent_tool_handling)
graph.add_conditional_edges("education_agent_node", agent_tool_handling)
graph.add_conditional_edges("lab_agent_node", agent_tool_handling)
graph.add_conditional_edges("industrial_agent_node", agent_tool_handling)

graph.add_node("general_agent_node",     general_agent_node)
graph.add_node("education_agent_node",   education_agent_node)
graph.add_node("lab_agent_node",         lab_agent_node)
graph.add_node("industrial_agent_node",  industrial_agent_node)

graph.add_node("ToAgentEducation", create_entry_node("Contexto, Propósito y tipo de NDA", "education_agent_node"))
graph.add_node("ToAgentGeneral", create_entry_node("Información de las partes involucradas", "general_agent_node"))
graph.add_node("ToAgentLab", create_entry_node("Alcance de la información confidencial", "lab_agent_node"))
graph.add_node("ToAgentIndustrial", create_entry_node("Alcance de la información confidencial", "industrial_agent_node"))

graph.add_edge("ToAgentEducation", "education_agent_node")
graph.add_edge("ToAgentGeneral", "general_agent_node")
graph.add_edge("ToAgentLab", "lab_agent_node")
graph.add_edge("ToAgentIndustrial", "industrial_agent_node")


graph.add_node("leave_agent", pop_current_agent)
graph.add_edge("leave_agent", "router")













