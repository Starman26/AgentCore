from typing_extensions import TypedDict
from typing import Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition, ToolNode
from dotenv import load_dotenv; load_dotenv()
import os

from Settings.prompts import intent_prompt, general_prompt, education_prompt, lab_prompt, industrial_prompt
from Settings.tools import retrieve_context,get_student_profile, update_student_goals, update_learning_style, route_to
from Settings.state import State

GENERAL_TOOLS = [get_student_profile, update_student_goals, update_learning_style, route_to]
LAB_TOOLS     = [retrieve_context, route_to]
EDU_TOOLS       = [get_student_profile, update_learning_style, route_to]

# ---------- LLM base ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY en .env")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0, request_timeout=30)

# Clasificador con salida estructurada
intent_llm = llm
llm_runnable = intent_prompt | intent_llm

def classify_intent_node(state: State):
    label = llm_runnable.invoke({"messages": state["messages"]}).content.strip().upper()
    label = {"EDUCATION","LAB","INDUSTRIAL","GENERAL"} & {label} and label or "GENERAL"
    return {"next_node": label}

# ---------- Runnables por agente ----------
general_llm      = llm.bind_tools(GENERAL_TOOLS)
education_llm    = llm.bind_tools(EDU_TOOLS)     # para leer/actualizar perfil
lab_llm          = llm.bind_tools(LAB_TOOLS)
industrial_llm   = llm                           # sin tools por ahora

general_runnable     = general_prompt   | general_llm
education_runnable   = education_prompt | education_llm
lab_runnable         = lab_prompt       | lab_llm
industrial_runnable  = industrial_prompt| industrial_llm

def general_agent_node(state: State):     return {"messages": general_runnable.invoke({"messages": state["messages"]})}
def education_agent_node(state: State):   return {"messages": education_runnable.invoke({"messages": state["messages"]})}
def lab_agent_node(state: State):         return {"messages": lab_runnable.invoke({"messages": state["messages"]})}
def industrial_agent_node(state: State):  return {"messages": industrial_runnable.invoke({"messages": state["messages"]})}
def router_from_general(state: State):
    msg = state["messages"][-1].content
    route = msg.split("::",1)[1] if isinstance(msg,str) and msg.startswith("ROUTE::") else None
    return {"route_request": route}

def router_from_education(state: State):
    msg = state["messages"][-1].content
    route = msg.split("::",1)[1] if isinstance(msg,str) and msg.startswith("ROUTE::") else None
    return {"route_request": route}

def router_from_lab(state: State):
    msg = state["messages"][-1].content
    route = msg.split("::",1)[1] if isinstance(msg,str) and msg.startswith("ROUTE::") else None
    return {"route_request": route}

graph = StateGraph(State)

graph.add_node("classify_intent_node", classify_intent_node)
graph.add_node("general_agent_node",     general_agent_node)
graph.add_node("education_agent_node",   education_agent_node)
graph.add_node("lab_agent_node",         lab_agent_node)
graph.add_node("industrial_agent_node",  industrial_agent_node)
graph.add_node("router_general",   router_from_general)
graph.add_node("router_education", router_from_education)
graph.add_node("router_lab",       router_from_lab)

# ToolNodes por agente (limpio y seguro)
graph.add_node("general_tools",   ToolNode(GENERAL_TOOLS))
graph.add_node("education_tools", ToolNode(EDU_TOOLS))
graph.add_node("lab_tools",       ToolNode(LAB_TOOLS))

graph.set_entry_point("classify_intent_node")

# Invocación de tools por agente
graph.add_conditional_edges("general_agent_node",   tools_condition, {"tools": "general_tools",   "__end__": END})
graph.add_conditional_edges("education_agent_node", tools_condition, {"tools": "education_tools", "__end__": END})
graph.add_conditional_edges("lab_agent_node",       tools_condition, {"tools": "lab_tools",       "__end__": END})
graph.add_conditional_edges("industrial_agent_node", tools_condition, {"__end__": END})  # sin tools

# ToolNodes -> routers (un solo camino)
graph.add_edge("general_tools",   "router_general")
graph.add_edge("education_tools", "router_education")
graph.add_edge("lab_tools",       "router_lab")

# Routers deciden destino o regresan a su agente si no hubo route_to
graph.add_conditional_edges(
    "router_general",
    lambda s: s.get("route_request") or "__back__",
    {"GENERAL":"general_agent_node","EDUCATION":"education_agent_node","LAB":"lab_agent_node",
     "INDUSTRIAL":"industrial_agent_node","__back__":"general_agent_node"}
)
graph.add_conditional_edges(
    "router_education",
    lambda s: s.get("route_request") or "__back__",
    {"GENERAL":"general_agent_node","EDUCATION":"education_agent_node","LAB":"lab_agent_node",
     "INDUSTRIAL":"industrial_agent_node","__back__":"education_agent_node"}
)
graph.add_conditional_edges(
    "router_lab",
    lambda s: s.get("route_request") or "__back__",
    {"GENERAL":"general_agent_node","EDUCATION":"education_agent_node","LAB":"lab_agent_node",
     "INDUSTRIAL":"industrial_agent_node","__back__":"lab_agent_node"}
)


# Clasificador
graph.add_conditional_edges(
    "classify_intent_node",
    lambda s: s["next_node"],
    {"EDUCATION":"education_agent_node","LAB":"lab_agent_node","INDUSTRIAL":"industrial_agent_node",
     "GENERAL":"general_agent_node","NOT_IDENTIFIED":"general_agent_node"}
)