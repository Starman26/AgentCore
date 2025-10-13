from typing_extensions import TypedDict
from typing import Annotated, Literal
from langgraph.types import Runnable
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph.message import AnyMessage, add_messages
from pydantic.v1 import BaseModel, Field
from langchain_core.tools import tool
from rag.rag_logic import create_or_update_vectorstore
from langgraph.prebuilt import tools_condition, ToolNode

# -------------------------
# Estado base con tipo estricto de intención
# -------------------------
IntentType = Literal["EDUCATION", "LAB", "INDUSTRIAL", "NOT_IDENTIFIED"]

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_node: IntentType  # Tipo estricto para intención

class SupervisorOutput(BaseModel):
    next_node: IntentType


# -------------------------
# LLM base
# -------------------------
llm = ChatOpenAI(model="gpt-4o-mini")

# -------------------------
# Prompts de clasificación
# -------------------------
intent_prompt_template = ChatPromptTemplate.from_messages([
    {"role": "system", "content": (
         "You are a user intent classifier. "
        "There are 3 categories:\n"
        "1. EDUCATION: The user wants to learn something new or follow an educational roadmap related to PLC or industrial topics.\n"
        "2. LAB: The user wants to identify existing lab errors or add new ones.\n"
        "3. INDUSTRIAL: The user is seeking general knowledge or has a question about PLCs and industrial machines.\n"
        "If the message does not match any of these categories, respond with 'NOT_IDENTIFIED'.\n"
        "Return only the corresponding category name."
    )},
    {"role": "user", "content": "{messages}"}
])
llm_runnable: Runnable = intent_prompt_template | llm.with_structured_output(SupervisorOutput)

# -------------------------
# Nodo de clasificación
# -------------------------
def classify_intent_node(state: State):
    state = {**state}
    result = llm_runnable.invoke(state)
    return {
        "next_node": result.next_node
    }

# -------------------------
# Prompts de los agentes
# -------------------------
education_prompt = ChatPromptTemplate.from_messages([
    {"role": "system", "content": (
        "Eres un asistente educativo. Genera un texto claro, didáctico y estructurado para que el usuario aprenda algo nuevo "
        "sobre PLC o temas industriales. Sé breve pero completo."
    )},
    {"role": "user", "content": "{messages}"}
])


education_runnable: Runnable = education_prompt | llm

@tool
def retrieve_context(query: str):
    """Search for relevant robot problem records from persistent or auto-created vector DB."""
    vectorstore = create_or_update_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    results = retriever.invoke(query)
    
    # Construir respuesta legible
    response = []
    for doc in results:
        meta = doc.metadata
        response.append(
            f"📅 {meta['created_at']} | 🤖 {meta['robot_type']} | 🧠 {meta['problem_title']} | 👷 {meta['author']}\n"
            f"{doc.page_content}\n"
        )
    
    return "\n".join(response)

lab_prompt = ChatPromptTemplate.from_messages([
    {"role": "system", "content": (
        "Eres un asistente de laboratorio. Ayuda a identificar errores o sugiere nuevos errores de manera segura. "
        "Explica las posibles soluciones y causas de manera clara."
    )},
    {"role": "user", "content": "{messages}"}
])
tools = [retrieve_context]
llm_with_tools = llm.bind_tools(tools)
lab_runnable: Runnable = lab_prompt | llm_with_tools

industrial_prompt = ChatPromptTemplate.from_messages([
    {"role": "system", "content": (
        "Eres un asistente de conocimiento industrial. Proporciona información clara y precisa sobre PLC, máquinas o procesos industriales. "
        "Responde de manera general y profesional."
    )},
    {"role": "user", "content": "{messages}"}
])
industrial_runnable: Runnable = industrial_prompt | llm

# -------------------------
# Nodos de los agentes
# -------------------------
def education_agent_node(state: State):
    state = {**state}

    response = education_runnable.invoke(state)
    
    return {"messages": response}

def lab_agent_node(state: State):
    state = {**state}

    response = lab_runnable.invoke(state)
    
    return {"messages": response}

def industrial_agent_node(state: State):
    state = {**state}

    response = industrial_runnable.invoke(state)
  
    return {"messages": response}

# -------------------------
# Grafo completo
# -------------------------
graph = StateGraph(State)

graph.add_node("classify_intent_node", classify_intent_node)
graph.add_node("education_agent_node", education_agent_node)
graph.add_node("lab_agent_node", lab_agent_node)
graph.add_node("industrial_agent_node", industrial_agent_node)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("classify_intent_node")
graph.add_conditional_edges(
    "lab_agent_node",
    tools_condition,
)

graph.add_edge("tools", "lab_agent_node")

graph.add_conditional_edges(
    "classify_intent_node",
    lambda state: state["next_node"],
    {
        "EDUCATION": "education_agent_node",
        "LAB": "lab_agent_node",
        "INDUSTRIAL": "industrial_agent_node",
        "NOT_IDENTIFIED": END,
    }
)
