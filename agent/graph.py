from typing_extensions import TypedDict
from typing import Annotated, Literal, Optional, List, Callable
from datetime import datetime
from zoneinfo import ZoneInfo
import locale
import os
import re

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langchain_core.runnables.config import RunnableConfig

from pydantic.v1 import BaseModel, Field

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
    update_student_goals,
    update_learning_style,
    route_to,
    current_datetime,
    _submit_chat_history,
    get_student_profile,
    check_user_exists,
    register_new_student,
    update_student_info,
    _fetch_student,
    summarize_all_chats,
    retrieve_robot_support,
    get_project_tasks,
    get_task_steps,
    get_task_step_images,
    search_manual_images,
    complete_task_step,
    identify_user_from_message,
    search_in_db,
    list_agent_tables,
    describe_agent_table,
    gather_rag_context,
)

# Carga de entorno
load_dotenv()


# =====================================================
# Helpers para stack de agentes
# =====================================================
def update_current_agent_stack(left: list[str], right: Optional[str]) -> list[str]:
    if right is None:
        return left

    if isinstance(right, list):
        right_list = [r for r in right if isinstance(r, str)]
        if not right_list:
            print(
                "update_current_agent_stack: received empty/non-string list as right; ignoring"
            )
            return left
        return left + right_list

    if right == "pop":
        return left[:-1]

    if not isinstance(right, str):
        print(
            f"update_current_agent_stack: unexpected type for right: {type(right)}; coercing to str"
        )
        return left + [str(right)]

    return left + [right]


# =====================================================
# Estado del grafo
# =====================================================
class State(TypedDict, total=False):
    # Historial de mensajes
    messages: Annotated[List[AnyMessage], add_messages]

    # Perfil
    profile_summary: Optional[str]

    # Tiempo / zona horaria
    tz: str
    now_utc: str
    now_local: str
    now_human: str

    # Estilo del avatar
    avatar_style: Optional[str]

    # Config del widget (desde frontend o BD)
    widget_avatar_id: Optional[str]       # "cat" | "robot" | "duck" | "lab" | "astro" | "cora"
    widget_mode: Optional[str]            # "default" | "custom"
    widget_personality: Optional[str]
    widget_notes: Optional[str]

    # Pila de agentes activos
    current_agent: Annotated[
        List[
            Literal[
                "education_agent_node",
                "general_agent_node",
                "lab_agent_node",
                "industrial_agent_node",
                "router",
            ]
        ],
        update_current_agent_stack,
    ]

    # Info de usuario / sesión
    user_identified: Optional[bool]
    user_email: Optional[str]
    user_name: Optional[str]
    session_id: Optional[str]
    awaiting_user_info: Optional[str]
    team_id: Optional[str]  

    # Título de la sesión (para el frontend / Supabase)
    session_title: Optional[str]

    # Contexto de prácticas / proyecto
    chat_type: Optional[str]          # "practice", "general", etc.
    project_id: Optional[str]         # projects.id
    current_task_id: Optional[str]    # project_tasks.id
    current_step_number: Optional[int]
    practice_completed: Optional[bool]


# =====================================================
# Modelos de tools / control de flujo
# =====================================================
class CompleteOrEscalate(BaseModel):
    reason: str = Field(description="Motivo para finalizar o escalar.")
    cancel: bool = Field(
        default=False, description="True=cierra; False=continúa/escalado."
    )


class ToAgentEducation(BaseModel):
    reason: str = Field(
        description="Motivo de transferencia al agente educativo."
    )


class ToAgentGeneral(BaseModel):
    reason: str = Field(
        description="Motivo de transferencia al agente general."
    )


class ToAgentLab(BaseModel):
    reason: str = Field(
        description="Motivo de transferencia al agente de laboratorio."
    )


class ToAgentIndustrial(BaseModel):
    reason: str = Field(
        description="Motivo de transferencia al agente industrial."
    )


# =====================================================
# Helper: construir estilo según avatar
# =====================================================
def build_avatar_style(
    student: Optional[dict],
    override_avatar_id: Optional[str] = None,
    override_mode: Optional[str] = None,
    override_personality: Optional[str] = None,
    override_notes: Optional[str] = None,
) -> str:
    """
    Devuelve un texto con instrucciones de estilo para el asistente según el
    avatar seleccionado y, si existen, las personalizaciones guardadas
    en la tabla students (widget_*).
    """
    student = student or {}

    avatar_id = (
        override_avatar_id
        or student.get("widget_avatar_id")
        or "cora"   # default
    )
    mode = override_mode or student.get("widget_mode") or "default"
    custom_personality = (
        override_personality
        or student.get("widget_personality")
        or ""
    )
    custom_notes = override_notes or student.get("widget_notes") or ""

    # Estilos base por avatar
    if avatar_id == "cat":
        base_style = (
            "Modo Gato Analítico:\n"
            "- Tono tranquilo, cálido y paciente.\n"
            "- Prefiere explicaciones claras, ordenadas y con ejemplos cuando hagan falta.\n"
            "- Puedes hacer referencias suaves a gatos (curiosidad, flexibilidad, etc.) "
            "solo cuando encaje de forma natural, pero evita repetir siempre la misma palabra o sonido."
        )

    elif avatar_id == "robot":
        base_style = (
            "Modo Robot Industrial:\n"
            "- Tono técnico, claro y directo.\n"
            "- Prefiere listas y pasos cuando aportan claridad.\n"
            "- No uses frases de cierre fijas; adapta el final según la situación."
        )

    elif avatar_id == "duck":
        base_style = (
            "Modo Pato Creativo:\n"
            "- Tono imaginativo, optimista y con buena energía.\n"
            "- Usa ejemplos creativos pero mantén la precisión profesional.\n"
            "- Puedes mencionar patos o usar humor ligero ocasionalmente, "
            "pero sin repetir siempre 'cuack' ni un emoji específico."
        )

    elif avatar_id == "lab":
        base_style = (
            "Modo Asistente de Laboratorio:\n"
            "- Tono metódico, técnico y seguro.\n"
            "- Prefiere pasos, orden y buenas prácticas.\n"
            "- Puedes cerrar con una pregunta orientada a la acción cuando tenga sentido, "
            "no como obligación fija."
        )

    elif avatar_id == "astro":
        base_style = (
            "Modo Explorador XR:\n"
            "- Tono curioso, futurista y con analogías espaciales suaves.\n"
            "- Usa referencias a exploración o misiones solo cuando aporten claridad.\n"
            "- No repitas siempre la misma frase al final; mantén variedad natural."
        )

    else:
        base_style = (
            "Modo Cora (básico):\n"
            "- Tono profesional, amable y claro.\n"
            "- Priorizas neutralidad y precisión."
        )

    extra = ""
    if mode == "custom":
        if custom_personality:
            extra += (
                "\n\nInstrucciones personalizadas de personalidad definidas por el usuario:\n"
                f"{custom_personality}"
            )
        if custom_notes:
            extra += (
                "\n\nNotas adicionales del usuario sobre el comportamiento del asistente:\n"
                f"{custom_notes}"
            )

    return base_style + extra


# =====================================================
# LLM base
# =====================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY en .env")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0,
    request_timeout=30,
)


# =====================================================
# Tools por agente
# =====================================================
GENERAL_TOOLS = [
    CompleteOrEscalate,
    web_research,
    get_student_profile,
    update_student_goals,
    update_learning_style,
    retrieve_context,
    summarize_all_chats,
    route_to,
    current_datetime,
    identify_user_from_message,
    gather_rag_context, 
]

EDU_TOOLS = [
    CompleteOrEscalate,
    web_research,
    get_student_profile,
    update_learning_style,
    retrieve_context,
    route_to,
    current_datetime,
    get_project_tasks,
    get_task_steps,
    get_task_step_images,
    search_manual_images,
    complete_task_step,
    identify_user_from_message,
    gather_rag_context, 
]

LAB_TOOLS = [
    CompleteOrEscalate,
    web_research,
    retrieve_context,
    retrieve_robot_support,
    route_to,
    current_datetime,
    identify_user_from_message,
    gather_rag_context, 
]

IND_TOOLS = [
    CompleteOrEscalate,
    web_research,
    retrieve_context,
    retrieve_robot_support,
    route_to,              # <-- añadir
    current_datetime,
    identify_user_from_message,
    gather_rag_context, 
]



# =====================================================
# Runnables por agente
# =====================================================
general_llm = llm.bind_tools(GENERAL_TOOLS)
education_llm = llm.bind_tools(EDU_TOOLS)
lab_llm = llm.bind_tools(LAB_TOOLS)
industrial_llm = llm.bind_tools(IND_TOOLS)

identification_llm = llm.bind_tools(
    [check_user_exists, register_new_student, update_student_info]
)

general_runnable = general_prompt | general_llm
education_runnable = education_prompt | education_llm
lab_runnable = lab_prompt | lab_llm
industrial_runnable = industrial_prompt | industrial_llm
identification_runnable = identification_prompt | identification_llm


# =====================================================
# Helpers de ejecución de agentes
# =====================================================
def _invoke_runnable_as_messages(runnable, state: State) -> dict:
    """Envuelve la salida del runnable como lista de mensajes nuevos."""
    result = runnable.invoke(state)
    if isinstance(result, list):
        msgs = result
    else:
        msgs = [result]
    return {"messages": msgs}


def general_agent_node(state: State):
    return _invoke_runnable_as_messages(general_runnable, state)


def education_agent_node(state: State):
    return _invoke_runnable_as_messages(education_runnable, state)


def lab_agent_node(state: State):
    return _invoke_runnable_as_messages(lab_runnable, state)


def industrial_agent_node(state: State):
    return _invoke_runnable_as_messages(industrial_runnable, state)


# =====================================================
# Identificación de usuario
# =====================================================
def identify_user_node(state: State):
    """
    Identifica al usuario pidiendo nombre/correo si no está identificado.
    """
    if state.get("user_identified"):
        return {}

    messages = state.get("messages", [])
    has_asked_for_info = False

    for msg in messages:
        if hasattr(msg, "type") and msg.type == "ai":
            content = getattr(msg, "content", "")
            content_lower = str(content).lower()
            if (
                ("nombre" in content_lower and "correo" in content_lower)
                or ("carrera" in content_lower or "habilidades" in content_lower)
            ):
                has_asked_for_info = True
                break

    if not has_asked_for_info:
        from langchain_core.messages import AIMessage

        return {
            "messages": [
                AIMessage(
                    content=(
                        "¡Hola! Para poder ayudarte mejor, necesito conocerte primero. "
                        "¿Podrías decirme tu nombre completo y correo electrónico?"
                    )
                )
            ],
            "awaiting_user_info": "name_email",
        }

    result = identification_runnable.invoke(state)
    if isinstance(result, list):
        msgs = result
    else:
        msgs = [result]
    return {"messages": msgs}


def check_identification_status(
    state: State,
) -> Literal["identified", "tools", "await_user"]:
    """
    Verifica el estado de identificación y decide el siguiente paso.
    """
    messages = state.get("messages", [])

    if not state.get("user_identified"):
        if (
            messages
            and hasattr(messages[-1], "tool_calls")
            and messages[-1].tool_calls
        ):
            return "tools"

        if messages and hasattr(messages[-1], "type") and messages[-1].type == "ai":
            return "await_user"

        return "await_user"

    if messages and hasattr(messages[-1], "type") and messages[-1].type == "ai":
        last_content = getattr(messages[-1], "content", "")
        if (
            "Ya te tengo identificado" in str(last_content)
            or "¿En qué puedo ayudarte" in str(last_content)
        ):
            return "await_user"

    return "identified"


def check_after_identification_tools(
    state: State,
) -> Literal["identified", "continue_identifying", "await_user"]:
    """
    Después de usar herramientas de identificación, decide el siguiente paso.
    """
    messages = state.get("messages", [])
    if not messages:
        return "await_user"

    if state.get("user_identified"):
        last_msg = messages[-1]
        if hasattr(last_msg, "type") and last_msg.type == "ai":
            content = getattr(last_msg, "content", "")
            if (
                "Ya te tengo identificado" in str(content)
                or "¿En qué puedo ayudarte" in str(content)
            ):
                return "await_user"
        return "identified"

    last_msg = messages[-1]

    if hasattr(last_msg, "content") and getattr(last_msg, "content") == "NOT_FOUND":
        return "continue_identifying"

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "continue_identifying"

    if hasattr(last_msg, "type") and last_msg.type == "ai":
        return "await_user"

    return "continue_identifying"


def process_identification_tools(state: State):
    """
    Procesa las tool calls de identificación y actualiza el estado.
    """
    from langchain_core.messages import AIMessage

    messages = state.get("messages", [])
    if (
        not messages
        or not hasattr(messages[-1], "tool_calls")
        or not messages[-1].tool_calls
    ):
        return {}

    tool_call = messages[-1].tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call.get("args", {})
    tool_call_id = tool_call["id"]

    result_content = ""
    email = None

    try:
        if tool_name == "check_user_exists":
            email = tool_args.get("email", "")
            result_content = check_user_exists.invoke({"email": email})
        elif tool_name == "register_new_student":
            email = tool_args.get("email", "")
            result_content = register_new_student.invoke(tool_args)
        elif tool_name == "update_student_info":
            email = tool_args.get("email", "")
            result_content = update_student_info.invoke(tool_args)
    except Exception as e:
        result_content = f"ERROR:{str(e)}"

    tool_message = ToolMessage(content=result_content, tool_call_id=tool_call_id)

    if result_content == "OK" or "EXISTS:" in str(result_content):
        if email:
            student = _fetch_student(email)
            if student:
                profile_summary = get_student_profile.invoke(
                    {"name_or_email": email}
                )
                confirmation_msg = AIMessage(
                    content=(
                        f"¡Perfecto, {student.get('full_name', 'usuario')}! "
                        "Ya te tengo identificado. ¿En qué puedo ayudarte hoy?"
                    )
                )
                return {
                    "messages": [tool_message, confirmation_msg],
                    "user_identified": True,
                    "user_email": email,
                    "user_name": student.get("full_name", ""),
                    "profile_summary": profile_summary,
                    "awaiting_user_info": None,
                }

    return {"messages": [tool_message]}


# =====================================================
# Nodo inicial: tiempo + perfil + estilo de avatar
# =====================================================
def _inject_time_fields(state: State) -> None:
    tz = state.get("tz") or "America/Monterrey"
    state["tz"] = tz

    try:
        locale.setlocale(locale.LC_TIME, "es_MX.UTF-8")
    except Exception:
        pass

    now_local_dt = datetime.now(ZoneInfo(tz))
    state["now_local"] = now_local_dt.isoformat()
    state["now_utc"] = datetime.utcnow().isoformat() + "Z"
    state["now_human"] = now_local_dt.strftime("%A, %d %b %Y, %H:%M")


def initial_node(state: State, config: RunnableConfig) -> State:
    """
    Inyecta tiempo, session_id, perfil, project/team context y avatar_style por defecto.
    """
    # Clonar para no mutar el state original
    state = dict(state or {})

    # 1) Tiempo / zona horaria
    _inject_time_fields(state)

    # 2) Perfil básico por defecto
    if not state.get("profile_summary"):
        state["profile_summary"] = "Perfil aún no registrado."

    # 3) Conectar session_id con thread_id si viene desde config
    configurable = config.get("configurable", {}) if config else {}

    if not state.get("session_id"):
        thread_id = configurable.get("thread_id")
        if thread_id:
            state["session_id"] = thread_id

    # 4) Si el usuario ya está identificado, cargar su perfil completo
    student = None
    if state.get("user_identified") and state.get("user_email"):
        user_info = state.get("user_email")
        try:
            # Resumen legible del perfil
            summary = get_student_profile.invoke({"name_or_email": user_info})
            state["profile_summary"] = summary
        except Exception as e:
            print(f"[initial_node] Error en get_student_profile: {e}")

        try:
            # Registro completo para sacar avatar, team, project, etc.
            student = _fetch_student(user_info)
        except Exception as e:
            print(f"[initial_node] Error al traer student para avatar: {e}")

    # 5) Inyectar contexto de proyecto / equipo desde student (si existe)
    if student:
        # Solo setea si aún no vienen en el state
        if not state.get("project_id"):
            # Usa el campo que tengas en tu tabla (ajusta nombres si es distinto)
            state["project_id"] = (
                student.get("default_project_id")
                or student.get("project_id")
                or state.get("project_id")
            )
        if not state.get("team_id"):
            state["team_id"] = student.get("team_id") or state.get("team_id")

        # También podrías inferir un chat_type por defecto desde el student si quisieras
        # (ej. según curso / práctica), pero lo dejo opcional:
        # if not state.get("chat_type") and student.get("default_chat_type"):
        #     state["chat_type"] = student["default_chat_type"]

    # 6) Sobrescribir chat_type / project_id / team_id desde el frontend (configurable)
    #    → lo que mande el frontend tiene prioridad sobre lo inferido
    if configurable.get("chat_type") and not state.get("chat_type"):
        state["chat_type"] = configurable["chat_type"]

    if configurable.get("project_id") and not state.get("project_id"):
        state["project_id"] = configurable["project_id"]

    if configurable.get("team_id") and not state.get("team_id"):
        state["team_id"] = configurable["team_id"]

    # 7) Overrides de avatar / modo desde state o config.configurable
    override_avatar_id = (
        state.get("widget_avatar_id")
        or configurable.get("avatar_id")
        or configurable.get("widget_avatar_id")
    )
    override_mode = (
        state.get("widget_mode")
        or configurable.get("widget_mode")
    )
    override_personality = (
        state.get("widget_personality")
        or configurable.get("widget_personality")
    )
    override_notes = (
        state.get("widget_notes")
        or configurable.get("widget_notes")
    )

    # 8) Construir y fijar avatar_style
    state["avatar_style"] = build_avatar_style(
        student=student,
        override_avatar_id=override_avatar_id,
        override_mode=override_mode,
        override_personality=override_personality,
        override_notes=override_notes,
    )

    return state



# =====================================================
# Guardado de historial
# =====================================================
def _flatten_message_content(content) -> str:
    """Convierte content (str o lista de bloques) en texto plano."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text_parts.append(str(item.get("text", "")))
            else:
                text_parts.append(str(item))
        return " ".join(text_parts).strip()

    return str(content)


def save_user_input(state: State):
    """Guarda el input del usuario en la BD."""
    session_id = state.get("session_id")
    if not session_id:
        return {}

    msgs = state.get("messages") or []
    if not msgs:
        return {}

    last = msgs[-1]

    if hasattr(last, "type") and hasattr(last, "content"):
        role = "student" if last.type == "human" else "agent"
        content = _flatten_message_content(last.content)
    else:
        role = "student"
        content = last.get("content") if isinstance(last, dict) else str(last)

    user_email = state.get("user_email")

    try:
        _submit_chat_history(
            session_id=session_id,
            role=role,
            content=content,
            user_email=user_email,
        )
    except Exception as e:
        print(f"[save_user_input] Error al guardar chat: {e}")

    return {}


def generate_session_title_from_history(messages: List[AnyMessage]) -> str:
    """
    Genera un título breve usando el primer mensaje del usuario.
    """
    first_user_text = None

    for msg in messages:
        msg_type = getattr(msg, "type", None) or getattr(msg, "role", None)
        if msg_type in ("human", "user", "student"):
            first_user_text = _flatten_message_content(
                getattr(msg, "content", "")
            )
            if first_user_text:
                break

    if not first_user_text:
        return "Sesión sin título"

    first_user_text = first_user_text.strip()
    if len(first_user_text) > 60:
        first_user_text = first_user_text[:60] + "…"

    return first_user_text or "Sesión sin título"


def save_agent_output(state: State):
    """
    Guarda el output del agente en la BD y, si existe,
    también el último ToolMessage (contexto RAG usado).
    Además genera un título de sesión basado en el historial.
    """
    session_id = state.get("session_id")
    if not session_id:
        return {}

    msgs = state.get("messages") or []
    if not msgs:
        return {}

    user_email = state.get("user_email")

    # --- 1) Identificar qué mensajes guardar en este turno ---
    to_persist = []

    # Último mensaje (normalmente el AI final)
    last = msgs[-1]
    if hasattr(last, "type") and hasattr(last, "content"):
        role = "agent"  # este siempre queremos que sea la respuesta final
        content = _flatten_message_content(last.content)
        to_persist.append((role, content))

    # Mensaje anterior: si es ToolMessage, guárdalo como 'tool'
    if len(msgs) >= 2:
        prev = msgs[-2]
        prev_type = getattr(prev, "type", None)
        if prev_type == "tool":
            tool_content = _flatten_message_content(getattr(prev, "content", ""))
            # Si quieres distinguir, puedes meter prefijo [TOOL] aquí
            to_persist.insert(0, ("tool", tool_content))

    # --- 2) Insertar en chat_message ---
    for role, content in to_persist:
        try:
            _submit_chat_history(
                session_id=session_id,
                role=role,          # 'agent' o 'tool'
                content=content,
                user_email=user_email,
            )
        except Exception as e:
            print(f"[save_agent_output] Error al guardar chat ({role}): {e}")

    # --- 3) Generar título de sesión para el frontend ---
    try:
        title = generate_session_title_from_history(msgs)
    except Exception as e:
        print(f"[save_agent_output] Error generando título de sesión: {e}")
        title = None

    if title:
        return {"session_title": title}

    return {}


def initial_routing(state: State) -> Literal["router"]:
    return "router"


# =====================================================
# Router
# =====================================================
def _fallback_pick_agent(text: str) -> str:
    t = str(text).lower()

    if re.search(
        r"\b(plc|robot|hmi|scada|opc|ladder|siemens|allen-bradley|automatización)\b",
        t,
    ):
        return "ToAgentIndustrial"

    if re.search(
        r"\bnda|confidencial|alcance|categorías|clasificar info|laboratorio|experimento|sensor|muestra\b",
        t,
    ):
        return "ToAgentLab"

    if re.search(
        r"\bplan de estudios|tarea|examen|aprender|clase|curso|proyecto escolar|estudio\b",
        t,
    ):
        return "ToAgentEducation"

    if re.search(
        r"\bpartes|rfc|domicilio|contrato|datos de contacto|coordinador|registro\b",
        t,
    ):
        return "ToAgentGeneral"

    return "ToAgentGeneral"


def intitial_route_function(
    state: State,
) -> Literal[
    "ToAgentEducation", "ToAgentIndustrial", "ToAgentGeneral", "ToAgentLab", "__end__"
]:
    from langgraph.prebuilt import tools_condition as _tools_condition

    # 0) Si es chat de práctica, fuerza educación
    chat_type = (state.get("chat_type") or "").lower()
    if chat_type == "practice":
        print("[Router] chat_type='practice' → forzando ToAgentEducation")
        return "ToAgentEducation"

    # 1) Lógica normal de tools_condition
    tools = _tools_condition(state)
    if tools == END:
        return END

    # 2) Si el último mensaje tiene tool_calls de ruteo, respétalos
    tool_calls = getattr(state["messages"][-1], "tool_calls", []) or []
    if tool_calls:
        name = tool_calls[0]["name"]
        if name in {
            "ToAgentEducation",
            "ToAgentIndustrial",
            "ToAgentGeneral",
            "ToAgentLab",
        }:
            return name

    # 3) Fallback por texto
    last_message = getattr(state["messages"][-1], "content", "")
    forced = _fallback_pick_agent(last_message)
    print(f"[Router fallback] No tool call detectada → Dirigiendo a {forced}")
    return forced


router_runnable = agent_route_prompt | llm.bind_tools(
    [ToAgentEducation, ToAgentGeneral, ToAgentLab, ToAgentIndustrial],
    tool_choice="any",
)


class Assistant:
    def __init__(self, runnable):
        self.runnable = runnable

    def __call__(self, state: State, config):
        result = self.runnable.invoke(state)
        if isinstance(result, list):
            msgs = result
        else:
            msgs = [result]
        return {"messages": msgs}


# =====================================================
# ToolNode y helper para identify_user_from_message
# =====================================================
def update_user_from_identify_tool(state: State) -> dict:
    """
    Si la última ejecución de tools incluyó identify_user_from_message
    y devolvió algo tipo 'FOUND:email:nombre',
    actualiza user_email, user_name y profile_summary en el state.
    """
    msgs = state.get("messages") or []
    if not msgs:
        return {}

    last_found = None

    # Buscamos desde el final algún mensaje con contenido 'FOUND:...'
    for msg in reversed(msgs):
        content = getattr(msg, "content", "")
        content = _flatten_message_content(content)
        if isinstance(content, str) and content.startswith("FOUND:"):
            last_found = content.strip()
            break

    if not last_found:
        # No hubo identify_user_from_message o no encontró usuario
        return {}

    try:
        # Formato: FOUND:email:nombre completo
        _, email, full_name = last_found.split(":", 2)
        email = email.strip()
        full_name = full_name.strip()
    except ValueError:
        # Formato raro → ignoramos
        return {}

    new_state: dict = {
        "user_email": email,
        "user_name": full_name,
    }

    # Opcional: actualizar también el profile_summary
    try:
        profile_summary = get_student_profile.invoke({"name_or_email": email})
        if isinstance(profile_summary, str):
            new_state["profile_summary"] = profile_summary
    except Exception as e:
        print("[update_user_from_identify_tool] error get_student_profile:", e)

    # Aquí NO marcamos user_identified=True para no chocar con el pipeline formal
    return new_state


def return_to_current_agent(state: State) -> str:
    """
    Devuelve el nombre del nodo del agente actual en la cima del stack.
    """
    stack = state.get("current_agent") or []
    return stack[-1] if stack else "general_agent_node"


# =====================================================
# Construcción del grafo
# =====================================================
graph = StateGraph(State)

# Nodos base
graph.add_node("initial_node", initial_node)
graph.add_node("identify_user", identify_user_node)
graph.add_node("identification_tools", process_identification_tools)
graph.add_node("save_user_input", save_user_input)
graph.add_node("save_agent_output", save_agent_output)

graph.set_entry_point("initial_node")
graph.add_edge("initial_node", "identify_user")

# Flujo de identificación
graph.add_conditional_edges(
    "identify_user",
    check_identification_status,
    {
        "identified": "save_user_input",
        "tools": "identification_tools",
        "await_user": END,  # Termina y espera respuesta del usuario
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

# Nodo router
graph.add_node("router", Assistant(router_runnable))
graph.add_conditional_edges("router", intitial_route_function)

# Nodos de agentes
graph.add_node("general_agent_node", general_agent_node)
graph.add_node("education_agent_node", education_agent_node)
graph.add_node("lab_agent_node", lab_agent_node)
graph.add_node("industrial_agent_node", industrial_agent_node)

# Nodos de entrada por tool-call del router
def create_entry_node(assistant_name: str, current_agent: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        ca = current_agent

        if isinstance(ca, list):
            ca_list = [x for x in ca if isinstance(x, str)]
            ca = ca_list[-1] if ca_list else None

        msg = ToolMessage(
            tool_call_id=tool_call_id,
            content=(
                f"Ahora eres {assistant_name}. Revisa el contexto y continúa "
                "con la intención del usuario."
            ),
        )
        if ca is None:
            return {"messages": [msg]}
        return {"messages": [msg], "current_agent": ca}

    return entry_node


graph.add_node(
    "ToAgentEducation",
    create_entry_node("Agente Educativo", "education_agent_node"),
)
graph.add_node(
    "ToAgentGeneral",
    create_entry_node("Agente General", "general_agent_node"),
)
graph.add_node(
    "ToAgentLab",
    create_entry_node("Agente de Laboratorio", "lab_agent_node"),
)
graph.add_node(
    "ToAgentIndustrial",
    create_entry_node("Agente Industrial", "industrial_agent_node"),
)

graph.add_edge("ToAgentEducation", "education_agent_node")
graph.add_edge("ToAgentGeneral", "general_agent_node")
graph.add_edge("ToAgentLab", "lab_agent_node")
graph.add_edge("ToAgentIndustrial", "industrial_agent_node")

# ToolNode (tools compartidas entre agentes)
tools_node = ToolNode(
    tools=[
        web_research,
        retrieve_context,
        retrieve_robot_support,
        get_student_profile,
        update_student_goals,
        update_learning_style,
        summarize_all_chats,
        route_to,
        current_datetime,
        get_project_tasks,
        get_task_steps,
        get_task_step_images,
        search_manual_images,
        complete_task_step,
        identify_user_from_message,
        search_in_db,
        list_agent_tables,
        describe_agent_table,
    ]
)
graph.add_node("tools", tools_node)

# Nodo para actualizar user_name / user_email desde identify_user_from_message
graph.add_node("update_user_from_identify_tool", update_user_from_identify_tool)

# Después de cada agente: tools → update_user_from_identify_tool → agente actual, o save_agent_output
for agent in [
    "general_agent_node",
    "education_agent_node",
    "lab_agent_node",
    "industrial_agent_node",
]:
    graph.add_conditional_edges(
        agent,
        tools_condition,
        {"tools": "tools", "__end__": "save_agent_output"},
    )

# tools siempre pasa por el nodo que actualiza el usuario
graph.add_edge("tools", "update_user_from_identify_tool")

# Desde update_user_from_identify_tool regresamos al agente actual (cima del stack)
graph.add_conditional_edges("update_user_from_identify_tool", return_to_current_agent)

# Guardar output y terminar
graph.add_edge("save_agent_output", END)


# =====================================================
# Nodo opcional para "salir" del agente (pop del stack)
# =====================================================
def pop_current_agent(state: State) -> dict:
    messages = []

    if state.get("messages") and getattr(
        state["messages"][-1], "tool_calls", None
    ):
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        messages.append(
            ToolMessage(
                tool_call_id=tool_call_id,
                content=(
                    "Reanuda con el asistente principal. Continúa ayudando al usuario."
                ),
            )
        )

    return {"current_agent": "pop", "messages": messages}


graph.add_node("leave_agent", pop_current_agent)

# Fin del archivo
