from typing import Annotated, Literal, Optional, List
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from pydantic.v1 import BaseModel

IntentType = Literal["EDUCATION", "LAB", "INDUSTRIAL", "GENERAL", "NOT_IDENTIFIED"]

def update_current_agent_stack(left: list[str], right: Optional[str]) -> list[str]:
    if right is None: return left
    if isinstance(right, list):
        right_list = [r for r in right if isinstance(r, str)]
        return left + right_list if right_list else left
    if right == "pop": return left[:-1]
    return left + [str(right) if not isinstance(right, str) else right]

class State(TypedDict, total=False):
    # Mensajes
    messages: Annotated[List[AnyMessage], add_messages]

    # Perfil usuario
    profile_summary: Optional[str]
    user_identified: Optional[bool]
    user_email: Optional[str]
    user_name: Optional[str]

    # Temporal
    tz: str
    now_utc: str
    now_local: str
    now_human: str

    # Widget / avatar
    widget_avatar_id: Optional[str]
    widget_mode: Optional[str]
    widget_personality: Optional[str]
    widget_notes: Optional[str]
    avatar_style: Optional[str]

    # Sesión
    session_id: Optional[str]
    session_title: Optional[str]
    awaiting_user_info: Optional[str]

    # Routing stack
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

    # ===== Prácticas =====
    chat_type: Optional[str]
    project_id: Optional[str]
    current_task_id: Optional[str]
    current_step_number: Optional[int]
    practice_completed: Optional[bool]


class SupervisorOutput(BaseModel):
    next_node: IntentType
