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
    # Router moderno (tool-call)
    profile_summary: Optional[str]
    tz: str
    now_utc: str
    now_local: str
    now_human: str
    current_agent: Annotated[List[Literal[
        "education_agent_node","general_agent_node","lab_agent_node","industrial_agent_node","router"
    ]], update_current_agent_stack]
    # Clasificador antiguo (compat)
    next_node: IntentType
    route_request: Optional[Literal["EDUCATION","LAB","INDUSTRIAL","GENERAL"]]
    session_id: Optional[str]  # Cambiado de int a str para usar thread_id
    user_identified: Optional[bool]
    user_email: Optional[str]
    user_name: Optional[str]

class SupervisorOutput(BaseModel):
    next_node: IntentType
