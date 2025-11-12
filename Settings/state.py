from typing import Annotated, Literal, Optional, TypedDict, List
from langgraph.graph.message import AnyMessage, add_messages
from pydantic.v1 import BaseModel

# Intenciones posibles
IntentType = Literal["EDUCATION", "LAB", "INDUSTRIAL", "GENERAL", "NOT_IDENTIFIED"]

# Estado del grafo
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    next_node: IntentType
    route_request: Optional[Literal["EDUCATION", "LAB", "INDUSTRIAL", "GENERAL"]]  
    session_id: Optional[int]
    user_identified: Optional[bool]
    user_email: Optional[str]
    user_name: Optional[str]

# Output del clasificador
class SupervisorOutput(BaseModel):
    next_node: IntentType
