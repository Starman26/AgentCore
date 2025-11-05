from typing import Annotated, Literal, Optional, TypedDict, List
from langgraph.graph.message import AnyMessage, add_messages
from pydantic.v1 import BaseModel



# Estado del grafo
def update_current_agent_stack(left: list[str], right: Optional[str]) -> list[str]:
  """Push or pop the state with defensive handling.

  See agent.graph.update_current_agent_stack for details. This version
  mirrors the same protection so the state logic is consistent across the
  codebase.
  """
  if right is None:
    return left

  if isinstance(right, list):
    right_list = [r for r in right if isinstance(r, str)]
    if not right_list:
      print("update_current_agent_stack (Settings): received empty/non-string list as right; ignoring")
      return left
    return left + right_list

  if right == "pop":
    return left[:-1]

  if not isinstance(right, str):
    print(f"update_current_agent_stack (Settings): unexpected type for right: {type(right)}; coercing to str")
    return left + [str(right)]

  return left + [right]

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    profile_summary: Optional[str]  
    current_agent: Annotated[list[Literal[
        "education_agent_node",
        "general_agent_node",
        "lab_agent_node",
        "industrial_agent_node",
        "router"
    ]], update_current_agent_stack]


