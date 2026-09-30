

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class EmailState(TypedDict):
    messages: Annotated[list, add_messages]
    request: str 
    emails: list[str]
    tool: str
    tool_args: dict
    tool_result: object



        



        

    
