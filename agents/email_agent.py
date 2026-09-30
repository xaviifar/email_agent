import asyncio
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from langgraph.graph.state import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from external.google_auth_service import google_auth_service
from external.gmail_client import GmailClient

from states.email_state import EmailState
from tools.gmail_tools import GmailTools

load_dotenv()

def should_continue(state: EmailState ):

    last_msg = state["messages"][-1]

    if last_msg.tool_calls:
        return "tools"

    return END
    
def agent_node(state: EmailState, model):

    try:
        response = model.invoke(state["messages"])   
        return {"messages": [response]}

    except Exception as e:
        print(f"Error {e}")
        raise




async def main():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("Falta OPENROUTER_API_KEY")

    credentials = google_auth_service.get_credentials()

    gmail_client = GmailClient(credentials=credentials)

    gmail_tools = GmailTools(gmail_client)

    tools = [
        gmail_tools.read_inbox,
        gmail_tools.delete_batch,
        gmail_tools.search_emails
    ]


    model = ChatOpenRouter(
        model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
        api_key=api_key,
    )

    model_tools = model.bind_tools(tools)

    tool_node = ToolNode(tools)

    graph = StateGraph(EmailState)

    graph.add_node("agent", lambda state: agent_node(state, model_tools))
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
            "agent", should_continue, {
                "tools": "tools",
                END: END,
            }
            
    )
    graph.add_edge("tools", "agent")


    agent = graph.compile()

    result = await agent.ainvoke({
        "messages": 
        [{
            "role": "user", 
            "content": "Que correos tengo"
        }]
    })

    print(result)

if __name__ == "__main__":
    asyncio.run(main())