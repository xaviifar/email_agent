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
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode

from external.google_auth_service import google_auth_service
from external.gmail_client import GmailClient
from external.elevenlabs_client import ElevenLabsClient

from states.email_state import EmailState
from tools.gmail_tools import GmailTools

load_dotenv()


def load_prompt(file = "./prompt.txt"):

    with open(file, "+r", encoding="utf-8") as f:
        prompt = f.read()

    return prompt

def should_continue(state: EmailState ):

    last_msg = state["messages"][-1]

    if last_msg.tool_calls:
        return "tools"

    return END


    
def agent_node(state: EmailState, model):

    prompt = load_prompt()

    try:

        messages = [
            SystemMessage(content=prompt),

            *state["messages"]]

        response = model.invoke(messages)   
        return {"messages": [response]}

    except Exception as e:
        print(f"Error {e}")
        raise



async def email_agent():

    msg = input("Que necesitas? ")


    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("Falta OPENROUTER_API_KEY")

    credentials = google_auth_service.get_credentials()

    gmail_client = GmailClient(credentials=credentials)

    gmail_tools = GmailTools(gmail_client)

    tools = [
        gmail_tools.read_inbox,
        gmail_tools.delete_batch,
        gmail_tools.search_emails,
        gmail_tools.delete_email
    ]


    model = ChatOpenRouter(
        model=os.getenv("OPENROUTER_MODEL", "qwen/qwen3.7-flash"),
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
            "content": msg
        }]
    })

    msg_r = next(
        message 
        for message in reversed(result["messages"])
        if isinstance(message, AIMessage) and message.content
    )

    v_client = ElevenLabsClient(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
    voice=os.getenv("ELEVENLABS_VOICE"),
    model= os.getenv("ELEVENLABS_MODEL_TTS"),
    )

    print(msg_r.content)

    # v_client.tts(text = msg_r.content)


if __name__ == "__main__":
    asyncio.run(email_agent())