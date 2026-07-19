import asyncio
import sys
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

# Adds the folder above 'agent' to your system path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Now you can use a clean, absolute import
from agent import graph
from settings import settings

def execute_agent_stream(user_input: str, session_id: str, status_container):
    """
    Runs the LangGraph instance asynchronously using astream to feed 
    status updates directly to the UI container while processing.
    """
    config = {"configurable": {"thread_id": session_id}}
    inputs = {
        "user_input"         : user_input,
        "query_plan"         : None,
        "retrieved_contexts" : {},
        "context_sufficient" : False,
        "final_response"     : "",
        "chat_history"       : [],
        "status_mssg"        : [],
    }
    # 1. Instantiate LLM & MCP inside the live event loop 
    # This fixes: conflict between Streamlit's threading, asyncio.run(), and global async HTTP clients by initializing within the Streamlit event-loop.
    # ── LLM setup ─────────────────────────────────────────────────────────────────
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", 
        google_api_key=settings.GEMINI_API_KEY.get_secret_value(),
        temperature=0.0
    )
    # llm = init_chat_model(model='openai/gpt-oss-120b', model_provider='groq', 
    #                       temperature=0, api_key= settings.GROQ_API_KEY.get_secret_value())

    # ── MCP client ─────────────────────────────────────────────────────────────────
    # Single shared client for the lifetime of the process.
    mcp_client = MultiServerMCPClient({
        "rag-retrieval": {
            "transport": "streamable_http",
            "url": settings.SERVER_URL,
            "headers": {"X-API-Key": settings.MCP_SECRET_KEY.get_secret_value()},
        }
    })

    # 2. Inject into the configurable run context
    config["configurable"]["llm"] = llm
    config["configurable"]["mcp_client"] = mcp_client



    async def _stream():
        async for event in graph.astream(inputs, config=config, stream_mode="updates"):
            if not event:
                continue
            node_name = list(event.keys())[0]
            node_output = event[node_name]
            
            if "status_mssg" in node_output and node_output["status_mssg"]:
                for msg in node_output["status_mssg"]:
                    status_container.write(f"⚙️ {msg}")

        final_state = await graph.aget_state(config)
        return final_state.values

    return asyncio.run(_stream())