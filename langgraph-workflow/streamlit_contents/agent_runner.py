import asyncio
import sys
from pathlib import Path

# Adds the folder above 'agent' to your system path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Now you can use a clean, absolute import
from agent import graph

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