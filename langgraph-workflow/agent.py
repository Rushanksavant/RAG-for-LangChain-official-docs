"""
LangGraph RAG agent for LangChain/LangGraph/LangSmith documentation.

Graph flow:
    plan_query
        ├── guardrail      → END
        ├── no retrieval   → generate_answer → END
        └── needs retrieval → retrieve_contexts → evaluate_context → generate_answer → END
"""

import asyncio
import json
import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from schemas import AgentGraphState, QueryPlan
from prompts import PLANNER_PROMPT, ANSWER_PROMPT
from retrieve_and_clean import fetch_one

import sys
from pathlib import Path
# Adds the project root directory directly to Python's search path
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agent")

# ── LLM setup ─────────────────────────────────────────────────────────────────

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
#                             google_api_key=settings.GEMINI_API_KEY.get_secret_value(),
#                             temperature=0.0)
from langchain.chat_models import init_chat_model
llm = init_chat_model(model='openai/gpt-oss-120b', model_provider='groq', 
                      temperature=0, api_key= settings.GROQ_API_KEY.get_secret_value())


# ── MCP client ─────────────────────────────────────────────────────────────────
# Single shared client for the lifetime of the process.
# Re-created on ConnectionError by retrieve_contexts node.

mcp_client = MultiServerMCPClient({
    "rag-retrieval": {
        "transport": "streamable_http",
        "url": settings.SERVER_URL,
        "headers": {"X-API-Key": settings.MCP_SECRET_KEY.get_secret_value()},
    }
})

# ── History helper ─────────────────────────────────────────────────────────────

def get_recent_history(history: list, max_exchanges: int = 10) -> list:
    """Returns the last N exchanges (2 messages per exchange) from chat history."""
    return history[-(max_exchanges * 2):]


# ── Node 1: plan_query ─────────────────────────────────────────────────────────

async def plan_query(state: AgentGraphState) -> dict:
    logger.info("plan_query: start")

    history  = get_recent_history(state.get("chat_history"))
    messages = [SystemMessage(content=PLANNER_PROMPT)] + history + [HumanMessage(content=state["user_input"])]

    plan: QueryPlan = await llm.with_structured_output(QueryPlan, strict = True, method = "json_schema").ainvoke(messages)

    logger.info(f"plan_query: guardrail={plan.guardrail!r}, needs_retrieval={plan.needs_retrieval}, sub_queries={plan.sub_queries}")

    status = state.get("status_mssg", []) + [f"Query plan ready — {len(plan.sub_queries)} sub-quer{'y' if len(plan.sub_queries) == 1 else 'ies'}"]
    return {"query_plan": plan, "status_mssg": status}


# ── Node 2: retrieve_contexts ──────────────────────────────────────────────────

async def retrieve_contexts(state: AgentGraphState) -> dict:
    logger.info("retrieve_contexts: start")

    # Fire all sub-queries in parallel
    sub_queries = state["query_plan"].sub_queries
    all_results = await asyncio.gather(*[fetch_one(mcp_client, q) for q in sub_queries])

    # Merge results, deduplicate by parent_id
    seen   = set()
    merged = []
    for chunks in all_results:
        for chunk in chunks:
            pid = chunk.get("parent_id")
            if pid and pid not in seen:
                seen.add(pid)
                merged.append(chunk)

    logger.info(f"retrieve_contexts: {len(merged)} unique chunks from {len(sub_queries)} sub-queries")

    status = state.get("status_mssg", []) + [f"Retrieved {len(merged)} context chunks"]
    return {"retrieved_contexts": merged, "status_mssg": status}


# ── Node 3: evaluate_context ───────────────────────────────────────────────────

async def evaluate_context(state: AgentGraphState) -> dict:
    logger.info("evaluate_context: start")

    # Reranker disabled — presence check is the honest signal for now.
    sufficient = len(state.get("retrieved_contexts", [])) > 0

    status = state.get("status_mssg", []) + ["Context sufficient" if sufficient else "No context found"]
    return {"context_sufficient": sufficient, "status_mssg": status}


# ── Node 4: generate_answer ────────────────────────────────────────────────────

async def generate_answer(state: AgentGraphState) -> dict:
    logger.info("generate_answer: start")

    plan       = state["query_plan"]
    contexts   = state.get("retrieved_contexts")
    sufficient = state.get("context_sufficient")
    history    = get_recent_history(state.get("chat_history"))

    # Allow general knowledge if retrieval wasn't needed
    if not plan.needs_retrieval:
        context_text = "No retrieval needed. Answer the user's general question using your pre-trained knowledge."
        sufficient = True # Satisfies prompt safety constraint
    # Else if context retrieved, bind it together    
    elif sufficient and contexts:
        context_text = "\n\n".join(
            f"[Chunk {i+1}]\n{c.get('text', '')}"
            for i, c in enumerate(contexts)
        )
    # Else fixed answer to avoid hallucination   
    else:
        context_text = "No relevant documentation context retrieved."

    user_prompt = f"""
                Translated query: {plan.translated_query}

                Sub-queries used:
                {chr(10).join(f"- {q}" for q in plan.sub_queries)}

                Context (context_sufficient={sufficient}):
                {context_text}
                """.strip()

    messages  = [SystemMessage(content=ANSWER_PROMPT)] + history + [HumanMessage(content=user_prompt)]
    response  = await llm.ainvoke(messages)
    answer    = response.content

    status = state.get("status_mssg", []) + ["Answer ready"]
    return {
        "response": answer,
        "status_mssg": status,
        # append this exchange to chat history
        "chat_history": [HumanMessage(content=state["user_input"]), AIMessage(content=answer)]}


# ── Node 5: guardrail ──────────────────────────────────────────────────────────

async def end_with_guardrail(state: AgentGraphState) -> dict:
    msg    = state["query_plan"].guardrail
    status = state.get("status_mssg", []) + ["Guardrail triggered"]
    logger.info(f"end_with_guardrail: {msg!r}")
    return {
        "response": msg,
        "status_mssg": status,
        "chat_history": [HumanMessage(content=state["user_input"]), AIMessage(content=msg)]}


# ── Routing ────────────────────────────────────────────────────────────────────

def route_after_plan(state: AgentGraphState) -> str:
    plan = state["query_plan"]
    if plan.guardrail:
        return "end_with_guardrail"
    if not plan.needs_retrieval:
        return "generate_answer"
    return "retrieve_contexts"


# ── Graph assembly ─────────────────────────────────────────────────────────────

builder = StateGraph(AgentGraphState)

builder.add_node("plan_query",         plan_query)
builder.add_node("retrieve_contexts",  retrieve_contexts)
builder.add_node("evaluate_context",   evaluate_context)
builder.add_node("generate_answer",    generate_answer)
builder.add_node("end_with_guardrail", end_with_guardrail)

builder.add_edge(START, "plan_query")
builder.add_conditional_edges("plan_query", route_after_plan, {"retrieve_contexts":"retrieve_contexts",
                                                               "generate_answer":"generate_answer",
                                                               "end_with_guardrail":"end_with_guardrail"})
builder.add_edge("retrieve_contexts",  "evaluate_context")
builder.add_edge("evaluate_context",   "generate_answer")
builder.add_edge("generate_answer",    END)
builder.add_edge("end_with_guardrail", END)

graph = builder.compile(checkpointer=MemorySaver())

# Visualize graph
graph_img = graph.get_graph().draw_mermaid_png()
with open("langgraph-workflow/graph.png", "wb") as f:
    f.write(graph_img)


# ── Public entry point ─────────────────────────────────────────────────────────

async def run_agent(user_input: str, session_id: str) -> dict:
    """
    Called by the Streamlit frontend.
    session_id maps to a LangGraph thread — each browser session gets its own memory.
    """
    config = {"configurable": {"thread_id": session_id}}

    result = await graph.ainvoke(
        {
            "user_input"         : user_input,
            "query_plan"         : None,
            "retrieved_contexts" : [],
            "context_sufficient" : False,
            "response"           : "",
            "chat_history"       : [],
            "status_mssg"        : [],
        },
        config=config,
    )

    return {
        "response":          result["response"],
        "status_mssg": result["status_mssg"],
        "query_plan":      result["query_plan"],
    }