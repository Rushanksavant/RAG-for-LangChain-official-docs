"""
LangGraph RAG agent for LangChain/LangGraph/LangSmith documentation.
"""

import asyncio
import json
import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from agent_contents.schemas import AgentGraphState, QueryPlan
from agent_contents.prompts import PLANNER_PROMPT, ANSWER_PROMPT
from agent_contents.utilities import fetch_one, process_packet

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

def get_recent_history(history: list, max_exchanges: int = 3) -> list:
    """Returns the last N exchanges (2 messages per exchange) from chat history."""
    return history[-(max_exchanges * 2):]


# ── Node 1: plan_query ─────────────────────────────────────────────────────────

async def plan_query(state: AgentGraphState) -> dict:
    logger.info("plan_query: start")

    history  = get_recent_history(state.get("chat_history"))
    messages = [SystemMessage(content=PLANNER_PROMPT)] + history + [HumanMessage(content=state["user_input"])]

    plan: QueryPlan = await llm.with_structured_output(QueryPlan, strict = True, method = "json_schema").ainvoke(messages)

    logger.info(f"plan_query: guardrail={plan.guardrail!r}, needs_retrieval={plan.needs_retrieval}, sub_queries={plan.sub_queries}")

    status = [f"Query plan ready — {len(plan.sub_queries)} sub-quer{'y' if len(plan.sub_queries) == 1 else 'ies'}"]
    return {"query_plan": plan, 
            "status_mssg": status,
#### Resetting the schema (to avoid context pollution) #### IMPORTANT
            "retrieved_contexts": {},
            "context_sufficient": False,
            "mapped_insights": [],
            "final_response": ""}


# ── Node 2: retrieve_contexts ──────────────────────────────────────────────────

async def retrieve_contexts(state: AgentGraphState) -> dict:
    logger.info("retrieve_contexts: start")

    # Fire all sub-queries in parallel (each returns {query: [chunks]})
    sub_queries = state["query_plan"].sub_queries
    all_packets = await asyncio.gather(*[fetch_one(mcp_client, q) for q in sub_queries]) # all_packets = [{sub-query: [retrieved chunks texts]}, ..]

    # Merge individual query packets into a single master dictionary
    subquery_contextList = {}
    for packet in all_packets:
        subquery_contextList.update(packet)

    logger.info(f"retrieve_contexts: {len(sub_queries) * 2} chunks from {len(sub_queries)} sub-queries")

    status = [f"Retrieved {len(sub_queries) * 2} context chunks"]
    return {"retrieved_contexts": subquery_contextList, "status_mssg": status}


# ── Node 3: evaluate_context ───────────────────────────────────────────────────

async def evaluate_context(state: AgentGraphState) -> dict:
    logger.info("evaluate_context: start")

    # Reranker disabled — presence check is the honest signal for now.
    contexts = state.get("retrieved_contexts", {})
    sufficient = all(len(chunks) > 0 for chunks in contexts.values()) # stricter: needs retrieval for all sub-queries
    # sufficient = any(len(chunks) > 0 for chunks in contexts.values()) # less-strict: needs retrieval for atleast 1 sub-query

    status = state.get("status_mssg", []) + ["Context sufficient" if sufficient else "No context found"]
    return {"context_sufficient": sufficient, "status_mssg": status}


# ── Node 5: generate_subquery_answer ────────────────────────────────────────────────────

async def generate_subquery_answer(state: AgentGraphState) -> dict:
    logger.info("generate_subquery_answer: start")
    subquery_contextList = state["retrieved_contexts"] 

    # Parallel processing using standard .items() unpacking
    insights = await asyncio.gather(*[
        process_packet(llm, query, chunks) for query, chunks in subquery_contextList.items()
    ])
    
    return {"mapped_insights": list(insights)}


# ── Node 4: generate_final_answer ────────────────────────────────────────────────────

async def generate_final_answer(state: AgentGraphState) -> dict:
    logger.info("generate_final_answer: start")

    status             = state.get("status_mssg")[:]
    plan               = state["query_plan"]
    mapped_insights    = state.get("mapped_insights")
    retrieved_contexts = state.get("retrieved_contexts")
    chat_history       = state.get("chat_history")
    
    # 2. Dynamic Prompt Construction
    # PATH A: Multi-subqueries-outputs (Map-Reduce route taken)
    if mapped_insights:
        combined_insights = "\n\n".join(mapped_insights)
        prompt = f"""
        Original User Query: {plan.translated_query} 

        Synthesize these researched components into a cohesive final answer.
        Research Insights:
        {combined_insights}

        CRITICAL RULES:
        1. Base your synthesis strictly on the provided Research Insights above.
        2. If any of the insights note that documentation was missing or insufficient for a component, explicitly tell the user which parts are undocumented instead of inventing setup instructions.
        3. Do not invent or assume class properties, import locations, or configuration schemas outside what is explicitly stated in the insights.
        """
        status.append("Path A: Synthesizing Multi-query insights")
        
    # PATH B: Single-Query (Bypassed map step, use dictionary directly)
    elif plan.needs_retrieval and not mapped_insights:
        # flatten all retrieved chunks
        all_chunks = []
        for chunks in retrieved_contexts.values():
            all_chunks.extend(chunks)

        ctx_text = "\n\n".join(f"[Chunk {i+1}]\n{text}" for i, text in enumerate(all_chunks))
        prompt = f"""
        You must answer the query strictly using only the facts provided in the Context below.
        
        Query: {plan.translated_query}
        
        Context:
        {ctx_text}
        
        CRITICAL RULES:
        1. If the user query asks about specific terms, classes, or frameworks that are NOT explicitly documented 
        in the Context above, you must state: "I cannot find documentation for those components."
        2. Do not invent packages, imports, or code architectures from outside the provided Context blocks.
        3. If an exact method signature or import path is not visibly supported by a chunk, state that it is unavailable.
        """
        status.append("Path B: Generating final answer from single-query context")
        
    # PATH C: No Retrieval Needed (General Knowledge)
    else:
        prompt = f"""Query: {plan.translated_query}
                    This is a general knowledge question. Answer directly from your training knowledge — no documentation context is needed.
                    """
        status.append("Path C: Generating final answer from pre-trained knowledge")

    # 3. LLM Invocation
    messages = [SystemMessage(content=ANSWER_PROMPT)] + chat_history + [HumanMessage(content=prompt)]
    
    resp = await llm.ainvoke(messages)
    answer = resp.content
    
    status.append("Final answer generated")
    
    return {
        "final_response": answer,
        "chat_history": [HumanMessage(content=state["user_input"]), AIMessage(content=answer)],
        "status_mssg": status
    }


# ── Node 5: guardrail ──────────────────────────────────────────────────────────

async def end_with_guardrail(state: AgentGraphState) -> dict:
    msg = state["query_plan"].guardrail or "I'm sorry, but no relevant documentation could be retrieved to securely answer your question."
    status = state.get("status_mssg", []) + ["Guardrail triggered / Insufficient context"]
    logger.info(f"end_with_guardrail: {msg!r}")
    return {
        "final_response": msg,
        "status_mssg": status,
        "chat_history": [HumanMessage(content=state["user_input"]), AIMessage(content=msg)]}


# ── Routing ────────────────────────────────────────────────────────────────────

def route_after_plan(state: AgentGraphState) -> str:
    """
    Routing after plan_query:
     - end_with_guardrail: if irrelevant question
     - retrieve context: if retrieval required
     - generate_final_answer: if basic/simple question
    """
    plan = state["query_plan"]
    if plan.guardrail:
        return "irrelevant query"
    if not plan.needs_retrieval:
        return "basic query"
    return "requires retrieval"

def if_subqueries(state: AgentGraphState) -> str:
    """"
    Routing after evaluate_context:
     - verifies if context is sufficient
     - generate_final_answer: if context map contains single packet (subquery = translated query)
     - generate_subquery_answer: if context map containes >1 subqueries

    """
    # 1. If evaluation flagged the context as insufficient/empty, abort immediately
    if not state.get("context_sufficient"):
        return "insufficient context"
    
    subquery_contextList = state.get("retrieved_contexts")
    if len(subquery_contextList.keys()) == 1:
        return "single translated-query"
    return "multiple sub-query"


# ── Graph assembly ─────────────────────────────────────────────────────────────

builder = StateGraph(AgentGraphState)

builder.add_node("plan_query",         plan_query)
builder.add_node("retrieve_contexts",  retrieve_contexts)
builder.add_node("evaluate_context",   evaluate_context)
builder.add_node("generate_subquery_answer", generate_subquery_answer)
builder.add_node("generate_final_answer",    generate_final_answer)
builder.add_node("end_with_guardrail", end_with_guardrail)

builder.add_edge(START, "plan_query")
builder.add_conditional_edges("plan_query", route_after_plan, {"requires retrieval":"retrieve_contexts",
                                                               "basic query":"generate_final_answer",
                                                               "irrelevant query":"end_with_guardrail"})
builder.add_edge("retrieve_contexts",  "evaluate_context")
builder.add_conditional_edges("evaluate_context",   if_subqueries, {"single translated-query":"generate_final_answer",
                                                                    "multiple sub-query":"generate_subquery_answer",
                                                                    "insufficient context": "end_with_guardrail"})
builder.add_edge("generate_subquery_answer", "generate_final_answer")
builder.add_edge("generate_final_answer",    END)
builder.add_edge("end_with_guardrail", END)

graph = builder.compile(checkpointer=MemorySaver())

# Visualize graph
# graph_img = graph.get_graph().draw_mermaid_png()
# with open("agent_contents/graph.png", "wb") as f:
#     f.write(graph_img)


# ── Public entry point ─────────────────────────────────────────────────────────

async def run_agent(user_input: str, session_id: str) -> dict:
    """
    Called by the Streamlit frontend.
    session_id maps to a LangGraph thread — each browser session gets its own memory.
    """
    config = {"configurable": {"thread_id": session_id}}

    result = await graph.ainvoke(
        {"user_input": user_input}, config=config)

    return {"final_response":          result["final_response"],
        "status_mssg": result["status_mssg"],
        "query_plan":      result["query_plan"]}