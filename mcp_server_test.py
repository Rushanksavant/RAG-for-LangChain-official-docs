"""
test.py — MCP server integration test.

Tests the full client→server round-trip using langchain-mcp-adapters,
exactly as the LangGraph agent will consume the MCP server in production.

Usage:
  # Start the server first in a separate terminal:
  python server.py

  # Then run this test:
  python test.py

  # To test against deployed HF Space:
  SERVER_URL=https://your-space.hf.space/mcp 
  MCP_SECRET_KEY=your-key 
  python test.py
"""

import os
import asyncio
import json
import logging

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Config — override via environment variables for production testing ─────────
SERVER_URL = os.environ.get("SERVER_URL")
MCP_SECRET_KEY = os.environ.get("MCP_SECRET_KEY")


async def run_test():
    from langchain_mcp_adapters.client import MultiServerMCPClient

    print("\n" + "=" * 60)
    print("MCP SERVER INTEGRATION TEST")
    print(f"  Server URL : {SERVER_URL}")
    print(f"  Auth key   : {'*' * len(MCP_SECRET_KEY)}")
    print("=" * 60 + "\n")

    # ── Client config — same pattern used by LangGraph agent in production ────
    client_config = {
        "rag-retrieval": {
            "transport": "streamable_http",
            "url": SERVER_URL,
            "headers": {
                "X-API-Key": MCP_SECRET_KEY,
            },
        }
    }

    # ── Test 1: Tool discovery ─────────────────────────────────────────────────
    print("[1/4] Fetching available tools...")
    client = MultiServerMCPClient(client_config)
    tools = await client.get_tools()
    tool_names = [t.name for t in tools]
    print(f"✅ Tools discovered: {tool_names}")

    if "retrieve_context" not in tool_names:
        print("❌ FAIL: 'retrieve_context' not found in tool list.")
        print(f"   Available tools: {tool_names}")
        return

    for tool in tools:
        if tool.name == "retrieve_context":
            print(f"\n   Tool description: {tool.description}")
            print(f"   Tool schema:      {json.dumps(tool.args_schema, indent=4)}")
    print()

    # ── Test 2: Basic invocation ───────────────────────────────────────────────
    print("[2/4] Invoking retrieve_context (basic call)...")
    retrieve_tool = next(t for t in tools if t.name == "retrieve_context")
    result = await retrieve_tool.ainvoke(
        {"query": "How to configure a multi-agent setup in LangGraph?", 
         "top_k_child": 3, 
         "top_k_parent": 3}
    )
    print(f"✅ Response received:")
    print(f"   {result}\n")

    # ── Test 3: Concurrent invocations ────────────────────────────────────────
    print("[3/4] Testing concurrent invocations (5 parallel requests)...")
    queries = [
        "What is LangGraph?",
        "How does LangSmith tracing work?",
        "How to use memory in LangChain agents?",
        "What are LangChain runnables?",
        "How to stream responses in LangChain?",
    ]

    async def invoke_single(query: str, index: int):
        result = await retrieve_tool.ainvoke({"query": query, "top_k_child": 15, "top_k_parent": 3})
        return index, result

    tasks = [invoke_single(q, i) for i, q in enumerate(queries)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = 0
    for item in results:
        if isinstance(item, Exception):
            print(f"   ❌ Request failed: {item}")
        else:
            idx, res = item
            print(f"   ✅ Request {idx}: {str(res)[:80]}...")
            success_count += 1

    print(f"\n   Concurrent: {success_count}/{len(queries)} succeeded\n")

    # ── Test 4: Auth rejection ─────────────────────────────────────────────────
    print("[4/4] Testing auth rejection with wrong API key...")
    bad_client = MultiServerMCPClient({
        "rag-retrieval": {
            "transport": "streamable_http",
            "url": SERVER_URL,
            "headers": {"X-API-Key": "wrong-key-12345"},
        }
    })
    try:
        await bad_client.get_tools()
        print("❌ FAIL: Server accepted invalid API key — auth is broken.")
    except Exception as e:
        print(f"✅ Auth correctly rejected invalid key: {type(e).__name__}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("""
Under-the-hood in production (HF Spaces):

  1. LangGraph agent calls client.get_tools() once at startup
     → MultiServerMCPClient sends POST /mcp to discover tool schemas
     → Server responds with retrieve_context schema (name, description, args)
     → LangGraph binds this as a standard LangChain tool

  2. When agent decides to retrieve context:
     → LangGraph calls tool.ainvoke({query, top_k})
     → MultiServerMCPClient sends POST /mcp with tool name + args
     → FastAPI auth middleware validates X-API-Key
     → FastMCP routes to retrieve_context() function
     → Function runs (BGE-M3 embed → Qdrant hybrid search → HF reranker)
     → Result returned as MCP tool response
     → LangGraph receives result as tool message in conversation state

  3. Concurrency:
     → Each tool call is stateless (stateless_http=True on server)
     → Multiple agent calls can fire in parallel safely
     → HF Spaces free tier (2 vCPU, 16GB RAM) supports ~3-4 concurrent
       BGE-M3 embedding calls before RAM becomes the bottleneck
""")


if __name__ == "__main__":
    asyncio.run(run_test())