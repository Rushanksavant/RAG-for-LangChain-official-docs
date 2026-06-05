"""
MCP Server — Production entry point.

Exposes a single MCP tool: retrieve_context (dummy for now, retrieval logic added later).
Deployed on HF Spaces behind a FastAPI wrapper that handles:
  - Auth via X-API-Key header
  - Health/probe endpoints that bypass auth (required by HF Spaces proxy)
  - Active connection tracking

MCP endpoint: POST /mcp
Health endpoint: GET /health
Root probe:    GET /
"""

import os
import uuid
import logging
import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastmcp import FastMCP

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("mcp_server")


# ── Active connection tracking ────────────────────────────────────────────────
# Simple counter — good enough for a single-process deployment on HF Spaces.
# For multi-worker deployments, replace with Redis or shared memory.
_active_connections: int = 0


# ── MCP server definition ─────────────────────────────────────────────────────
# stateless_http=True: each request is independent, no session state maintained. needs to be False
# Required for HF Spaces where requests may hit different uvicorn workers.
mcp = FastMCP("LangChain-RAG-Retrieval")


@mcp.tool()
async def retrieve_context(query: str, top_k: int = 5) -> str:
    """
    Retrieves the most relevant LangChain documentation context for a given query.

    Uses hybrid dense+sparse search over Qdrant Cloud, fetches parent chunks
    for full context, and applies cross-encoder reranking.

    Args:
        query: The user's question or search query
        top_k:  Number of top parent chunks to return after reranking (default 5)

    Returns:
        JSON string containing ranked documentation chunks with relevance scores
    """
    # ── Dummy implementation — retrieval pipeline integrated later ────────────
    logger.info(f"retrieve_context called: query='{query}', top_k={top_k}")
    return (
        f"[DUMMY] Retrieved {top_k} context chunks for query: '{query}'. "
        f"Retrieval pipeline will be integrated here."
    )


# ── Build the MCP ASGI sub-app ────────────────────────────────────────────────
# path="/mcp" means the MCP protocol endpoint lives at /mcp on the outer app.
# We pass lifespan to FastAPI so FastMCP's session manager initialises correctly.
# This is critical for streamable-http — without it, sessions silently fail.
mcp_asgi_app = mcp.http_app(path="/", stateless_http=False)


# ── FastAPI wrapper ───────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Shares FastMCP's lifespan with FastAPI.
    Required for streamable-http transport — FastMCP's session manager
    must be initialised before any requests arrive.
    """
    async with mcp_asgi_app.lifespan(mcp_asgi_app):
        logger.info("MCP session manager initialised.")
        yield
    logger.info("MCP session manager shut down.")


app = FastAPI(
    title="LangChain RAG — Remote MCP Server",
    lifespan=lifespan,
    redirect_slashes=False
)


# ── Auth middleware ───────────────────────────────────────────────────────────
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    global _active_connections

    # HF Spaces health probes must bypass auth — they don't send API keys.
    # Without this, HF marks the Space as unhealthy and stops routing traffic.
    if request.url.path in ["/", "/health"]:
        return await call_next(request)

    expected_key = os.environ.get("MCP_SECRET_KEY")
    provided_key = request.headers.get("X-API-Key")

    if not expected_key:
        logger.error("MCP_SECRET_KEY environment variable is not set.")
        return JSONResponse(
            status_code=401,
            content={"error": "Server misconfigured: MCP_SECRET_KEY missing."}
        )

    if provided_key != expected_key:
        logger.warning(
            f"Unauthorized request to {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized: Invalid or missing X-API-Key."}
        )

    _active_connections += 1
    try:
        response = await call_next(request)
        return response
    finally:
        _active_connections -= 1


# ── Health + probe endpoints ──────────────────────────────────────────────────
@app.get("/")
async def root_probe():
    """HF Spaces startup probe — must return 200 without auth."""
    return {"status": "online", "service": "LangChain RAG MCP Server"}


@app.get("/health")
async def health_check():
    """
    Lightweight health check used by HF Spaces and external monitors.
    Returns active connection count for basic load visibility.
    """
    return {
        "status": "healthy",
        "active_connections": _active_connections,
    }


# ── Mount MCP under /mcp ──────────────────────────────────────────────────────
# All MCP protocol traffic (tool discovery, invocation) hits /mcp.
# Auth middleware above applies to all /mcp requests automatically.
app.mount("/mcp", mcp_asgi_app)


# ── Local dev entrypoint ──────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=7860,
        reload=False,   # Never use reload=True in production
    )