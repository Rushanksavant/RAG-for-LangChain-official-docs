"""
MCP Server — Production entry point.

Exposes a single MCP tool: retrieve_context (dummy for now).
Retrieval pipeline integrated in next phase.

MCP endpoint:    POST /mcp   (handled by FastMCP mounted at /)
Health endpoint: GET  /health
"""

import os
import logging

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

_active_connections: int = 0

# ── MCP definition ────────────────────────────────────────────────────────────
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
    logger.info(f"retrieve_context called: query='{query}', top_k={top_k}")
    return (
        f"[DUMMY] Retrieved {top_k} context chunks for query: '{query}'. "
        f"Retrieval pipeline will be integrated here."
    )


# ── MCP ASGI sub-app ──────────────────────────────────────────────────────────
mcp_app = mcp.http_app(path="/mcp")

# ── FastAPI wrapper ───────────────────────────────────────────────────────────
# lifespan=mcp_app.lifespan shares FastMCP's session manager lifecycle
# with FastAPI. Required for streamable-http — without this, the session
# manager never initialises and every request returns "Session terminated".
app = FastAPI(
    title="LangChain RAG — Remote MCP Server",
    lifespan=mcp_app.lifespan,
    redirect_slashes=False,
)


# ── Auth middleware ───────────────────────────────────────────────────────────
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    global _active_connections

    # HF Spaces health probes bypass auth — they don't send API keys.
    if request.url.path in ["/health"]:
        return await call_next(request)

    expected_key = os.environ.get("MCP_SECRET_KEY")
    provided_key = request.headers.get("X-API-Key")

    if not expected_key:
        logger.critical("MCP_SECRET_KEY is not set.")
        return JSONResponse(status_code=401, content={"error": "Server auth not configured."})

    if provided_key != expected_key:
        logger.warning(f"Unauthorized: {request.url.path}")
        return JSONResponse(status_code=401, content={"error": "Unauthorized: Invalid X-API-Key."})

    _active_connections += 1
    try:
        return await call_next(request)
    finally:
        _active_connections -= 1


# ── Health + probe endpoints ──────────────────────────────────────────────────
@app.get("/")
async def root_probe():
    return {"status": "online", "service": "LangChain RAG MCP Server"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "active_connections": _active_connections}


# ── Mount MCP ─────────────────────────────────────────────────────────────────
app.mount("/", mcp_app)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)