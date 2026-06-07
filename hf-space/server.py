"""
MCP Server — Production entry point.

Exposes retrieve_context tool backed by the RAG pipeline.

MCP endpoint:    POST /mcp  (FastMCP mounted at /)
Health endpoint: GET  /health
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastmcp import FastMCP, Context
from dotenv import load_dotenv

from retrieval_pipeline import execute_retrieval

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("mcp_server")

_active_connections: int = 0

# ── Keep-alive config ─────────────────────────────────────────────────────────
# HF Spaces free tier sleeps after 15 minutes of inactivity.
# Pinging /health every 10 minutes keeps the space awake.
# Uses localhost — zero network cost, bypasses auth middleware.
_KEEP_ALIVE_INTERVAL_SECONDS = 10 * 60  # 10 minutes
_KEEP_ALIVE_URL = "http://localhost:7860/health"


async def _keep_alive_loop():
    """
    Pings /health on a fixed interval to prevent HF Spaces from sleeping.

    - Waits one full interval before the first ping so the server has time
      to finish startup (avoids a spurious connection error on boot).
    - Uses a short timeout — if the server is somehow unresponsive, we log
      and continue rather than crashing the background task.
    - Any exception is caught and logged; the loop always continues.
    """
    await asyncio.sleep(_KEEP_ALIVE_INTERVAL_SECONDS)  # skip first interval
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                resp = await client.get(_KEEP_ALIVE_URL)
                logger.info(f"Keep-alive ping → {resp.status_code}")
            except Exception as e:
                logger.warning(f"Keep-alive ping failed: {e}")
            await asyncio.sleep(_KEEP_ALIVE_INTERVAL_SECONDS)


# ── MCP definition ────────────────────────────────────────────────────────────
mcp = FastMCP("LangChain-RAG-Retrieval")


@mcp.tool()
async def retrieve_context(
    query        : str,
    top_k_child  : int = 15,
    top_k_parent : int = 3,
    ctx          : Context = None) -> str:
    """
    Retrieves the most relevant LangChain documentation context for a given query.

    Performs hybrid dense+sparse search over Qdrant Cloud, fetches parent chunks
    for full context, and applies cross-encoder reranking. Falls back to RRF
    ordering if the reranker API is unavailable.

    Args:
        query:        The user's question or search query
        top_k_child:  Number of child chunks to retrieve via hybrid search (default 15)
        top_k_parent: Number of top parent chunks to return after reranking (default 3)

    Returns:
        JSON string containing ranked parent documentation chunks with relevance scores
    """

    async def _mcp_log(msg: str):
        logger.info(msg)
        if ctx:
            await ctx.info(msg)

    return await execute_retrieval(
        query=query,
        top_k_child=top_k_child,
        top_k_parent=top_k_parent,
        log_callback=_mcp_log
    )


# ── MCP ASGI sub-app ──────────────────────────────────────────────────────────
mcp_app = mcp.http_app(path="/mcp")


# ── Composed lifespan ─────────────────────────────────────────────────────────
# We can't pass mcp_app.lifespan directly to FastAPI because we need to run
# our own startup/shutdown logic (keep-alive task) alongside it.
# Solution: wrap mcp_app.lifespan in our own asynccontextmanager so both
# lifecycles run together. FastMCP's session manager starts inside the
# `async with mcp_app.lifespan(app)` block — this must not be removed.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start keep-alive background task
    keep_alive_task = asyncio.create_task(_keep_alive_loop())
    logger.info(
        f"Keep-alive started — pinging {_KEEP_ALIVE_URL} "
        f"every {_KEEP_ALIVE_INTERVAL_SECONDS // 60} minutes."
    )

    # Hand control to FastMCP's session manager lifespan
    async with mcp_app.lifespan(app):
        yield

    # Shutdown: cancel keep-alive cleanly
    keep_alive_task.cancel()
    try:
        await keep_alive_task
    except asyncio.CancelledError:
        pass
    logger.info("Keep-alive stopped.")


# ── FastAPI wrapper ───────────────────────────────────────────────────────────
app = FastAPI(
    title="LangChain RAG — Remote MCP Server",
    lifespan=lifespan,          # composed lifespan, not mcp_app.lifespan directly
    redirect_slashes=False,
)


@app.middleware("http")
async def strip_trailing_slash(request: Request, call_next):
    """
    HF Spaces' NGINX proxy appends a trailing slash to some paths (e.g.
    /mcp → /mcp/). fastmcp 2.9.2 returns 400 on /mcp/ because it only
    registers the exact path /mcp. Strip the trailing slash before the
    request reaches fastmcp. Exclude root "/" to avoid infinite redirect.
    """
    if request.url.path != "/" and request.url.path.endswith("/"):
        stripped = request.url.path.rstrip("/")
        scope = dict(request.scope)
        scope["path"] = stripped
        request = Request(scope, request.receive, request._send)
    return await call_next(request)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    global _active_connections

    if request.url.path in ["/health"]:
        return await call_next(request)

    expected_key = os.environ.get("MCP_SECRET_KEY")
    provided_key = request.headers.get("X-API-Key")

    if not expected_key:
        logger.critical("MCP_SECRET_KEY is not set.")
        return JSONResponse(status_code=401, content={"error": "Server auth not configured."})

    if provided_key != expected_key:
        logger.warning(f"Unauthorized request to {request.url.path}")
        return JSONResponse(status_code=401, content={"error": "Unauthorized: Invalid X-API-Key."})

    _active_connections += 1
    try:
        return await call_next(request)
    finally:
        _active_connections -= 1


@app.get("/health")
async def health_check():
    return {"status": "healthy", "active_connections": _active_connections}


app.mount("/", mcp_app)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)