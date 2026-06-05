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
from fastapi import FastAPI
from fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("mcp_server")


# ── Active connection tracking ────────────────────────────────────────────────
_active_connections: int = 0


# ── MCP server definition ─────────────────────────────────────────────────────
mcp = FastMCP("LangChain-RAG-Retrieval")


@mcp.tool()
async def retrieve_context(query: str, top_k: int = 5) -> str:
    """
    Retrieves the most relevant LangChain documentation context for a given query.
    """
    logger.info(f"retrieve_context called: query='{query}', top_k={top_k}")
    return (
        f"[DUMMY] Retrieved {top_k} context chunks for query: '{query}'. "
        f"Retrieval pipeline will be integrated here."
    )


# Keep path="/mcp" so FastMCP expects /mcp/sse and /mcp/messages internally
mcp_asgi_app = mcp.http_app(
    path="/mcp", 
    transport="streamable-http", 
    stateless_http=False
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp_asgi_app.lifespan(mcp_asgi_app):
        logger.info("MCP session manager initialised.")
        yield
    logger.info("MCP session manager shut down.")


app = FastAPI(
    title="LangChain RAG — Remote MCP Server",
    lifespan=lifespan,
    redirect_slashes=False
)


# ── FIX: Pure ASGI Middleware to prevent streaming / SSE buffering chokes ─────
class ASGIAuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        global _active_connections

        # Pass through non-HTTP protocols (e.g., lifespans, websockets if any)
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        # Always bypass auth for the root probe and health check endpoints
        if path in ["/", "/health"]:
            return await self.app(scope, receive, send)

        # Extract incoming headers (ASGI normalizes header keys to lowercase bytes)
        headers = dict(scope.get("headers", []))
        provided_key = headers.get(b"x-api-key", b"").decode("utf-8")
        expected_key = os.environ.get("MCP_SECRET_KEY")

        if not expected_key:
            logger.error("MCP_SECRET_KEY environment variable is missing on server.")
            await self._send_json_error(send, 500, b'{"error": "Server misconfigured."}')
            return

        if provided_key != expected_key:
            logger.warning(f"Unauthorized request blocked for path: {path}")
            await self._send_json_error(send, 401, b'{"error": "Unauthorized."}')
            return

        # Safely track concurrent connections
        _active_connections += 1
        try:
            await self.app(scope, receive, send)
        finally:
            _active_connections -= 1

    async def _send_json_error(self, send, status_code: int, body: bytes):
        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": [(b"content-type", b"application/json")]
        })
        await send({
            "type": "http.response.body",
            "body": body,
            "more_body": False
        })


app.add_middleware(ASGIAuthMiddleware)


# ── Health + probe endpoints ──────────────────────────────────────────────────
@app.get("/")
async def root_probe():
    return {"status": "online", "service": "LangChain RAG MCP Server"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "active_connections": _active_connections}


# ── FIX: Mount to root so path prefixes remain intact for FastMCP ─────────────
app.mount("/", mcp_asgi_app)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860)