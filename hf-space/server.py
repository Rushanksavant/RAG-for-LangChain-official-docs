import os
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
    return (f"[DUMMY] Retrieved {top_k} context chunks for query: '{query}'. "
        f"Retrieval pipeline will be integrated here.")


# ── Build the MCP ASGI sub-app ────────────────────────────────────────────────
mcp_asgi_app = mcp.http_app(
    path="/mcp", 
    transport="streamable-http", 
    stateless_http=False)


# ── FastAPI wrapper ───────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp_asgi_app.lifespan(mcp_asgi_app):
        logger.info("MCP session manager initialised.")
        yield
    logger.info("MCP session manager shut down.")

app = FastAPI(
    title="LangChain RAG — Remote MCP Server",
    lifespan=lifespan,
    redirect_slashes=False)


# ── Pure ASGI Auth Middleware (SSE-Safe) ──────────────────────────────────────
class ASGIAuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        global _active_connections

        # Only intercept HTTP requests; allow lifespan/websockets to pass through
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # HF Spaces health probes must bypass auth
        path = scope.get("path", "")
        if path in ["/", "/health"]:
            return await self.app(scope, receive, send)

        # Extract headers safely
        headers = dict(scope.get("headers", []))
        provided_key = headers.get(b"x-api-key", b"").decode("utf-8")
        expected_key = os.environ.get("MCP_SECRET_KEY")

        if not expected_key:
            logger.error("MCP_SECRET_KEY environment variable is not set.")
            await self._send_json_error(send, 401, b'{"error": "Server misconfigured: MCP_SECRET_KEY missing."}')
            return

        if provided_key != expected_key:
            logger.warning("Unauthorized request blocked.")
            await self._send_json_error(send, 401, b'{"error": "Unauthorized: Invalid or missing X-API-Key."}')
            return

        # Track connection and process request
        _active_connections += 1
        try:
            await self.app(scope, receive, send)
        finally:
            _active_connections -= 1

    async def _send_json_error(self, send, status_code: int, body: bytes):
        """Helper to send raw ASGI JSON responses without buffering."""
        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": [(b"content-type", b"application/json")]
            })
        await send({
            "type": "http.response.body",
            "body": body,
            "more_body": False})

# Register the raw ASGI middleware
app.add_middleware(ASGIAuthMiddleware)


# ── Health + probe endpoints ──────────────────────────────────────────────────
@app.get("/")
async def root_probe():
    return {"status": "online", "service": "LangChain RAG MCP Server"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_connections": _active_connections}


# ── Inject MCP Routes ─────────────────────────────────────────────────────────
# FIX: Flattening the routes directly into the main app prevents trailing-slash
# drops from the HF Spaces proxy and avoids prefix duplication.
for route in mcp_asgi_app.routes:
    app.router.routes.append(route)


# ── Local dev entrypoint ──────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )