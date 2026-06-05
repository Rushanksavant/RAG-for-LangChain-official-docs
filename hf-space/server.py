import os
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastmcp import FastMCP
from dotenv import load_dotenv

# Loads environment variables from a .env file (if present) into os.environ.
# On Hugging Face Spaces, this silently does nothing because the secrets are injected directly.
load_dotenv()

# Set up logging so we can track requests and errors in the Hugging Face Spaces logs tab.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("mcp_server")


# ── Active connection tracking ────────────────────────────────────────────────
# A simple global counter. Because we run a single Uvicorn worker (--workers 1) 
# on HF Spaces, this memory is shared across all incoming requests safely.
_active_connections: int = 0


# ── MCP server definition ─────────────────────────────────────────────────────
# Initializes the FastMCP instance. This is the engine that actually understands
# the Model Context Protocol (tool discovery, JSON-RPC formatting, etc.)
mcp = FastMCP("LangChain-RAG-Retrieval")

# The @mcp.tool() decorator automatically inspects the type hints (query: str, top_k: int)
# and docstring to generate a JSON Schema. When LangChain calls `client.get_tools()`, 
# FastMCP sends this exact schema back so the agent knows how to use it.
@mcp.tool()
async def retrieve_context(query: str, top_k: int = 5) -> str:
    """
    Retrieves the most relevant LangChain documentation context for a given query.
    """
    logger.info(f"retrieve_context called: query='{query}', top_k={top_k}")
    
    # This is where your BGE-M3 embedding and Qdrant search logic will go.
    # Whatever string you return here is sent directly back to the LangGraph agent's memory.
    return (
        f"[DUMMY] Retrieved {top_k} context chunks for query: '{query}'. "
        f"Retrieval pipeline will be integrated here."
    )


# ── Build the MCP ASGI sub-app ────────────────────────────────────────────────
# This converts the FastMCP engine into a standard web application (ASGI) that FastAPI can host.
# - transport="streamable-http": Tells FastMCP to use Server-Sent Events (SSE).
# - stateless_http=False: Forces the server to keep the session alive in memory 
#   so the LangChain client can hold a persistent connection open.
mcp_asgi_app = mcp.http_app(
    path="/mcp", 
    transport="streamable-http", 
    stateless_http=False
)


# ── FastAPI wrapper ───────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # FastMCP has internal startup tasks (like initializing its session manager).
    # This lifespan context manager ensures FastMCP boots up properly right before 
    # FastAPI starts accepting internet traffic, preventing silent session drops.
    async with mcp_asgi_app.lifespan(mcp_asgi_app):
        logger.info("MCP session manager initialised.")
        yield # The server runs during this yield
    logger.info("MCP session manager shut down.")

# The main outer application that acts as the reverse proxy, handling auth and health checks
# before handing MCP traffic off to the FastMCP sub-app.
app = FastAPI(
    title="LangChain RAG — Remote MCP Server",
    lifespan=lifespan,
    redirect_slashes=False # Prevents FastAPI from auto-adding slashes and breaking MCP POST requests
)


# ── Pure ASGI Auth Middleware (SSE-Safe) ──────────────────────────────────────
class ASGIAuthMiddleware:
    """
    We use raw ASGI middleware instead of FastAPI's @app.middleware("http") because 
    FastAPI's HTTP middleware physically buffers responses in memory. Buffering breaks 
    Server-Sent Events (SSE), which require bytes to stream continuously over the wire.
    """
    def __init__(self, app):
        self.app = app # The next app in the chain (our FastAPI app)

    async def __call__(self, scope, receive, send):
        global _active_connections

        # ASGI handles more than just HTTP (e.g., websockets, lifespan events).
        # We only want to intercept and authenticate HTTP traffic.
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        
        # Hugging Face Spaces regularly pings "/" and "/health" to check if the app crashed.
        # If we demand an API key for these, HF thinks we are down and shuts off our traffic.
        if path in ["/", "/health"]:
            return await self.app(scope, receive, send)

        # Extract headers safely from the raw ASGI byte strings
        headers = dict(scope.get("headers", []))
        provided_key = headers.get(b"x-api-key", b"").decode("utf-8")
        expected_key = os.environ.get("MCP_SECRET_KEY")

        # Security Check 1: Did the developer forget to set the secret in HF settings?
        if not expected_key:
            logger.error("MCP_SECRET_KEY missing on server.")
            await self._send_json_error(send, 401, b'{"error": "Server misconfigured."}')
            return

        # Security Check 2: Does the client's key match our secret?
        if provided_key != expected_key:
            logger.warning("Unauthorized request blocked.")
            await self._send_json_error(send, 401, b'{"error": "Unauthorized."}')
            return

        # If auth passes, track the connection and hand the request over to FastAPI/FastMCP
        _active_connections += 1
        try:
            await self.app(scope, receive, send)
        finally:
            # Regardless of success or failure, decrement the counter when the client disconnects
            _active_connections -= 1

    async def _send_json_error(self, send, status_code: int, body: bytes):
        """Helper function to send HTTP error responses directly via ASGI without buffering."""
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

# Attach our custom unbuffered middleware to the main app
app.add_middleware(ASGIAuthMiddleware)


# ── Health + probe endpoints ──────────────────────────────────────────────────
@app.get("/")
async def root_probe():
    """Returns 200 OK instantly. Required by HF Spaces to verify the container booted."""
    return {"status": "online", "service": "LangChain RAG MCP Server"}

@app.get("/health")
async def health_check():
    """Allows us to monitor load by checking how many LangGraph agents are currently connected."""
    return {"status": "healthy", "active_connections": _active_connections}


# ── Mount MCP ─────────────────────────────────────────────────────────────────
# This acts like a router. If a request path starts with /mcp, FastAPI hands it 
# entirely over to the mcp_asgi_app. 
# It creates two hidden endpoints for us automatically:
# 1. GET /mcp/sse (Where the client establishes the streaming connection)
# 2. POST /mcp/messages (Where the client actually sends tool execution commands)
app.mount("/mcp", mcp_asgi_app)


# ── Local dev entrypoint ──────────────────────────────────────────────────────
# This only runs if you execute `python server.py` directly on your machine.
# In Hugging Face, the Dockerfile command (`uvicorn server:app ...`) bypasses this block.
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860)