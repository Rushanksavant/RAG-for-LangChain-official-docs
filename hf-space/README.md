---
title: LangChain RAG MCP Server
emoji: 🚀
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
---

# Remote MCP Server for LangChain RAG Pipeline
This Space exposes a production-ready Model Context Protocol (MCP) server over HTTP/SSE.
It serves as the highly optimized retrieval layer for the corresponding LangGraph agentic workflow.

# GitHub Actions configured
for this folder `hf-space/`, we have configured a GitHub action (`.github/workflows/hf_sync.yml`)to push changes in code to HF space repo.
Hf space should be created.

### Other required steps:
- We need a hugging face **Write** token
- Go-to: GitHub project repository (on web) -->  Settings --> Secrets and variables --> Actions
- Click the **New repository** secret button
- Name the secret exactly: `HF_TOKEN_WRITE`. Paste token in **Secret field** and click **Add secret**

**Setup/Use CI-CD:**
In local-project terminal
```bash
# 1. Stage the workflow file and your hf-space folder
git add .github/workflows/mcp_deploy.yml hf-space/

# 2. Commit the infrastructure changes
git commit -m "chore: setup automated CI/CD for remote MCP server"

# 3. Push to your main branch (or whichever branch is set in the hf_sync.yml)
git push origin main
```
- Go-to HF-space page --> Settings --> Variables and secrets | Add the MCP_SECRET_KEY as new secret

Alternative to this was using Git Subtree (Manual Command Line), initializing git remote for hf-face repo and pushing all changes manually to hf-face repo.


# MCP Server — Architecture & Working Guide

---

## What this server does

This server exposes the RAG retrieval pipeline as a **remote tool** that any MCP-compatible client (like a LangGraph agent) can call over HTTP. Instead of the agent running retrieval logic locally, it calls this server and gets back ranked documentation chunks.

```
LangGraph Agent  →  POST /mcp  →  MCP Server  →  Qdrant Cloud
                 ←  JSON result  ←             ←  ranked chunks
```

---

## server.py

### The three layers

```
┌─────────────────────────────────────────┐
│  FastAPI (outer app)                    │  ← handles auth, health
│  ┌───────────────────────────────────┐  │
│  │  FastMCP (mounted at /)           │  │  ← handles MCP protocol
│  │  ┌─────────────────────────────┐  │  │
│  │  │  retrieve_context() tool    │  │  │  ← your actual logic
│  │  └─────────────────────────────┘  │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**FastAPI** is the outer HTTP server. It handles:
- Auth middleware — validates `X-API-Key` on every request except `/health`
- Health endpoint — returns `{"status": "healthy"}` for HF Spaces probe

**FastMCP** is an ASGI (Asynchronous Server Gateway Interface) sub-application mounted inside FastAPI. It handles the MCP protocol layer — tool discovery, serialization, session management. It knows nothing about auth; FastAPI handles that before requests reach it.

**The tool** is a plain Python async function decorated with `@mcp.tool()`. When a client calls `retrieve_context`, FastMCP deserializes the arguments, calls this function, and serializes the return value back.

---

### The mount trick

```python
mcp_app = mcp.http_app(path="/mcp")   # internal path = /mcp
app.mount("/", mcp_app)               # mounted at root
```

FastMCP's internal path and FastAPI's mount path **combine**. If you set `path="/mcp"` and mount at `/`, the public URL becomes `/mcp`. If you set `path="/"` and mount at `/mcp`, FastAPI creates a double-path `/mcp/mcp` — which is the bug we hit repeatedly. The working rule: **set the path inside `http_app()`, mount at `/`**.

---

### Lifespan sharing

```python
app = FastAPI(lifespan=mcp_app.lifespan)
```

FastMCP's streamable-http transport uses an internal session manager (a task group) that must be started before any requests arrive. FastAPI doesn't automatically start sub-app lifecycles — you must pass `mcp_app.lifespan` explicitly. Without this, every request returns `Session terminated` because the session manager task group was never initialized.

---

### Auth middleware

```python
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path in ["/health"]:
        return await call_next(request)   # bypass auth for health probe
    # validate X-API-Key header...
```

Every request passes through this middleware before reaching any route or the mounted FastMCP sub-app. `/health` bypasses auth because HF Spaces sends probe requests without API keys — if the probe gets a 401, HF marks the Space as unhealthy and stops routing traffic to it.

---

## Dockerfile

```dockerfile
FROM python:3.11-slim

# Create non-root user — required by HF Spaces security policy
RUN useradd -m -u 1000 mcpuser

# Install uv (fast dependency resolver)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install dependencies as root, then drop to non-root user
RUN uv pip install --system fastapi "uvicorn[standard]" fastmcp ...

COPY --chown=mcpuser:mcpuser . .
USER mcpuser

# Single worker — when BGE-M3 is integrated (~4-5GB RAM),
# multiple workers would each load their own model copy and OOM
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

**Why non-root user:** HF Spaces enforces UID 1000. Running as root causes the Space to fail security checks.

**Why single worker:** Uvicorn workers are separate processes. Each would load BGE-M3 independently — 4 workers × 5GB = 20GB, exceeding HF Spaces' 16GB free tier RAM. Single worker + async concurrency handles parallel requests without multiplying RAM usage.

**Why port 7860:** HF Spaces' NGINX proxy routes all external traffic to port 7860 internally. Using any other port means the app is unreachable.

---

## .dockerignore

Prevents unnecessary files from being copied into the Docker image:
- `.env` — never ship secrets in an image
- `__pycache__`, `.git`, `*.pyc` — build artifacts and version control internals
- Test files, local notebooks — not needed at runtime

Smaller image = faster builds and deploys on HF Spaces.

---

## mcp_server_test.py — how the 4 tests work

### Test 1: Tool discovery
```python
client = MultiServerMCPClient(config)
tools = await client.get_tools()
```
Client sends a `GET /mcp` → server responds with a list of available tools and their JSON schemas (name, description, argument types). This is how LangGraph discovers what tools are available at startup — it calls `get_tools()` once and binds them as LangChain tools.

### Test 2: Basic invocation
```python
result = await retrieve_tool.ainvoke({"query": "...", "top_k": 3})
```
Client sends `POST /mcp` with the tool name and serialized arguments → FastAPI middleware validates `X-API-Key` → FastMCP routes to `retrieve_context()` → function runs → result returned as JSON → client deserializes and returns the string. This is the exact same call path the LangGraph agent uses every time it needs retrieval.

### Test 3: Concurrent invocations
```python
tasks = [invoke_single(q, i) for i in range(5)]
results = await asyncio.gather(*tasks)
```
Fires 5 tool calls simultaneously using `asyncio.gather`. Since both the client and server are async, these run in parallel — 5 HTTP requests in flight at the same time. Tests that the server handles concurrent load without crashing or blocking. With the dummy tool this is trivial; after BGE-M3 is integrated, this test reveals the actual RAM/compute limits (~3-4 concurrent requests on free tier).

### Test 4: Auth rejection
```python
bad_client = MultiServerMCPClient({"headers": {"X-API-Key": "wrong-key"}})
await bad_client.get_tools()  # should raise
```
Deliberately sends the wrong API key and expects an exception. Confirms the auth middleware correctly rejects unauthorized clients with 401. If this test passes (exception raised), auth is working. If the server accepts the wrong key, auth is broken.

---

## Request lifecycle in production (HF Spaces)

```
User query
    ↓
LangGraph agent decides to call retrieve_context
    ↓
MultiServerMCPClient serializes {query, top_k} → POST /mcp
    ↓
HF Spaces NGINX proxy → forwards to port 7860
    ↓
FastAPI auth_middleware → validates X-API-Key
    ↓
FastMCP session manager → deserializes MCP protocol message
    ↓
retrieve_context(query, top_k) runs
    [dummy now → BGE-M3 embed + Qdrant search + HF reranker after integration]
    ↓
FastMCP serializes result → HTTP response
    ↓
LangGraph receives result as tool message in conversation state
    ↓
Agent uses context to generate answer
```

---

## CI/CD — GitHub Actions → HF Spaces

The `hf_sync.yml` workflow:
1. Triggers on push to `rag_pipeline_v2` branch when files in `hf-space/` change
2. Clones the HF Space git repo into a temp folder
3. `rsync`s the `hf-space/` directory contents into it (strips the `hf-space/` prefix)
4. Commits and force-pushes to the HF Space repo
5. HF Spaces detects the push, rebuilds the Docker image, redeploys

HF Spaces is just a git repo with a Docker build system attached. Every push triggers a full rebuild and redeploy.

---

## Key environment variables

| Variable | Where set | Purpose |
|---|---|---|
| `MCP_SECRET_KEY` | HF Spaces secret / local `.env` | Auth key validated on every request |
| `SERVER_URL` | local `.env` only | MCP client connection URL |
| `QDRANT_CLOUD_KEY` | HF Spaces secret | Qdrant auth (needed after retrieval integration) |
| `HF_API_KEY` | HF Spaces secret | HF reranker API auth (needed after retrieval integration) |
| `HF_TOKEN_WRITE` | HF write token / local `.env` | for builds on HF space (via github actions) |


## Queries answered:

**Wrapping FastMCP inside FastAPI**

Yes, exactly. FastMCP can run standalone with `mcp.run(transport="streamable-http")` and that gives you a working MCP server on its own. The only reasons to wrap it in FastAPI are: custom auth, additional HTTP endpoints like `/health`, or future needs like rate limiting or logging middleware. If none of those were needed, standalone FastMCP would be simpler.

---

**Session manager and lifespan sharing**

Yes — all streamable-http MCP servers use an internal session manager. It's part of the MCP SDK's streamable-http implementation, not FastMCP-specific. The session manager is a task group that manages the lifecycle of concurrent MCP sessions. Whenever you mount any streamable-http MCP app inside FastAPI or Starlette, you must share the lifespan — otherwise the task group is never started and every request gets `Session terminated`. SSE transport has the same requirement.

---

**`/health` endpoint**

HF Spaces' infrastructure sends periodic HTTP GET requests to your running container to check if it's alive. If it gets anything other than a 200 response, it marks the Space as unhealthy and stops routing user traffic to it. Without `/health`, the probe hits `/` which is occupied by FastMCP — FastMCP would return an MCP protocol response, not a clean 200 JSON, and HF might misinterpret it. Removing `/health` is technically fine if HF's probe is satisfied by the root `/` response, but it's fragile and you lose the `active_connections` visibility. It also makes the server harder to monitor externally.

---

**Root vs non-root user**

Root user (UID 0) inside a container has full system privileges — it can install packages, modify system files, and if the container is ever compromised, the attacker has root access to the host in misconfigured setups. Non-root user (UID 1000) has no system privileges — it can only read/write files it owns. HF Spaces enforces UID 1000 as a security policy. Beyond HF, running as non-root is standard container security practice — principle of least privilege.

---

**BGE-M3 concurrent requests**

BGE-M3 does not handle concurrent requests by default. `BGEM3FlagModel.encode()` is a synchronous blocking call — it occupies the CPU/GPU for its entire duration. In an async FastAPI server, calling it directly inside an async function blocks the entire event loop, meaning all other requests wait until it finishes.

When integrating BGE-M3 you must offload it to a thread pool:

```python
import asyncio
result = await asyncio.get_event_loop().run_in_executor(
    None, 
    lambda: bge_model.encode([query], return_dense=True, return_sparse=True)
)
```

`run_in_executor` runs the blocking call in a separate thread, freeing the event loop to handle other requests concurrently. On GPU, the GPU itself serializes operations — two concurrent encode calls will queue on the GPU, not truly run in parallel. On CPU with multiple cores, thread-level parallelism is possible but limited by Python's GIL for CPU-bound work. Realistic concurrency on HF Spaces free tier: 3-4 requests before latency degrades noticeably.

---

**Auth on GET vs POST**

Your current middleware applies auth to **all** requests except `/health` — including `GET /mcp`. So tool discovery also requires the API key. This is correct behavior for a production server — you don't want unauthenticated clients to even see what tools are available.

The middleware doesn't distinguish between GET and POST — it checks the path, not the method. Any request to `/mcp` (GET or POST) goes through the key validation. Only `/health` bypasses it.


# Adding the original retrieval pipeline

- Add these to HF Spaces secrets:
QDRANT_CLOUD_CLUSTER_ENDPOINT = https://your-cluster.qdrant.io
QDRANT_CLOUD_KEY = your-key
MCP_SECRET_KEY = your-decided-key
HF_API_KEY = your-hf-token



# Testing in local machine

**Method 1: local dev env**
- run server.py in one terminal and test scrip in another
- make sure SERVER_URL points to localhost server link

**Method 2: Docker build**
To verify docker image is designed correctly
- cd hf-space

- Step 1: Build the Docker Image: `docker build -t mcp-server-local .` (if major changes, `docker build --no-cache -t mcp-server-local .`)

- Step 2: Spin Up the Container Locally
Run the container, map port 7860, and feed it the local .env configuration file so it has access to your Qdrant cluster and credentials.
`docker run --rm --name mcp-test-container -p 7860:7860 --env-file ../.env --dns 1.1.1.1 mcp-server-local`
Uvicorn will bind successfully to http://0.0.0.0:7860 under the `mcpuser` profile

- Step 3: Run the Test Script
In new terminal execute test script targeted at the local container's endpoint. 
Make sure to replace `mcp_secret_key` with the actual value inside `.env` file