FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/mcpuser

# Create non-root user — required by HF Spaces security policy
RUN useradd -m -u 1000 mcpuser
WORKDIR $HOME

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install dependencies as root before switching user
# fastmcp pulls in the MCP SDK, starlette, httpx automatically
RUN uv pip install --system \
    fastapi \
    "uvicorn[standard]" \
    fastmcp \
    qdrant-client \
    requests \
    python-dotenv \
    python-multipart

COPY --chown=mcpuser:mcpuser . .

USER mcpuser
EXPOSE 7860

# Single worker only — BGE-M3 will occupy ~4-5GB RAM when integrated.
# Multiple uvicorn workers would each load their own model copy and OOM
# on HF Spaces free tier (16GB). Single worker + async handles concurrency.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]