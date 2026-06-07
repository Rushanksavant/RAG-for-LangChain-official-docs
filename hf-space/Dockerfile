FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/mcpuser \
    HF_HOME=/home/mcpuser/.cache/huggingface

RUN useradd -m -u 1000 mcpuser
WORKDIR $HOME

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Layer 1: torch CPU-only, isolated on its own index.
# torch==2.6.0 is the minimum to avoid CVE-2025-32434 — FlagEmbedding calls
# torch.load internally and versions <2.6 raise ValueError at model load.
RUN uv pip install --system \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.6.0+cpu"

# Layer 2: everything else, PyPI only.
#
# Dependency pin rationale (hard-won — do not loosen without testing):
#
# pydantic==2.10.6:
#   fastmcp 2.9.2 settings.py uses nested_model_default_partial_update=True
#   which triggers a default+default_factory conflict in pydantic>=2.11.
#   Pinning to 2.10.6 (last stable before 2.11) avoids the TypeError at startup.
#
# fastmcp==2.9.2 + mcp==1.9.4:
#   fastmcp>=2.10 requires mcp>=1.10 which requires pydantic>=2.11 — conflicts
#   with the pin above. fastmcp 2.9.2 caps mcp<1.10.0 so mcp 1.9.4 is resolved.
#   fastmcp>=2.10 also changed http_app()+lifespan wiring causing 409 on every
#   request. 2.9.2 matches the server.py mount pattern exactly.
#
# FlagEmbedding==1.4.0 requires ALL of these at import time or raises
# "Could not import module TrainingArguments":
#   transformers, datasets, accelerate, peft, sentence_transformers,
#   sentencepiece, protobuf
RUN uv pip install --system \
    "pydantic==2.10.6" \
    "pydantic-settings>=2.5.2,<3.0.0" \
    "transformers>=4.44.2,<5.0.0" \
    "datasets>=2.19.0" \
    "accelerate>=0.20.1" \
    "peft" \
    "sentence_transformers" \
    "sentencepiece" \
    "protobuf" \
    "FlagEmbedding==1.4.0" \
    "fastapi" \
    "uvicorn[standard]" \
    "fastmcp==2.9.2" \
    "mcp==1.9.4" \
    "qdrant-client==1.18.0" \
    "requests" \
    "python-dotenv" \
    "python-multipart"

# Create HF cache dir and hand ownership to mcpuser before switching users.
# Without this, BGE-M3 download fails with PermissionError on first request.
RUN mkdir -p /home/mcpuser/.cache/huggingface && \
    chown -R mcpuser:mcpuser /home/mcpuser

COPY --chown=mcpuser:mcpuser . .

USER mcpuser
EXPOSE 7860

# Single worker — BGE-M3 ~4-5 GB RAM at runtime. Multiple workers would OOM.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]


# ── Bug-fixes ───────────────────────────────────────────
# Missing FlagEmbedding 1.4.0 runtime deps — peft, sentence_transformers, protobuf, datasets
# torch==2.4.1 blocked by CVE-2025-32434 — upgraded to torch==2.6.0+cpu
# fastmcp==2.14.7 broke http_app()+lifespan wiring — pinned to 2.9.2
# HF cache dir owned by root — added mkdir + chown -R mcpuser before USER mcpuser
# pytorch index polluting PyPI resolver — split into two RUN layers, pytorch index isolated to Layer 1 only
# --dns 1.1.1.1 — ISP DNS doesn't resolve HF/Qdrant subdomains
# Docker --env-file passes quotes literally, unlike python-dotenv