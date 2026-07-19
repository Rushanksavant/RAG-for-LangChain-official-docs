<div align="center">

# 🦜 LangChain RAG Assistant

**Ask anything about LangChain, LangGraph, LangSmith, or DeepAgents — get grounded answers with code examples.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-1C3C3C?style=flat)](https://langchain-ai.github.io/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C?style=flat)](https://qdrant.tech)
[![HuggingFace](https://img.shields.io/badge/🤗-HF_Spaces-FFD21E?style=flat)](https://huggingface.co/spaces)

</div>

---

## What it does

Indexes the entire official LangChain documentation ecosystem (~27K chunks) into a hybrid vector database and exposes it through a multi-step LangGraph agent that plans, retrieves, and synthesizes answers — not just keyword matches.

---

## How it works

![How it works](project_flow.PNG)

---

## Architecture decisions

| Decision | Choice | Why |
|---|---|---|
| **Chunking** | Parent-child hierarchy | Children give precise retrieval hits. Parents give LLM rich context. One chunk size forces a tradeoff — two sizes don't. |
| **Embedding** | BGE-M3 | Single model for both dense semantic vectors and sparse keyword vectors. Handles code + technical terms well. 8,192 token input limit covers full doc sections. |
| **Search** | Hybrid (dense + sparse) via RRF | Dense alone misses exact class names. Sparse alone misses semantic queries. RRF fusion gets both without tuning a weight parameter. |
| **Vector DB** | Qdrant | Native hybrid search, Rust-speed, metadata filtering, free cloud tier, portable storage files. |
| **Agent** | LangGraph | Clean map-reduce pattern for parallel sub-query retrieval. Conditional routing without spaghetti if/else. LangSmith tracing out of the box. |
| **LLM** | Gemini 3.1 Flash Lite | 500 free req/day, 1M token context, Jan 2026 training cutoff — most current LangChain knowledge. |

---

## Project structure

```
├── src/
│   ├── chunk-pipeline/          # MDX → structured chunks (parser.py, run_parser.py)
│   └── indexing-pipeline/       # Colab notebook: embed + upload to Qdrant
│
├── dB-maintenance/
│   ├── chunks_diff.py           # Diffs old vs new chunks → addition + removal lists
│   └── upsert_dB.py             # Syncs local Qdrant → Qdrant Cloud (patch or full rebuild)
│
├── hf-space/                    # 🤗 Deployed: FastAPI + FastMCP retrieval server
│   ├── server.py                # MCP tool endpoint with auth middleware
│   └── retrieval_pipeline.py    # BGE-M3 embed → hybrid search → parent fetch
│
├── langgraph-workflow/          # 🤗 Deployed: Streamlit + LangGraph agent
│   ├── agent.py                 # Graph: plan → retrieve → map → reduce → answer
│   └── app.py                   # Streamlit chat UI
│
├── bin/                         # Shell scripts: data_pull.sh, chunking.sh, upsert_dB.sh
└── docker/                      # Local Qdrant via Docker Compose
```

---

## Retrieval pipeline (in detail)

```
Query: "How do I add memory to a LangGraph agent?"
    │
    ▼  BGE-M3 encodes into:
    ├─ dense vector  [0.02, -0.31, 0.88, ...]  (1024 floats — semantic meaning)
    └─ sparse vector {token_47: 0.21, token_1830: 0.63, ...}  (keyword weights)
    │
    ▼  Qdrant hybrid search
    ├─ dense prefetch:  top 30 children by cosine similarity
    ├─ sparse prefetch: top 30 children by keyword overlap
    └─ RRF fusion:      rewards chunks ranking high in BOTH → top 15 children
    │
    ▼  Extract unique parent_ids from top 15 children
    │
    ▼  Fetch parent chunks by ID  (full doc sections, ~2-4K chars each)
    │
    ▼  Return to LangGraph agent for synthesis
```

---

## Running locally

**Prerequisites:** Docker Desktop, `uv`, API keys in `langgraph-workflow/.env`

```bash
# 1. Start the agent
uv run --with streamlit streamlit run langgraph-workflow/app.py
# → http://localhost:8501
```

**Updating the index** when LangChain docs change:

```bash
bin/data_pull.sh          # pull latest docs
bin/chunking.sh           # rechunk + diff against previous version
# → upload addition_*.json to Colab indexing notebook
# → download qdrant_storage.zip → place in data/
bin/upsert_dB.sh          # patch Qdrant Cloud with only changed chunks
```

---

## Deployment

Two HF Spaces, each auto-deployed via GitHub Actions on push:

| Space | What it runs | Trigger |
|---|---|---|
| `hf-space/` | FastMCP retrieval server (FastAPI, port 7860) | changes to `hf-space/` |
| `langgraph-workflow/` | Streamlit agent UI (port 8501) | changes to `langgraph-workflow/` |