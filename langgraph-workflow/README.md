---
title: LangGraph RAG Documentation Assistant
emoji: 🦜
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# LangGraph RAG Documentation Assistant

An asynchronous, parallelized multi-agent RAG workflow built with **LangGraph** and optimized for querying official conceptual frameworks for LangChain, LangGraph, and LangSmith. 

This application connects securely to a remote Model Context Protocol (MCP) server running on Hugging Face Spaces to retrieve highly accurate contextual document chunks.

## 🚀 Features
* **Query Planner Node:** Generates structured sub-queries utilizing advanced terminology decomposition.
* **Parallel Asynchronous Retrieval:** Fetches multi-source contexts concurrently using `asyncio.gather`.
* **Guardrail Controls:** Automatically captures off-topic or malformed inquiries without breaking runtime structures.
* **Streamlit Interface:** High-scannability frontend displaying conversational states and dynamic context extraction counts.

## 🛠️ Local Development

To spin up this application locally with full containerization tracking:

```bash
# Build the image locally
docker build -t langgraph-assistant:latest .

# Run the container with environment variables
docker run -it \
  -p 7860:7860 \
  --env-file .env \
  --name langgraph-test \
  langgraph-assistant:latest


## Keep Container alive
To keep docker container alive in hf-space after long hours of inactivity, we have created a cron-job that hits `https://rushank-langgraph-rag-agent.hf.space/_stcore/health` every 10 mins and receives **ok** status
`/_stcore/health` is generated my streamlit automatically.
**Cron-job url:** `https://console.cron-job.org/jobs/7778794/history`