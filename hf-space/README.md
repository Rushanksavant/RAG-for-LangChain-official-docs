---
title: LangChain RAG MCP Server
emoji: 🚀
colorFrom: green
colorTo: white
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

Alternative to this was using Git Subtree (Manual Command Line), initializing git remote for hf-face repo and pushing all changes manually to hf-face repo.