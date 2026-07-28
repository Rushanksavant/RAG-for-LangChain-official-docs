import json
from langchain_mcp_adapters.client import MultiServerMCPClient # importing just for typehint

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retrieve_and_clean")


# ────── fetch parent docs for sub-query ────────────────────────────────────────────────────
async def fetch_one(mcp_client: MultiServerMCPClient, query: str) -> list[dict]:
        """
        This function is used:
         -  to retrieve chunks from RAG MCP for 1 sub-query
         - clean the MCP output to extract parent chunk text

        This function will be used to async retrieve chunks
        for multiple sub-queries - in the retrieve-contexts node.

        Despite adding keep_alive in HF-Space, it might sleep. Hence,
        we need a retry with backoff. Current set to 3 retries with 
        3 second intervals. This gives HF-space 9 secs to wake-up.
        """
        max_retries = 3
        retry_delay = 3  # seconds

        for attempt in range(max_retries):
            try:
                tools  = await mcp_client.get_tools()
                tool   = next(t for t in tools if t.name == "retrieve_context")
                result = await tool.ainvoke({"query": query, "top_k_child": 15, "top_k_parent": 2})

                # ... rest of existing unpacking logic unchanged ...
                if isinstance(result, list) and len(result) > 0:
                    first_item = result[0]
                    if isinstance(first_item, dict) and "text" in first_item:
                        result = first_item["text"]
                    elif hasattr(first_item, "text"):
                        result = first_item.text

                if isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                        raw_chunks = parsed if isinstance(parsed, list) else [parsed]
                    except json.JSONDecodeError:
                        raw_chunks = [{"text": result}]
                else:
                    raw_chunks = result if isinstance(result, list) else [result]

                return {query: [c["text"] for c in raw_chunks]}

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retrieval attempt {attempt + 1} failed for '{query}': {e} — retrying in {retry_delay}s")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff: 2s, 4s
                else:
                    logger.error(f"Retrieval failed after {max_retries} attempts for '{query}': {e}")
                    return {query: []}
        

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseLanguageModel # importing just for type-hint
from agent_contents.prompts  import ANSWER_PROMPT
# ──────────────────────────────────────────────────────────
async def process_packet(llm: BaseLanguageModel, query: str, chunks: list) -> str:
    """Processes an isolated sub-query and its chunks map-reduce style."""
    ctx_text = "\n\n".join(
        f"[Chunk {i+1}]\n{c if isinstance(c, str) else c.get('text', '')}" 
        for i, c in enumerate(chunks))
    
    prompt = f"""
    Analyze following context to resolve the sub-query: "{query}"
    
    Context:
    {ctx_text}

    GROUNDING RULES:
    1. Extract answers strictly from the context blocks provided.
    2. If the context contains NO relevant information whatsoever for this sub-query, output: 'Insufficient documentation available for this sub-component.' If partial information exists, use it and note what is missing.
    3. Do not assume or extrapolate syntax based on other frameworks.
    4. Do not miss any detail that might directly/indirectly answer the sub-query
    5. Preserve:
        - all code examples exactly as written.
        - all markdown tables exactly as written.
    5. Your response will be used as context to next agentic-node, hence answer's accuracy to the context is highly important to avoid hallucinations. 
    6. Never assume, infer, or hallucinate deprecation timelines, architectural migrations, or legacy framework replacements unless they are explicitly stated in the context blocks.
    """

    resp = await llm.ainvoke([SystemMessage(content=ANSWER_PROMPT), HumanMessage(content=prompt)])
    raw = resp.content
    answer = raw[0]["text"] if isinstance(raw, list) else raw
    return f"### Topic: {query}\n{answer}\n"  # For gemini
    # return f"### Topic: {query}\n{resp.content}\n"  # For gpt-oss-120b