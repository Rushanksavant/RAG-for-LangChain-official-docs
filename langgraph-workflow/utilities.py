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
        """
        try:
            tools  = await mcp_client.get_tools()
            tool   = next(t for t in tools if t.name == "retrieve_context")
            result = await tool.ainvoke({"query": query, "top_k_child": 15, "top_k_parent": 2})
            
            # 1. Unpack the MCP tool content block envelope
            if isinstance(result, list) and len(result) > 0:
                first_item = result[0]
                if isinstance(first_item, dict) and "text" in first_item:
                    result = first_item["text"]
                elif hasattr(first_item, "text"):
                    result = first_item.text

            # 2. Safely parse the double-serialized inner JSON string
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                    raw_chunks = parsed if isinstance(parsed, list) else [parsed]
                except json.JSONDecodeError:
                    raw_chunks = [{"text": result}]
            else:
                raw_chunks = result if isinstance(result, list) else [result]
                    
            # 3. return as a {query: [retreived-chunks text]} packet
            return {query: [c["text"] for c in raw_chunks]}
        
        except Exception as e:
            logger.error(f"Retrieval failed for '{query}': {e}")
            return {query: []}
        

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseLanguageModel # importing just for type-hint
from prompts  import ANSWER_PROMPT
# ──────────────────────────────────────────────────────────
async def process_packet(llm: BaseLanguageModel, query: str, chunks: list) -> str:
    """Processes an isolated sub-query and its chunks map-reduce style."""
    ctx_text = "\n\n".join(f"[Chunk {i+1}]\n{c.get('text', '')}" for i, c in enumerate(chunks))
    
    prompt = f"""
    Analyze following context to answer: "{query}"
    
    Context:
    {ctx_text}
    """

    resp = await llm.ainvoke([SystemMessage(content=ANSWER_PROMPT), HumanMessage(content=prompt)])
    return f"### Topic: {query}\n{resp.content}\n"