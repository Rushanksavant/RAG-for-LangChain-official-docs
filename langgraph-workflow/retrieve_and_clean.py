import json
from langchain_mcp_adapters.client import MultiServerMCPClient # importing just for typehint

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retrieve_and_clean")



async def fetch_one(mcp_client: MultiServerMCPClient, query: str) -> list[dict]:
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
                    return parsed if isinstance(parsed, list) else [parsed]
                except json.JSONDecodeError:
                    return [{"text": result}]
                    
            return result if isinstance(result, list) else [result]
        except Exception as e:
            logger.error(f"Retrieval failed for '{query}': {e}")
            return []