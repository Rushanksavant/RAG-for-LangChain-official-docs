"""
RAG Retrieval Pipeline.

Handles:
  - BGE-M3 local embeddings (dense + sparse)
  - Qdrant Cloud hybrid search (RRF fusion)
  - HF Inference API reranker (bge-reranker-v2-m3)
"""

import os
import time
import uuid
import asyncio
import logging
import json
from typing import Callable, Awaitable

import requests
from qdrant_client import QdrantClient, models

logger = logging.getLogger("retrieval_pipeline")

# ── Configuration ─────────────────────────────────────────────────────────────
CHILDREN_COLLECTION = "documentation_child_chunks"
PARENTS_COLLECTION  = "documentation_parent_chunks"


# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_bge_model = None
_qdrant    = None


def _deterministic_uuid(source_string: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, source_string))


def _get_bge_model():
    global _bge_model
    if _bge_model is None:
        logger.info("Loading BGE-M3 model (~2.2 GB, first request only)...")
        from FlagEmbedding import BGEM3FlagModel
        _bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, device="cpu")
        logger.info("BGE-M3 ready.")
    return _bge_model


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(
            url=os.environ["QDRANT_CLOUD_CLUSTER_ENDPOINT"],
            api_key=os.environ["QDRANT_CLOUD_KEY"],
            timeout=30.0,
        )
    return _qdrant


# ── Reranker ──────────────────────────────────────────────────────────────────
def _rerank_with_api(query: str, documents: list, max_retries: int = 6) -> list | None:
    hf_token = os.environ.get("HF_API_KEY", "")
    headers  = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    # api-inference.huggingface.co was deprecated and decommissioned.
    # New endpoint: router.huggingface.co/hf-inference
    # New payload format: {"query": str, "texts": [str, ...]}
    # Response format: [{"index": int, "score": float}, ...]  sorted by index
    api_url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-reranker-v2-m3/v1/rerank"

    payload = {
        "query": query,
        "texts": documents,
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)

            if response.status_code == 503:
                wait_time = min(response.json().get("estimated_time", 15), 20)
                logger.warning(f"HF model loading, retrying in {wait_time}s ({attempt}/{max_retries})")
                time.sleep(wait_time)
                continue
            elif response.status_code == 429:
                time.sleep(attempt * 4)
                continue
            elif response.status_code != 200:
                if attempt == max_retries:
                    logger.error(f"HF API failed after {max_retries} attempts: {response.status_code} {response.text}")
                    return None
                time.sleep(attempt * 2)
                continue

            data = response.json()
            # Response: [{"index": 0, "score": 0.92}, {"index": 1, "score": 0.45}, ...]
            # Sorted by score descending by the API — we need scores in original
            # document order so zip(documents, scores) aligns correctly.
            scores_by_index = {item["index"]: item["score"] for item in data}
            return [scores_by_index[i] for i in range(len(documents))]

        except requests.exceptions.ConnectionError as ce:
            logger.error(f"Fatal connection/DNS error to HF Reranker: {ce}")
            return None  # Triggers instant fallback to RRF ordering

        except Exception as e:
            logger.warning(f"Reranker attempt {attempt} failed: {e}")
            if attempt == max_retries:
                return None
            time.sleep(attempt * 2)

    return None


# ── Main Pipeline Execution ───────────────────────────────────────────────────
async def execute_retrieval(
    query: str,
    top_k_child: int,
    top_k_parent: int,
    log_callback: Callable[[str], Awaitable[None]]
) -> str:
    """
    Executes the full hybrid retrieval and reranking pipeline.
    """

    await log_callback(f"Query received: '{query}'")

    # ── Step 1: Embed query ───────────────────────────────────────────────────
    await log_callback("Step 1/5: Generating query embeddings (BGE-M3)...")
    
    bge = _get_bge_model()
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(
        None,
        lambda: bge.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
    )

    dense_vector = embeddings["dense_vecs"][0].tolist()
    sparse_dict  = embeddings["lexical_weights"][0]
    qdrant_sparse = models.SparseVector(
        indices=[int(k) for k in sparse_dict.keys()],
        values=[float(v) for v in sparse_dict.values()],
    )
    await log_callback("Step 1/5: Embeddings ready.")

    # ── Step 2: Hybrid search ─────────────────────────────────────────────────
    await log_callback(f"Step 2/5: Running hybrid search (RRF fusion, prefetch={top_k_child * 2})...")
    prefetch_limit = top_k_child * 2

    qdrant = _get_qdrant()
    # Run inside executor to prevent event loop blocking
    search_results = await loop.run_in_executor(
        None,
        lambda: qdrant.query_points(
            collection_name=CHILDREN_COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_vector,  using="dense",  limit=prefetch_limit),
                models.Prefetch(query=qdrant_sparse, using="sparse", limit=prefetch_limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k_child,
            with_payload=True))
    await log_callback(f"Step 2/5: Retrieved {len(search_results.points)} child chunks.")

    # ── Step 3: Deduplicate parent IDs ────────────────────────────────────────
    await log_callback("Step 3/5: Extracting unique parent IDs...")
    parent_ids   = []
    seen_parents = set()
    for point in search_results.points:
        parent_id = point.payload.get("parent_id")
        if parent_id and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent_ids.append(parent_id)

    if not parent_ids:
        await log_callback("No parent IDs found — returning empty result.")
        return "[]"

    await log_callback(f"Step 3/5: Found {len(parent_ids)} unique parent chunks.")

    # ── Step 4: Fetch parent texts ────────────────────────────────────────────
    await log_callback("Step 4/5: Fetching full parent chunk texts from Qdrant...")
    parent_uuids  = [_deterministic_uuid(pid) for pid in parent_ids]
    parent_points = await loop.run_in_executor(
                                    None,
                                    lambda: qdrant.retrieve(
                                        collection_name=PARENTS_COLLECTION,
                                        ids=parent_uuids,
                                        with_payload=True,
                                        with_vectors=False))

    parent_documents = []
    for point in parent_points:
        p = point.payload
        if p and "text" in p:
            parent_documents.append({"id": p.get("doc_id"), "text": p["text"]})

    if not parent_documents:
        await log_callback("Parent texts not found — returning empty result.")
        return "[]"

    await log_callback(f"Step 4/5: Fetched {len(parent_documents)} parent texts.")

    # ── Step 5: Rerank ────────────────────────────────────────────────────────
    await log_callback("Step 5/5: Reranking with bge-reranker-v2-m3 (HF Inference API)...")
    texts_to_rerank = [doc["text"] for doc in parent_documents]

    scores = await loop.run_in_executor(
        None,
        lambda: _rerank_with_api(query, texts_to_rerank)
    )

    if scores is None:
        # Fallback: RRF ordering
        await log_callback("⚠️  Reranker unavailable — falling back to RRF ordering.")
        fallback_results = parent_documents[:top_k_parent]
        return json.dumps([
            {
                "parent_id":       doc["id"],
                "text":            doc["text"],
                "relevance_score": None,
                "source":          "rrf_fallback",
            }
            for doc in fallback_results
        ], ensure_ascii=False)

    # Assemble and sort by reranker score
    ranked = sorted(
        [{
        "parent_id":       doc["id"],
        "text":            doc["text"],
        "relevance_score": float(score),
        "source":          "reranker"}
            for doc, score in zip(parent_documents, scores)],
        key=lambda x: x["relevance_score"], reverse=True
        )[:top_k_parent]

    await log_callback(f"Step 5/5: Reranking complete. "
                    f"Top score: {ranked[0]['relevance_score']:.4f}")

    return json.dumps(ranked, ensure_ascii=False)