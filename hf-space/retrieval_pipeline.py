"""
RAG Retrieval Pipeline.

Handles:
  - BGE-M3 local embeddings (dense + sparse)
  - Qdrant Cloud hybrid search (RRF fusion)
  - HF Inference API reranker (bge-reranker-v2-m3)
"""

import os
import uuid
import asyncio
import logging
import json
from typing import Callable, Awaitable

from qdrant_client import QdrantClient, models

logger = logging.getLogger("retrieval_pipeline")

# ── Configuration ─────────────────────────────────────────────────────────────
CHILDREN_COLLECTION = "documentation_child_chunks"
PARENTS_COLLECTION  = "documentation_parent_chunks"


# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_bge_model  = None
_reranker   = None
_qdrant     = None


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


def _get_reranker():
    global _reranker
    if _reranker is None:
        logger.info("Loading BGE-Reranker-v2-m3 (~1.1 GB, first request only)...")
        from FlagEmbedding import FlagReranker
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False, device="cpu")
        logger.info("Reranker ready.")
    return _reranker


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
def _rerank_local(query: str, documents: list) -> list | None:
    """
    Reranks documents locally using FlagReranker (BAAI/bge-reranker-v2-m3).

    Replaces the old HF Inference API call — api-inference.huggingface.co was
    decommissioned and bge-reranker-v2-m3 is not currently supported by any
    hf-inference router endpoint. Running locally avoids all external API
    dependencies for reranking and is faster (no network round-trip).

    FlagReranker.compute_score() is synchronous/blocking — caller must
    run it in an executor to avoid blocking the async event loop.

    Returns scores in the same order as the input documents list.
    """
    try:
        reranker = _get_reranker()
        pairs = [[query, doc] for doc in documents]
        scores = reranker.compute_score(pairs, normalize=True)
        # compute_score returns a single float when len(pairs)==1, else a list
        if isinstance(scores, float):
            scores = [scores]
        return scores
    except Exception as e:
        logger.error(f"Local reranker failed: {e}")
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
    # DISABLED — two reasons:
    #   1. API: api-inference.huggingface.co decommissioned; bge-reranker-v2-m3
    #      not supported by any current router.huggingface.co provider endpoint.
    #   2. Latency + RAM: running FlagReranker locally adds ~3-4 sec on CPU and
    #      ~1.1 GB RAM on top of BGE-M3's ~4.5 GB. On HF Spaces free tier
    #      (2 vCPU, 16 GB) this pushes single-request latency to 10-12 sec and
    #      risks OOM on concurrent requests. bge-reranker-v2-m3 requires 8192
    #      token input limit (parent chunks go up to 8000+ chars) so smaller
    #      alternatives like bge-reranker-base (512 token limit) would silently
    #      truncate most chunks, making scores unreliable.
    #
    # Re-enable when one of these becomes viable:
    #   a) HF Inference Providers add bge-reranker-v2-m3 support
    #   b) Upgrade to HF Spaces paid tier (more RAM/CPU headroom)
    #   c) Eval pipeline confirms reranking meaningfully improves retrieval
    #      quality, justifying the latency+cost tradeoff
    #
    # scores = await loop.run_in_executor(
    #     None,
    #     lambda: _rerank_local(query, texts_to_rerank)
    # )
    #
    # if scores is None:
    #     await log_callback("⚠️  Reranker unavailable — falling back to RRF ordering.")
    #     fallback_results = parent_documents[:top_k_parent]
    #     return json.dumps([
    #         {
    #             "parent_id":       doc["id"],
    #             "text":            doc["text"],
    #             "relevance_score": None,
    #             "source":          "rrf_fallback",
    #         }
    #         for doc in fallback_results
    #     ], ensure_ascii=False)
    #
    # ranked = sorted(
    #     [
    #         {
    #             "parent_id":       doc["id"],
    #             "text":            doc["text"],
    #             "relevance_score": float(score),
    #             "source":          "reranker",
    #         }
    #         for doc, score in zip(parent_documents, scores)
    #     ],
    #     key=lambda x: x["relevance_score"],
    #     reverse=True,
    # )[:top_k_parent]
    #
    # await log_callback(
    #     f"Step 5/5: Reranking complete. "
    #     f"Top score: {ranked[0]['relevance_score']:.4f}"
    # )
    # return json.dumps(ranked, ensure_ascii=False)

    # Returning top_k_parent results in RRF order (hybrid dense+sparse fusion)
    await log_callback("Step 5/5: Skipped — returning top results in RRF order.")
    rrf_results = parent_documents[:top_k_parent]
    return json.dumps([
        {
            "parent_id":       doc["id"],
            "text":            doc["text"],
            "relevance_score": None,
            "source":          "rrf",
        }
        for doc in rrf_results
    ], ensure_ascii=False)