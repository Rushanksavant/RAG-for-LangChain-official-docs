import json
import os
import uuid
import logging
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Constants ─────────────────────────────────────────────────────────────────
CHILDREN_COLLECTION = "documentation_chunks"
PARENTS_COLLECTION  = "documentation_parents"
CHUNKS_PATH         = "data/processed/chunks.json"
EMBED_BATCH_SIZE    = 64   # texts sent to embedding model at once
UPLOAD_BATCH_SIZE   = 64   # points uploaded to Qdrant at once


# ── Unique Ids ─────────────────────────────────────────────────────────────────
def deterministic_uuid(source_string: str) -> str:
    """Stable UUID from chunk ID string — same input always yields same UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, source_string))

# ── Get Batch ─────────────────────────────────────────────────────────────────
def batch(iterable: list, size: int):
    """Yield successive fixed-size slices of a list."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]



# ── Batch Indexing of Child Chunks ────────────────────────────────────────────
def index_children(client: QdrantClient, child_chunks: list, 
                   dense_model: TextEmbedding, sparse_model: SparseTextEmbedding,
                    ) -> None:
    """
    Embed and upload all child chunks to documentation_chunks.

    Each point:
      vector  : {"dense": [...1024 floats], "sparse": {indices, values}}
      payload : chunk_id, text, parent_id, framework, content_category,
                breadcrumbs, source_file
    """
    logging.info(f"Indexing {len(child_chunks)} children → '{CHILDREN_COLLECTION}'...")
    total = len(child_chunks)

    ## Batch-wise execution:
    for batch_idx, chunk_batch in enumerate(batch(child_chunks, EMBED_BATCH_SIZE)):
        # 1. Texts of all chunks in the batch
        texts = [c["text"] for c in chunk_batch]

        # 2. Batch embed — FastEmbed processes the whole list in one ONNX forward pass
        dense_vectors  = list(dense_model.embed(texts))   # list of np.ndarray (1024,)
        sparse_vectors = list(sparse_model.embed(texts))  # list of SparseEmbedding

        # 3. Batch Points creation 
        points = []
        for chunk, dv, sv in zip(chunk_batch, dense_vectors, sparse_vectors): # Loop for entire batch
            points.append(
                models.PointStruct(
                    id=deterministic_uuid(chunk["id"]),

                    vector={"dense": dv.tolist(),
                            "sparse": models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())},

                    payload={ # meta-data mapping
                        "chunk_id":         chunk["id"],
                        "text":             chunk["text"],
                        "parent_id":        chunk["metadata"].get("parent_id"),
                        "framework":        chunk["metadata"].get("framework"),
                        "content_category": chunk["metadata"].get("content_category"),
                        "breadcrumbs":      chunk["metadata"].get("breadcrumbs"),
                        "source_file":      chunk["metadata"].get("source_file")}
                    ))

        # 4. Batch Points Upload to Qdrant dB 
        client.upload_points(collection_name=CHILDREN_COLLECTION, points=points)
        done = min((batch_idx + 1) * EMBED_BATCH_SIZE, total)
        logging.info(f"  Children uploaded: {done}/{total}")

    logging.info(f"✅ Children indexing complete.")


# ── Batch Indexing of Parent Chunks ──────────────────────────────────────────
def index_parents(client: QdrantClient, parent_chunks: list, dense_model: TextEmbedding) -> None:
    """
    Embed and upload all parent chunks to documentation_parents collection.

    Parents use dense vectors only — they are fetched by ID after child
    retrieval and passed to the re-ranker. No sparse vector needed for parents.

    Each point:
      vector  : {"dense": [...1024 floats]}
      payload : doc_id, text, 
                framework, global_title, 
                section_heading, source_file
    """
    logging.info(f"Indexing {len(parent_chunks)} parents → '{PARENTS_COLLECTION}'...")
    total = len(parent_chunks)

    ## Batch-wise execution:
    for batch_idx, chunk_batch in enumerate(batch(parent_chunks, EMBED_BATCH_SIZE)):
        # 1. Texts of all chunks in the batch
        texts = [c["text"] for c in chunk_batch]

        # 2. Batch embed — FastEmbed processes the whole list in one ONNX forward pass
        dense_vectors = list(dense_model.embed(texts))

        # 3. Batch Points creation 
        points = []
        for chunk, dv in zip(chunk_batch, dense_vectors):
            points.append(
                models.PointStruct(
                    id=deterministic_uuid(chunk["id"]),
                    vector={"dense": dv.tolist()},
                    payload={
                        "doc_id":          chunk["id"],
                        "text":            chunk["text"],
                        "framework":       chunk["metadata"].get("framework"),
                        "global_title":    chunk["metadata"].get("global_title"),
                        "section_heading": chunk["metadata"].get("section_heading"),
                        "source_file":     chunk["metadata"].get("source_file")}
                ))

        # 4. Batch Points Upload to Qdrant dB 
        client.upload_points(collection_name=PARENTS_COLLECTION, points=points)
        done = min((batch_idx + 1) * EMBED_BATCH_SIZE, total)
        logging.info(f"  Parents uploaded: {done}/{total}")

    logging.info(f"✅ Parents indexing complete.")


# ── Run entire indexing pipeline ───────────────────────────────────────────────────────
def run_indexing_pipeline() -> None:
    # Verify chunks path-location
    if not os.path.exists(CHUNKS_PATH):
        logging.error(f"chunks.json not found at '{CHUNKS_PATH}'. Run the parser first.")
        return

    # Connect client to Qdrant Engine in docker container
    client = QdrantClient(url="http://localhost:6333")

    # ── Load models ───────────────────────────────────────────────────────────
    # FastEmbed downloads weights once to ~/.cache/fastembed/
    logging.info("Loading dense model  : BAAI/bge-large-en-v1.5")
    dense_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")

    logging.info("Loading sparse model : Qdrant/bm42-all-minilm-l6-v2-attentions")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")

    # ── Load chunks ───────────────────────────────────────────────────────────
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    parent_chunks = [c for c in all_chunks if c["type"] == "parent"]
    child_chunks  = [c for c in all_chunks if c["type"] == "child"]
    logging.info(f"Loaded {len(parent_chunks)} parents and {len(child_chunks)} children.")

    # ── Perform Indexing ──────────────────────────────────────────────────────
    index_children(client, child_chunks, dense_model, sparse_model)
    index_parents(client, parent_chunks, dense_model)

    logging.info("🎉 Indexing pipeline complete. Both collections populated.")


if __name__ == "__main__":
    run_indexing_pipeline()