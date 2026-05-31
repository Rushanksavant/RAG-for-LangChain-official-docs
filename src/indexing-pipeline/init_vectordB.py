import logging
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Collection names ──────────────────────────────────────────────────────────
CHILDREN_COLLECTION = "documentation_child_chunks"   # hybrid dense+sparse, 18K child chunks
PARENTS_COLLECTION  = "documentation_parent_chunks"  # dense-only, 8.5K parent chunks

DENSE_DIM = 1024  # BAAI/bge-large-en-v1.5 output dimension


def _create_children_collection(client: QdrantClient) -> None:
    """
    Hybrid collection for child chunks.
      - dense  : BGE-large-en-v1.5 (1024d, cosine) — semantic similarity search
      - sparse : BM42 attentions — keyword / code-token matching

    Payload indexes enable fast filtered search without full collection scans:
      - framework        (keyword) : filter by langchain / langgraph / langsmith / deepagents
      - content_category (keyword) : filter by code_snippet / descriptive_text / structured_table
      - parent_id        (keyword) : bulk-fetch all children of a parent in one call
    """
    logging.info(f"Creating '{CHILDREN_COLLECTION}' (hybrid dense+sparse)...")

    client.create_collection(
        collection_name=CHILDREN_COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_DIM,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        },
    )

    # Payload indexes — keyword type for exact-match filtering
    for field in ("framework", "content_category", "parent_id"):
        client.create_payload_index(
            collection_name=CHILDREN_COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD, # asking Qdrant to allow keyword filtering 
        )
        logging.info(f"  Payload index created: {field}")


def _create_parents_collection(client: QdrantClient) -> None:
    """
    Dense-only collection for parent chunks.
    Parents are never searched directly — they are fetched by ID after a
    child search, then passed to the reranker for full-context scoring.
    So no sparse vector needed, and the only payload index needed is
    framework (for optional pre-filtering during re-fetch).

    Payload fields stored per parent point:
      - doc_id           : matches global_doc_id from parser (same as parent chunk id) (indexed)
      - framework        : langchain / langgraph / etc. (indexed)
      - global_title     : document title
      - section_heading
      - source_file
      - text             : full parent text (read by reranker)
    """
    logging.info(f"Creating '{PARENTS_COLLECTION}' (dense-only)...")

    client.create_collection(
        collection_name=PARENTS_COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_DIM,
                distance=models.Distance.COSINE,
            )
        },
    )

    for field in ("framework", "doc_id"):
        client.create_payload_index(
            collection_name=PARENTS_COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        logging.info(f"  Payload index created: {field}")


# ── Initialize vectordB ──────────────────────────────────────────────────────────
def initialize_production_schema() -> None:
    container_url = "http://localhost:6333"
    logging.info(f"Connecting to Qdrant engine at {container_url}...")

    client = QdrantClient(url=container_url) # client connecting with Qdrant Engine (in docker container)

    # Verify connection is alive before doing anything
    try:
        client.get_collections()
    except UnexpectedResponse:
        logging.error("Could not connect to Qdrant. Is the Docker container running?")
        return

    # ── Create Children Collection ───────────────────────────────────────────────────
    if client.collection_exists(CHILDREN_COLLECTION):
        logging.info(f"'{CHILDREN_COLLECTION}' already exists — skipping.")
    else:
        _create_children_collection(client)
        logging.info(f"✅ '{CHILDREN_COLLECTION}' ready.")

    # ── Create Parents Collection ────────────────────────────────────────────────────
    if client.collection_exists(PARENTS_COLLECTION):
        logging.info(f"'{PARENTS_COLLECTION}' already exists — skipping.")
    else:
        _create_parents_collection(client)
        logging.info(f"✅ '{PARENTS_COLLECTION}' ready.")

    logging.info("🚀 Schema initialization complete.")


if __name__ == "__main__":
    initialize_production_schema()