# filepath: src/resume_indexing.py
import json
import os
import logging
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

# Import only the stable UUID generator from your untouched script
from index_chunks import generate_deterministic_uuid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_all_existing_ids(client, collection_name): ## Extracts all vector ids from qdrant dB
    """Scrolls through Qdrant to download all indexed IDs in a single upfront sweep."""
    existing_ids = set()
    next_page_offset = None
    
    logging.info("Scanning Qdrant storage to fetch indexed point IDs...")
    
    while True:
        # Pull 5,000 IDs at a time with no vectors/payload to keep it blazing fast
        records, next_page_offset = client.scroll(
            collection_name=collection_name,
            limit=5000,
            with_payload=False,
            with_vectors=False,
            offset=next_page_offset
        )
        
        for record in records:
            existing_ids.add(record.id)
            
        if next_page_offset is None:
            break
            
    logging.info(f"Successfully cached {len(existing_ids)} existing point IDs locally in memory.")
    return existing_ids

def run_resume_pipeline(): # same as run_indexing_pipeline in index_chunks.py
    chunks_path = "data/processed/chunks.json"
    collection_name = "documentation_chunks"
    
    if not os.path.exists(chunks_path):
        logging.error("Missing chunks.json! Run your parser runner first.")
        return
        
    client = QdrantClient(url="http://localhost:6333")
    
    # 1. Gather all points currently living in Docker storage
    existing_ids = get_all_existing_ids(client, collection_name)
    
    logging.info("Loading BGE-M3 localized computation workers...")
    dense_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
        
    child_chunks = [c for c in all_chunks if c["type"] == "child"]
    total_chunks = len(child_chunks)
    logging.info(f"Evaluating {total_chunks} total chunks against local ID cache...")
    
    points = []
    
    # 2. pipeline loop with id uniqueness verification
    for idx, chunk in enumerate(child_chunks):
        point_id = generate_deterministic_uuid(chunk["id"])
        
        # Immediate local lookup - completely avoids hitting Qdrant network ports
        if point_id in existing_ids:
            continue
            
        # Code execution lands here ONLY for new chunks
        text_content = chunk["text"]
        
        payload = {
            "chunk_id": chunk["id"],
            "text": text_content,
            "parent_id": chunk["metadata"].get("parent_id"),
            "framework": chunk["metadata"].get("framework"),
            "content_category": chunk["metadata"].get("content_category"),
            "breadcrumbs": chunk["metadata"].get("breadcrumbs"),
            "source_file": chunk["metadata"].get("source_file")
        }
        
        dense_vector = next(dense_model.embed([text_content])).tolist()
        
        sparse_output = next(sparse_model.embed([text_content]))
        sparse_vector = models.SparseVector(
            indices=sparse_output.indices.tolist(),
            values=sparse_output.values.tolist()
        )
        
        points.append(
            models.PointStruct(
                id=point_id,
                payload=payload,
                vector={
                    "dense": dense_vector,
                    "sparse": sparse_vector
                }
            )
        )
        
        if len(points) >= 64:
            logging.info(f"Streaming fresh batch to container... (Progress: {idx+1}/{total_chunks})")
            client.upload_points(collection_name=collection_name, points=points)
            points = []
            
    if points:
        logging.info(f"Streaming final residue batch of {len(points)} points...")
        client.upload_points(collection_name=collection_name, points=points)
        
    logging.info("🎉 System fully caught up! All remaining chunks securely indexed.")

if __name__ == "__main__":
    run_resume_pipeline()