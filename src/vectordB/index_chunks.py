# filepath: src/index_chunks.py
import json
import os
import uuid
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def generate_deterministic_uuid(source_string: str) -> str:
    """Generates a consistent UUIDv5 based on a string name to ensure idempotency."""
    namespace = uuid.NAMESPACE_DNS
    return str(uuid.uuid5(namespace, source_string))

def build_and_index_vectors():
    chunks_path = "data/processed/chunks.json"
    collection_name = "documentation_chunks"
    
    if not os.path.exists(chunks_path):
        logging.error(f"Missing chunks database at {chunks_path}. Run your parser pipeline first!")
        return
        
    # 1. Connect to local DB and load the embedding model
    client = QdrantClient(path="./data/qdrant_storage")
    logging.info("Loading local embedding model ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Read parsed chunks file
    with open(chunks_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
        
    # REMEMBER STRATEGY: We ONLY embed 'child' chunks to keep the index lean and hyper-focused
    child_chunks = [c for c in all_chunks if c["type"] == "child"]
    logging.info(f"Found {len(child_chunks)} target child chunks ready for vectorization.")
    
    points = []
    
    # 3. Vectorize and transform chunks into Qdrant Points
    for idx, chunk in enumerate(child_chunks):
        text_to_embed = chunk["text"]
        
        # Compute vector embeddings via CPU/GPU
        vector = model.encode(text_to_embed).tolist()
        
        # Generate a stable UUID based on chunk ID so we can run updates safely without duplicates
        point_id = generate_deterministic_uuid(chunk["id"])
        
        # Build the Qdrant Point Payload structure
        payload = {
            "chunk_id": chunk["id"],
            "text": text_to_embed,
            "parent_id": chunk["metadata"].get("parent_id"),
            "framework": chunk["metadata"].get("framework"),
            "content_category": chunk["metadata"].get("content_category"),
            "breadcrumbs": chunk["metadata"].get("breadcrumbs"),
            "source_file": chunk["metadata"].get("source_file")
        }
        
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        
        # Batch upsert every 500 records to maximize throughput and keep memory low
        if len(points) >= 500:
            logging.info(f"Upserting batch of {len(points)} points... (Progress: {idx+1}/{len(child_chunks)})")
            client.upsert(collection_name=collection_name, points=points)
            points = []
            
    # Upsert any remaining points in the buffer
    if points:
        logging.info(f"Upserting final residue batch of {len(points)} points...")
        client.upsert(collection_name=collection_name, points=points)
        
    logging.info("🎉 Vector indexing complete! Your local database is fully populated.")

if __name__ == "__main__":
    build_and_index_vectors()