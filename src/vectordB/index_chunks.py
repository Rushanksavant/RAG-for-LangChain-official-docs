# filepath: src/index_chunks.py
import json
import os
import uuid
import logging
from qdrant_client import QdrantClient, models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def generate_deterministic_uuid(source_string: str) -> str:
    """Generating a stable UUID matching the chunk ID."""
    namespace = uuid.NAMESPACE_DNS
    return str(uuid.uuid5(namespace, source_string))

def run_indexing_pipeline():
    chunks_path = "data/processed/chunks.json"
    collection_name = "documentation_chunks" # as initialized in init_vectordB.py
    
    if not os.path.exists(chunks_path): # verify if chunks exist
        logging.error("Missing chunks.json! Run your parser runner first.")
        return
        
    client = QdrantClient(url="http://localhost:6333") # connect local Qdrant client to Qdrant engine (running in docker container)
    
    # Register the BGE-M3 engine wrapper within the Qdrant client framework
    # FastEmbed library in client downloads the weights once to your machine (~/.fastembed/models/)
    BGE_M3_MODEL = "BAAI/bge-m3"
    client.set_model(BGE_M3_MODEL) # for dense vectors
    client.set_sparse_model(BGE_M3_MODEL) # for sparse vectors
    
    # loading chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    # picking only child chunks for indexing    
    child_chunks = [c for c in all_chunks if c["type"] == "child"]
    logging.info(f"Vectorizing {len(child_chunks)} child nodes via local BGE-M3 processor...")
    
    points = [] # vectors here
    
    for idx, chunk in enumerate(child_chunks):
        text_content = chunk["text"] # extracting chunk content
        point_id = generate_deterministic_uuid(chunk["id"]) # generating id
        
        payload = { # mapping meta data from child chunks to be stored in vectordB
            "chunk_id": chunk["id"],
            "text": text_content,
            "parent_id": chunk["metadata"].get("parent_id"),
            "framework": chunk["metadata"].get("framework"),
            "content_category": chunk["metadata"].get("content_category"),
            "breadcrumbs": chunk["metadata"].get("breadcrumbs"),
            "source_file": chunk["metadata"].get("source_file")
        }
        
        # Document maps text to both dense and sparse vectors via underlying ONNX model
        points.append(
            models.PointStruct(
                id=point_id,
                payload=payload,
                vector={
                    "dense": models.Document(text=text_content, model=BGE_M3_MODEL),
                    "sparse": models.Document(text=text_content, model=BGE_M3_MODEL)
                }
            )
        )
        
        # Batch upload to maximize throughput over the network API boundary
        if len(points) >= 64:
            logging.info(f"Streaming batch to container... (Progress: {idx+1}/{len(child_chunks)})")
            client.upload_points(collection_name=collection_name, points=points)
            points = []
            
    if points: # if left-over chunks (since we are uploading in batches of 64)
        logging.info(f"Streaming final residue batch of {len(points)} points...")
        client.upload_points(collection_name=collection_name, points=points)
        
    logging.info("🎉 Database population complete! All chunks securely indexed in the container system.")

if __name__ == "__main__":
    run_indexing_pipeline()