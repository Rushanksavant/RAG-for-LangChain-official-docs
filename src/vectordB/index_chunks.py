# filepath: src/index_chunks.py
import json
import os
import uuid
import logging
from qdrant_client import QdrantClient, models

from fastembed import TextEmbedding, SparseTextEmbedding # for dense and sparse vectors

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
    
    # initiating both model components separately
    # FastEmbed library downloads the weights once to your machine (~/.fastembed/models/)
    logging.info("Loading BGE-M3 localized computation workers...")
    dense_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
    
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
        
        ## Generating Vectors
        # We wrap it in next() because .embed() outputs a generator
        dense_vector = next(dense_model.embed([text_content])).tolist()
        
        # Convert fastembed's sparse format into the format Qdrant API expects
        sparse_output = next(sparse_model.embed([text_content]))
        sparse_vector = models.SparseVector(
            indices=sparse_output.indices.tolist(),
            values=sparse_output.values.tolist()
        )

        # Document maps text to both dense and sparse vectors via underlying ONNX model
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
    # print(SparseTextEmbedding.list_supported_models())