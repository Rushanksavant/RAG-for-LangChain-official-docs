# filepath: src/init_vector_db.py
import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def initialize_qdrant():
    db_path = "./data/qdrant_storage"
    collection_name = "documentation_chunks"
    
    logging.info(f"Connecting to self-hosted local Qdrant instance at: {db_path}")
    # Initialize client in local disk-persisted mode
    client = QdrantClient(path=db_path)
    
    # We are using a standard local embedding model ('all-MiniLM-L6-v2') 
    # It outputs vectors with exactly 384 dimensions.
    VECTOR_DIMENSION = 384 
    
    # Check if collection already exists to prevent overwriting
    if client.collection_exists(collection_name):
        logging.info(f"Collection '{collection_name}' already exists. Skipping initialization.")
        return client
        
    logging.info(f"Creating collection '{collection_name}' with Cosine Distance...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_DIMENSION,
            distance=Distance.COSINE
        )
    )
    logging.info("🚀 Qdrant collection initialized successfully and ready for vectors!")
    return client

if __name__ == "__main__":
    initialize_qdrant()