import logging
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def initialize_production_schema():
    # Connect to your centralized Docker container engine API
    container_url = "http://localhost:6333" # port used by qdrant engine by default
    collection_name = "documentation_chunks"
    
    logging.info(f"Connecting to Qdrant Engine container at: {container_url}")
    client = QdrantClient(url=container_url)
    
    try:
        if client.collection_exists(collection_name): # if collection already exists in Qdrant engine
            logging.info(f"Collection '{collection_name}' already exists in the engine. Schema is safe.")
            return
    except UnexpectedResponse: # if connection with Qdrant engine fails
        logging.error("Could not connect to Qdrant Docker container. Is it running via 'docker run'?")
        return

    logging.info(f"Declaring dual-vector hybrid schema for '{collection_name}'...")
    
    client.create_collection( # Create the db (initialize the collection) 
        collection_name=collection_name,
        # Config 1: Dense Semantic Space 
        vectors_config={
            "dense": models.VectorParams(
                size=1024, # since BGE-M3 outputs 1024 dimension vectors
                distance=models.Distance.COSINE # using cosine similarity 
            )
        },
        # Config 2: Sparse Keyword Space 
        sparse_vectors_config={
            "sparse": models.SparseVectorParams() # Enables token matching for code snippets
        }
    )
    logging.info("🚀 Production-ready collection initialized inside Qdrant container!")

if __name__ == "__main__":
    initialize_production_schema()