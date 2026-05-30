import json
import os
import logging
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration paths
CHUNKS_JSON_PATH = "data/processed/chunks.json"
COLLECTION_NAME = "documentation_chunks"
QDRANT_URL = "http://localhost:6333"

class RetrievalPipeline:
    def __init__(self):
        logging.info("Initializing Dense, Sparse, and Reranking models...")

        # 1. Initializing the same models used during indexing-pipeline
        self.dense_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        
        # 2. Initializing the Cross-Encoder Reranker
        self.reranker = TextCrossEncoder(model_name="BAAI/bge-reranker-base")
        
        # 3. Initializing Qdrant Client
        logging.info(f"Connecting to Qdrant at {QDRANT_URL}...")
        self.qdrant_client = QdrantClient(url=QDRANT_URL)
        
        # 4. Load lookup database (chunks.json) into memory for fast parent retrieval
        logging.info(f"Loading mapping database from {CHUNKS_JSON_PATH}...")
        with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
            
        # Build an easy-access dict for parents: { parent_chunk_id: text }
        self.parent_lookup = {
            c["id"]: c["text"] for c in all_chunks if c.get("type") == "parent"
        }
        logging.info(f"Successfully cached {len(self.parent_lookup)} parent chunks in memory.")


    def retrieve(self, query_text: str, top_k_child: int = 15, top_k_parent: int = 3):
        logging.info(f"Processing query: '{query_text}'")
        
        # Step 1: Generate embeddings for the incoming query
        dense_vector = list(self.dense_model.embed([query_text]))[0].tolist()
        sparse_vector = list(self.sparse_model.embed([query_text]))[0]
        
        # Format sparse vector structure for Qdrant
        qdrant_sparse = models.SparseVector(
            indices=sparse_vector.indices.tolist(),
            values=sparse_vector.values.tolist()
        )
        
        # Step 2: Execute Hybrid Search (Dense + Sparse combined)
        logging.info(f"Executing hybrid search in Qdrant (fetching top {top_k_child} child chunks)...")
        search_results = self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=dense_vector, using="dense", limit=top_k_child),
                models.Prefetch(query=qdrant_sparse, using="sparse", limit=top_k_child),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF), # Combines dense/sparse lists safely
            limit=top_k_child,
            with_payload=True
        )
        
        # Step 3: Extract and deduplicate parent IDs from child payloads
        parent_ids_to_fetch = []
        seen_parents = set()
        
        for point in search_results.points:
            payload = point.payload
            parent_id = payload.get("parent_id")
            
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_ids_to_fetch.append(parent_id)
                
        logging.info(f"Found {len(parent_ids_to_fetch)} unique parent contexts from child chunks.")
        
        # Step 4: Map back to actual Parent text blocks
        parent_documents = []
        for pid in parent_ids_to_fetch:
            parent_text = self.parent_lookup.get(pid)
            if parent_text:
                parent_documents.append({"id": pid, "text": parent_text})
            else:
                logging.warning(f"Parent ID {pid} referenced by child but missing in chunks.json!")
                
        if not parent_documents:
            logging.error("No valid parent texts could be retrieved.")
            return []

        # Step 5: Cross-Encoder Reranking over the Parent text blocks
        logging.info("Passing parent text fragments to Cross-Encoder Reranker...")
        texts_to_rerank = [doc["text"] for doc in parent_documents]
        
        # Reranker scores everything relative to the raw query string
        rerank_results = list(self.reranker.rerank(query=query_text, documents=texts_to_rerank))
        
        # Re-assemble our original structural data with their new scores safely
        final_results = []
        for i, res in enumerate(rerank_results):
            # FastEmbed returns scores either as a list of floats matching input order,
            # or objects with a .score property. We handle both cleanly here:
            score = res.score if hasattr(res, "score") else float(res)
            
            final_results.append({
                "parent_id": parent_documents[i]["id"],
                "text": parent_documents[i]["text"],
                "relevance_score": score
            })
            
        # Sort by best score descending and slice to our requested threshold
        final_results = sorted(final_results, key=lambda x: x["relevance_score"], reverse=True)
        return final_results[:top_k_parent]




# Quick test interface execution
if __name__ == "__main__":
    # print(TextCrossEncoder.list_supported_models())
    # Start the engine
    pipeline = RetrievalPipeline()

    print("\n" + "="*50)
    print("Pipeline Ready! Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")
    
    while True:
        user_query = input("Enter your search query: ")
        if user_query.strip().lower() in ["exit", "quit"]:
            break
            
        if not user_query.strip():
            continue
            
        results = pipeline.retrieve(user_query)
        
        print("\n" + "█"*30 + " TOP RERANKED PARENT CONTEXTS " + "█"*30)
        for i, match in enumerate(results, 1):
            print(f"\n[RANK {i}] 🔥 Score: {match['relevance_score']:.4f} | Parent ID: {match['parent_id']}")
            print("─" * 80)
            
            raw_text = match['text']
            
            # Smart Truncation: Show the first 600 characters cleanly
            if len(raw_text) > 600:
                # Find the nearest clean line break or sentence end so we don't cut off mid-word
                snippet = raw_text[:600]
                last_period = snippet.rfind(".")
                if last_period > 400:
                    snippet = snippet[:last_period + 1]
                else:
                    snippet = snippet.rstrip() + "..."
            else:
                snippet = raw_text
                
            # Print the cleaned snippet
            print(snippet)
            print("─" * 80)
        print("█"*86 + "\n")