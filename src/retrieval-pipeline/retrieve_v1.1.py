import os
import sys

# 1. Stop the silent OpenMP multi-runtime crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# 2. Prevent Rust tokenizer thread-forking deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 3. Force Hugging Face to be completely transparent about downloads
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import uuid
import logging
import torch
from qdrant_client import QdrantClient, models

# Setup hyper-verbose logging to catch exactly where the engine stalls
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Explicitly pull the huggingface download tracker into logs
logging.getLogger("huggingface_hub").setLevel(logging.INFO)

print("🚀 Starting RAG Retrieval Engine...")
print("Checking for torch availability...")
print(f"  -> PyTorch version: {torch.__version__}")
print(f"  -> CUDA Available: {torch.cuda.is_available()}")

# Force deferred import to ensure environment variables are locked down first
print("Importing FlagEmbedding modules...")
try:
    from FlagEmbedding import BGEM3FlagModel
    # from FlagEmbedding import FlagReranker # uses pytorch, execution is a bit slower on cpu
    from fastembed.rerank.cross_encoder import TextCrossEncoder # uses ONNX runtime rather than pytorch, faster execution with lower RAM consumption
    print("✅ FlagEmbedding libraries imported successfully.")
except Exception as e:
    print(f"❌ Failed to import libraries: {e}")
    sys.exit(1)

# Configuration constants
CHILDREN_COLLECTION = "documentation_child_chunks"
PARENTS_COLLECTION  = "documentation_parent_chunks"
QDRANT_URL          = "http://localhost:6333"

def deterministic_uuid(source_string: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, source_string))


class RetrievalPipeline:
    def __init__(self):
        is_cuda = torch.cuda.is_available()
        
        print("\n" + "-"*50)
        print("STAGE 1: Loading Unified BGE-M3 Model")
        print(f"Targeting Local Device Acceleration -> CUDA: {is_cuda} | FP16: {is_cuda}")
        print("Note: If running for the first time, this will download 2.2 GB of weights.")
        print("-"*50)
        
        try:
            # We explicitly pass the device to remove any internal guessing logic
            target_device = "cuda" if is_cuda else "cpu"
            
            self.bge_model = BGEM3FlagModel(
                'BAAI/bge-m3', 
                use_fp16=is_cuda,
                device=target_device # FIXED: Changed from 'devices' to 'device'
            )
            print("✅ STAGE 1 COMPLETE: BGE-M3 loaded into memory.")
        except Exception as e:
            print(f"❌ STAGE 1 CRASHED: Failed inside BGEM3FlagModel constructor.")
            print(f"Error details: {e}")
            sys.exit(1)

        print("\nSTAGE 2: Loading Cross-Encoder Reranker...")
        # self.reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=is_cuda)
        self.reranker = TextCrossEncoder(model_name="BAAI/bge-reranker-base")
        print("✅ STAGE 2 COMPLETE: Reranker operational.")
        
        print("\nSTAGE 3: Connecting to Qdrant Docker Engine...")
        self.qdrant_client = QdrantClient(url=QDRANT_URL)
        print("✅ STAGE 3 COMPLETE: Connected to database.")



    def retrieve(self, query_text: str, top_k_child: int = 15, top_k_parent: int = 3):
        logging.info(f"Processing query: '{query_text}'")
        
        # Step 1: Generate matching embeddings using the unified BGE-M3 model
        embeddings = self.bge_model.encode([query_text], return_dense=True, return_sparse=True)
        dense_vector = embeddings['dense_vecs'][0].tolist()
        sparse_dict = embeddings['lexical_weights'][0]
        
        # Preparing sparse vectors for keyword-search 
        qdrant_sparse = models.SparseVector(
        indices=[int(k) for k in sparse_dict.keys()],
        values=[float(v) for v in sparse_dict.values()])
        
        # Step 2: Execute Hybrid Search (Dense + Sparse combined via RRF)
        prefetch_limit = top_k_child * 2  # 15*2 candidates for RRF to work with.

        logging.info(f"Executing hybrid search in Qdrant (fetching top {top_k_child} child chunks)...")
        search_results = self.qdrant_client.query_points(
            collection_name=CHILDREN_COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_vector, using="dense", limit= prefetch_limit),
                models.Prefetch(query=qdrant_sparse, using="sparse", limit= prefetch_limit)
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
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
                
        logging.info(f"Found {len(parent_ids_to_fetch)} unique parent IDs from child chunks.")
        
        if not parent_ids_to_fetch:
            logging.warning("No parent connections found within child payloads.")
            return []

        # Step 4: Fetch actual Parent text records natively from Qdrant
        logging.info(f"Fetching parent contexts directly from Qdrant '{PARENTS_COLLECTION}'...")
        parent_uuids = [deterministic_uuid(pid) for pid in parent_ids_to_fetch]
        
        parent_points = self.qdrant_client.retrieve(
            collection_name=PARENTS_COLLECTION,
            ids=parent_uuids,
            with_payload=True,
            with_vectors=False
        )
        
        parent_documents = []
        for point in parent_points:
            p_payload = point.payload
            if p_payload and "text" in p_payload:
                parent_documents.append({
                    "id": p_payload.get("doc_id"), 
                    "text": p_payload.get("text")
                })

        if not parent_documents:
            logging.error("No valid parent texts could be pulled from the database records.")
            return []

        # Step 5: Cross-Encoder Reranking over the Parent text blocks
        logging.info("Passing parent text records to Cross-Encoder Reranker...")
        texts_to_rerank = [doc["text"] for doc in parent_documents]
        
        rerank_results = list(self.reranker.rerank(query=query_text, documents=texts_to_rerank)) ## this returns 
        # print(rerank_results)
        
        # Re-assemble structural data with their context scores
        final_results = []
        # Safe 1-to-1 zip mapping of the original documents and their corresponding reranker scores
        for doc, res in zip(parent_documents, rerank_results):
            score = float(res)
            final_results.append({
                "parent_id": doc["id"],
                "text": doc["text"],
                "relevance_score": score
            })
            
        # Sort by best score descending and slice to requested threshold
        final_results = sorted(final_results, key=lambda x: x["relevance_score"], reverse=True)
        return final_results[:top_k_parent]


# Interactive terminal verification interface
if __name__ == "__main__":
    pipeline = RetrievalPipeline()

    print("\n" + "="*50)
    print("Production Pipeline Ready! Type 'exit' or 'quit' to stop.")
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
            if len(raw_text) > 600:
                snippet = raw_text[:600]
                last_period = snippet.rfind(".")
                if last_period > 400:
                    snippet = snippet[:last_period + 1]
                else:
                    snippet = snippet.rstrip() + "..."
            else:
                snippet = raw_text
                
            print(snippet)
            print("─" * 80)
        print("█"*86 + "\n")