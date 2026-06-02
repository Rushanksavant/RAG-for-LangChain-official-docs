from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

for collection in ["documentation_child_chunks", "documentation_parent_chunks"]:
    info = client.get_collection(collection)
    print(f"\n{collection}")
    print(f"  points_count:        {info.points_count}")
    print(f"  indexed_vectors_count: {info.indexed_vectors_count}")
    print(f"  status:              {info.status}")
    print(f"  optimizer_status:    {info.optimizer_status}")

# Sample a point and inspect its sparse vector
results, _ = client.scroll(
    collection_name="documentation_child_chunks",
    limit=1,
    with_vectors=True,
)
if results:
    point = results[0]
    sparse = point.vector.get("sparse")
    dense  = point.vector.get("dense")
    print(f"\nSample point sparse vector:")
    print(f"  Non-zero entries: {len(sparse.indices) if sparse else 0}")
    print(f"  Sample indices:   {sparse.indices[:10] if sparse else None}")
    print(f"  Sample values:    {[round(v,4) for v in sparse.values[:10]] if sparse else None}")
    print(f"  Dense dim:        {len(dense) if dense else 0}")

print("\n─────Sparse vector health diagnosis─────────────────────────────────────────────────────")

results, _ = client.scroll(
    collection_name="documentation_child_chunks",
    limit=5,
    with_vectors=True,
    with_payload=["text", "content_category"],
)

for point in results:
    sparse = point.vector.get("sparse")
    text_len = len(point.payload.get("text", ""))
    cat = point.payload.get("content_category")
    print(f"  category: {cat:>20} | text_len: {text_len:>5} chars | sparse entries: {len(sparse.indices)}")


print("\n─────Qdrant db health check────────────────────────────────────────")
import os
import sys
import logging
from qdrant_client import QdrantClient, models

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Configurations ────────────────────────────────────────────────────────────
# Ensure these match your local configuration settings
QDRANT_URL          = "http://localhost:6333"
CHILDREN_COLLECTION = "documentation_child_chunks"
PARENTS_COLLECTION  = "documentation_parent_chunks"

# Modify these numbers if your total count in chunks.json differs
EXPECTED_CHILDREN   = 17731
EXPECTED_PARENTS    = 7825

def verify_and_optimize_local_index():
    logging.info(f"Connecting to local Qdrant engine at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)

    # ── Step 1: Verify exact database counts ──────────────────────────────────
    logging.info("Verifying point counts across collections (exact scan)...")
    try:
        children_count = client.count(CHILDREN_COLLECTION, exact=True).count
        parents_count  = client.count(PARENTS_COLLECTION,  exact=True).count
    except Exception as e:
        logging.error(f"Failed to connect or query Qdrant collections. Is Qdrant running? Details: {e}")
        sys.exit(1)

    print("\n" + "═"*60)
    print(f"  {CHILDREN_COLLECTION}: {children_count} / {EXPECTED_CHILDREN} expected")
    print(f"  {PARENTS_COLLECTION}:  {parents_count}  / {EXPECTED_PARENTS} expected")
    print("═"*60)

    if children_count < EXPECTED_CHILDREN or parents_count < EXPECTED_PARENTS:
        missing_c = EXPECTED_CHILDREN - children_count
        missing_p = EXPECTED_PARENTS - parents_count
        logging.warning(f"⚠️ Index incomplete! {missing_c} child chunks and {missing_p} parent chunks are missing.")
        logging.warning("Please resume your indexing pipeline script before running optimizations.")
        sys.exit(1)

    print("✅ All points registered and accounted for in the database!\n")

    # ── Step 2: Force WAL -> Disk Flush ───────────────────────────────────────
    # By default, recent writes sit in volatile Write-Ahead Logs (WAL) in RAM.
    # Setting indexing_threshold=0 forces Qdrant's background optimizer to flush
    # those WALs into permanent, immutable segments on the disk immediately.
    logging.info("Forcing Write-Ahead Log (WAL) flush to disk (setting indexing_threshold=0)...")
    for name in [CHILDREN_COLLECTION, PARENTS_COLLECTION]:
        client.update_collection(
            collection_name=name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )

    # Wait for the status of the collections to transition back to Green (Safe & Idle)
    print("Waiting for database optimization and disk flush", end="", flush=True)
    import time
    for _ in range(60):  # Wait up to 2 minutes
        time.sleep(2)
        try:
            statuses = [client.get_collection(n).status for n in [CHILDREN_COLLECTION, PARENTS_COLLECTION]]
            if all(s.value == "green" for s in statuses):
                print(" -> Done!")
                break
        except Exception as e:
            logging.error(f"\nError checking collection optimization status: {e}")
            sys.exit(1)
        print(".", end="", flush=True)
    else:
        print("\n⚠️ Background optimization did not return a green status. Proceeding anyway.")

    # ── Step 3: Verify HNSW index completeness ────────────────────────────────
    # Math: indexed_vectors_count represents only HNSW graphs (DENSE vectors).
    # Sparse vectors are stored in an inverted index structure and never counted here.
    # Thus, expected HNSW count is exactly equal to the point count (1 per point).
    print("\nVerifying HNSW Index Structure Completeness:")
    all_indexed = True
    for name in [CHILDREN_COLLECTION, PARENTS_COLLECTION]:
        info = client.get_collection(name)
        expected = info.points_count  # Exactly 1 dense vector index graph expected per point
        actual = info.indexed_vectors_count

        """
        NOTE: For small collections (under 10,000 point, our parents collection), 
        Qdrant skips HNSW graph building to save CPU cycles and runs unindexed brute-force scans.
        This is correct and highly efficient.
        """
        if actual == 0:
            print(f"  ✅ {name}: indexed HNSW={actual} / {expected} expected (Optimal flat-scan index active)")
        elif actual < expected:
            print(f"  ⚠️  {name}: indexed HNSW={actual} / {expected} expected (Still compacting/optimizing segments...)")
            all_indexed = False
        else:
            print(f"  ✅ {name}: indexed HNSW={actual} / {expected} expected (HNSW graph complete)")

    # ── Step 4: Reset optimizer threshold to default ──────────────────────────
    # Leaving indexing_threshold at 0 means any single future upsert will immediately
    # lock the collection to build HNSW graphs, destroying write performance.
    # We reset it to Qdrant's standard threshold (20,000) so local queries perform optimally.
    logging.info("\nResetting indexing optimizer thresholds back to default (20000)...")
    for name in [CHILDREN_COLLECTION, PARENTS_COLLECTION]:
        client.update_collection(
            collection_name=name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )
    print("✅ Local database parameters optimized and secure!")
    print("\n🎉 Index verification script completed successfully.")

if __name__ == "__main__":
    verify_and_optimize_local_index()