import sys
from pathlib import Path
import time

# Adds the project root directory directly to Python's search path
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from settings import settings


import os
import math
from qdrant_client import QdrantClient, models
from tqdm import tqdm


# ==============================================================================
# CONFIGURATION
# ==============================================================================
LOCAL_QDRANT_URL = "http://localhost:6333"
BATCH_SIZE = 100
MAX_RETRIES = 5  # Maximum number of retry attempts for write operations


# ── UPDATE MODE CONFIGURATION ──────────────────────────────────────────────────
# If True: Keeps the Cloud collection intact. Adds new vectors and updates existing ones.
# If False: Completely DROPS the Cloud collection and uploads a fresh copy.
UPDATE_MODE = True 


def migrate_database():
    # 1. Initialize Clients
    print(f"🔌 Connecting to local and cloud database instances (UPDATE_MODE={UPDATE_MODE})...")
    local_client = QdrantClient(url=LOCAL_QDRANT_URL)    

    cloud_client = QdrantClient(
        url = settings.QDRANT_CLOUD_CLUSTER_ENDPOINT.get_secret_value(),
        api_key = settings.QDRANT_CLOUD_KEY.get_secret_value(),
        timeout = 120.0) # default timeout = 5 sec, too less for processing large vectors in batch of 100


    # 2. Discover local collections
    try:
        collections = [c.name for c in local_client.get_collections().collections]
        print(f"📦 Found local collections to process: {collections}\n")
    except Exception as e:
        print(f"❌ Failed to connect to local Qdrant. Is Docker running? Error: {e}")
        sys.exit(1)


    for collection_name in collections:
        print("═" * 70)
        print(f"🔄 Processing collection: '{collection_name}'")
        print("═" * 70)
        

        # Fetch local configuration and exact total point count
        local_info = local_client.get_collection(collection_name)
        total_points = local_info.points_count
        print(f"   📊 Total points in local database: {total_points:,}")
        

        # 3. Collection Schema Management (Create vs Update)
        collection_exists = cloud_client.collection_exists(collection_name)     
        if collection_exists and UPDATE_MODE:
            print(f"   ⏩ UPDATE MODE: Collection '{collection_name}' exists on Cloud. Skipping schema recreation.")
        else:
            if collection_exists:
                print(f"   ⚠️  OVERWRITE MODE: Dropping existing cloud collection '{collection_name}'...")
                cloud_client.delete_collection(collection_name)


            print("   🏗️ Creating collection schema on Qdrant Cloud...")

            # PERFORMANCE SHIELD: Temporarily disable indexing threshold during initial bulk migration
            cloud_params = local_info.config.params
            cloud_optimizer = local_info.config.optimizer_config
            if not UPDATE_MODE:
                # Disables background indexing threads during initial stream to maximize upload speeds
                cloud_optimizer = models.OptimizersConfigDiff(indexing_threshold=0)


            cloud_client.create_collection( 
                collection_name       = collection_name,
                vectors_config        = cloud_params.vectors,
                sparse_vectors_config = cloud_params.sparse_vectors,
                quantization_config   = local_info.config.quantization_config,
                optimizers_config      = cloud_optimizer)


            # Safe allocation loop ensuring the remote collection is ready
            while cloud_client.get_collection(collection_name).status != models.CollectionStatus.GREEN:
                time.sleep(0.5)
                
            print("   ✅ Base collection schema created on cloud.")            

            # Replicate Payload Indexes (Only needed when creating a fresh collection)
            if local_info.payload_schema:
                print("   🔍 Replicating local payload keyword indexes...")
                for field_name, schema_info in local_info.payload_schema.items():
                    data_type_value = schema_info.data_type                    

                    # Convert the internal datatype string/enum into a valid schema parameter
                    if "text" in str(data_type_value).lower():
                        text_params = getattr(schema_info, 'params', None)                       
                        tokenizer_val = getattr(text_params, 'tokenizer', models.TokenizerType.WORD)

                        if hasattr(tokenizer_val, 'value'):
                            tokenizer_val = tokenizer_val.value                            

                        safe_field_schema = models.TextIndexParams(
                            type=models.TextIndexType.TEXT,
                            tokenizer=models.TokenizerType(tokenizer_val),
                            lowercase=getattr(text_params, 'lowercase', True),
                            min_token_len=getattr(text_params, 'min_token_len', None),
                            max_token_len=getattr(text_params, 'max_token_len', None)
                        ) if text_params else models.PayloadSchemaType.TEXT
                    else:
                        raw_enum_val = data_type_value.value if hasattr(data_type_value, 'value') else data_type_value
                        safe_field_schema = models.PayloadSchemaType(raw_enum_val)
                        

                    cloud_client.create_payload_index(
                        collection_name = collection_name,
                        field_name      = field_name,
                        field_schema    = safe_field_schema)
                print("   ✅ Payload indexing replication complete.")
        

        # 4. Execute Data Migration Stream
        if total_points == 0:
            print(f"   ℹ️  Collection '{collection_name}' is empty locally. Skipping data migration.\n")
            continue
            

        next_page_offset = None
        action_text = "Updating" if (collection_exists and UPDATE_MODE) else "Uploading"
        print(f"   🚀 Starting live data stream...")        

        with tqdm(total=total_points, desc=f"   {action_text} '{collection_name}'", unit="point") as pbar:
            while True: 
                records, next_page_offset = local_client.scroll(
                    collection_name = collection_name,
                    limit           = BATCH_SIZE,
                    offset          = next_page_offset,
                    with_payload    = True,    
                    with_vectors    = True)                

                if not records:
                    break

                points_to_upload = []
                for record in records:
                    normalized_vector = {}
                    if isinstance(record.vector, dict):
                        for v_name, v_data in record.vector.items():
                            # Flexible serialization handler for sparse / custom objects
                            if hasattr(v_data, "indices") and hasattr(v_data, "values"):
                                normalized_vector[v_name] = models.SparseVector(indices=list(v_data.indices), values=list(v_data.values))
                            elif hasattr(v_data, "tolist"):
                                normalized_vector[v_name] = v_data.tolist()
                            else:
                                normalized_vector[v_name] = v_data
                    else:
                        normalized_vector = record.vector.tolist() if hasattr(record.vector, "tolist") else record.vector

                    points_to_upload.append(models.PointStruct(
                            id=record.id,
                            vector=normalized_vector,
                            payload=record.payload))                

                # Cloud Upsert operation block with Exponential Backoff Retries
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        cloud_client.upsert(
                            collection_name=collection_name,
                            points=points_to_upload
                        )
                        break  # Success, break retry loop
                    except Exception as e:
                        if attempt == MAX_RETRIES:
                            print(f"\n❌ Permanent failure on upsert after {MAX_RETRIES} attempts.")
                            raise e
                        sleep_time = attempt * 2  # Exponential backoff (2s, 4s, 6s, 8s...)
                        time.sleep(sleep_time)               

                pbar.update(len(points_to_upload))                

                if next_page_offset is None:
                    break                    


        # 5. POST-MIGRATION OPTIMIZER RESTORATION
        # Re-enables HNSW background indexing threads on the cloud if they were disabled
        if not UPDATE_MODE:
            print("   🛠️ Re-enabling background indexing thresholds for optimal cloud search speeds...")
            cloud_client.update_collection(
                collection_name=collection_name,
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000))
            print("   ✅ Cloud indices optimization triggered successfully.")            

        print(f"✅ Finished processing collection: '{collection_name}'\n")

    print("🎉 Sync process finished successfully!")

if __name__ == "__main__":
    migrate_database()