# Ways to run indexing pipeline

1. **Run pipeline of local computer (CPU-intense operation)**
    - Run a docker container as Qdrant Engine 
    - Connect your client to the docker container at port 6333 (used by Qdrant)
    - Store calculate vectors in vectordB in a local folder

2. **Run Qdrant inside Google Colab**
    - Perform all the computation using colab resources
    - Download the generated vectordB

3. **Use Qdrant Cloud (free & paid tier available)**
    - Point Google Colab to the cloud
    - Peform indexing in colab and store dB in cloud



## Implementation of Option 1 and 2


###  <ins>1. Run pipeline of local computer (CPU-intense operation)</ins>

**Docker commands should be runned while keeping terminal in docker folder (or where dockerfile/docker-compose lives)**

1. **Start qdrant engine** in docker container: (Make sure docker engine is open)
```bash
cd docker 
docker compose up -d
```
To verify it is running, open your web browser and go to http://localhost:6333/dashboard. Qdrant comes with a beautiful, built-in Web UI dashboard!

2. **Initialize the vectordB** 
```bash
cd ..
uv run src/vectordB/init_vectordB.py
```
Check your Qdrant browser dashboard; you will see the documentation_chunks collection appear instantly with 0 points

3. **Run the indexing pipeline**
```bash
uv run src/vectordB/index_chunks.py
```
As this script runs, you will see your terminal log batches of 64 points being transmitted over port 6333. Once complete, refresh your browser dashboard to see your total vector count fill up.

**Done**

5. **Stop the Engine** after indexing/retrieval session
```bash
cd docker
docker compose down
```


### <ins>2. Run Qdrant inside Google Colab</ins>

**Colab VM**
    |
    ├── pip install qdrant-client fastembed
    |
    ├── wget qdrant binary + run it locally inside Colab
    |
    ├── run init_vectordB.py  (creates collections inside Colab's Qdrant)
    |
    ├── run index_chunks.py   (embeds 27K chunks, uploads to Colab's Qdrant)
    |
    ├── zip /qdrant_storage/
    |
    └── download to your machine → unzip into data/qdrant_dB/

**Your machine**
    |
    └── docker compose up -d  (Qdrant reads the pre-built storage → instant, no re-indexing)

**The key insights:**

- Qdrant storage is just files. 
- Whatever Qdrant writes during indexing on Colab, you copy those files to your local volume mount and Docker picks them up as-is. 
- Zero re-indexing/processing needed locally.

**Start qdrant engine** in docker container and you can run retrieval pipeline.
```bash
cd docker 
docker compose up -d
```