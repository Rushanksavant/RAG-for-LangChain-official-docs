## Eval Pipeline — Summary

---

### 1. What is Ragas and why are we using it

Ragas is an evaluation framework for RAG systems. It does two things:
- **Generates synthetic test data** from your document corpus (no manual labeling needed)
- **Scores your RAG pipeline** on metrics like Context Precision and Context Recall

We use it because manually writing 200 test questions for LangChain docs is impractical. Ragas automates this using an LLM to read documents and generate realistic questions + ground truth answers.

---

### 2. The Knowledge Graph — what it is and why Ragas builds one

Ragas doesn't generate questions directly from raw text. It first builds a **Knowledge Graph (KG)** — a structured network where:
- Each document chunk becomes a **Node**
- Extracted entities, summaries, themes become **Additional nodes**
- Semantic relationships between nodes become **Edges**

A KG starting from 20 documents typically expands to ~100+ nodes with hundreds of relationships after enrichment.

**IMPORTANT: Why a KG?** It enables multi-hop questions — questions that require reasoning across multiple connected documents. For example: *"How does LangSmith tracing integrate with LangGraph state management?"* — this connects two separate doc sections via a relationship edge. Flat vector search can't generate this. The KG structure makes it possible.

---

### 3. KG construction — what happens under the hood

When `apply_transforms(kg, transforms)` runs, it executes a pipeline of operations on each node:

- **Summarizer** — LLM reads the chunk, writes a 2-3 sentence summary → 1 API call per node
- **NER extractor** — LLM extracts named entities (class names, functions, concepts) → 1 API call per node
- **Keyphrase extractor** — LLM extracts important phrases → 1 API call per node
- **Embedding** — embedding model encodes each node for similarity computation → 1 API call per node for embedding model 
- **Relationship builder** — computes cosine similarity between all node pairs using cached embeddings → no LLM call, pure compute

**Cost: ~4-5 LLM calls per document, zero LLM calls for relationship building. If using remote embedding model: +1 API call per node**

---

### 4. Idempotency — why incremental updates are cheap

Ragas transformations are **idempotent** — before processing a node, each extractor checks if the target property already exists. If it does, the node is skipped entirely.

This means:
- Load saved KG with 200 enriched nodes
- Append 20 new raw nodes
- Call `apply_transforms` again
- Result: only the 20 new nodes get processed, 200 existing nodes are skipped

**Cost stays flat at ~4-5 API calls per new document regardless of total KG size.**

Relationship building reruns across all nodes (quadratic growth in compute), but this can be achieved using cached embeddings — no API calls, just CPU/GPU time.

---

### 5. Question synthesis — what happens and why it's expensive

After the KG is built, `generator.generate(testset_size=20)` samples scenarios from the graph and generates questions. Three question types are produced:

- **SingleHopSpecific (50%)** — question answerable from one node. ~3-4 LLM calls (question generation + answer generation + quality check)
- **MultiHopAbstract (25%)** — abstract question requiring reasoning across 2+ connected nodes. ~5-7 LLM calls
- **MultiHopSpecific (25%)** — specific question requiring 2+ connected nodes. ~5-7 LLM calls

**Cost: ~5-8 LLM calls per question. For 20 questions ≈ 100-160 API calls.**

This is more expensive than KG construction per run.

---

### 6. Daily API call budget

| Phase | Cost per day | Notes |
|---|---|---|
| KG construction | ~100 calls | Only new 20 documents processed |
| Question synthesis | ~100-160 calls | 20 questions × 5-8 calls each |
| **Total** | **~200-260 calls/day** | Well within Groq's 1,000 RPD |

---

### 7. Model strategy — two models, two phases

KG construction and question synthesis use LLMs for different tasks:

- **KG construction** — entity extraction, summarization. Needs good reading comprehension of technical docs. Using **Groq Llama 3.3 70B** (strong, fast, 1,000 RPD free)
- **Question synthesis** — multi-hop reasoning, ground truth generation. Needs stronger reasoning. Uses **Gemini 2.5 Flash** (good reasoning, 250 RPD free)

This split is officially supported by Ragas. The `transformer_llm` used in `apply_transforms` is completely separate from the `generator_llm` passed to `TestsetGenerator`. The KG is saved to disk between phases — no rebuilding needed.

---

### 8. Duplicate question prevention

Ragas has no built-in deduplication across generation runs. Each `generate()` call samples the KG independently and can produce questions similar to previous runs.

**Solution:** After each synthesis run, before appending to `eval_dataset.json`, run a fuzzy similarity check using `rapidfuzz` against all existing questions. Threshold of ~0.85 similarity = duplicate, discard it.

---

### 9. The incremental pipeline — full daily workflow

```
daily_run.py — runs once per day
│
├── STEP 1: Fetch new documents
│   ├── Load seen_doc_ids.json (tracks already-processed Qdrant point IDs)
│   ├── Query Qdrant documentation_parent_chunks with optional filters:
│   │   └── framework: langchain / langgraph / langsmith / deepagents
│   ├── Exclude any doc whose ID is already in seen_doc_ids.json
│   ├── Take first 20 unique unseen documents
│   └── Append their IDs to seen_doc_ids.json
│
├── STEP 2: Update Knowledge Graph
│   ├── Load knowledge_graph.json (or init fresh KnowledgeGraph on Day 1)
│   ├── Append 20 new Document nodes to kg.nodes manually
│   ├── Run apply_transforms(kg, transforms, llm=groq_llm)
│   │   ├── Existing 200 nodes → skipped (already enriched)
│   │   └── New 20 nodes → processed (~100 LLM calls)
│   └── Save updated knowledge_graph.json
│
├── STEP 3: Generate new questions
│   ├── Load knowledge_graph.json
│   ├── Initialize TestsetGenerator with llm=gemini_flash
│   ├── Call generator.generate(testset_size=20)
│   │   └── ~100-160 LLM calls
│   └── Raw 20 questions produced
│
├── STEP 4: Deduplicate
│   ├── Load eval_dataset.json (all previously accepted questions)
│   ├── For each new question, compute fuzzy similarity against all existing
│   ├── Discard if similarity > 0.85 with any existing question
│   └── Accept remaining unique questions
│
└── STEP 5: Persist
    ├── Append accepted questions to eval_dataset.json
    └── Log: date, questions added, questions discarded, total count
```

---

### 10. The two objectives and how this pipeline serves both

**Objective 1 — Build a comprehensive 200-question eval set:**
Run the pipeline daily for 10 days. Each day adds ~20 clean unique questions covering progressively more of the corpus (stratified across frameworks via Qdrant metadata filtering). After 10 days you have a diverse, LLM-generated eval dataset covering LangChain, LangGraph, LangSmith, and DeepAgents content.

**Objective 2 — Monitor retrieval quality as docs evolve:**
When LangChain releases new docs:
1. Run `run_parser.py` → fresh `chunks.json`
2. Re-index Qdrant with updated chunks
3. Run `run_evaluation.py` using the **fixed** `eval_dataset.json` against the **new** Qdrant index
4. Compare Context Precision and Context Recall scores against previous run

The eval dataset doesn't need to change when docs update — you're testing whether your retrieval still finds the right context for the same questions. The dataset only grows when you want broader coverage, not when the underlying docs change.

---

