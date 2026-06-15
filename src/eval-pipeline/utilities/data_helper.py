
import json
import random
import logging
import re
from pathlib import Path
from collections import defaultdict
from utilities.to_use import GENERATION_PROMPT

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset_gen")

# ── Configuration ──────────────────────────────────────────────────────────────
root_dir = Path(__file__).resolve().parent.parent.parent.parent
CHUNKS_DIR = root_dir / "data" / "processed"
MIN_CHUNK_LENGTH = 500      # skip chunks too short to generate good questions
BATCH_SIZE       = 3        # chunks per LLM call (3 enables multi-hop questions)
RANDOM_SEED      = 42       # reproducible sampling


# How many parents to sample per framework
# Total = 35 parents × ~3 questions each ≈ 105 retrieval questions
SAMPLE_PER_FRAMEWORK = {
    "langchain":  22,
    "langgraph":  14,
    "langsmith":  14,
    "deepagents": 10
}

# ── Gemini setup ───────────────────────────────────────────────────────────────

# 1. Define primary and fallback models
llm_primary = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.0)
llm_fallback = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.0)

# 2. Chain them together using fallbacks
# This will automatically try the fallback if the primary fails with a ServerError (503)
llm = llm_primary.with_fallbacks([llm_fallback])


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_latest_chunks_file() -> Path:
    """Finds the most recent chunks_*.json file in data/processed/."""
    latest_chunks_file = sorted(CHUNKS_DIR.glob("chunks_*.json"))[-1]
    return latest_chunks_file


def load_parent_chunks(chunks_file: Path) -> dict[str, list[dict]]:
    """
    Loads parent chunks from the JSON file.
    Returns {framework: [chunk, ...]} filtered to chunks long enough
    to generate meaningful questions.
    """
    logger.info(f"Loading chunks from: {chunks_file.name}")
    with open(chunks_file, encoding="utf-8") as f:
        all_chunks = json.load(f)

    # Keep only parents with enough content
    by_framework = defaultdict(list)
    for chunk in all_chunks:
        if chunk["type"] != "parent":
            continue
        if len(chunk["text"]) < MIN_CHUNK_LENGTH:
            continue
        fw = chunk["metadata"]["framework"]
        by_framework[fw].append(chunk)

    for fw, chunks in by_framework.items():
        logger.info(f"  {fw}: {len(chunks)} usable parents")

    return dict(by_framework)


def stratified_sample(by_framework: dict[str, list[dict]]) -> list[dict]:
    """
    Samples parent chunks proportionally across frameworks.
    Uses a fixed seed for reproducibility.
    """
    random.seed(RANDOM_SEED)
    sampled = []
    for fw, n in SAMPLE_PER_FRAMEWORK.items():
        pool = by_framework.get(fw, [])
        if not pool:
            logger.warning(f"No usable chunks for framework: {fw}")
            continue
        # Sample min(n, available) to handle small framework corpora
        take = min(n, len(pool))
        sampled.extend(random.sample(pool, take))
        logger.info(f"  Sampled {take} parents from {fw}")

    # Shuffle so batches are mixed across frameworks
    random.shuffle(sampled)
    return sampled


def build_prompt(batch: list[dict]) -> str:
    """
    Formats a batch of chunks into the generation prompt.
    2 multi-hop + (batch_size - 1) single-hop questions per batch.
    """
    n         = len(batch)
    n_multi   = 2   # 2 multi-hop per batch (requires >= 2 chunks)
    n_single  = 3   # 3 single-hop per batch (one per chunk)
    n_questions = n_multi + n_single

    # Preparing chunks as "Chunk 3 (langchain - checkpointers) \n ......"
    chunks_text = ""
    for i, chunk in enumerate(batch):
        fw      = chunk["metadata"]["framework"]
        heading = chunk["metadata"].get("section_heading", "")
        chunks_text += f"\n[Chunk {i}] ({fw} — {heading})\n{chunk['text']}\n"

    return GENERATION_PROMPT.format( # Augmenting prompt
        n           = n,
        n_questions = n_questions,
        n_single    = n_single,
        n_multi     = n_multi,
        chunks_text = chunks_text,
    )


def call_gemini(prompt: str) -> list[dict] | None:
    """
    Calls Gemini and parses the JSON response.
    Returns list of question dicts, or None on failure.
    """
    try:
        response = llm.invoke(prompt).content
        raw = response[0]["text"] if isinstance(response, list) else response
        raw = raw.strip()

        # Strip markdown code fences if Gemini wraps the JSON anyway
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        # print(raw)

        return json.loads(raw)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        return None


def build_dataset_row(
    question_data: dict,
    batch: list[dict],
    question_type: str,
) -> dict:
    """
    Builds one complete dataset row from a generated question
    and the batch of chunks it came from.
    """
    indices  = question_data.get("source_chunk_indices", [])
    # Guard against out-of-range indices from LLM
    indices  = [i for i in indices if 0 <= i < len(batch)]
    sources  = [batch[i] for i in indices]

    return {
        "question":          question_data["question"],
        "question_type":     question_type,
        "framework":         sources[0]["metadata"]["framework"] if sources else "unknown",
        "source_parent_ids": [c["id"] for c in sources],
        "reference_answer":  question_data.get("reference_answer", ""),
        "relevant_context":  [c["text"] for c in sources],

        # These fields are filled later when the agent processes the question
        "query_plan":        None,   # populated during eval run
        "retrieved_chunks":  None,   # populated during eval run
        "final_response":    None,   # populated during eval run
    }