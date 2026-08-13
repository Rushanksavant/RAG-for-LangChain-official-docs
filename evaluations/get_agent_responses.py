"""
get_agent_responses.py
---------
Runs the agent on every question in the golden test dataset.
Saves results incrementally to a JSONL file so the run can be resumed
if it crashes or hits a rate limit mid-way.

Each saved result includes:
  - The original question + reference answer
  - The agent's final response
  - Whether retrieval was performed
  - The retrieved chunks (flattened, for RAGAS)
  - The agent's status messages (used for categorization later)

Run from project root:
    uv run python evaluations/get_agent_responses.py
"""

import sys
import asyncio
import json
import uuid
import logging
from pathlib import Path

# ── resolve langgraph-workflow onto the path so agent.py is importable ────────
ROOT_DIR     = Path(__file__).resolve().parent.parent.parent
LANGGRAPH_WORKFLOW_DIR = ROOT_DIR / "langgraph-workflow"
sys.path.insert(0, str(LANGGRAPH_WORKFLOW_DIR))

## For Ollama x Kaggle pipeline
DATASET_PATH = Path(__file__).parent / "data" / "golden_dataset.json" 
RESULTS_PATH = Path(__file__).parent / "data" / "results.jsonl"


# Agent makes ~3 LLM calls per question on Gemini free tier (10 RPM).
# 22s gap keeps us safely under that ceiling.
DELAY_SECONDS = 22

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("get_agent_responses")


def load_completed_ids() -> set:
    """Return IDs already saved in results.jsonl so we can resume interrupted runs."""
    if not RESULTS_PATH.exists():
        return set()
    completed = set()
    with open(RESULTS_PATH) as f:
        for line in f:
            row = json.loads(line)
            completed.add(row["id"])
    return completed


def flatten_retrieved_contexts(retrieved_contexts: dict) -> list[str]:
    """
    Agent stores chunks as {sub_query: [chunk1, chunk2, ...]}.
    RAGAS expects a flat list — merge all sub-query chunks here.
    """
    flat = []
    for chunks in retrieved_contexts.values():
        flat.extend(chunks)
    return flat


def retrieval_was_performed(result: dict) -> bool:
    """
    Infer whether the agent ran retrieval by inspecting its status messages.
    Adjust the keywords here if your status messages change.
    """
    status = " ".join(result.get("status_mssg", []))
    return any(kw in status for kw in ("Retrieved", "Path A", "Path B"))


async def run_one(item: dict) -> dict:
    """Run a single question through the agent and return a structured result dict."""
    import os
    os.chdir(LANGGRAPH_WORKFLOW_DIR) # changing working directory to langgraph-workflow (else it will take root/.env)

    from agent import run_agent  # imported here to keep module-level imports clean

    session_id = str(uuid.uuid4()) # new session for each question as current golden-data holds independent questions.  
    try:
        result = await run_agent(item["user_input"], session_id)

        return {
            "id"                  : item["id"],
            "user_input"          : item["user_input"],
            "reference"           : item["reference"],
            "reference_contexts"  : item.get("reference_contexts", []),
            "question_type"       : item.get("question_type", "unknown"),
            "agent_response"      : result["final_response"],
            "retrieved_contexts"  : flatten_retrieved_contexts(
                                        result.get("retrieved_contexts", {})
                                    ),
            "retrieval_performed" : retrieval_was_performed(result),
            "status_mssg"         : result.get("status_mssg", []),
            "error"               : None,
        }

    except Exception as e:
        logger.error(f"Failed on question {item['id']}: {e}")
        return {
            "id"                  : item["id"],
            "user_input"          : item["user_input"],
            "reference"           : item["reference"],
            "reference_contexts"  : item.get("reference_contexts", []),
            "question_type"       : item.get("question_type", "unknown"),
            "agent_response"      : None,
            "retrieved_contexts"  : [],
            "retrieval_performed" : False,
            "status_mssg"         : [],
            "error"               : str(e),
        }


async def main():
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    # Assign stable numeric IDs if the dataset doesn't already have them
    for i, item in enumerate(dataset):
        if "id" not in item:
            item["id"] = str(i)

    completed = load_completed_ids()
    pending   = [item for item in dataset if item["id"] not in completed]
    logger.info(f"Total: {len(dataset)} | Completed: {len(completed)} | Pending: {len(pending)}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "a") as out:
        for i, item in enumerate(pending):
            logger.info(f"[{i+1}/{len(pending)}] {item['user_input'][:70]}...")

            result = await run_one(item)
            out.write(json.dumps(result) + "\n")
            out.flush()  # write immediately so no results are lost on crash

            if i < len(pending) - 1:
                await asyncio.sleep(DELAY_SECONDS)

    logger.info(f"Done. Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())