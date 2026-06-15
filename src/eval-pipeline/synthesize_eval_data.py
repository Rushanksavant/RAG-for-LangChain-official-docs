"""
Synthetic evaluation dataset generator.

Samples parent chunks from the latest chunks JSON, sends them to Gemini
in batches of 3, and asks it to generate questions + reference answers.
Outputs a JSONL file ready for DeepEval-based evaluation.

Dataset row structure:
    question          : str         — the generated question
    question_type     : str         — single_hop | multi_hop | edge_case
    framework         : str         — langchain | langgraph | langsmith | deepagents
    source_parent_ids : list[str]   — chunk IDs used to generate the question
    reference_answer  : str         — ground truth answer (from chunk text)
    relevant_context  : list[str]   — chunk texts needed to answer

Usage:
    python eval-pipeline/synthesize_eval_data.py

Output:
    eval-pipeline/eval_dataset.jsonl
"""

import json
import random
import time
import logging
import re
from pathlib import Path
from collections import defaultdict
from utilities.to_use import EDGE_CASE_QUESTIONS
from utilities import data_helper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset_gen")


# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_FILE      = Path("eval_dataset.jsonl")
BATCH_SIZE       = 3        # chunks per LLM call (3 enables multi-hop questions)
# Delay between API calls to stay within Gemini free tier rate limits
API_CALL_DELAY_SECONDS = 16


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load and sample chunks
    chunks_file  = data_helper.find_latest_chunks_file()
    by_framework = data_helper.load_parent_chunks(chunks_file)
    sampled      = data_helper.stratified_sample(by_framework)

    logger.info(f"Total sampled parents: {len(sampled)}")
    logger.info(f"Expected batches: {len(sampled) // BATCH_SIZE}")

    dataset = []

    # 2. Process in batches of BATCH_SIZE
    batches = [sampled[i:i+BATCH_SIZE] for i in range(0, len(sampled), BATCH_SIZE)]

    for batch_num, batch in enumerate(batches, 1):
        logger.info(f"Batch {batch_num}/{len(batches)} — frameworks: "
                    f"{[c['metadata']['framework'] for c in batch]}")

        prompt    = data_helper.build_prompt(batch)
        questions = data_helper.call_gemini(prompt)

        if not questions:
            logger.warning(f"Batch {batch_num} failed — skipping.")
            time.sleep(API_CALL_DELAY_SECONDS)
            continue

        for q in questions:
            row = data_helper.build_dataset_row(
                question_data = q,
                batch         = batch,
                question_type = q.get("question_type", "single_hop"),
            )
            dataset.append(row)

        logger.info(f"  Generated {len(questions)} questions. Dataset total: {len(dataset)}")

        # Respect Gemini free tier rate limits between calls
        time.sleep(API_CALL_DELAY_SECONDS)

    # 3. Add edge case questions (no chunks, no LLM call needed)
    for edge in EDGE_CASE_QUESTIONS:
        dataset.append({
            "question":          edge["question"],
            "question_type":     "edge_case",
            "framework":         "all",
            "source_parent_ids": [],
            "reference_answer":  edge.get("expected_guardrail", ""),
            "relevant_context":  [],
            "expected_guardrail": edge.get("expected_guardrail"),
            "is_conversation_start": edge.get("is_conversation_start", False),
            "follows":           edge.get("follows"),   # for multi-turn questions

            # Filled during eval run
            "query_plan":        None,
            "retrieved_chunks":  None,
            "final_response":    None,
        })

    logger.info(f"Edge cases added: {len(EDGE_CASE_QUESTIONS)}")

    # 4. Write to JSONL — one JSON object per line, easy to stream and append
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(f"")
    logger.info(f"── Dataset generation complete ─────────────────")
    logger.info(f"  Total questions:    {len(dataset)}")
    logger.info(f"  Retrieval questions:{len(dataset) - len(EDGE_CASE_QUESTIONS)}")
    logger.info(f"  Edge cases:         {len(EDGE_CASE_QUESTIONS)}")
    logger.info(f"  Written to:         {OUTPUT_FILE}")
    logger.info(f"────────────────────────────────────────────────")


if __name__ == "__main__":
    run()