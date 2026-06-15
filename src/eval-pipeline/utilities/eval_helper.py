"""
eval_helpers.py — Pure utility functions for the eval pipeline.

Contains:
  - load_dataset()       : reads eval_dataset.jsonl
  - load_done_ids()      : reads existing eval_results.csv for resume support
  - build_session_map()  : assigns thread_ids (multi-turn questions share one)
  - save_results()       : writes incremental CSV after each question
  - compute_summary()    : aggregates scores and writes eval_summary.json

Nothing here calls the agent or the judge — pure data in, data out.
"""

import json
import csv
import logging
from pathlib import Path


def load_dataset(dataset_file: Path) -> list[dict]:
    """
    Reads eval_dataset.jsonl and returns a list of question dicts.
    Each line in the file is one JSON object (one question row).
    """
    rows = [
        json.loads(line)
        for line in open(dataset_file, encoding="utf-8")
        if line.strip()
    ]
    logging.info(f"Loaded {len(rows)} questions from {dataset_file.name}")
    return rows


def load_done_ids(results_file: Path) -> tuple[set[int], list[dict]]:
    """
    Reads an existing eval_results.csv (if it exists) and returns:
      - done_ids : set of question_ids already evaluated
      - results  : list of already-scored result dicts

    This enables resume: if the eval run crashed at question 50,
    re-running will skip questions 1-49 and continue from 50.
    """
    done_ids = set()
    results  = []

    if not results_file.exists():
        return done_ids, results

    with open(results_file, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done_ids.add(int(row["question_id"]))
            # CSV stores everything as strings — convert score columns back to float/None
            for k, v in row.items():
                if k not in ("question_id", "question_type"):
                    row[k] = float(v) if v else None
            results.append(row)

    logging.info(f"Resuming — {len(done_ids)} questions already done, skipping them.")
    return done_ids, results


def build_session_map(dataset: list[dict]) -> dict[str, str]:
    """
    Assigns a LangGraph thread_id to every question.

    Standalone questions each get their own unique session (clean memory).
    Multi-turn follow-up questions share a session with their parent so the
    agent has the correct conversation history when answering.

    The 'follows' field in the dataset row points to the parent question text.

    Example:
        "How do I create a ReAct agent?"   → session "eval-session-0"
        "How do I add memory to it?"       → session "eval-session-0"  (same — follows parent)
        "What checkpointer for prod?"      → session "eval-session-0"  (same — follows grandparent)
        "How to stream in LangChain?"      → session "eval-session-3"  (new standalone)

    Returns: {question_text: thread_id}
    """
    session_map = {}

    for j, row in enumerate(dataset):
        question = row["question"]
        follows  = row.get("follows")   # text of the parent question, or None

        if follows and follows in session_map:
            # This is a follow-up — inherit the parent's session ID
            session_map[question] = session_map[follows]
        else:
            # Standalone or conversation root — assign a new unique session
            session_map[question] = f"eval-session-{j}"

    return session_map


def save_results(results: list[dict], results_file: Path) -> None:
    """
    Writes all results so far to CSV.
    Called after every question so progress is never lost on a crash.
    """
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def compute_summary(results: list[dict], summary_file: Path) -> dict:
    """
    Averages scores across all questions, broken down by layer and question type.
    Writes the summary to JSON and returns it for terminal logging.
    """

    # Helper: average a list, ignoring None values (failed metric calls)
    def avg(values):
        clean = [v for v in values if v is not None]
        return round(sum(clean) / len(clean), 3) if clean else None

    summary = {
        "total_evaluated": len(results),

        "layer_1_query_planner": {
            "guardrail_accuracy":   avg([r["guardrail"]   for r in results]),
            "translation_quality":  avg([r["translation"] for r in results]),
        },

        "layer_2_mcp_retriever": {
            "contextual_precision": avg([r["precision"] for r in results]),
            "contextual_recall":    avg([r["recall"]    for r in results]),
        },

        "layer_3_final_generation": {
            "faithfulness":         avg([r["faithfulness"] for r in results]),
            "answer_relevancy":     avg([r["relevancy"]    for r in results]),
        },

        # Same metrics broken down by question type for finer-grained analysis
        "by_question_type": {
            qt: {
                "n":                    sum(1 for r in results if r["question_type"] == qt),
                "contextual_precision": avg([r["precision"]    for r in results if r["question_type"] == qt]),
                "contextual_recall":    avg([r["recall"]       for r in results if r["question_type"] == qt]),
                "faithfulness":         avg([r["faithfulness"] for r in results if r["question_type"] == qt]),
                "answer_relevancy":     avg([r["relevancy"]    for r in results if r["question_type"] == qt]),
            }
            for qt in ("single_hop", "multi_hop", "edge_case")
        },
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary