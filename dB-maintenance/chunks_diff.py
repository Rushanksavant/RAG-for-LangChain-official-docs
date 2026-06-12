"""
chunks_diff.py — Parent-driven diff between two chunks snapshots.

Finds the two most recent chunks_*.json files in data/processed/,
compares them at the PARENT level only, and writes to dB-maintenance/:

    addition_{hash}_{date}.json  — chunks to upsert (parent + all its children)
    removal_{hash}_{date}.json   — chunk IDs to delete (parent + all its children)

Why parent-only comparison is sufficient and correct:
    Parent text = framework header + full section_body.
    Children are extracted directly from that same section_body.
    If any child's content changed, section_body changed, so parent text
    changed too. It is impossible for a child's text to change without its
    parent's text also changing. Therefore diffing children independently
    is redundant — parent comparison captures all changes.

What goes into addition.json:
    - New parent IDs (section added to docs)
    - Parent IDs with changed text (section updated)
    + ALL new children belonging to those parents

What goes into removal.json:
    - Parent IDs absent from new snapshot (section removed)
    + ALL old children belonging to removed parents
    + ALL old children belonging to updated parents (crucial for cleanup)

Usage:
    python dB-maintenance/chunks_diff.py
"""

import json
import hashlib
import re
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("chunks_diff")

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_DIR    = Path(__file__).resolve().parent  # dB-maintenance/


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_two_latest_chunks_files() -> tuple[Path, Path]:
    """
    Scans data/processed/ for files matching chunks_*.json,
    sorts by the YYYYMMDD date in the filename,
    and returns (older_file, newer_file).

    Filename format: chunks_{commit_hash}_{YYYYMMDD}.json
    """
    pattern = re.compile(r"^chunks_[a-z0-9]+_(\d{8})\.json$")

    candidates = []
    for f in PROCESSED_DIR.iterdir():
        match = pattern.match(f.name)
        if match:
            candidates.append((int(match.group(1)), f))

    if len(candidates) < 2:
        raise FileNotFoundError(
            f"Expected at least 2 chunks_*.json files in {PROCESSED_DIR}, "
            f"found {len(candidates)}. Run run_parser.py on the updated docs first."
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-2][1], candidates[-1][1]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_and_split(filepath: Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """
    Loads a chunks JSON file and returns two structures:

        parent_map   — {parent_id: parent_chunk}
        children_map — {parent_id: [child_chunk, ...]}
    """
    logger.info(f"Loading: {filepath.name}")
    with open(filepath, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    parent_map   = {}
    children_map = defaultdict(list)

    for chunk in chunks:
        if "id" not in chunk or "text" not in chunk:
            continue
        if chunk["type"] == "parent":
            parent_map[chunk["id"]] = chunk
        elif chunk["type"] == "child":
            pid = chunk["metadata"].get("parent_id")
            if pid:
                children_map[pid].append(chunk)

    logger.info(
        f"  {len(parent_map):,} parents, "
        f"{sum(len(v) for v in children_map.values()):,} children"
    )
    return parent_map, children_map


# ── Core diff ──────────────────────────────────────────────────────────────────

def compute_diff(
    old_parents:  dict[str, dict],
    old_children: dict[str, list],
    new_parents:  dict[str, dict],
    new_children: dict[str, list],
) -> tuple[list[dict], list[str], set[str], set[str], set[str]]:
    """
    Parent-driven diff. Returns:

        addition           — chunk dicts to upsert
        removal            — chunk IDs to delete
        added_parent_ids   — parents new in new snapshot
        updated_parent_ids — parents present in both with changed text
        removed_parent_ids — parents gone from new snapshot
    """
    old_ids = set(old_parents)
    new_ids = set(new_parents)

    added_parent_ids   = new_ids - old_ids
    removed_parent_ids = old_ids - new_ids
    updated_parent_ids = {
        pid for pid in (old_ids & new_ids)
        if content_hash(old_parents[pid]["text"]) != content_hash(new_parents[pid]["text"])
    }

    # addition: new + updated parents and ALL their NEW children
    addition = []
    for pid in sorted(added_parent_ids | updated_parent_ids):
        addition.append(new_parents[pid])
        addition.extend(new_children.get(pid, []))

    # removal: removed parents and ALL their OLD children IDs
    removal = []
    for pid in sorted(removed_parent_ids):
        removal.append(pid)
        removal.extend(c["id"] for c in old_children.get(pid, []))
        
    # BUG FIX: Also remove all OLD children IDs of UPDATED parents.
    for pid in sorted(updated_parent_ids):
        removal.extend(c["id"] for c in old_children.get(pid, []))

    return addition, removal, added_parent_ids, updated_parent_ids, removed_parent_ids


# ── Detailed terminal output ───────────────────────────────────────────────────

def log_stats(
    addition:           list[dict],
    removal:            list[str],
    old_parents:        dict,
    old_children:       dict,
    new_parents:        dict,
    new_children:       dict,
    added_parent_ids:   set,
    updated_parent_ids: set,
    removed_parent_ids: set,
) -> None:

    SEP  = "─" * 58
    SEP2 = "═" * 58

    # ── Snapshot overview ──────────────────────────────────────────────────
    n_unchanged = len(set(old_parents) & set(new_parents)) - len(updated_parent_ids)

    logger.info(SEP2)
    logger.info("  SNAPSHOT OVERVIEW")
    logger.info(SEP2)
    logger.info(f"  Old snapshot:  {len(old_parents):>6,} parents  │  {sum(len(v) for v in old_children.values()):>7,} children")
    logger.info(f"  New snapshot:  {len(new_parents):>6,} parents  │  {sum(len(v) for v in new_children.values()):>7,} children")
    logger.info(f"  Unchanged:     {n_unchanged:>6,} parents  (skipped — no re-indexing needed)")

    # ── New sections ───────────────────────────────────────────────────────
    logger.info(SEP)
    logger.info(f"  NEW SECTIONS  (+{len(added_parent_ids)} parents)")
    logger.info(SEP)
    if added_parent_ids:
        total_new_children = 0
        for pid in sorted(added_parent_ids):
            p         = new_parents[pid]
            n_ch      = len(new_children.get(pid, []))
            total_new_children += n_ch
            framework = p["metadata"].get("framework", "")
            heading   = p["metadata"].get("section_heading", "")
            source    = p["metadata"].get("source_file", "")
            logger.info(f"  + [{framework}]  {heading}")
            logger.info(f"        file:      {source}")
            logger.info(f"        children:  {n_ch}")
        logger.info(f"  ┌ subtotal — parents: {len(added_parent_ids)}  children: {total_new_children}")
    else:
        logger.info("  (none)")

    # ── Updated sections ───────────────────────────────────────────────────
    logger.info(SEP)
    logger.info(f"  UPDATED SECTIONS  (~{len(updated_parent_ids)} parents)")
    logger.info(SEP)
    if updated_parent_ids:
        total_updated_children = 0
        for pid in sorted(updated_parent_ids):
            p         = new_parents[pid]
            n_ch      = len(new_children.get(pid, []))
            total_updated_children += n_ch
            framework = p["metadata"].get("framework", "")
            heading   = p["metadata"].get("section_heading", "")
            source    = p["metadata"].get("source_file", "")
            logger.info(f"  ~ [{framework}]  {heading}")
            logger.info(f"        file:      {source}")
            logger.info(f"        children:  {n_ch}")
        logger.info(f"  ┌ subtotal — parents: {len(updated_parent_ids)}  children: {total_updated_children}")
    else:
        logger.info("  (none)")

    # ── Removed sections ───────────────────────────────────────────────────
    logger.info(SEP)
    logger.info(f"  REMOVED SECTIONS  (-{len(removed_parent_ids)} parents)")
    logger.info(SEP)
    if removed_parent_ids:
        total_removed_children = 0
        for pid in sorted(removed_parent_ids):
            p         = old_parents[pid]
            n_ch      = len(old_children.get(pid, []))
            total_removed_children += n_ch
            framework = p["metadata"].get("framework", "")
            heading   = p["metadata"].get("section_heading", "")
            source    = p["metadata"].get("source_file", "")
            logger.info(f"  - [{framework}]  {heading}")
            logger.info(f"        file:      {source}")
            logger.info(f"        children:  {n_ch}")
        logger.info(f"  ┌ subtotal — parents: {len(removed_parent_ids)}  children: {total_removed_children}")
    else:
        logger.info("  (none)")

    # ── Final counts ───────────────────────────────────────────────────────
    n_addition_children = sum(1 for c in addition if c["type"] == "child")
    n_removal_parents   = sum(1 for rid in removal if rid in old_parents)
    n_removal_children  = len(removal) - n_removal_parents

    logger.info(SEP2)
    logger.info("  FINAL COUNTS")
    logger.info(SEP2)
    logger.info(f"  addition.json  →  {len(addition):,} chunks to upsert")
    logger.info(f"      parents (new):      {len(added_parent_ids):,}")
    logger.info(f"      parents (updated):  {len(updated_parent_ids):,}")
    logger.info(f"      children:           {n_addition_children:,}")
    logger.info(f"  removal.json   →  {len(removal):,} IDs to delete")
    logger.info(f"      parents:            {n_removal_parents:,}")
    logger.info(f"      children:           {n_removal_children:,}")
    logger.info(SEP2)


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    # 1. Find the two most recent chunks files
    older_file, newer_file = find_two_latest_chunks_files()
    logger.info(f"Old snapshot: {older_file.name}")
    logger.info(f"New snapshot: {newer_file.name}")

    # 2. Load and split into parent/children maps
    old_parents, old_children = load_and_split(older_file)
    new_parents, new_children = load_and_split(newer_file)

    # 3. Compute diff (returns change sets for logging)
    logger.info("Computing diff...")
    addition, removal, added_pids, updated_pids, removed_pids = compute_diff(
        old_parents, old_children,
        new_parents, new_children,
    )

    # 4. Detailed terminal output
    log_stats(
        addition, removal,
        old_parents, old_children,
        new_parents, new_children,
        added_pids, updated_pids, removed_pids,
    )

    # 5. Write outputs named after the new snapshot for traceability
    new_suffix    = newer_file.stem.replace("chunks_", "")  # e.g. d27603b_20260529
    addition_path = OUTPUT_DIR / f"addition_{new_suffix}.json"
    removal_path  = OUTPUT_DIR / f"removal_{new_suffix}.json"

    with open(addition_path, "w", encoding="utf-8") as f:
        json.dump(addition, f, indent=2, ensure_ascii=False)
    logger.info(f"Written: {addition_path.name}  ({len(addition):,} chunks)")

    with open(removal_path, "w", encoding="utf-8") as f:
        json.dump(removal, f, indent=2, ensure_ascii=False)
    logger.info(f"Written: {removal_path.name}  ({len(removal):,} IDs)")

    if not addition and not removal:
        logger.info("✅ No changes — Qdrant DB is already up to date.")
    else:
        logger.info("✅ Diff complete. Feed addition.json to the Colab indexing notebook,")
        logger.info("   then run migrate.py to sync additions and deletions to Qdrant Cloud.")


if __name__ == "__main__":
    run()