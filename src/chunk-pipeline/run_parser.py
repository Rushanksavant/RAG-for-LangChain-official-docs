import os
import json
import logging
from parser import DocumentParser
from helper import get_doc_repo_git_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# ── Version map ──────────────────────────────────────────────────────────────
# Update these when new releases drop — they get embedded into every chunk
# so filtered retrieval ("show me langgraph 0.3.x docs") stays accurate.
FRAMEWORK_VERSIONS = {
    "langchain":  "1.3.2",
    "langgraph":  "1.2.2",
    "langsmith":  "0.5.1",
    "deepagents": "0.6.7",  # included in oss/ — index or exclude as needed
}

# ── Files to skip (exact filename match, lowercased) ─────────────────────────
# These are either repo-management files or Mintlify config — not doc content.
EXCLUDED_FILES = {"readme.md", "license.md", "contributing.md",
    "claude.md",          # Mintlify AI assistant config — not user-facing docs
    "code_of_conduct.md", "changelog.md", "index.md", "index.mdx",
    "template.mdx",       # Integration page templates — boilerplate, not real docs
    "_template.mdx", "google_imagen.mdx",
    "docs.json",          # Mintlify nav config — not a content file
}

# ── Directory keywords that signal non-content folders ───────────────────────
# Any path component matching one of these causes the file to be skipped.
EXCLUDED_DIR_KEYWORDS = {"node_modules", "venv", ".git", "static", "images", "_static",
    "snippets",    # Reusable MDX fragments — not standalone pages; they lack context
    "build",       # Mintlify build output — generated from src/, never edit directly
    "pipeline",    # Build pipeline scripts
    "scripts", "tests", "fonts",
    "reference",   # API reference pages are auto-generated; low signal for RAG
}

# ── Framework routing ─────────────────────────────────────────────────────────
# The langchain-ai/docs repo uses following path structure under src/:
#
#   src/langsmith/          → langsmith
#   src/oss/langchain/      → langchain
#   src/oss/langgraph/      → langgraph
#   src/oss/deepagents/     → deepagents 
#   src/oss/integrations/   → langchain   (integration pages belong to langchain)
#   src/oss/concepts/       → langchain   (cross-cutting concepts)
#   src/oss/contributing/   → langchain
#


def detect_framework(file_path: str) -> str:
    """
    Determines which framework a doc file belongs to based on its path.
    Returns one of: "langchain", "langgraph", "langsmith", "deepagents".
    Defaults to "langchain" for unrecognised paths under oss/.
    """
    path_lower = file_path.lower().replace("\\", "/")

    # Most specific matches first
    if "/langsmith/" in path_lower:
        return "langsmith"
    if "/langgraph/" in path_lower:
        return "langgraph"
    if "/deepagents/" in path_lower:
        return "deepagents"
    # integrations/, concepts/, contributing/, releases/ all belong to langchain
    return "langchain"


def is_valid_doc_file(file_path: str) -> bool:
    """
    Returns True only for .md/.mdx files that are real user-facing documentation.

    Filters applied (in order):
      1. Must end with .md or .mdx
      2. Filename must not be in EXCLUDED_FILES
      3. No path segment may be in EXCLUDED_DIR_KEYWORDS
    """
    file_name = os.path.basename(file_path).lower()

    if not file_name.endswith((".md", ".mdx")):
        return False

    if file_name in EXCLUDED_FILES:
        return False

    # Normalise separators for cross-platform safety
    normalised = file_path.lower().replace("\\", "/")
    for keyword in EXCLUDED_DIR_KEYWORDS:
        if f"/{keyword}/" in normalised or normalised.endswith(f"/{keyword}"):
            return False

    return True


def run_pipeline() -> None:
    """
    Recursively crawls the langchain-ai/docs src/ directory, parses every
    valid .md/.mdx file into parent-child chunks, and writes two output files:

      data/processed/chunks.json        - all parsed chunks
      data/processed/failed_files.json  - files that raised an exception
    """
    # ── Path setup ───────────────────────────────────────────────────────
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Primary target: the src/ directory inside the cloned langchain-ai/docs repo.
    # Your local clone lives at: data/raw/conceptual_docs/
    # The actual MDX content is under: data/raw/conceptual_docs/src/
    target_docs_path = os.path.join(raw_dir, "conceptual_docs", "src")

    if not os.path.exists(target_docs_path):
        logging.error(
            f"Target directory not found: {target_docs_path}\n"
            f"Expected the langchain-ai/docs clone at: data/raw/conceptual_docs/\n"
            f"Clone it with: git clone https://github.com/langchain-ai/docs data/raw/conceptual_docs"
        )
        return

    # ── Crawl ────────────────────────────────────────────────────────────
    all_chunks = []
    failed_files = []
    processed_count = 0
    skipped_count = 0

    logging.info(f"Starting crawl: {target_docs_path}")

    for root, dirs, files in os.walk(target_docs_path):
        # Prune excluded directories IN-PLACE so os.walk doesn't descend into them.
        # This is more efficient than checking the full path on every file.
        dirs[:] = [
            d for d in dirs
            if d.lower() not in EXCLUDED_DIR_KEYWORDS
        ]

        for file_name in files:
            file_path = os.path.join(root, file_name)

            if not is_valid_doc_file(file_path):
                skipped_count += 1
                continue

            framework = detect_framework(file_path)
            version = FRAMEWORK_VERSIONS.get(framework, "unknown")

            try:
                parser = DocumentParser(
                    file_path=file_path,
                    framework=framework,
                    version=version,
                )
                chunks = parser.parse()

                if chunks:
                    all_chunks.extend(chunks)
                    processed_count += 1
                    logging.debug(
                        f"Parsed {file_path} → {len(chunks)} chunks "
                        f"({sum(1 for c in chunks if c['type']=='parent')} parents, "
                        f"{sum(1 for c in chunks if c['type']=='child')} children)"
                    )
                else:
                    # File parsed without error but produced no chunks — log it
                    logging.warning(f"Zero chunks produced for: {file_path}")
                    failed_files.append({"file": file_path, "error": "zero chunks produced"})

            except Exception as exc:
                logging.error(f"Parse failed: {file_path} | {exc}")
                failed_files.append({"file": file_path, "error": str(exc)})

    # ── Stats summary ────────────────────────────────────────────────────
    total_parents = sum(1 for c in all_chunks if c["type"] == "parent")
    total_children = sum(1 for c in all_chunks if c["type"] == "child")

    # Framework breakdown
    from collections import Counter
    fw_counts = Counter(
        c["metadata"]["framework"]
        for c in all_chunks
        if c["type"] == "child"
    )

    logging.info("── Crawl complete ──────────────────────────────")
    logging.info(f"  Files processed:  {processed_count}")
    logging.info(f"  Files skipped:    {skipped_count}")
    logging.info(f"  Files failed:     {len(failed_files)}")
    logging.info(f"  Total chunks:     {len(all_chunks)}")
    logging.info(f"    Parents:        {total_parents}")
    logging.info(f"    Children:       {total_children}")
    logging.info(f"  Child breakdown:  {dict(fw_counts)}")
    logging.info("────────────────────────────────────────────────")

    # ── Write outputs ────────────────────────────────────────────────────
    # File naming based on git commit for repo-cloning id and date
    # or chunks file formation date (if git commit extraction throws error)

    # 1. Fetch the Git metadata from the raw docs folder
    commit_date, commit_hash = get_doc_repo_git_info(target_docs_path)
    # 2. Construct the dynamic filename
    output_filename = f"chunks_{commit_hash}_{commit_date}.json" 

    # chunks_path = os.path.join(processed_dir, "chunks.json")
    chunks_path = os.path.join(processed_dir, output_filename)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    logging.info(f"Chunks written to: {chunks_path}")

    failed_path = os.path.join(processed_dir, "failed_files.json")
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(failed_files, f, indent=2, ensure_ascii=False)

    if failed_files:
        logging.warning(
            f"{len(failed_files)} files failed. Audit log: {failed_path}"
        )
    else:
        logging.info("Zero parsing failures.")


if __name__ == "__main__":
    run_pipeline()