import os
import json
import logging
from parser import DocumentParser

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# O(1) Hash Sets for rapid filtering
EXCLUDED_FILES = {
    "readme.md", "license.md", "contributing.md", "claude.md", 
    "code_of_conduct.md", "changelog.md", "index.md", "index.mdx"
}

EXCLUDED_DIR_KEYWORDS = {
    "node_modules", "venv", ".git", "static", "images", "_static", "snippets", "build"
}

def is_valid_doc_file(file_path: str) -> bool:
    """Applies defensive filters to isolate high-value Markdown guides."""
    file_name = os.path.basename(file_path).lower()
    
    if not file_name.endswith(('.md', '.mdx')): # if file not end with .md/.mdx
        return False
    if file_name in EXCLUDED_FILES: # if file name in excluded list
        return False
        
    for keyword in EXCLUDED_DIR_KEYWORDS: # if file path contains excluded directory name
        if f"{os.sep}{keyword}{os.sep}" in file_path.lower():
            return False
            
    return True

def run_pipeline():
    """Crawls directories, parses Markdown files, and outputs a Parent-Child JSON tree."""

    raw_dir = "data/raw" # where cloned official repos exists
    processed_dir = "data/processed" # to store output
    os.makedirs(processed_dir, exist_ok=True)
    
    target_docs_path = os.path.join(raw_dir, "conceptual_docs") # data/raw/conceptual_docs: official documentation repo stored
    if not os.path.exists(target_docs_path):
        target_docs_path = raw_dir
        
    all_chunks = []
    failed_files = []
    processed_count = 0
    
    # Recursive filesystem crawl
    for root, dirs, files in os.walk(target_docs_path):
        # for optimization: Prune excluded directories in-place to prevent disk reads
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDED_DIR_KEYWORDS] # not to look in excluded directories
        
        for file in files:
            file_path = os.path.join(root, file)
            
            if is_valid_doc_file(file_path):

                # Get framework from file path
                framework = "langchain" # set default
                path_lower = file_path.lower()
                if "langgraph" in path_lower:
                    framework = "langgraph"
                elif "langsmith" in path_lower:
                    framework = "langsmith"
                
                # Fault-isolated execution
                try:
                    parser = DocumentParser(file_path=file_path, framework=framework, version="1.3.x")
                    chunks = parser.parse() # calling parse method (contains entire pipeline for parent-child) of DocumentParser
                    all_chunks.extend(chunks)
                    processed_count += 1
                except Exception as e:
                    logging.error(f"Failed to parse file: {file_path} | Error: {str(e)}")
                    failed_files.append({"file": file_path, "error": str(e)})

    # Store parent-child chunks
    output_path = os.path.join(processed_dir, "chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
    # Store failed file-parsing paths for developer audits
    failed_output_path = os.path.join(processed_dir, "failed_files.json")
    with open(failed_output_path, "w", encoding="utf-8") as f:
        json.dump(failed_files, f, indent=2, ensure_ascii=False)
        
    logging.info(f"Pipeline complete. Processed: {processed_count}")
    logging.info(f"Successfully wrote chunks database to: {output_path}")
    
    if failed_files:
        logging.warning(f"Failed to process {len(failed_files)} files. Audit log saved to: {failed_output_path}")
    else:
        logging.info("Zero parsing failures recorded.")



if __name__ == "__main__":
    run_pipeline()