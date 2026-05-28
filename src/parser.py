import os
import re
from typing import List, Dict, Any, Tuple

class DocumentParser:
    """
    Layout-aware documentation parser that ingests Markdown/MDX
    and splits it into hierarchical Parent (Sections) and Child (Code, Tables, Paragraphs)
    chunks with full metadata lineage, context propagation, and internal link mapping.
    """
    def __init__(self, file_path: str, framework: str, version: str):
        self.file_path = file_path
        self.framework = framework  # "langchain", "langgraph", "langsmith"
        self.version = version      # e.g., "1.3.x"
        
    def _read_file(self) -> str:
        """Reads file contents securely with UTF-8 encoding."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def parse_yaml_front_matter(self, content: str) -> Tuple[Dict[str, str], str]:
        """
        Extracts YAML front matter tags (like title, description) at the start of MD/MDX files.
        Returns a dictionary of front matter metadata and the remaining stripped content.
        """
        meta = {"title": "", "description": ""}
        # Match YAML block at the absolute start of the file
        match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if match:
            front_matter_text = match.group(1)
            # Parse simple key-value pairs (avoiding external yaml library overhead)
            for line in front_matter_text.split('\n'):
                if ':' in line:
                    key, val = line.split(':', 1)
                    key = key.strip().lower()
                    val = val.strip().strip('"').strip("'")
                    if key in meta:
                        meta[key] = val
            # Strip the YAML block from the main content body
            content = content[match.end():]
        return meta, content

    def _extract_child_elements(self, section_text: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Isolates child fragments within a section: code blocks, tables, and remaining paragraphs.
        """
        # Define 3 backticks dynamically using hex values to prevent markdown parsing errors in UI
        bt = '\x60\x60\x60'
        
        # 1. Extract Python Code Blocks (using dynamically constructed backticks regex)
        code_pattern = rf'({bt}python\s+.*?{bt})'
        code_blocks = re.findall(code_pattern, section_text, re.DOTALL)
        
        # Strip code blocks to parse tables easily
        stripped_text = re.sub(code_pattern, '', section_text, flags=re.DOTALL)
        
        # 2. Extract Markdown Tables
        table_pattern = r'(\|[^\n]+\|\r?\n\|[ \t]*:?---:?[ \t]*(?:\|[ \t]*:?---:?[ \t]*)*\|\r?\n(?:\|[^\n]+\|\r?\n?)*)'
        tables = re.findall(table_pattern, stripped_text, re.MULTILINE)
        
        # Strip tables to leave only plain descriptive text paragraphs
        stripped_text = re.sub(table_pattern, '', stripped_text, flags=re.MULTILINE)
        
        # 3. Extract Paragraphs (split by double newlines, clean out excess spaces)
        raw_paragraphs = stripped_text.split('\n\n')
        paragraphs = []
        for p in raw_paragraphs:
            cleaned = p.strip()
            # Avoid keeping tiny fragment strings or leftover headers
            if len(cleaned) > 20 and not cleaned.startswith('#'):
                paragraphs.append(cleaned)
                
        return code_blocks, tables, paragraphs

    def parse(self) -> List[Dict[str, Any]]:
        """
        Transforms a document into a structured list of hierarchical parent-child chunks.
        """
        raw_content = self._read_file()
        file_base = os.path.basename(self.file_path).replace('.', '_')
        global_doc_id = f"{self.framework}_{file_base}"
        
        # Extract metadata headers and strip YAML if present
        front_matter, content = self.parse_yaml_front_matter(raw_content)
        doc_title = front_matter.get("title") or file_base.replace('_md', '').replace('_mdx', '').title()
        
        # Split document by Heading 2 (##) boundaries
        # This keeps heading-level chunks as our manageable structural Parents
        sections = re.split(r'\n(##\s+[^\n]+)\n', '\n' + content)
        
        # Handle Edge Case: No Heading 2 found -> Treat entire file body as single parent
        if len(sections) <= 1:
            sections = ["## Introduction", content]
            
        chunks = []
        
        # Process split segments (Header line matches are indexed as dividers)
        for i in range(1, len(sections), 2):
            header = sections[i].replace('##', '').strip()
            section_body = sections[i+1] if (i+1) < len(sections) else ""
            
            # Generate deterministic Parent Section ID
            heading_slug = re.sub(r'[^a-zA-Z0-9_]', '', header.lower().replace(' ', '_'))
            parent_id = f"{global_doc_id}_{heading_slug}"
            
            # Extract nested sub-heading (###) breadcrumbs within this parent body
            sub_headings = [sh.replace('###', '').strip() for sh in re.findall(r'###\s+([^\n]+)', section_body)]
            
            # Append Parent Chunk (Context Reservoir)
            chunks.append({
                "id": parent_id,
                "type": "parent",
                "text": f"# {doc_title} - {header}\n\n{section_body.strip()}",
                "metadata": {
                    "source_file": self.file_path,
                    "framework": self.framework,
                    "version": self.version,
                    "global_title": doc_title,
                    "section_heading": header,
                    "sub_headings": sub_headings,
                    "doc_id": global_doc_id
                }
            })
            
            # Extract granular children from this section's body
            codes, tables, paragraphs = self._extract_child_elements(section_body)
            
            # Add Code Children
            for idx, code in enumerate(codes):
                chunks.append({
                    "id": f"{parent_id}_child_code_{idx}",
                    "type": "child",
                    "text": code,
                    "metadata": {
                        "parent_id": parent_id,
                        "source_file": self.file_path,
                        "framework": self.framework,
                        "version": self.version,
                        "content_category": "code_snippet",
                        "breadcrumbs": f"{doc_title} > {header}"
                    }
                })
                
            # Add Table Children
            for idx, table in enumerate(tables):
                chunks.append({
                    "id": f"{parent_id}_child_table_{idx}",
                    "type": "child",
                    "text": table,
                    "metadata": {
                        "parent_id": parent_id,
                        "source_file": self.file_path,
                        "framework": self.framework,
                        "version": self.version,
                        "content_category": "structured_table",
                        "breadcrumbs": f"{doc_title} > {header}"
                    }
                })
                
            # Add Explanatory Paragraph Children (Eliminates the "Ghost Text" problem)
            for idx, para in enumerate(paragraphs):
                para_breadcrumb = f"{doc_title} > {header}"
                chunks.append({
                    "id": f"{parent_id}_child_para_{idx}",
                    "type": "child",
                    "text": para,
                    "metadata": {
                        "parent_id": parent_id,
                        "source_file": self.file_path,
                        "framework": self.framework,
                        "version": self.version,
                        "content_category": "descriptive_text",
                        "breadcrumbs": para_breadcrumb
                    }
                })
                
        return chunks

if __name__ == "__main__":
    print("Documentation Parser module compiled successfully.")