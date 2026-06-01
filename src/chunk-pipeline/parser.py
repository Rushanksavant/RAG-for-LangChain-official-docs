import os
import re
from typing import List, Dict, Any, Tuple


class DocumentParser:
    """
    MDX-aware documentation parser for the langchain-ai/docs repository.

    Produces hierarchical Parent (sections) + Child (code, table, text) chunks
    with full metadata lineage on every chunk.

    Fixes vs original:
      - Strips Mintlify/MDX noise BEFORE chunking (:::python fences, JSX components, import statements, icon= metadata on code fences, frontmatter extra keys)
      - Handles pre-first-## intro content as its own parent (was silently dropped)
      - Code-block regex accepts fences with NO language tag (+ → *)
      - Text accumulator target raised to 900 chars for better embedding quality
      - Framework + version injected inline into parent text for reranker visibility
    """

    def __init__(self, file_path: str, framework: str, version: str):
        self.file_path = file_path
        self.framework = framework   # "langchain" | "langgraph" | "langsmith"
        self.version = version       # e.g. "0.3.x"

    # ------------------------------------------------------------------ 
    #  1. Read the file from provided file-path                                                              
    # ------------------------------------------------------------------ 
    def _read_file(self) -> str:
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read()

    # ------------------------------------------------------------------ 
    #  2. YAML front matter                                                 
    # ------------------------------------------------------------------ 
    def parse_yaml_front_matter(self, content: str) -> Tuple[Dict[str, str], str]:
        """
        Strips the YAML block at the top of an MDX file and returns (metadata_dict, remaining_content).

        Captures: title, sidebarTitle, description.
        All other front-matter keys (mode, canonical, etc.) are discarded.
        """
        meta = {"title": "", "description": "", "sidebarTitle": ""}
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if match:
            for line in match.group(1).split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip().lower()
                    val = val.strip().strip('"').strip("'")
                    if key in meta:
                        meta[key] = val
            content = content[match.end():]
        return meta, content

    # ------------------------------------------------------------------ 
    #  3. MDX / Mintlify noise removal                                   
    # ------------------------------------------------------------------ 
    def _clean_mdx(self, content: str) -> str:
        """
        Removes Mintlify-specific MDX syntax that adds zero semantic value to embeddings.  
        Operates on the full document BEFORE section splitting
        so that every parent and child chunk is already clean.

        What is removed:
          - base64 embedded images — long binary data
          - MDX import/export statements at top of the file 
          - :::python / :::js  language-fence wrappers  (but NOT the code inside)
          - Mintlify JSX block component: 
            - <Tooltip ...>text</Tooltip>, we will remove tags and keep text intact
            - self-closing tags 
            - Open tags
            - closing tags
            Eg: <CodeGroup>, <Columns>, <Card ...>, <Steps>, <Step ...>, <Expandable ...>, <Tip>, <Note>, <Warning>, <Info>, <Check>, <Icon ...>
          - shield.io and other badge image URLs inside table cells
          - Inline icon= / arrow= / cta= metadata on code fence labels

        What is KEPT:
          - All ``` code fences and their contents (untouched)
          - Markdown tables (structure carries meaning)
          - Bold / italic / inline-code spans
          - Blockquotes  (> ...)
          - Heading lines (# / ## / ###)
          - Plain paragraphs
        """
        # 0. Strip base64 embedded images — binary data, zero semantic value
        content = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '[image]', content)

        # 1. MDX import / export lines (comes first before other passes) e.g.  import { Foo } from "@/components/Foo"
        content = re.sub(r"^(?:import|export)\s+.*$", "", content, flags=re.MULTILINE)

        # 2. :::python / :::js wrapper fences — strip the fence line only, keep everything between them (code blocks) intact
        content = re.sub(r"^:::[a-zA-Z]+\s*$", "", content, flags=re.MULTILINE)

        # 3. Inline <Tooltip tip="...">visible text</Tooltip>  → keep visible text intact
        content = re.sub(r'<Tooltip[^>]*>([^<]*)</Tooltip>', r'\1', content)

        # 4. Self-closing JSX tags with attributes, e.g. <Icon icon="wand" size={20} />
        content = re.sub(r"<[A-Z][A-Za-z]*(?:\s[^>]*)?\s*/>", "", content)

        # 5. Block-level JSX open tags (may span one line), e.g. <Card title="..." href="...">
        #    We strip the tag itself but keep the content between open and close tags.
        #    Handles: <Tip>, <Note>, <Warning>, <Info>, <Check>, <Steps>, <Step ...>,
        #              <Expandable ...>, <Columns ...>, <CodeGroup>, <Card ...>
        content = re.sub(r"<(?:Tip|Note|Warning|Info|Check|Steps|Expandable|Columns|CodeGroup)(?:\s[^>]*)?>", "", content)
        content = re.sub(r"<Step(?:\s[^>]*)?>", "", content)
        content = re.sub(r"<Card(?:\s[^>]*)?>", "", content)

        # 6. Closing JSX tags
        content = re.sub(
            r"</(?:Tip|Note|Warning|Info|Check|Steps|Step|Expandable|Columns|CodeGroup|Card)>",
            "",
            content
        )

        # 7. Code-fence label metadata: ```python Set API key icon="key" → normalise to  ```python
        #    The label text (e.g. "Set API key") carries some signal so we keep it
        #    as a comment inside the fence via post-processing in _extract_child_elements.
        #    Here we just strip the icon= / arrow= / cta= key-value pairs.
        content = re.sub(
            r"(```[a-zA-Z0-9_\-]*)\s+([^\n`{]+?)\s+(?:icon|arrow|cta)=[^\n]*",
            r"\1  # \2",   # keep label as inline comment, drop icon= junk
            content
        )

        # 8. Badge / shield.io image URLs inside table cells — they never embed usefully
        content = re.sub(r"!\[.*?\]\(https://img\.shields\.io[^)]*\)", "", content)

        # 9. Collapse runs of 3+ blank lines left behind by the above stripping
        content = re.sub(r"\n{3,}", "\n\n", content)

        return content

    # ------------------------------------------------------------------ 
    #  4. Child element extraction                                          
    # ------------------------------------------------------------------ 
    def _extract_child_elements(self, section_text: str) -> Tuple[List[Dict], List[str], List[str]]:
        """
        Splits a section body into three child groups:
          code_blocks  - list of dicts {language, raw_text}
          tables       - list of raw markdown table strings
          paragraphs   - list of accumulated prose strings (~900 chars each)

        Key fixes vs original:
          - Language tag is now OPTIONAL in code-fence regex (was mandatory, so
            unlabelled fences were stripped but never captured → silent data loss)
          - TARGET_MIN_CHARS raised from 450 → 900 for richer embeddings
        """
        bt = "\x60\x60\x60"  # triple backtick, hex-encoded to avoid UI rendering issues

        # ── 1. Code blocks ──────────────────────────────────────────────
        # Language tag is optional (*) — fixes the silent-drop bug
        code_pattern = rf"{bt}([a-zA-Z0-9_\-]*)\s*?\n(.*?)\n{bt}"
        raw_code_matches = re.findall(code_pattern, section_text, re.DOTALL)

        code_blocks = []
        for lang, code_body in raw_code_matches:
            lang_clean = lang.lower().strip() if lang.strip() else "text"
            raw_markdown = f"```{lang_clean}\n{code_body.strip()}\n```"
            code_blocks.append({
                "language": lang_clean,
                "raw_text": raw_markdown})

        # Replacing codeblocks with empty string before looking for tables/prose
        strip_code_pattern = rf"{bt}[a-zA-Z0-9_\-]*\s*?\n.*?\n{bt}"
        stripped = re.sub(strip_code_pattern, "", section_text, flags=re.DOTALL) 

        # ── 2. Markdown tables ───────────────────────────────────────────
        table_pattern = (
            r"(\|[^\n]+\|\r?\n"
            r"\|[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|\r?\n"
            r"(?:\|[^\n]+\|\r?\n?)*)"
        )
        tables = re.findall(table_pattern, stripped, re.MULTILINE)
        stripped = re.sub(table_pattern, "", stripped, flags=re.MULTILINE)

        # ── 3. Paragraphs (accumulated to ~900 chars) ─────────────
        # Raised from 450 → 900: BGE-large embeds richer context more accurately
        TARGET_MIN_CHARS = 900

        raw_paragraphs = stripped.split("\n\n")
        paragraphs: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for p in raw_paragraphs:
            cleaned = p.strip()
            # Skip empty lines, very short fragments, and heading lines
            if not cleaned or len(cleaned) < 10 or cleaned.startswith("#"):
                continue

            current_chunk.append(cleaned)
            current_len += len(cleaned)

            if current_len >= TARGET_MIN_CHARS:
                paragraphs.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0

        # Flush residual buffer
        if current_chunk:
            combined = "\n\n".join(current_chunk)
            if len(combined) > 20:
                paragraphs.append(combined)

        return code_blocks, tables, paragraphs

    # ------------------------------------------------------------------ #
    #  Main parse driver                                                   #
    # ------------------------------------------------------------------ #
    def parse(self) -> List[Dict[str, Any]]:
        """
        Full parse pipeline:
          1. Read file
          2. Strip YAML front matter
          3. Clean MDX/Mintlify noise from the entire document
          4. Split on ## headings into sections
          5. Handle pre-## intro block (was silently dropped before)
          6. For each section → create parent chunk + child chunks
        """
        raw_content = self._read_file()
        # file_base = os.path.basename(self.file_path).replace(".", "_")
        # global_doc_id = f"{self.framework}_{file_base}"
        rel_path = os.path.relpath(self.file_path).replace("\\", "/")
        file_slug = re.sub(r"[^a-zA-Z0-9]", "_", rel_path).strip("_")
        global_doc_id = f"{self.framework}_{file_slug}"

        # ── Step 1: Front matter ─────────────────────────────────────────
        front_matter, content = self.parse_yaml_front_matter(raw_content)
        doc_title = (front_matter.get("title") or front_matter.get("sidebarTitle") or 
                     os.path.basename(self.file_path).replace(".", "_").replace("_md", "").replace("_mdx", "").replace("_", " ").title()) # take tile from either front matter "title"/"sidebarTitle" (or) file location

        # ── Step 2: Clean MDX noise from the whole document ──────────────
        content = self._clean_mdx(content)

        # ── Step 3: Split on ## headings ─────────────────────────────────
        # re.split with a capturing group includes the delimiter in the result list:
        #   sections[0]           = text BEFORE first ## (intro block)
        #   sections[1], [3], ... = ## heading lines
        #   sections[2], [4], ... = body text after each heading
        sections = re.split(r"\n(##\s+[^\n]+)\n", "\n" + content)

        # Edge case: document has no ## headings at all
        if len(sections) <= 1:
            sections = ["## Overview", content]

        chunks: List[Dict[str, Any]] = []

        # ── Step 4: Handle intro block (content before first ##) ─────────
        # FIX: in original code sections[0] was completely ignored — intro text lost.
        intro_body = sections[0].strip()
        if intro_body:
            intro_id = f"{global_doc_id}_intro"
            # Inline framework/version so reranker sees them without needing metadata
            parent_text = (f"Framework: {self.framework} | Version: {self.version}\n"f"# {doc_title}\n\n{intro_body}")

            chunks.append({
                "id": intro_id,
                "type": "parent",
                "text": parent_text,
                "metadata": {
                    "source_file": self.file_path,
                    "framework": self.framework,
                    "version": self.version,
                    "global_title": doc_title,
                    "section_heading": "Introduction",
                    "sub_headings": [],
                    "doc_id": global_doc_id,
                },
            })
            codes, tables, paragraphs = self._extract_child_elements(intro_body)
            chunks.extend(self._build_children(
                codes, tables, paragraphs, intro_id, doc_title, "Introduction"
            ))

        # ── Step 5: Process each ## section ──────────────────────────────
        for section_num, i in enumerate(range(1, len(sections), 2)):
            header_line = sections[i]
            section_body = sections[i + 1] if (i + 1) < len(sections) else ""

            header = header_line.replace("##", "").strip()
            heading_slug = re.sub(r"[^a-zA-Z0-9]", "_", header.lower())
            heading_slug = re.sub(r"_+", "_", heading_slug).strip("_")
            parent_id = f"{global_doc_id}_{heading_slug}_{section_num}"

            sub_headings = re.findall(r"###\s+([^\n]+)", section_body)

            # Inline framework/version into parent text body
            # This ensures the reranker sees "langchain", "langgraph", etc.
            # even when those words don't naturally appear in the section prose.
            parent_text = (
                f"Framework: {self.framework} | Version: {self.version}\n"
                f"# {doc_title} > {header}\n\n{section_body.strip()}"
            )

            chunks.append({
                "id": parent_id,
                "type": "parent",
                "text": parent_text,
                "metadata": {
                    "source_file": self.file_path,
                    "framework": self.framework,
                    "version": self.version,
                    "global_title": doc_title,
                    "section_heading": header,
                    "sub_headings": sub_headings,
                    "doc_id": global_doc_id,
                },
            })

            codes, tables, paragraphs = self._extract_child_elements(section_body)
            chunks.extend(self._build_children(
                codes, tables, paragraphs, parent_id, doc_title, header
            ))

        return chunks

    # ------------------------------------------------------------------ #
    #  Child chunk builder (shared by intro + sections)                   #
    # ------------------------------------------------------------------ #

    def _build_children(self,
        codes: List[Dict],
        tables: List[str],
        paragraphs: List[str],
        parent_id: str,
        doc_title: str,
        section_header: str,) -> List[Dict[str, Any]]:
        """
        Converts raw extracted elements into well-formed child chunk dicts.
        Centralised here so intro and section blocks share identical structure.

        Filters applied:
          - MIN_CHILD_CHARS = 80:  drops empty code fences, single-word fragments
          - MAX_CHILD_CHARS = 50000: drops base64 blobs and other non-text noise
        """
        MIN_CHILD_CHARS = 80
        MAX_CHILD_CHARS = 50_000

        children = []
        breadcrumb = f"{doc_title} > {section_header}"

        base_meta = {
            "parent_id": parent_id,
            "source_file": self.file_path,
            "framework": self.framework,
            "version": self.version,
            "breadcrumbs": breadcrumb,
        }

        for idx, code in enumerate(codes):
            if not (MIN_CHILD_CHARS <= len(code["raw_text"]) <= MAX_CHILD_CHARS):
                continue
            children.append({
                "id": f"{parent_id}_child_code_{idx}",
                "type": "child",
                "text": code["raw_text"],
                "metadata": {
                    **base_meta,
                    "content_category": "code_snippet",
                    "code_language": code["language"],
                },
            })

        for idx, table in enumerate(tables):
            if not (MIN_CHILD_CHARS <= len(table) <= MAX_CHILD_CHARS):
                continue
            children.append({
                "id": f"{parent_id}_child_table_{idx}",
                "type": "child",
                "text": table,
                "metadata": {
                    **base_meta,
                    "content_category": "structured_table",
                },
            })

        for idx, para in enumerate(paragraphs):
            if not (MIN_CHILD_CHARS <= len(para) <= MAX_CHILD_CHARS):
                continue
            children.append({
                "id": f"{parent_id}_child_para_{idx}",
                "type": "child",
                "text": para,
                "metadata": {
                    **base_meta,
                    "content_category": "descriptive_text",
                },
            })

        return children


if __name__ == "__main__":
    print("DocumentParser module loaded successfully.")