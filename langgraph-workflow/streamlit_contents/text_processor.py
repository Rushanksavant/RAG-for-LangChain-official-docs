import re

def format_cell_content(cell_str: str) -> str:
    """
    Parses and styles text elements, inline code snippets, and multi-line 
    code blocks inside HTML table cells. Thoroughly cleans escaped quotes,
    translates raw list markers into HTML tokens, and runs a localized token-highlighter.
    """
    if not isinstance(cell_str, str):
        return str(cell_str)

    # 1. CRITICAL SANITIZATION: Strip out leaked JSON/Markdown escape backslashes
    cell_str = cell_str.replace('\\"', '"').replace("\\'", "'")
    cell_str = cell_str.replace('\xa0', ' ')
    cell_str = cell_str.replace("\\n", "\n").replace(r"\n", "\n")
    
    code_blocks = []
    def placeholder_code(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_PLACEHOLDER_{len(code_blocks)-1}__"
        
    # Isolate all code targets from the general text formatting passes
    cell_str = re.sub(r'```.*?```', placeholder_code, cell_str, flags=re.DOTALL)
    cell_str = re.sub(r'`[^`]+`', placeholder_code, cell_str)
    
    # 2. RESOLVE LIST MARKERS: Convert raw markdown bullets sitting outside code blocks
    lines = cell_str.split("\n")
    processed_lines = []
    for line in lines:
        stripped = line.lstrip()
        # Intercept markdown bullet configurations (- or *) and swap to crisp HTML dots
        if stripped.startswith("- ") or stripped.startswith("* "):
            leading_spaces = len(line) - len(stripped)
            line = "&nbsp;" * leading_spaces + "&bull; " + stripped[2:]
        processed_lines.append(line)
    cell_str = "\n".join(processed_lines)
    
    # 3. TRANSLATE TYPOGRAPHY: Handle bold, italic, and newline breaks safely
    cell_str = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', cell_str)
    cell_str = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', cell_str)
    cell_str = cell_str.replace("\n", "<br>")
    
    # 4. TOKEN HIGHLIGHTER ENGINE: Format code blocks to look like an IDE layout
    for idx, raw_code in enumerate(code_blocks):
        if raw_code.startswith("```"):
            match = re.match(r'```(python|ts|js|bash|json|sh)?\s*(.*?)\s*```', raw_code, flags=re.DOTALL)
            code_content = match.group(2) if match else raw_code
            
            # Wipe out any nested escape sequences locked inside the block
            code_content = code_content.replace('\\"', '"').replace("\\'", "'")
            code_content = code_content.replace("\\n", "\n").replace(r"\n", "\n")
            code_content = code_content.strip("\n")
            
            # Escape HTML brackets to guarantee DOM structure doesn't fragment
            code_content = code_content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            raw_lines = code_content.split("\n")
            formatted_lines = []
            
            for raw_line in raw_lines:
                leading_spaces = len(raw_line) - len(raw_line.lstrip(' '))
                line_text = raw_line.lstrip(' ')
                
                if not line_text:
                    formatted_lines.append('<div style="min-height: 1.25em;"></div>')
                    continue
                
                # Isolate line comments to prevent coloration spillover
                comment_part = ""
                if "#" in line_text:
                    parts = line_text.split('#')
                    accumulated = ""
                    for p_idx, part in enumerate(parts):
                        accumulated += part
                        if accumulated.count('"') % 2 == 0 and accumulated.count("'") % 2 == 0:
                            comment_part = "#" + "#".join(parts[p_idx+1:])
                            line_text = parts[0] if p_idx == 0 else "#".join(parts[:p_idx+1])
                            break
                
                # Protect string literals from matching regular language keywords
                string_store = []
                def protect_strings(m):
                    string_store.append(m.group(0))
                    return f"__STR_TOKEN_{len(string_store)-1}__"
                
                line_text = re.sub(r'(".*?"|\'.*?\')', protect_strings, line_text)
                
                # Highlight core framework keywords
                py_keywords = ['def', 'import', 'from', 'with', 'as', 'return', 'if', 'else', 'for', 'in', 'and', 'or', 'not', 'class', 'try', 'except', 'pass']
                for kw in py_keywords:
                    line_text = re.sub(rf'\b{kw}\b', f'<span style="color: #569cd6; font-weight: bold;">{kw}</span>', line_text)
                
                # Highlight global system types, constants, and custom framework structures
                py_constants = ['True', 'False', 'None', 'dict', 'list', 'str', 'int', 'len', 'agent', 'config', 'DB_URI']
                for cn in py_constants:
                    line_text = re.sub(rf'\b{cn}\b', f'<span style="color: #4ec9b0;">{cn}</span>', line_text)
                    
                # Highlight invoked method/function names (e.g., .invoke, .get_user_info)
                line_text = re.sub(r'\b(\w+)(?=\s*\()', r'<span style="color: #dcdcaa;">\1</span>', line_text)
                
                # Re-inject the pristine strings back with vibrant orange theme highlighting
                for s_idx, s_val in enumerate(string_store):
                    styled_str = f'<span style="color: #ce9178;">{s_val}</span>'
                    line_text = line_text.replace(f"__STR_TOKEN_{s_idx}__", styled_str)
                
                # Attach comments back to the end of the line element
                if comment_part:
                    line_text += f'<span style="color: #6a9955; font-style: italic;">{comment_part}</span>'
                
                indent_prefix = "&nbsp;" * leading_spaces
                formatted_lines.append(f'<div style="line-height: 1.5; font-family: monospace;">{indent_prefix}{line_text}</div>')
                
            final_code_html = "".join(formatted_lines)
            
            # Build wrapper container for clean spacing inside table parameters
            html_block = f"""
            <div style="margin: 6px 0; padding: 12px; background-color: #1e1e1e; 
                        color: #d4d4d4; border-radius: 6px; font-family: 'Courier New', Courier, monospace; 
                        font-size: 13px; border: 1px solid #333; overflow-x: auto; white-space: nowrap;">
                {final_code_html}
            </div>
            """
            cell_str = cell_str.replace(f"__CODE_BLOCK_PLACEHOLDER_{idx}__", html_block)
        else:
            # Inline snippet configurations (`code`)
            content = raw_code.strip("`").replace('\\"', '"').replace("\\'", "'")
            content = content.replace("\\n", " ").replace(r"\n", " ")
            html_inline = f'<code style="background-color: rgba(255,255,255,0.12); padding: 2px 5px; border-radius: 4px; font-family: monospace; color: #ff79c6; font-size: 13px;">{content}</code>'
            cell_str = cell_str.replace(f"__CODE_BLOCK_PLACEHOLDER_{idx}__", html_inline)
            
    return cell_str


def convert_md_table_to_html(md_lines: list) -> str:
    """
    Parses a group of raw markdown lines representing a table 
    and transforms them into a single beautifully styled HTML table block.
    """
    if len(md_lines) < 2:
        return "\n".join(md_lines)
        
    def parse_row(row_str):
        row_str = row_str.strip()
        if row_str.startswith("|"):
            row_str = row_str[1:]
        if row_str.endswith("|"):
            row_str = row_str[:-1]
        return [cell.strip() for cell in row_str.split("|")]
        
    headers = parse_row(md_lines[0])
    separator = parse_row(md_lines[1])
    
    # Verify if the second line is a legitimate markdown divider row (e.g., |---|)
    is_delimiter = all(all(c in "-: " for c in cell) and len(cell) > 0 for cell in separator)
    
    start_row_idx = 2 if is_delimiter else 1
    if not is_delimiter:
        headers = []
        start_row_idx = 0
        
    # Build complete responsive HTML table structure wrapper
    html = ['<div style="overflow-x: auto; margin: 20px 0;"><table style="width: 100%; border-collapse: collapse; font-size: 14px; border: 1px solid #444; color: #f0f0f0;">']
    
    # Append Table Header
    if headers:
        html.append('<thead style="background-color: rgba(255, 255, 255, 0.06); border-bottom: 2px solid #555;"><tr>')
        for h in headers:
            clean_header = re.sub(r'\*\*([^*]+)\*\*', r'\1', h)
            html.append(f'<th style="padding: 12px; text-align: left; font-weight: bold; border: 1px solid #444;">{clean_header}</th>')
        html.append('</tr></thead>')
        
    # Append Table Body Rows
    html.append('<tbody>')
    for i in range(start_row_idx, len(md_lines)):
        cells = parse_row(md_lines[i])
        if headers and len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
            
        html.append('<tr style="border-bottom: 1px solid #3a3a3a;">')
        for cell_data in cells:
            formatted_cell = format_cell_content(cell_data)
            html.append(f'<td style="padding: 12px; vertical-align: top; border: 1px solid #444; line-height: 1.5;">{formatted_cell}</td>')
        html.append('</tr>')
        
    html.append('</tbody></table></div>')
    return "".join(html)


def clean_markdown_stream(text: str) -> str:
    """
    Main entry point. Automatically un-smashes compressed streaming lines, heals multi-line 
    table rows, and translates code tables into clean native HTML containers.
    """
    if not isinstance(text, str):
        return text
        
    # 1. Un-smash row-concatenation errors where a block ends and a new row starts on the same line
    text = re.sub(r'```\s*\|\|\s*', r'``` |\n| ', text)
    
    # 2. Row Healing Loop: Collect multi-line table blocks split by syntax returns
    raw_lines = text.split("\n")
    healed_lines = []
    current_row = ""
    inside_cell_code = False
    
    for line in raw_lines:
        backtick_count = line.count("```")
        
        if inside_cell_code:
            current_row += "\n" + line
            if backtick_count % 2 != 0:
                inside_cell_code = False
            if not inside_cell_code:
                healed_lines.append(current_row)
                current_row = ""
        else:
            stripped = line.strip()
            is_table = "|" in stripped and (stripped.startswith("|") or stripped.endswith("|") or stripped.count("|") >= 2)
            
            if is_table and backtick_count % 2 != 0:
                inside_cell_code = True
                current_row = line
            else:
                healed_lines.append(line)
                
    if inside_cell_code and current_row:
        healed_lines.append(current_row)
        
    # 3. Process structural rows into unified tables
    processed_output = []
    table_lines = []
    in_table_block = False
    
    for line in healed_lines:
        stripped_line = line.strip()
        is_table_row = "|" in stripped_line and (stripped_line.startswith("|") or stripped_line.endswith("|") or stripped_line.count("|") >= 2 or in_table_block)
        
        if is_table_row:
            in_table_block = True
            table_lines.append(line)
        else:
            if in_table_block:
                processed_output.append(convert_md_table_to_html(table_lines))
                table_lines = []
                in_table_block = False
            processed_output.append(line)
            
    if in_table_block:
        processed_output.append(convert_md_table_to_html(table_lines))
        
    return "\n".join(processed_output)