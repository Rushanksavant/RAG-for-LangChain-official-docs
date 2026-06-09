import streamlit as st

def inject_layout_styling():
    """
    Injects global CSS stylesheets to configure center alignments, 
    custom widths, and polish the container presentation profiles.
    """
    st.markdown(
        """
        <style>
        /* 1. Configuration for standard st.text_input */
        div[data-testid="stTextInput"] {
            max-width: 550px;             
            margin: 10px auto !important; 
        }

        /* 2. Configuration for bottom-anchored st.chat_input */
        div[data-testid="stChatInput"] {
            max-width: 650px !important;      
            margin-left: auto !important;     
            margin-right: auto !important;    
        }
        
        /* 3. Center layout content and optimize width for maximum readability */
        .main .block-container {
            max-width: 880px;
            padding-top: 2rem;
            padding-bottom: 7rem;
            margin: 0 auto;
        }
        
        /* 4. Polish chat status logs spacing */
        .stStatusContainer {
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def get_live_tracker_script(max_chars: int, max_tokens: int) -> str:
    """
    Hard-enforces character limits on the browser textarea and dynamically
    injects an error warning directly below the chat input widget.
    """
    script = f"""
    <script>
    function enforceInputBoundaryRules() {{
        const doc = window.parent.document;
        const textarea = doc.querySelector('textarea[data-testid="stChatInputTextArea"]');
        const inputContainer = doc.querySelector('div[data-testid="stChatInput"]');
        
        if (!textarea || !inputContainer) return;

        // 1. Force browser-level character lock
        if (!textarea.hasAttribute('maxlength')) {{
            textarea.setAttribute('maxlength', '{max_chars}');
        }}
        
        // 2. Inject or fetch the dynamic error warning element
        let alertLabel = doc.getElementById('custom-max-char-alert');
        if (!alertLabel) {{
            alertLabel = doc.createElement('div');
            alertLabel.id = 'custom-max-char-alert';
            alertLabel.innerText = 'Input query is too large, please shorten your query.';
            alertLabel.style.color = '#FF4B4B';
            alertLabel.style.fontSize = '13px';
            alertLabel.style.marginTop = '8px';
            alertLabel.style.textAlign = 'center';
            alertLabel.style.width = '100%';
            alertLabel.style.fontFamily = 'sans-serif';
            alertLabel.style.display = 'none';
            inputContainer.appendChild(alertLabel);
        }}

        // 3. Evaluate content length on every keystroke
        if (textarea.value.length >= {max_chars}) {{
            alertLabel.style.display = 'block';
        }} else {{
            alertLabel.style.display = 'none';
        }}
    }}
    
    // Continually bind to catch Streamlit state refreshes/re-renders
    setInterval(enforceInputBoundaryRules, 300);
    </script>
    """
    return script