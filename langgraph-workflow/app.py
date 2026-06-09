# uv run --with streamlit streamlit run langgraph-workflow/app.py
import streamlit as st
import uuid
import datetime
import streamlit.components.v1 as components

# Import modular helper dependencies
from streamlit_contents.text_processor import clean_markdown_stream
from streamlit_contents.agent_runner import execute_agent_stream
from streamlit_contents.ui_components import inject_layout_styling, get_live_tracker_script

# ── 1. INITIALIZE SYSTEM STATE & RENDERS ─────────────────────────────────────
st.set_page_config(
    page_title="LangChain RAG Assistant",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global layout adjustments
inject_layout_styling()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "processing" not in st.session_state:
    st.session_state.processing = False

MAX_ALLOWED_TOKENS = 800
MAX_ALLOWED_CHARS = 2000


# ── 2. SIDEBAR METRICS & OPERATION CONTROLS ─────────────────────────────────
st.sidebar.title("🛠️ Assistant Controls")
st.sidebar.markdown("Manage your interactive documentation research terminal threads.")
st.sidebar.markdown(f"**Session Thread ID:**\n`{st.session_state.session_id}`")
st.sidebar.markdown("---")

if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.toast("Conversation thread cleared!")
    st.rerun()

if st.session_state.messages:
    export_lines = [
        "# LangChain RAG Assistant — Conversation Export",
        f"**Date:** {datetime.date.today().strftime('%Y-%m-%d')}\n---"
    ]
    for msg in st.session_state.messages:
        speaker = "**You:**" if msg["role"] == "user" else "**Assistant:**"
        export_lines.append(f"\n{speaker}\n{msg['content']}\n\n---")
        
    st.sidebar.download_button(
        label="📥 Export Conversation (.md)",
        data="\n".join(export_lines),
        file_name=f"rag_conversation_{st.session_state.session_id[:8]}.md",
        mime="text/markdown",
        use_container_width=True
    )


# ── 3. MAIN HEADER ──────────────────────────────────────────────────────────
st.title("🦜 LangChain RAG Assistant")
st.markdown(
    "Deep-context search terminal for *LangChain*, *LangGraph*, and *LangSmith* frameworks. "
    "Features live multi-query map-reduce orchestration traces."
)
st.markdown("---")


# ── 4. CHAT HISTORY DISPLAY ─────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(clean_markdown_stream(msg["content"]), unsafe_allow_html=True)
        
        status_messages = msg.get("meta", {}).get("status_mssg", [])
        if status_messages:
            trace_line = "  •  ".join(status_messages)
            st.caption(f"🏁 *System Trace:* {trace_line}")


# ── 5. ACTIVE RUNNING SYSTEM MONITOR ────────────────────────────────────────
if st.session_state.processing and st.session_state.messages:
    last_user_query = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.status("Analyzing graph pipelines...", expanded=True) as status_box:
            try:
                # Fire the modular agent stream runner
                result = execute_agent_stream(last_user_query, st.session_state.session_id, status_box)
                status_box.update(label="Analysis complete!", state="complete", expanded=False)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.get("final_response", "No response returned."),
                    "meta": {
                        "status_mssg": result.get("status_mssg", [])
                    }
                })
            except Exception as err:
                status_box.update(label="Execution Encountered Fault", state="error", expanded=True)
                st.error(f"Processing Exception: {str(err)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ **System Execution Failure:** An internal script error occurred.",
                    "meta": {"status_mssg": ["Internal Execution Exception Checked"]}
                })
            finally:
                st.session_state.processing = False
                st.rerun()


# ── 6. CHAT INPUT BAR & LIVE METRIC HOOKS ───────────────────────────────────
user_prompt = st.chat_input(
    placeholder="Ask anything about LangChain, LangGraph state channels, or custom checkpointers...",
    disabled=st.session_state.processing
)

if user_prompt:
    # Strict server-side length validation
    if len(user_prompt) > MAX_ALLOWED_CHARS:
        st.error(
            f"⚠️ **Query Rejected:** Your input contains {len(user_prompt)} characters, "
            f"exceeding the maximum allowed safety limit of {MAX_ALLOWED_CHARS} characters. "
            f"Please truncate your query and try again."
        )
    else:
        st.session_state.processing = True
        st.session_state.messages.append({
            "role": "user",
            "content": user_prompt,
            "meta": {}
        })
        st.rerun()

# Inject JavaScript live metrics listener in the background
tracker_js = get_live_tracker_script(MAX_ALLOWED_CHARS, MAX_ALLOWED_TOKENS)
components.html(tracker_js, height=0)