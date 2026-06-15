# Edge case questions — manually defined, no chunks needed
# These test guardrails and query translation, not retrieval
EDGE_CASE_QUESTIONS = [
    # Off-topic guardrail (5)
    {"question": "What is the capital of France?",
     "expected_guardrail": "Cannot produce results for anything other than LangChain, LangGraph, or LangSmith related topics."},
    {"question": "Who won the FIFA World Cup in 2022?",
     "expected_guardrail": "Cannot produce results for anything other than LangChain, LangGraph, or LangSmith related topics."},
    {"question": "Write me a poem about the ocean.",
     "expected_guardrail": "Cannot produce results for anything other than LangChain, LangGraph, or LangSmith related topics."},
    {"question": "What is the best Python web framework?",
     "expected_guardrail": "Cannot produce results for anything other than LangChain, LangGraph, or LangSmith related topics."},
    {"question": "Explain quantum computing.",
     "expected_guardrail": "Cannot produce results for anything other than LangChain, LangGraph, or LangSmith related topics."},

    # API reference guardrail (5)
    {"question": "What are the parameters of BaseMessage.__init__?",
     "expected_guardrail": "The current RAG pipeline only indexes the official conceptual documentation"},
    {"question": "What does ChatOpenAI._generate return?",
     "expected_guardrail": "The current RAG pipeline only indexes the official conceptual documentation"},
    {"question": "Give me the exact method signature of RunnableSequence.invoke.",
     "expected_guardrail": "The current RAG pipeline only indexes the official conceptual documentation"},
    {"question": "What are all the attributes of AgentState dataclass?",
     "expected_guardrail": "The current RAG pipeline only indexes the official conceptual documentation"},
    {"question": "List all keyword arguments accepted by StateGraph.compile().",
     "expected_guardrail": "The current RAG pipeline only indexes the official conceptual documentation"},

    # Vague queries that need translation (5) — no guardrail, needs retrieval
    {"question": "how do i make my agent remember things",
     "expected_guardrail": None},
    {"question": "my langgraph thing keeps breaking",
     "expected_guardrail": None},
    {"question": "how do i see what my agent is doing",
     "expected_guardrail": None},
    {"question": "can langchain work with other llms",
     "expected_guardrail": None},
    {"question": "how do i save progress in my graph",
     "expected_guardrail": None},

    # Multi-turn follow-ups (5) — need to be run in sequence
    {"question": "How do I create a ReAct agent in LangChain?",
     "expected_guardrail": None, "is_conversation_start": True},
    {"question": "How do I add memory to it?",
     "expected_guardrail": None, "follows": "How do I create a ReAct agent in LangChain?"},
    {"question": "What checkpointer should I use for production?",
     "expected_guardrail": None, "follows": "How do I add memory to it?"},
    {"question": "How do I add memory and stream responses in LangGraph?",
     "expected_guardrail": None, "is_conversation_start": True},
    {"question": "What about for a multi-agent setup?",
     "expected_guardrail": None, "follows": "How do I add memory and stream responses in LangGraph?"},
]



# ── Prompt ─────────────────────────────────────────────────────────────────────
GENERATION_PROMPT = """
You are creating an evaluation dataset for a RAG system built over LangChain,
LangGraph, and LangSmith documentation.

You are given {n} documentation chunks below. Generate exactly {n_questions} questions:
    - {n_single} single-hop questions: answerable using ONE of the chunks
    - {n_multi}  multi-hop questions:  require combining info from TWO or more chunks

Rules:
1. Every question must be clearly answerable from the provided chunks.
   Do not ask about things not covered in the text.
2. Questions must sound like real developer questions — natural, practical.
3. For each question, write a reference_answer drawn ONLY from the chunk text.
   Do not use outside knowledge. Quote or closely paraphrase the chunks.
4. For each question, list the source_chunk_indices (0-indexed) needed to answer it.
   Multi-hop questions must reference at least 2 different chunk indices.
5. Label each question: single_hop or multi_hop.

Return ONLY valid JSON — no markdown, no explanation, no backticks.
Format:
[
  {{
    "question": "...",
    "question_type": "single_hop",
    "reference_answer": "...",
    "source_chunk_indices": [0]
  }},
  {{
    "question": "...",
    "question_type": "multi_hop",
    "reference_answer": "...",
    "source_chunk_indices": [0, 2]
  }}
]

--- CHUNKS ---
{chunks_text}
""".strip()