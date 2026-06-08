"""
Manual test script for the LangGraph agent.
"""

import asyncio
from agent import run_agent

# ── Test cases ─────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "label": "Basic question (no retrieval expected)",
        "input": "What is LangChain?",
    },
    {
        "label": "Retrieval question — single topic",
        "input": "How do I add memory to a LangGraph agent?",
    },
    {
        "label": "Retrieval question — multi topic",
        "input": "How do I add memory and stream responses in LangGraph?",
    },
    {
        "label": "Guardrail — off topic",
        "input": "What is the capital of France?",
    },
    {
        "label": "Guardrail — API reference",
        "input": "What are the parameters of BaseMessage.__init__?",
    },
    {
        "label": "Follow-up (tests history — run after case 2)",
        "input": "What about for a multi-agent setup?",
    },
]

# Use a fixed session ID so all test cases share the same memory thread
SESSION_ID = "test-session-001"


async def run_tests():
    print("\n" + "=" * 60)
    print("LANGGRAPH AGENT TEST")
    print("=" * 60)

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {case['label']}")
        print(f"Input: {case['input']}")
        print("-" * 40)

        result = await run_agent(case["input"], SESSION_ID)

        # Query plan summary
        plan = result["query_plan"]
        if plan:
            print(f"Guardrail:        {plan.guardrail or 'None'}")
            print(f"Needs retrieval:  {plan.needs_retrieval}")
            print(f"Translated query: {plan.translated_query}")
            print(f"Sub-queries:      {plan.sub_queries}")

        # Status messages
        print(f"Status:           {' → '.join(result['status_mssg'])}")

        # Answer
        print(f"\nAnswer:\n{result['response']}")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_tests())