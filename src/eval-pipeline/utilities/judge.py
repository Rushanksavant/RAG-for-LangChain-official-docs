"""
judge.py — LLM judge and DeepEval metric definitions.

DeepEval needs a custom LLM wrapper to use any model not natively supported.
We use DeepSeek R1 via Groq as the judge for all 6 metrics.

Imported by run_eval.py — nothing here needs to be changed unless
you want to swap the judge model or adjust metric thresholds/criteria.
"""

import logging
from langchain.chat_models import init_chat_model
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    GEval,
)
from deepeval.test_case import LLMTestCaseParams
from dotenv import load_dotenv
load_dotenv()
import os
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

JUDGE_MODEL = "deepseek/deepseek-r1"


# ── LLM Judge wrapper ──────────────────────────────────────────────────────────
# DeepEval requires 4 methods: load_model, get_model_name, generate, a_generate.
# We wrap LangChain's init_chat_model so we can point at any provider easily.

class LangChainJudge(DeepEvalBaseLLM):

    def __init__(self):
        self.model = init_chat_model(model = JUDGE_MODEL,model_provider = "openrouter", 
                                    api_key = OPENROUTER_API_KEY, temperature = 0)

    def load_model(self):
        return self.model

    def get_model_name(self) -> str:
        return JUDGE_MODEL

    def generate(self, prompt: str) -> str:
        # Synchronous call — DeepEval uses this internally
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        # Async call — DeepEval uses this for async metric evaluation
        return (await self.model.ainvoke(prompt)).content


# Instantiate once — reused across all metric calls in the eval run
judge = LangChainJudge()


# ── Metric definitions ─────────────────────────────────────────────────────────
# Each metric is instantiated once here and imported by run_eval.py.
# Thresholds: precision/recall at 0.5 (lenient), faithfulness/relevancy at 0.7 (strict).

metrics = {

    # Layer 1a — did the guardrail fire correctly?
    # Only evaluated for edge_case questions.
    "guardrail": GEval(
        name             = "GuardrailAccuracy",
        model            = judge,
        evaluation_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        criteria          = (
            "Score 1.0 if the guardrail decision is correct:\n"
            "- Off-topic questions (not about LangChain/LangGraph/LangSmith) SHOULD trigger the guardrail.\n"
            "- API reference questions (method signatures, parameters, return types) SHOULD trigger.\n"
            "- Valid conceptual questions about LangChain/LangGraph/LangSmith should NOT trigger.\n"
            "Score 0.0 if the decision is wrong."
        ),
    ),

    # Layer 1b — did the planner translate the query well?
    # Only evaluated for single_hop and multi_hop questions.
    "translation": GEval(
        name             = "QueryTranslationQuality",
        model            = judge,
        evaluation_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        criteria          = (
            "Score 0.0 to 1.0 based on:\n"
            "- Does the translated query preserve the user's original intent?\n"
            "- Is the specific framework (LangChain / LangGraph / LangSmith) named?\n"
            "- Are sub-queries self-contained and independently meaningful?\n"
            "1.0 = perfect translation, 0.0 = intent completely lost."
        ),
    ),

    # Layer 2a — of the retrieved chunks, how many were actually relevant?
    "precision": ContextualPrecisionMetric(
        model        = judge,
        threshold    = 0.5,
        verbose_mode = False,
    ),

    # Layer 2b — of the chunks that should have been retrieved, how many were?
    "recall": ContextualRecallMetric(
        model        = judge,
        threshold    = 0.5,
        verbose_mode = False,
    ),

    # Layer 3a — are the claims in the answer supported by the retrieved chunks?
    "faithfulness": FaithfulnessMetric(
        model        = judge,
        threshold    = 0.7,
        verbose_mode = False,
    ),

    # Layer 3b — does the answer actually address what was asked?
    "relevancy": AnswerRelevancyMetric(
        model        = judge,
        threshold    = 0.7,
        verbose_mode = False,
    ),
}


# ── Safe metric call ───────────────────────────────────────────────────────────

def safe_measure(metric, test_case) -> float | None:
    """
    Calls metric.measure(test_case) and returns the score.
    Returns None on any failure (provider 502, timeout, parse error)
    so a single bad API call never crashes the whole eval run.
    """
    try:
        metric.measure(test_case)
        return metric.score
    except Exception as e:
        logging.warning(f"{metric.__class__.__name__} failed: {e}")
        return None