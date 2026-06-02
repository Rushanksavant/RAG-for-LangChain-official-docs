import os
import pandas as pd
from datasets import Dataset
from qdrant_client import QdrantClient
from settings import Settings

# Ragas & Google imports
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas import evaluate
from ragas.metrics import context_precision, context_recall

from google import genai

# 1. Environment & API Keys (Get free key from Google AI Studio)
GEMINI_API_KEY = Settings.GEMINI_API_KEY
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
