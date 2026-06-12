#!/bin/bash
set -e 

echo "🧱 Step 1: Chunking data and generating chunks.json..."
uv run src/chunk-pipeline/run_parser.py

echo "🔄 Step 2: Running chunks_diff.py to calculate deltas..."
uv run dB-maintenance/chunks_diff.py

echo "🧹 Step 3: Clearing old local staging storage..."
# Ensure the folder exists first, then clear it safely
mkdir -p data/qdrant_storage
rm -rf data/qdrant_storage/*

echo "--------------------------------------------------------"
echo "🎉 Local processing complete!"
echo "👉 1. Upload the NEW chunks_*.json from data/processed/ to Colab."
echo "👉 2. Make sure the removal_*.json is sitting in dB-maintenance/."
echo "--------------------------------------------------------"