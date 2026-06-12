#!/bin/bash
set -e

# Target the compose file directly from the project root using the -f flag
echo "🛑 Stopping local engine if running..."
docker compose -f docker/docker-compose.yml down

# Verify the zip file is actually in the data/ directory before proceeding
if [ ! -f "data/qdrant_storage.zip" ]; then
    echo "❌ Error: qdrant_storage.zip not found in data/ folder!"
    echo "Please move the downloaded zip to data/ and run again."
    exit 1
fi

echo "📦 Extracting new patch database from data/ folder..."
# Targets data/qdrant_storage.zip and extracts it straight into your volume
unzip -o data/qdrant_storage.zip -d data/qdrant_storage/

echo "⚡ Booting local engine with patch data..."
docker compose -f docker/docker-compose.yml up -d

echo "🚀 Executing Cloud Migration Pipeline..."
uv run dB-maintenance/upsert_dB.py

echo "🛑 Cleaning up staging environment..."
docker compose -f docker/docker-compose.yml down

# Clean up the zip file automatically so you don't accidentally reuse it next time
rm -f data/qdrant_storage.zip

echo "✅ Cloud successfully patched and synchronized!"