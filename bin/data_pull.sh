#!/bin/bash
# filepath: bin/data_pull.sh

echo "Initializing data directories..."
mkdir -p data/raw/conceptual_docs
mkdir -p data/raw/api_reference

# Clone or update conceptual docs
if [ ! -d "data/raw/conceptual_docs/.git" ]; then ## first-time 
    echo "Cloning conceptual docs..."
    git clone --depth 1 https://github.com/langchain-ai/docs.git data/raw/conceptual_docs

else ## if git already present in folder
    echo "Updating conceptual docs..."
    cd data/raw/conceptual_docs && git pull && cd ../../../
fi

# Clone or update archived API docs (used as our static API reference)
if [ ! -d "data/raw/api_reference/.git" ]; then
    echo "Cloning API reference archives..."
    git clone --depth 1 https://github.com/langchain-ai/langchain-api-docs-html.git data/raw/api_reference
else
    echo "Updating API reference archives..."
    cd data/raw/api_reference && git pull && cd ../../../
fi

echo "Data ingestion complete!"
