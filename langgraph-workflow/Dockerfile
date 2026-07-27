FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Install native system utilities required for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Explicitly install all application dependencies inline
# (Modify this list if your agent uses other third-party utilities)
RUN pip install --no-cache-dir \
    streamlit \
    langchain \
    langchain-core \
    langchain-groq \
    langchain-google-genai \
    langchain-mcp-adapters \
    langgraph \
    pydantic-settings \
    pydantic 

# Copy the rest of the application files
COPY . .

EXPOSE 8501

# Container health monitoring configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Execute app.py directly from the workspace root context
CMD ["streamlit", "run", "app.py"]

## Local build:

# 1. Step into the workflow directory
# cd langgraph-workflow

# 2. Build the Docker image using the local directory context
# docker build -t langgraph-assistant:latest .

# 3. Run and Test the Container Natively
# docker run -it -p 8501:8501 --env-file .env --name langgraph-test langgraph-assistant:latest

# Streamlit app at: http://localhost:8501