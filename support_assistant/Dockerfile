FROM python:3.14.2

WORKDIR /app
 
# Install system dependencies needed by chromadb / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
 
# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy application code and the document corpus
COPY . .
 
# Default to mock mode -- this is the required, graded baseline.
# Override with `-e MOCK_LLM=0` only for the optional, ungraded real-LLM extension.
ENV MOCK_LLM=1
 
EXPOSE 7860
 
# Run ingestion once at container startup (populates/refreshes the
# persistent ChromaDB collection), then start the API server.
CMD sh -c "python ingest.py && uvicorn main:app --host 0.0.0.0 --port 7860"