# Deploy target: Hugging Face Spaces (Docker SDK) or any Docker host (Render, Fly.io, etc.)
# Build context = repo root (this file references ../acoustic, ../semantic, ../fusion).

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only what's needed to run the pipeline + demo -- not the Android app, datasets, etc.
COPY acoustic ./acoustic
COPY semantic ./semantic
COPY fusion ./fusion
COPY webdemo ./webdemo

# CPU-only torch from the official PyTorch index (the default PyPI wheel pulls in a
# multi-GB CUDA dependency stack that's wasted on a CPU-only web dyno).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r webdemo/requirements.txt

# sentence-transformers is intentionally omitted from the free-tier image.
# ScamClassifier falls back to its rules-only detector when it is unavailable.

WORKDIR /app/webdemo
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
