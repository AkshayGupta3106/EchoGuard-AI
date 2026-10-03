# Lightweight FastAPI web image; build context is the repository root.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the source package without pulling optional model/export dependencies.
COPY pyproject.toml README.md ./
COPY src ./src

# Runtime assets and browser frontend; local recordings/builds are excluded.
COPY assets ./assets
COPY webdemo ./webdemo

# ONNX acoustic inference and rules-only text scoring, without model-export dependencies.
RUN pip install --no-cache-dir -r webdemo/requirements.txt
RUN pip install --no-cache-dir --no-deps -e .

WORKDIR /app/webdemo
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
