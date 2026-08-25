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

# NOTE: this image intentionally never installs torch. The web service uses
# acoustic/spoof_detector_onnx.py (ONNX Runtime) instead of spoof_detector.py
# (PyTorch): `import torch` alone uses ~490MB RSS - nearly the entire 512MB
# free-tier RAM budget on its own - while `import onnxruntime` uses ~40MB,
# for numerically identical output from the same AASIST-L weights (verified
# to ~1e-7). That gap was causing OOM crash-restart loops under normal use.
# This needs acoustic/aasist_model/aasist_l.onnx to be committed to the repo
# (~600KB - generate it once with `python3 acoustic/export_onnx.py` in an
# environment that has torch, then `git add` the resulting .onnx file).
RUN pip install --no-cache-dir -r webdemo/requirements.txt

# sentence-transformers is intentionally omitted from the free-tier image.
# ScamClassifier falls back to its rules-only detector when it is unavailable.

WORKDIR /app/webdemo
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
