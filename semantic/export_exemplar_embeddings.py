"""
export_exemplar_embeddings.py
------------------
Owner: Person B (semantic stream)

Run this ONCE on a dev machine with internet access (needs to download
all-MiniLM-L6-v2 from Hugging Face). It embeds the fixed SCAM_EXEMPLARS and
BENIGN_EXEMPLARS lists from scam_classifier.py and dumps them to JSON.

Why this exists: the exemplar list never changes at runtime, so there's no
reason to re-embed it on every app launch, and no reason to ship a
tokenizer+model just to embed 14 fixed strings. Android only needs to embed
the LIVE transcript text - it loads this JSON once and compares against it.

Output: exemplar_embeddings.json -> bundle as an Android asset alongside
the MiniLM ONNX export.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scam_classifier import SCAM_EXEMPLARS, BENIGN_EXEMPLARS  # noqa: E402


def main():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    scam_emb = model.encode(SCAM_EXEMPLARS, normalize_embeddings=True).tolist()
    benign_emb = model.encode(BENIGN_EXEMPLARS, normalize_embeddings=True).tolist()

    out = {
        "model": "all-MiniLM-L6-v2",
        "embedding_dim": len(scam_emb[0]),
        "scam_exemplars": [
            {"text": t, "embedding": e} for t, e in zip(SCAM_EXEMPLARS, scam_emb)
        ],
        "benign_exemplars": [
            {"text": t, "embedding": e} for t, e in zip(BENIGN_EXEMPLARS, benign_emb)
        ],
    }

    out_path = Path(__file__).parent / "exemplar_embeddings.json"
    with open(out_path, "w") as f:
        json.dump(out, f)

    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB, "
          f"{len(SCAM_EXEMPLARS)} scam + {len(BENIGN_EXEMPLARS)} benign exemplars)")


if __name__ == "__main__":
    main()
