"""Export Python scam/benign exemplar embeddings with all-MiniLM-L6-v2.

Writes assets/semantic/exemplar_embeddings.json. Android copying is separate;
review compatibility before replacing packaged embeddings.
Run deliberately: python tools/models/export_exemplars.py.
"""

import json

from echoguard.paths import SEMANTIC_ASSETS_DIR
from echoguard.semantic.scam_classifier import SCAM_EXEMPLARS, BENIGN_EXEMPLARS


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

    out_path = SEMANTIC_ASSETS_DIR / "exemplar_embeddings.json"
    with open(out_path, "w") as f:
        json.dump(out, f)

    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB, "
          f"{len(SCAM_EXEMPLARS)} scam + {len(BENIGN_EXEMPLARS)} benign exemplars)")


if __name__ == "__main__":
    main()
