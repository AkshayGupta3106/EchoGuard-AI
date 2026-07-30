"""
export_minilm_onnx.py
------------------
Owner: Person B (semantic stream)

Run this ONCE on a dev machine with internet access. Exports MiniLM
(all-MiniLM-L6-v2) to ONNX and saves its WordPiece vocab, for on-device
embedding of the LIVE transcript on Android (the exemplar list is handled
separately - see export_exemplar_embeddings.py - since it never changes).

Output:
  minilm.onnx      -> bundle as an Android asset
  vocab.txt        -> bundle as an Android asset (WordPiece vocab, one
                       token per line, same format BERT/MiniLM tokenizers use)

NOT verified in the dev sandbox this was written in - that environment has
no access to huggingface.co, so the actual export couldn't be run there.
Run this yourself and confirm the two output files load correctly before
trusting them.
"""

from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = Path(__file__).parent / "minilm_model"
MAX_SEQ_LEN = 64  # scam-related sentences are short; keeps inference fast on-device


def main():
    OUT_DIR.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # Save the WordPiece vocab for the Kotlin-side tokenizer to load.
    vocab_path = OUT_DIR / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        # sort by token id so line number == token id, which is what the
        # Kotlin tokenizer assumes when loading this file
        for token, _ in sorted(tokenizer.vocab.items(), key=lambda kv: kv[1]):
            f.write(token + "\n")

    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, MAX_SEQ_LEN))
    dummy_attention_mask = torch.ones((1, MAX_SEQ_LEN), dtype=torch.long)

    onnx_path = OUT_DIR / "minilm.onnx"
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        opset_version=18,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
    )

    print(f"Wrote {onnx_path} and {vocab_path}")
    print("IMPORTANT: on the Kotlin side, mean-pool last_hidden_state over the "
          "attention_mask, then L2-normalize - that's how sentence-transformers "
          "produces its sentence embedding from this model's raw output. "
          "Don't just take the [CLS] token / index 0.")


if __name__ == "__main__":
    main()
