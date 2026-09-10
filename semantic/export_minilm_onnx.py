"""
export_minilm_onnx.py
------------------
Owner: Person B (semantic stream)

Run this ONCE on a dev machine with internet access. Exports MiniLM
(all-MiniLM-L6-v2) to ONNX, quantizes it to INT8, and writes both the
model and its WordPiece vocab DIRECTLY into the Android app's assets
folder under their final names - same pattern as download_kroko.py /
download_indicconformer.py, so there's no manual copy/rename step.
(The exemplar list is handled separately - see
export_exemplar_embeddings.py - since it never changes.)

The quantized model is a single self-contained file (no external
.onnx.data companion needed) - ScamClassifier.kt expects exactly this.

Before quantizing, the fp32 model is run through
onnxruntime.quantization.shape_inference.quant_pre_process. MiniLM has
a single output so this happened not to be strictly required the way it
was for AASIST-L's two differently-shaped outputs, but it's kept here
for robustness (and because torch's newer dynamo-based exporter can
still produce shape metadata that trips up quantize_dynamic's default
raw onnx.shape_inference pass on some torch/onnx version combinations).

Output:
  semantic/minilm_model/minilm.onnx       -> fp32, intermediate only,
                                              not bundled
  app/app/src/main/assets/models/minilm.onnx  -> the actual Android
                                              asset, written directly, int8
  app/app/src/main/assets/models/vocab.txt    -> WordPiece vocab,
                                              written directly

NOT verified in the dev sandbox this was written in - that environment has
no access to huggingface.co, so the actual export couldn't be run there.
Run this yourself and confirm the output files load correctly before
trusting them.
"""

import tempfile
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = Path(__file__).parent / "minilm_model"
MAX_SEQ_LEN = 64  # scam-related sentences are short; keeps inference fast on-device

# Same layout download_kroko.py / download_indicconformer.py assume:
# EchoGuard-AI/
# ├── semantic/export_minilm_onnx.py   (this file)
# └── app/app/src/main/assets/models/
_REPO_ROOT = Path(__file__).parent.parent
_ASSETS_MODELS_DIR = _REPO_ROOT / "app" / "app" / "src" / "main" / "assets" / "models"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    _ASSETS_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # Write the WordPiece vocab directly to the Android asset path -
    # sort by token id so line number == token id, which is what the
    # Kotlin tokenizer assumes when loading this file.
    vocab_path = _ASSETS_MODELS_DIR / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, _ in sorted(tokenizer.vocab.items(), key=lambda kv: kv[1]):
            f.write(token + "\n")
    print(f"Wrote vocab directly to Android asset: {vocab_path}")

    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, MAX_SEQ_LEN))
    dummy_attention_mask = torch.ones((1, MAX_SEQ_LEN), dtype=torch.long)

    fp32_path = OUT_DIR / "minilm.onnx"
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        opset_version=18,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
    )
    print(f"Exported fp32 to {fp32_path} ({fp32_path.stat().st_size / (1024 * 1024):.1f} MB) - intermediate only")

    asset_path = _ASSETS_MODELS_DIR / "minilm.onnx"

    with tempfile.TemporaryDirectory() as tmp_dir:
        preprocessed_path = Path(tmp_dir) / "minilm.preprocessed.onnx"
        print("Running shape-inference preprocessing...")
        quant_pre_process(
            input_model=str(fp32_path),
            output_model_path=str(preprocessed_path),
        )

        quantize_dynamic(
            model_input=str(preprocessed_path),
            model_output=str(asset_path),
            weight_type=QuantType.QInt8,
        )

    print(f"Exported int8 directly to Android asset: {asset_path} "
          f"({asset_path.stat().st_size / (1024 * 1024):.1f} MB)")

    print()
    print("IMPORTANT: on the Kotlin side, mean-pool last_hidden_state over the "
          "attention_mask, then L2-normalize - that's how sentence-transformers "
          "produces its sentence embedding from this model's raw output. "
          "Don't just take the [CLS] token / index 0.")
    print()
    print("Done. No manual copy/rename needed - the app will pick these up on next build.")


if __name__ == "__main__":
    main()