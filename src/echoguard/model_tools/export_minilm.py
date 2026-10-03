"""Export all-MiniLM-L6-v2 to FP32 ONNX and self-contained Android INT8 assets.

FP32 output: assets/semantic/minilm_model/minilm.onnx. Android output:
models/minilm.onnx and models/vocab.txt. Shape preprocessing precedes quantization.
Exemplar embeddings are exported separately. Run deliberately and validate
model/tokenizer compatibility: python tools/models/export_minilm.py.
"""

import tempfile
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
from echoguard.paths import REPO_ROOT, SEMANTIC_ASSETS_DIR

OUT_DIR = SEMANTIC_ASSETS_DIR / "minilm_model"
MAX_SEQ_LEN = 64  # scam-related sentences are short; keeps inference fast on-device

_REPO_ROOT = REPO_ROOT
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
