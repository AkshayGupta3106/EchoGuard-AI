"""Export AASIST-L to self-contained web FP32 and Android INT8 ONNX assets.

Writes assets/acoustic/aasist_l.onnx and Android models/aasist_l.onnx.
quant_pre_process is required for the embedding/logit output shape metadata.
Run deliberately: python tools/models/export_aasist.py (replaces model files).
"""

import json
import sys
import tempfile
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

from echoguard.paths import REPO_ROOT, ACOUSTIC_MODEL_DIR

_MODEL_DIR = ACOUSTIC_MODEL_DIR
sys.path.insert(0, str(_MODEL_DIR))

from models.AASIST import Model  # noqa: E402

WINDOW_SAMPLES = 64600

# Android model export destination.
_REPO_ROOT = REPO_ROOT
_ASSETS_MODELS_DIR = _REPO_ROOT / "app" / "app" / "src" / "main" / "assets" / "models"


def main():
    with open(_MODEL_DIR / "AASIST-L.conf") as f:
        config = json.load(f)

    model = Model(config["model_config"])
    state = torch.load(_MODEL_DIR / "models" / "AASIST-L.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(1, WINDOW_SAMPLES)
    fp32_path = _MODEL_DIR / "aasist_l.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(fp32_path),
        input_names=["waveform"],
        output_names=["embedding", "logits"],
        opset_version=18,
        dynamic_axes=None,  # fixed input size - matches the model's fixed window
        dynamo=False,  # Static tracing export for the two-output model contract.
    )
    print(f"Exported fp32 to {fp32_path} ({fp32_path.stat().st_size / 1024:.0f} KB) - intermediate only")

    _ASSETS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    asset_path = _ASSETS_MODELS_DIR / "aasist_l.onnx"

    with tempfile.TemporaryDirectory() as tmp_dir:
        preprocessed_path = Path(tmp_dir) / "aasist_l.preprocessed.onnx"
        print("Running shape-inference preprocessing (required for this "
              "model's two differently-shaped outputs)...")
        quant_pre_process(
            input_model=str(fp32_path),
            output_model_path=str(preprocessed_path),
            skip_symbolic_shape=True,  # AASIST-L has no dynamic axes, this is fine and faster
        )

        quantize_dynamic(
            model_input=str(preprocessed_path),
            model_output=str(asset_path),
            weight_type=QuantType.QInt8,
        )

    print(f"Exported int8 directly to Android asset: {asset_path} "
          f"({asset_path.stat().st_size / 1024:.0f} KB)")
    print()
    print("Done. No manual copy/rename needed - the app will pick this up on next build.")


if __name__ == "__main__":
    main()
