"""
export_onnx.py
------------------
Owner: Person A (acoustic stream)

One-time script: exports the pretrained AASIST-L PyTorch checkpoint to
ONNX, quantizes it to INT8, and writes the result DIRECTLY into the
Android app's assets folder under its final name - same pattern as
download_kroko.py / download_indicconformer.py, so there's no manual
copy/rename step.

The quantized model is a single self-contained file (no external
.onnx.data companion needed) - SpoofDetector.kt expects exactly this.

Before quantizing, the fp32 model is run through
onnxruntime.quantization.shape_inference.quant_pre_process. This is
NOT optional for this model: AASIST-L has two differently-shaped
outputs ("embedding" [1,160] and "logits" [1,2]), and quantize_dynamic's
default raw onnx.shape_inference pass throws
"Inferred shape and existing shape differ in dimension 0: (160) vs (2)"
without this preprocessing step - it's a real shape-inference limitation
for multi-output models, not a setup mistake.

Run once on a dev machine: python3 export_onnx.py
Output:
  acoustic/aasist_model/aasist_l.onnx  <- fp32, intermediate only, kept
                                           here for debugging, NOT bundled
  app/app/src/main/assets/models/aasist_l.onnx  <- the actual Android
                                           asset, written directly, int8
"""

import json
import sys
import tempfile
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

_THIS_DIR = Path(__file__).parent
_MODEL_DIR = _THIS_DIR / "aasist_model"
sys.path.insert(0, str(_MODEL_DIR))

from models.AASIST import Model  # noqa: E402

WINDOW_SAMPLES = 64600

# Same layout download_kroko.py / download_indicconformer.py assume:
# EchoGuard-AI/
# ├── acoustic/export_onnx.py   (this file)
# └── app/app/src/main/assets/models/
_REPO_ROOT = _THIS_DIR.parent
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
        dynamo=False,  # legacy exporter - the newer dynamo-based exporter
                       # (torch 2.x default) writes conflicting shape metadata
                       # for this model's two differently-shaped outputs
                       # ("embedding" [1,160] vs "logits" [1,2]), which then
                       # crashes quantize_dynamic's internal shape-inference
                       # reload even after quant_pre_process. Since this
                       # model is fully static (dynamic_axes=None) anyway,
                       # the legacy exporter is a clean fit and avoids the
                       # bug entirely rather than working around it.
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