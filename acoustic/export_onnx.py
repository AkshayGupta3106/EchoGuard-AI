"""
export_onnx.py
------------------
Owner: Person A (acoustic stream)

One-time script: exports the pretrained AASIST-L PyTorch checkpoint to
ONNX, which is what you'll actually load on Android via ONNX Runtime
Mobile (not raw PyTorch - that's not practical to ship in an APK).

Run once on a dev machine: python3 export_onnx.py
Output: aasist_model/aasist_l.onnx  <- bundle this in the Android app assets.
"""

import json
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).parent
_MODEL_DIR = _THIS_DIR / "aasist_model"
sys.path.insert(0, str(_MODEL_DIR))

from models.AASIST import Model  # noqa: E402

WINDOW_SAMPLES = 64600


def main():
    with open(_MODEL_DIR / "AASIST-L.conf") as f:
        config = json.load(f)

    model = Model(config["model_config"])
    state = torch.load(_MODEL_DIR / "models" / "AASIST-L.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(1, WINDOW_SAMPLES)
    out_path = _MODEL_DIR / "aasist_l.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=["waveform"],
        output_names=["embedding", "logits"],
        opset_version=18,
        dynamic_axes=None,  # fixed input size - matches the model's fixed window
    )
    print(f"Exported to {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
