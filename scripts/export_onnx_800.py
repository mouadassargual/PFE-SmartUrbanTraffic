"""
Export a static 800x800 ONNX from the final Step 3 960 PyTorch weights.

This keeps the official 960 ONNX untouched and creates a separate deployment
artifact for Raspberry Pi latency experiments.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"
TMP_PT = Path("/tmp/YOLO26n_step3_800_export_source.pt")


def main():
    params = yaml.safe_load(PARAMS_PATH.read_text())
    config = params["experimental_exports"]["onnx_800_from_step3_960"]

    source_pt = ROOT / config["source_pt"]
    output_onnx = ROOT / config["output_onnx"]
    imgsz = int(config["imgsz"])
    opset = int(config["opset"])
    simplify = bool(config["simplify"])

    if not source_pt.exists():
        raise FileNotFoundError(f"Source PT not found: {source_pt}")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smart-traffic")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    shutil.copy2(source_pt, TMP_PT)
    model = YOLO(str(TMP_PT))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
        )
    )
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, output_onnx)
    print(f"Exported {imgsz}x{imgsz} ONNX -> {output_onnx}")


if __name__ == "__main__":
    main()
