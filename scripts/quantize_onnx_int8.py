"""
Quantize the experimental 800x800 ONNX model to INT8 with video calibration.

The generated INT8 model is an experimental Raspberry Pi benchmark artifact.
It does not replace the FP32 deployment models.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)


ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"
QUANT_TYPES = {
    "QInt8": QuantType.QInt8,
    "QUInt8": QuantType.QUInt8,
}


class VideoCalibrationReader(CalibrationDataReader):
    def __init__(self, video_path, input_name, imgsz, frame_count, frame_stride):
        self.input_name = input_name
        self.imgsz = int(imgsz)
        self.frame_count = int(frame_count)
        self.frame_stride = max(1, int(frame_stride))
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open calibration video: {video_path}")
        self.frames_returned = 0

    def _preprocess(self, frame):
        image = cv2.resize(frame, (self.imgsz, self.imgsz))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        return np.expand_dims(image, axis=0)

    def get_next(self):
        if self.frames_returned >= self.frame_count:
            self.cap.release()
            return None

        ok = False
        frame = None
        while not ok:
            ok, frame = self.cap.read()
            if not ok:
                self.cap.release()
                return None
            for _ in range(self.frame_stride - 1):
                if not self.cap.grab():
                    break

        self.frames_returned += 1
        return {self.input_name: self._preprocess(frame)}


def main():
    params = yaml.safe_load(PARAMS_PATH.read_text())
    config = params["experimental_exports"]["onnx_800_int8"]

    source_onnx = ROOT / config["source_onnx"]
    output_onnx = ROOT / config["output_onnx"]
    calibration_video = ROOT / config["calibration_video"]
    imgsz = int(config["imgsz"])
    method = str(config.get("method", "dynamic")).lower()
    weight_type = QUANT_TYPES[str(config.get("weight_type", "QInt8"))]

    if not source_onnx.exists():
        raise FileNotFoundError(f"Source ONNX not found: {source_onnx}")
    if method == "static" and not calibration_video.exists():
        raise FileNotFoundError(f"Calibration video not found: {calibration_video}")

    import onnxruntime as ort

    session = ort.InferenceSession(str(source_onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Source ONNX: {source_onnx}")
    print(f"Input: {input_name} {input_shape}")

    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    if method == "dynamic":
        quantize_dynamic(
            model_input=str(source_onnx),
            model_output=str(output_onnx),
            weight_type=weight_type,
            per_channel=bool(config.get("per_channel", False)),
        )
    elif method == "static":
        reader = VideoCalibrationReader(
            calibration_video,
            input_name,
            imgsz,
            config["calibration_frames"],
            config["calibration_stride"],
        )
        quantize_static(
            model_input=str(source_onnx),
            model_output=str(output_onnx),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=weight_type,
            per_channel=bool(config.get("per_channel", False)),
            calibrate_method=CalibrationMethod.MinMax,
            calibration_providers=["CPUExecutionProvider"],
            extra_options={
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
            },
        )
    else:
        raise ValueError(f"Unsupported quantization method: {method}")
    print(f"INT8 ONNX -> {output_onnx}")


if __name__ == "__main__":
    main()
