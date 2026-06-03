#!/usr/bin/env python3
"""
Benchmark ONNX inference on Raspberry Pi.

Measures preprocessing + ONNX Runtime inference latency for one or more models.
This is the model-speed benchmark; run pipeline/main.py separately for end-to-end
FPS with tracking, anonymization, and MDP decision.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil


def preprocess(frame: np.ndarray, img_size: int) -> np.ndarray:
    img = cv2.resize(frame, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


def benchmark_model(model_path: Path, video_path: Path, img_size: int, frames: int) -> dict | None:
    if not model_path.exists():
        print(f"Missing model: {model_path}")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    process = psutil.Process(os.getpid())

    # Warmup
    for _ in range(3):
        ok, frame = cap.read()
        if not ok:
            break
        blob = preprocess(frame, img_size)
        session.run(None, {input_name: blob})

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    total_ms: list[float] = []
    ram_mb: list[float] = []

    count = 0
    print(f"\nModel: {model_path.name} | imgsz={img_size} | frames={frames}")
    while count < frames:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        blob = preprocess(frame, img_size)
        t1 = time.perf_counter()
        session.run(None, {input_name: blob})
        t2 = time.perf_counter()

        preprocess_ms.append((t1 - t0) * 1000)
        inference_ms.append((t2 - t1) * 1000)
        total_ms.append((t2 - t0) * 1000)
        ram_mb.append(process.memory_info().rss / 1024 / 1024)
        count += 1

        if count % 50 == 0:
            fps = 1000 / float(np.mean(total_ms[-50:]))
            print(f"  frame {count:4d}: {fps:.2f} FPS")

    cap.release()
    if not total_ms:
        return None

    result = {
        "model": model_path.name,
        "img_size": img_size,
        "frames": count,
        "fps": round(1000 / float(np.mean(total_ms)), 2),
        "preprocess_ms": round(float(np.mean(preprocess_ms)), 2),
        "inference_ms": round(float(np.mean(inference_ms)), 2),
        "total_latency_ms": round(float(np.mean(total_ms)), 2),
        "p95_latency_ms": round(float(np.percentile(total_ms, 95)), 2),
        "ram_avg_mb": round(float(np.mean(ram_mb)), 1),
        "ram_max_mb": round(float(np.max(ram_mb)), 1),
    }

    print(
        f"  FPS={result['fps']} | "
        f"lat={result['total_latency_ms']}ms | "
        f"p95={result['p95_latency_ms']}ms | "
        f"RAM={result['ram_avg_mb']}MB"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ONNX models on Raspberry Pi.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--models", nargs="+", required=True, type=Path)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("benchmark_onnx_results.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("ONNX Runtime:", ort.__version__)
    print("CPU cores:", psutil.cpu_count())
    print("RAM total MB:", psutil.virtual_memory().total // 1024 // 1024)

    results = []
    for model in args.models:
        result = benchmark_model(model, args.video, args.img_size, args.frames)
        if result:
            results.append(result)

    print("\nSummary")
    print(f"{'model':36} {'img':>5} {'fps':>8} {'lat':>9} {'p95':>9} {'ram':>8}")
    print("-" * 82)
    for result in sorted(results, key=lambda item: item["fps"], reverse=True):
        print(
            f"{result['model'][:36]:36} "
            f"{result['img_size']:>5} "
            f"{result['fps']:>8.2f} "
            f"{result['total_latency_ms']:>8.2f} "
            f"{result['p95_latency_ms']:>8.2f} "
            f"{result['ram_avg_mb']:>7.1f}"
        )

    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
