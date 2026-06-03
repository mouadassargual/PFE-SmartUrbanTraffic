#!/usr/bin/env python3
"""
Benchmark ONNX models on Raspberry Pi
Mesure : FPS, latence par frame, RAM utilisée
Usage   : python3 benchmark_rpi.py
"""

import onnxruntime as ort
import cv2
import numpy as np
import time
import psutil
import os
import json
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────
VIDEO_PATH   = os.path.expanduser("~/video_test.mp4")
MODELS_DIR   = os.path.expanduser("~/models/")
INPUT_SIZE   = (640, 640)          # résolution d'entrée du modèle
MAX_FRAMES   = 200                 # nombre de frames à traiter (None = toutes)
OUTPUT_JSON  = "benchmark_results.json"
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "YOLO26n":  "YOLO26n_best.onnx",
    "YOLOv11n": "YOLOv11n_best.onnx",
    "YOLOv8n":  "YOLOv8n_best.onnx",
    "YOLOv11s": "YOLOv11s_best.onnx",
    "YOLOv8s":  "YOLOv8s_best.onnx",
    "YOLO26s":  "YOLO26s_best.onnx",
}

def preprocess(frame, size):
    img = cv2.resize(frame, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))          # HWC → CHW
    img = np.expand_dims(img, axis=0)            # → NCHW
    return img

def benchmark_model(name, model_path, video_path):
    print(f"\n{'='*55}")
    print(f"  Modèle : {name}")
    print(f"  Fichier: {model_path}")
    print(f"{'='*55}")

    if not os.path.exists(model_path):
        print(f"  ⚠️  Fichier introuvable, ignoré.")
        return None

    # Chargement du modèle
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options,
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Ouverture vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Impossible d'ouvrir {video_path}")
        return None

    total_fps_video = cap.get(cv2.CAP_PROP_FPS)
    latencies = []
    frame_count = 0
    process = psutil.Process(os.getpid())
    ram_samples = []

    # Warm-up (3 frames)
    for _ in range(3):
        ret, frame = cap.read()
        if not ret:
            break
        blob = preprocess(frame, INPUT_SIZE)
        session.run(None, {input_name: blob})

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # retour au début

    print(f"  ▶  Benchmark en cours ({MAX_FRAMES or 'toutes'} frames)...")
    start_total = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if MAX_FRAMES and frame_count >= MAX_FRAMES:
            break

        blob = preprocess(frame, INPUT_SIZE)

        t0 = time.perf_counter()
        session.run(None, {input_name: blob})
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)   # en ms
        ram_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
        frame_count += 1

        if frame_count % 50 == 0:
            avg_fps = 1000 / np.mean(latencies[-50:])
            print(f"    frame {frame_count:4d} | {avg_fps:.1f} FPS | {latencies[-1]:.1f} ms")

    cap.release()
    elapsed = time.perf_counter() - start_total

    if not latencies:
        print("  ❌ Aucune frame traitée.")
        return None

    results = {
        "model":          name,
        "frames":         frame_count,
        "total_time_s":   round(elapsed, 2),
        "avg_fps":        round(1000 / np.mean(latencies), 2),
        "avg_latency_ms": round(np.mean(latencies), 1),
        "min_latency_ms": round(np.min(latencies), 1),
        "max_latency_ms": round(np.max(latencies), 1),
        "p95_latency_ms": round(np.percentile(latencies, 95), 1),
        "ram_avg_mb":     round(np.mean(ram_samples), 1),
        "ram_max_mb":     round(np.max(ram_samples), 1),
    }

    print(f"\n  ✅ Résultats {name}:")
    print(f"     FPS moyen       : {results['avg_fps']} FPS")
    print(f"     Latence moyenne : {results['avg_latency_ms']} ms")
    print(f"     Latence P95     : {results['p95_latency_ms']} ms")
    print(f"     RAM moyenne     : {results['ram_avg_mb']} MB")
    print(f"     RAM max         : {results['ram_max_mb']} MB")

    return results


def main():
    print("\n" + "="*55)
    print("  🔬 BENCHMARK ONNX — Raspberry Pi")
    print("="*55)
    print(f"  Vidéo      : {VIDEO_PATH}")
    print(f"  Résolution : {INPUT_SIZE}")
    print(f"  Frames max : {MAX_FRAMES}")
    print(f"  CPU cores  : {psutil.cpu_count()}")
    print(f"  RAM totale : {psutil.virtual_memory().total // 1024 // 1024} MB")

    all_results = []

    for name, filename in MODELS.items():
        model_path = os.path.join(MODELS_DIR, filename)
        result = benchmark_model(name, model_path, VIDEO_PATH)
        if result:
            all_results.append(result)

    # ── Tableau récapitulatif ──────────────────────────────────────────────
    print("\n\n" + "="*75)
    print(f"  {'Modèle':<12} {'FPS':>6} {'Lat.moy':>9} {'Lat.P95':>9} {'RAM moy':>9} {'RAM max':>9}")
    print("-"*75)
    for r in sorted(all_results, key=lambda x: x["avg_fps"], reverse=True):
        print(f"  {r['model']:<12} {r['avg_fps']:>6.1f} "
              f"{r['avg_latency_ms']:>8.1f}ms "
              f"{r['p95_latency_ms']:>8.1f}ms "
              f"{r['ram_avg_mb']:>8.1f}MB "
              f"{r['ram_max_mb']:>8.1f}MB")
    print("="*75)

    # Sauvegarde JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  💾 Résultats sauvegardés → {OUTPUT_JSON}\n")


if __name__ == "__main__":
    main()
