"""Run a local UrbanFlow dashboard demo with deterministic mock data.

Usage:
    python3 scripts/run_urbanflow_dashboard_mock.py --port 5066
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.dashboard import Dashboard


def make_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (32, 42, 50)
    cv2.putText(
        frame,
        "URBANFLOW TEST - IMAGE IA ANONYMISEE",
        (260, 90),
        cv2.FONT_HERSHEY_DUPLEX,
        1.1,
        (245, 245, 245),
        2,
    )
    cv2.rectangle(frame, (80, 150), (580, 620), (52, 78, 95), -1)
    cv2.rectangle(frame, (700, 150), (1200, 620), (42, 70, 86), -1)
    cv2.putText(frame, "N=8 / S=0", (190, 370), cv2.FONT_HERSHEY_DUPLEX, 1.2, (80, 255, 120), 2)
    cv2.putText(frame, "E=5 / W=6", (800, 370), cv2.FONT_HERSHEY_DUPLEX, 1.2, (80, 220, 255), 2)
    cv2.putText(frame, "Phase attendue: EW", (475, 675), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 230, 120), 2)
    return frame


def mock_state():
    return {
        "N": {
            "count": 8,
            "score": 8.0,
            "pedestrians": 0,
            "by_class": {"car": 8, "motorcycle": 0, "truck": 0, "bus": 0, "emergency_vehicle": 0},
        },
        "S": {
            "count": 0,
            "score": 0.0,
            "pedestrians": 0,
            "by_class": {"car": 0, "motorcycle": 0, "truck": 0, "bus": 0, "emergency_vehicle": 0},
        },
        "E": {
            "count": 5,
            "score": 5.0,
            "pedestrians": 0,
            "by_class": {"car": 5, "motorcycle": 0, "truck": 0, "bus": 0, "emergency_vehicle": 0},
        },
        "W": {
            "count": 6,
            "score": 6.0,
            "pedestrians": 0,
            "by_class": {"car": 6, "motorcycle": 0, "truck": 0, "bus": 0, "emergency_vehicle": 0},
        },
        "pedestrians": 0,
        "emergency": False,
        "anonymized": 2,
        "anonymized_total": 12,
        "detections_total": 19,
        "detections_by_class": {"car": 19, "person": 2},
        "active_tracks": 19,
        "tracks_by_class": {"car": 19, "person": 2},
        "frame_id": 128,
        "frame_size": {"width": 1280, "height": 720},
        "polygons": {
            "N": [[500, 40], [660, 40], [720, 250], [440, 250]],
            "E": [[730, 260], [1220, 280], [1180, 510], [760, 500]],
            "S": [[450, 520], [800, 510], [900, 690], [300, 690]],
            "W": [[40, 280], [430, 250], [430, 520], [40, 540]],
        },
        "model_info": {"name": "YOLO26n_step3_960_best.onnx", "imgsz": 960, "profile": "ne8th"},
    }


def mock_decision():
    return {
        "phase": "EW",
        "duration": 45,
        "score_NS": 8.0,
        "score_EW": 11.0,
        "reason": "Axe Est/Ouest dominant",
        "green_dirs": ["E", "W"],
        "red_dirs": ["N", "S"],
    }


def main():
    parser = argparse.ArgumentParser(description="UrbanFlow dashboard local demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5066)
    args = parser.parse_args()

    frame = make_frame()
    dashboard = Dashboard(host=args.host, port=args.port)
    dashboard.start()
    dashboard.update(
        frame,
        mock_state(),
        mock_decision(),
        6.4,
        analysis_frame=frame,
        video_meta={
            "fps": 30.0,
            "total_frames": 5400,
            "current_second": 42.5,
            "playback_mode": "local_mock",
            "source_ready": False,
        },
    )

    print(f"UrbanFlow dashboard local: http://{args.host}:{args.port}")
    print("Ctrl+C pour arreter.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        dashboard.stop()
        print("\nDashboard arrete.")


if __name__ == "__main__":
    main()
