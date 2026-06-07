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
from pipeline.config import DEFAULT_SCENE, SCENE_PROFILES


SCENE_COUNTS = {
    "ne8th": {"N": 8, "S": 0, "E": 5, "W": 6, "phase": "EW"},
    "116th": {"N": 2, "S": 7, "E": 3, "W": 1, "phase": "NS"},
    "150th": {"N": 4, "S": 1, "E": 8, "W": 3, "phase": "EW"},
}


def make_frame(scene_id=DEFAULT_SCENE):
    counts = SCENE_COUNTS.get(scene_id, SCENE_COUNTS[DEFAULT_SCENE])
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (32, 42, 50)
    cv2.putText(
        frame,
        f"URBANFLOW TEST - {scene_id.upper()} - IMAGE IA ANONYMISEE",
        (260, 90),
        cv2.FONT_HERSHEY_DUPLEX,
        1.1,
        (245, 245, 245),
        2,
    )
    cv2.rectangle(frame, (80, 150), (580, 620), (52, 78, 95), -1)
    cv2.rectangle(frame, (700, 150), (1200, 620), (42, 70, 86), -1)
    cv2.putText(frame, f"N={counts['N']} / S={counts['S']}", (190, 370), cv2.FONT_HERSHEY_DUPLEX, 1.2, (80, 255, 120), 2)
    cv2.putText(frame, f"E={counts['E']} / W={counts['W']}", (800, 370), cv2.FONT_HERSHEY_DUPLEX, 1.2, (80, 220, 255), 2)
    cv2.putText(frame, f"Phase attendue: {counts['phase']}", (475, 675), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 230, 120), 2)
    return frame


def mock_state(scene_id=DEFAULT_SCENE):
    counts = SCENE_COUNTS.get(scene_id, SCENE_COUNTS[DEFAULT_SCENE])
    scene = SCENE_PROFILES.get(scene_id, SCENE_PROFILES[DEFAULT_SCENE])
    return {
        "N": {
            "count": counts["N"],
            "score": float(counts["N"]),
            "pedestrians": 0,
            "by_class": {"car": counts["N"], "motorcycle": 0, "truck": 0, "bus": 0, "emergency_vehicle": 0},
        },
        "S": {
            "count": counts["S"],
            "score": float(counts["S"]),
            "pedestrians": 0,
            "by_class": {"car": counts["S"], "motorcycle": 0, "truck": 0, "bus": 0, "emergency_vehicle": 0},
        },
        "E": {
            "count": counts["E"],
            "score": float(counts["E"]),
            "pedestrians": 0,
            "by_class": {"car": counts["E"], "motorcycle": 0, "truck": 0, "bus": 0, "emergency_vehicle": 0},
        },
        "W": {
            "count": counts["W"],
            "score": float(counts["W"]),
            "pedestrians": 0,
            "by_class": {"car": counts["W"], "motorcycle": 0, "truck": 0, "bus": 0, "emergency_vehicle": 0},
        },
        "pedestrians": 0,
        "emergency": False,
        "anonymized": 2,
        "anonymized_total": 12,
        "detections_total": sum(counts[z] for z in ("N", "S", "E", "W")) + 2,
        "detections_by_class": {"car": sum(counts[z] for z in ("N", "S", "E", "W")), "person": 2},
        "active_tracks": sum(counts[z] for z in ("N", "S", "E", "W")) + 2,
        "tracks_by_class": {"car": sum(counts[z] for z in ("N", "S", "E", "W")), "person": 2},
        "frame_id": 128,
        "frame_size": {"width": 1280, "height": 720},
        "polygons": {
            "N": [[500, 40], [660, 40], [720, 250], [440, 250]],
            "E": [[730, 260], [1220, 280], [1180, 510], [760, 500]],
            "S": [[450, 520], [800, 510], [900, 690], [300, 690]],
            "W": [[40, 280], [430, 250], [430, 520], [40, 540]],
        },
        "model_info": {
            "name": "YOLO26n_step3_960_best.onnx",
            "imgsz": 960,
            "profile": scene["profile"],
            "scene": scene_id,
            "scene_label": scene["label"],
        },
    }


def mock_decision(scene_id=DEFAULT_SCENE):
    counts = SCENE_COUNTS.get(scene_id, SCENE_COUNTS[DEFAULT_SCENE])
    score_ns = counts["N"] + counts["S"]
    score_ew = counts["E"] + counts["W"]
    phase = counts["phase"]
    return {
        "phase": phase,
        "duration": 45,
        "score_NS": float(score_ns),
        "score_EW": float(score_ew),
        "reason": "Axe Nord/Sud dominant" if phase == "NS" else "Axe Est/Ouest dominant",
        "green_dirs": ["N", "S"] if phase == "NS" else ["E", "W"],
        "red_dirs": ["E", "W"] if phase == "NS" else ["N", "S"],
    }


def publish_scene(dashboard, scene_id):
    frame = make_frame(scene_id)
    scene = SCENE_PROFILES.get(scene_id, SCENE_PROFILES[DEFAULT_SCENE])
    if Path(scene["video"]).exists():
        dashboard.set_source_video(scene["video"])
    dashboard.set_current_scene(scene_id)
    dashboard.update(
        frame,
        mock_state(scene_id),
        mock_decision(scene_id),
        6.4,
        analysis_frame=frame,
        video_meta={
            "fps": 30.0,
            "total_frames": 5400,
            "current_second": 42.5,
            "playback_mode": "local_mock",
            "source_ready": Path(scene["video"]).exists(),
            "scene": scene_id,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="UrbanFlow dashboard local demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5066)
    args = parser.parse_args()

    dashboard = Dashboard(host=args.host, port=args.port)
    dashboard.start()
    current_scene = DEFAULT_SCENE
    publish_scene(dashboard, current_scene)

    print(f"UrbanFlow dashboard local: http://{args.host}:{args.port}")
    print("Ctrl+C pour arreter.")
    try:
        while True:
            request = dashboard.pop_scene_request()
            if request and request.get("scene") in SCENE_PROFILES:
                current_scene = request["scene"]
                publish_scene(dashboard, current_scene)
                print(f"Scene mock -> {current_scene}")
            time.sleep(1)
    except KeyboardInterrupt:
        dashboard.stop()
        print("\nDashboard arrete.")


if __name__ == "__main__":
    main()
