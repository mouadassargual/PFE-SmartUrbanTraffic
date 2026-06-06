#!/usr/bin/env python3
"""
Prepare SUMO demand windows from pipeline_results.json.

The pipeline output contains per-frame zone counts. SUMO needs arrivals over
time, so this script groups frames into short windows and exports compact
directional demand for N/S/E/W.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "pipeline_results.json"
DEFAULT_OUT_DIR = ROOT / "sumo" / "simple_intersection" / "demand"
DIRECTIONS = ("N", "S", "E", "W")


def zone_count(frame: dict, direction: str) -> int:
    return int(frame.get("zones", {}).get(direction, {}).get("count", 0) or 0)


def zone_score(frame: dict, direction: str) -> float:
    return float(frame.get("zones", {}).get(direction, {}).get("score", 0.0) or 0.0)


def chunks(items: list[dict], size: int):
    for index in range(0, len(items), size):
        yield index // size, items[index : index + size]


def build_windows(
    frames: list[dict],
    window_frames: int,
    fps: float,
    seconds_per_window: float | None,
) -> list[dict]:
    windows = []
    for window_id, group in chunks(frames, window_frames):
        if not group:
            continue
        start_frame = int(group[0].get("frame", 0) or 0)
        end_frame = int(group[-1].get("frame", start_frame) or start_frame)
        directions = {}
        for direction in DIRECTIONS:
            counts = [zone_count(frame, direction) for frame in group]
            scores = [zone_score(frame, direction) for frame in group]
            directions[direction] = {
                "mean_count": round(mean(counts), 3),
                "max_count": max(counts),
                "mean_score": round(mean(scores), 3),
                "max_score": max(scores),
            }

        if seconds_per_window and seconds_per_window > 0:
            start_second = window_id * seconds_per_window
            end_second = (window_id + 1) * seconds_per_window
        else:
            start_second = (start_frame - 1) / fps if fps > 0 else 0.0
            end_second = end_frame / fps if fps > 0 else 0.0

        windows.append(
            {
                "window": window_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_second": round(start_second, 3),
                "end_second": round(end_second, 3),
                "directions": directions,
                "pedestrians_max": max(int(frame.get("pedestrians", 0) or 0) for frame in group),
                "emergency": any(bool(frame.get("emergency", False)) for frame in group),
            }
        )
    return windows


def write_csv(windows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["window", "start_second", "end_second"]
        for direction in DIRECTIONS:
            header.extend(
                [
                    f"{direction}_mean_count",
                    f"{direction}_max_count",
                    f"{direction}_mean_score",
                    f"{direction}_max_score",
                ]
            )
        header.extend(["pedestrians_max", "emergency"])
        writer.writerow(header)

        for window in windows:
            row = [window["window"], window["start_second"], window["end_second"]]
            for direction in DIRECTIONS:
                values = window["directions"][direction]
                row.extend(
                    [
                        values["mean_count"],
                        values["max_count"],
                        values["mean_score"],
                        values["max_score"],
                    ]
                )
            row.extend([window["pedestrians_max"], int(window["emergency"])])
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SUMO demand from pipeline results")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window-frames", type=int, default=30)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--seconds-per-window",
        type=float,
        default=30.0,
        help="SUMO duration represented by each grouped demand window; set 0 to derive from fps",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.window_frames < 1:
        raise SystemExit("--window-frames must be >= 1")
    if not args.input.exists():
        raise SystemExit(f"Missing pipeline results: {args.input}")

    data = json.loads(args.input.read_text())
    frames = data.get("frames", [])
    if not frames:
        raise SystemExit(f"No frames found in {args.input}")

    seconds_per_window = args.seconds_per_window if args.seconds_per_window > 0 else None
    windows = build_windows(frames, args.window_frames, args.fps, seconds_per_window)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "source": str(args.input),
        "window_frames": args.window_frames,
        "fps": args.fps,
        "seconds_per_window": seconds_per_window,
        "windows": windows,
    }
    json_path = args.out_dir / "pipeline_demand.json"
    csv_path = args.out_dir / "pipeline_demand.csv"
    json_path.write_text(json.dumps(payload, indent=2))
    write_csv(windows, csv_path)

    total = sum(
        int(window["directions"][direction]["max_count"])
        for window in windows
        for direction in DIRECTIONS
    )
    print(f"Demand windows : {len(windows)}")
    print(f"Max-count sum  : {total}")
    print(f"JSON           : {json_path}")
    print(f"CSV            : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
