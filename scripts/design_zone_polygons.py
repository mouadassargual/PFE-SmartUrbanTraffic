#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ZONE_ORDER = ["N", "E", "S", "W"]
ZONE_COLORS = {
    "N": (255, 120, 0),
    "E": (0, 220, 255),
    "S": (0, 180, 80),
    "W": (255, 0, 180),
}


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def read_image(image_path: Path) -> np.ndarray:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return frame


def load_existing(path: Path) -> dict[str, list[list[int]]]:
    data = json.loads(path.read_text())
    zones = data.get("zones", data)
    return {
        zone: [[int(x), int(y)] for x, y in zones.get(zone, [])]
        for zone in ZONE_ORDER
    }


def zones_have_points(zones: dict[str, list[list[int]]]) -> bool:
    return any(len(zones.get(zone, [])) > 0 for zone in ZONE_ORDER)


def zones_are_complete(zones: dict[str, list[list[int]]]) -> bool:
    return all(len(zones.get(zone, [])) >= 3 for zone in ZONE_ORDER)


def load_profile_defaults(profile: str) -> dict[str, list[list[int]]] | None:
    try:
        from pipeline.config import ZONE_PROFILES
    except Exception:
        return None

    if profile not in ZONE_PROFILES:
        return None
    return {
        zone: ZONE_PROFILES[profile][zone].astype(int).tolist()
        for zone in ZONE_ORDER
        if zone in ZONE_PROFILES[profile]
    }


class ZoneDesigner:
    def __init__(
        self,
        frame: np.ndarray,
        output: Path,
        profile: str,
        source: str,
        frame_index: int,
        allow_incomplete: bool = False,
    ):
        self.frame = frame
        self.output = output
        self.profile = profile
        self.source = source
        self.frame_index = frame_index
        self.allow_incomplete = allow_incomplete
        self.zone_index = 0
        self.points: dict[str, list[list[int]]] = {zone: [] for zone in ZONE_ORDER}
        self.window = "Zone polygon designer"

    @property
    def current_zone(self) -> str:
        return ZONE_ORDER[self.zone_index]

    def set_existing(self, zones: dict[str, list[list[int]]]) -> None:
        for zone in ZONE_ORDER:
            self.points[zone] = zones.get(zone, [])

    def mouse(self, event, x, y, flags, userdata) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points[self.current_zone].append([int(x), int(y)])
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.undo()

    def undo(self) -> None:
        if self.points[self.current_zone]:
            self.points[self.current_zone].pop()

    def reset_current(self) -> None:
        self.points[self.current_zone] = []

    def next_zone(self) -> None:
        self.zone_index = min(len(ZONE_ORDER) - 1, self.zone_index + 1)

    def previous_zone(self) -> None:
        self.zone_index = max(0, self.zone_index - 1)

    def save(self) -> bool:
        missing = [zone for zone in ZONE_ORDER if len(self.points[zone]) < 3]
        if missing:
            print(f"Warning: incomplete zones with <3 points: {missing}")
            if not self.allow_incomplete:
                print("Not saved. Complete all zones or pass --allow-incomplete.")
                return False

        h, w = self.frame.shape[:2]
        payload = {
            "profile": self.profile,
            "source": self.source,
            "frame_index": self.frame_index,
            "frame_size": {"width": w, "height": h},
            "zones": self.points,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(payload, indent=2))
        print(f"Saved: {self.output}")
        print("\nUse it with:")
        print(f"python3 -m pipeline.main --video <video> --model <model.onnx> --zones-json {self.output}")
        return True

    def draw(self) -> np.ndarray:
        canvas = self.frame.copy()
        overlay = canvas.copy()

        for zone in ZONE_ORDER:
            pts = self.points[zone]
            color = ZONE_COLORS[zone]
            if len(pts) >= 3:
                poly = np.array(pts, np.int32)
                cv2.fillPoly(overlay, [poly], color)
                cv2.polylines(canvas, [poly], True, color, 2)
                center = poly.mean(axis=0).astype(int)
                cv2.putText(
                    canvas,
                    zone,
                    tuple(center),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.2,
                    (255, 255, 255),
                    2,
                )
            for idx, point in enumerate(pts, start=1):
                cv2.circle(canvas, tuple(point), 5, color, -1)
                cv2.putText(
                    canvas,
                    str(idx),
                    (point[0] + 6, point[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )

        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

        self._draw_help(canvas)
        return canvas

    def _draw_help(self, canvas: np.ndarray) -> None:
        status = " | ".join(
            f"{zone}:{len(self.points[zone])}" for zone in ZONE_ORDER
        )
        lines = [
            f"Current zone: {self.current_zone}   ({status})",
            "Left click: add point | Right click/backspace: undo | r: reset current",
            "Enter/space/n: next zone | p: previous | s: save | q/esc: quit",
        ]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (8, 8), (900, 88), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        for idx, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (18, 32 + idx * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (245, 245, 245),
                1,
            )

    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1280, 720)
        cv2.setMouseCallback(self.window, self.mouse)

        while True:
            cv2.imshow(self.window, self.draw())
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (8, 127):
                self.undo()
            elif key == ord("r"):
                self.reset_current()
            elif key == ord("p"):
                self.previous_zone()
            elif key in (13, 32, ord("n")):
                self.next_zone()
            elif key == ord("s"):
                if self.save():
                    break

        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively draw N/E/S/W zone polygons.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Video file to sample a frame from.")
    source.add_argument("--image", type=Path, help="Image file to annotate.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index when using --video.")
    parser.add_argument("--profile", default="custom", help="Profile name saved in JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/zones/custom_zones.json"),
        help="Output JSON path.",
    )
    parser.add_argument("--load", type=Path, default=None, help="Existing zones JSON to edit.")
    parser.add_argument(
        "--blank",
        action="store_true",
        help="Start with empty polygons instead of loading output/default profile polygons.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow saving even if one or more zones have fewer than 3 points.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.video:
        frame = read_frame(args.video, args.frame)
        source = str(args.video)
    else:
        frame = read_image(args.image)
        source = str(args.image)

    designer = ZoneDesigner(
        frame=frame,
        output=args.output,
        profile=args.profile,
        source=source,
        frame_index=args.frame,
        allow_incomplete=args.allow_incomplete,
    )
    initial_zones = None
    initial_source = None
    if args.load:
        initial_zones = load_existing(args.load)
        initial_source = str(args.load)
    elif not args.blank and args.output.exists():
        output_zones = load_existing(args.output)
        if zones_have_points(output_zones):
            initial_zones = output_zones
            initial_source = str(args.output)
    if initial_zones is None and not args.blank:
        default_zones = load_profile_defaults(args.profile)
        if default_zones is not None and zones_are_complete(default_zones):
            initial_zones = default_zones
            initial_source = f"profile:{args.profile}"

    if initial_zones is not None:
        designer.set_existing(initial_zones)
        counts = {zone: len(initial_zones.get(zone, [])) for zone in ZONE_ORDER}
        print(f"Loaded polygons from {initial_source}: {counts}")
    else:
        print("Starting with empty polygons.")
    designer.run()


if __name__ == "__main__":
    main()
