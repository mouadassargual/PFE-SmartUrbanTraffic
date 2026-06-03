#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


CLASS_NAMES = ["bus", "car", "emergency_vehicle", "motorcycle", "person", "truck"]
PERSON_ID = CLASS_NAMES.index("person")


def read_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in label_path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        boxes.append((cls, x, y, w, h))
    return boxes


def bucket(area: float) -> str:
    if area < 0.0005:
        return "tiny <0.05%"
    if area < 0.001:
        return "very_small 0.05-0.10%"
    if area < 0.0025:
        return "small 0.10-0.25%"
    if area < 0.01:
        return "medium 0.25-1.00%"
    return "large >=1.00%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit person box sizes in a YOLO dataset.")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    root = args.dataset
    for split in ("train", "val", "test"):
        labels = sorted((root / "labels" / split).glob("*.txt"))
        counts = Counter()
        images_with_person = 0
        total_person = 0

        for label in labels:
            person_in_image = 0
            for cls, _, _, w, h in read_boxes(label):
                if cls != PERSON_ID:
                    continue
                area = w * h
                counts[bucket(area)] += 1
                total_person += 1
                person_in_image += 1
            images_with_person += person_in_image > 0

        print(f"\n{split}")
        print(f"  labels: {len(labels)}")
        print(f"  images_with_person: {images_with_person}")
        print(f"  person_instances: {total_person}")
        for name in ("tiny <0.05%", "very_small 0.05-0.10%", "small 0.10-0.25%", "medium 0.25-1.00%", "large >=1.00%"):
            value = counts[name]
            pct = 100 * value / total_person if total_person else 0
            print(f"  {name:22} {value:6d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
