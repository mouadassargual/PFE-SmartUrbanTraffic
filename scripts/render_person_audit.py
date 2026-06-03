#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ["bus", "car", "emergency_vehicle", "motorcycle", "person", "truck"]
PERSON_ID = CLASS_NAMES.index("person")
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def image_for_label(dataset: Path, split: str, stem: str) -> Path | None:
    image_dir = dataset / "images" / split
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def read_person_boxes(label_path: Path) -> list[tuple[float, float, float, float, float]]:
    boxes = []
    for line in label_path.read_text(errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        if class_id != PERSON_ID:
            continue
        x, y, w, h = map(float, parts[1:5])
        boxes.append((x, y, w, h, w * h))
    return boxes


def draw_boxes(image_path: Path, boxes: list[tuple[float, float, float, float, float]]) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    for index, (x, y, box_w, box_h, area) in enumerate(boxes, start=1):
        x1 = int((x - box_w / 2) * width)
        y1 = int((y - box_h / 2) * height)
        x2 = int((x + box_w / 2) * width)
        y2 = int((y + box_h / 2) * height)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        color = (255, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"p{index} {area * 100:.3f}%"
        cv2.putText(
            image,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    title = f"{image_path.name} | persons={len(boxes)}"
    cv2.rectangle(image, (0, 0), (min(width, 900), 28), (0, 0, 0), -1)
    cv2.putText(image, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return image


def make_contact_sheet(images: list[np.ndarray], out_path: Path, thumb_width: int = 320) -> None:
    if not images:
        return
    thumbs = []
    for image in images:
        height, width = image.shape[:2]
        scale = thumb_width / width
        thumb_height = max(1, int(height * scale))
        thumbs.append(cv2.resize(image, (thumb_width, thumb_height)))

    columns = 3
    rows = (len(thumbs) + columns - 1) // columns
    thumb_height = max(thumb.shape[0] for thumb in thumbs)
    sheet = np.full((rows * thumb_height, columns * thumb_width, 3), 245, dtype=np.uint8)

    for idx, thumb in enumerate(thumbs):
        row = idx // columns
        col = idx % columns
        y = row * thumb_height
        x = col * thumb_width
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render person-label audit images.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--out", type=Path, default=Path("data/audit/person_boxes"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for split in args.splits:
        split_out = args.out / split
        split_out.mkdir(parents=True, exist_ok=True)
        rendered_images = []

        labels = sorted((args.dataset / "labels" / split).glob("*.txt"))
        for label_path in labels:
            boxes = read_person_boxes(label_path)
            if not boxes:
                continue

            image_path = image_for_label(args.dataset, split, label_path.stem)
            if image_path is None:
                continue

            image = draw_boxes(image_path, boxes)
            out_image = split_out / f"{label_path.stem}_person_audit.jpg"
            cv2.imwrite(str(out_image), image)
            rendered_images.append(image)

            areas = sorted(box[-1] for box in boxes)
            rows.append(
                {
                    "split": split,
                    "image": image_path.name,
                    "audit_image": str(out_image),
                    "person_count": len(boxes),
                    "tiny_lt_0_05_pct": sum(area < 0.0005 for area in areas),
                    "very_small_lt_0_10_pct": sum(area < 0.001 for area in areas),
                    "min_area_pct": round(areas[0] * 100, 5),
                    "median_area_pct": round(areas[len(areas) // 2] * 100, 5),
                    "max_area_pct": round(areas[-1] * 100, 5),
                }
            )

        make_contact_sheet(rendered_images, args.out / f"contact_sheet_{split}.jpg")
        print(f"{split}: rendered {len(rendered_images)} audit images")

    csv_path = args.out / "person_audit.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "split",
                "image",
                "audit_image",
                "person_count",
                "tiny_lt_0_05_pct",
                "very_small_lt_0_10_pct",
                "min_area_pct",
                "median_area_pct",
                "max_area_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
