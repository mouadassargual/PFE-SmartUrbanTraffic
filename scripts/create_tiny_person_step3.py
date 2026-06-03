#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2


CLASS_NAMES = ["bus", "car", "emergency_vehicle", "motorcycle", "person", "truck"]
PERSON_ID = CLASS_NAMES.index("person")
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


@dataclass(frozen=True)
class Box:
    class_id: int
    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    def to_xyxy(self, image_w: int, image_h: int) -> tuple[float, float, float, float]:
        x1 = (self.x - self.w / 2) * image_w
        y1 = (self.y - self.h / 2) * image_h
        x2 = (self.x + self.w / 2) * image_w
        y2 = (self.y + self.h / 2) * image_h
        return x1, y1, x2, y2


@dataclass(frozen=True)
class CropRecord:
    image_name: str
    label_name: str
    source_image: str
    source_tiny_persons: int
    output_labels: int
    output_persons: int
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int


def read_boxes(label_path: Path) -> list[Box]:
    boxes = []
    for line in label_path.read_text(errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        if not 0 <= class_id < len(CLASS_NAMES):
            continue
        if w <= 0 or h <= 0:
            continue
        boxes.append(Box(class_id, x, y, w, h))
    return boxes


def image_for_label(dataset: Path, split: str, stem: str) -> Path | None:
    image_dir = dataset / "images" / split
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def clamp_crop(x1: float, y1: float, side: int, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    side = min(side, image_w, image_h)
    x1 = max(0, min(int(round(x1)), image_w - side))
    y1 = max(0, min(int(round(y1)), image_h - side))
    return x1, y1, x1 + side, y1 + side


def crop_for_person(
    box: Box,
    image_w: int,
    image_h: int,
    min_crop_frac: float,
    max_crop_frac: float,
    context_scale: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.to_xyxy(image_w, image_h)
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    min_side = min(image_w, image_h)
    side = max(box_w * context_scale, box_h * context_scale, min_side * min_crop_frac)
    side = min(side, min_side * max_crop_frac, min_side)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return clamp_crop(cx - side / 2, cy - side / 2, int(round(side)), image_w, image_h)


def crop_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(1, area_a + area_b - inter)


def clipped_yolo_line(
    box: Box,
    crop: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    min_visibility: float,
    min_box_px: float,
) -> str | None:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop
    box_x1, box_y1, box_x2, box_y2 = box.to_xyxy(image_w, image_h)

    ix1 = max(crop_x1, box_x1)
    iy1 = max(crop_y1, box_y1)
    ix2 = min(crop_x2, box_x2)
    iy2 = min(crop_y2, box_y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    original_area = max(1.0, (box_x2 - box_x1) * (box_y2 - box_y1))
    clipped_area = (ix2 - ix1) * (iy2 - iy1)
    if clipped_area / original_area < min_visibility:
        return None
    if ix2 - ix1 < min_box_px or iy2 - iy1 < min_box_px:
        return None

    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    x = ((ix1 + ix2) / 2 - crop_x1) / crop_w
    y = ((iy1 + iy2) / 2 - crop_y1) / crop_h
    w = (ix2 - ix1) / crop_w
    h = (iy2 - iy1) / crop_h
    return f"{box.class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def copy_base_dataset(src: Path, out: Path, overwrite: bool) -> None:
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {out}")
        shutil.rmtree(out)
    shutil.copytree(src, out)


def count_dataset(root: Path) -> dict[str, Counter[str]]:
    rows: dict[str, Counter[str]] = {}
    for split in ("train", "val", "test"):
        counts: Counter[str] = Counter()
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        counts["images"] = sum(1 for path in image_dir.iterdir() if path.is_file())
        counts["labels"] = sum(1 for path in label_dir.glob("*.txt"))
        for label_path in label_dir.glob("*.txt"):
            for box in read_boxes(label_path):
                counts[CLASS_NAMES[box.class_id]] += 1
        rows[split] = counts
    return rows


def person_size_counts(root: Path, split: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for label_path in (root / "labels" / split).glob("*.txt"):
        for box in read_boxes(label_path):
            if box.class_id != PERSON_ID:
                continue
            if box.area < 0.0005:
                counts["tiny_lt_0_05_pct"] += 1
            elif box.area < 0.001:
                counts["very_small_0_05_0_10_pct"] += 1
            elif box.area < 0.0025:
                counts["small_0_10_0_25_pct"] += 1
            elif box.area < 0.01:
                counts["medium_0_25_1_00_pct"] += 1
            else:
                counts["large_ge_1_00_pct"] += 1
    return counts


def make_crops(
    dataset: Path,
    max_person_area: float,
    max_crops: int,
    max_crops_per_image: int,
    skip_prefix: str,
    min_crop_frac: float,
    max_crop_frac: float,
    context_scale: float,
    crop_iou_skip: float,
    min_visibility: float,
    min_box_px: float,
) -> list[CropRecord]:
    image_out = dataset / "images" / "train"
    label_out = dataset / "labels" / "train"
    records: list[CropRecord] = []

    labels = sorted(label_out.glob("*.txt"))
    ranked: list[tuple[int, float, Path, Path, list[Box]]] = []
    for label_path in labels:
        if skip_prefix and label_path.stem.startswith(skip_prefix):
            continue
        boxes = read_boxes(label_path)
        tiny_people = [
            box for box in boxes if box.class_id == PERSON_ID and box.area <= max_person_area
        ]
        if not tiny_people:
            continue
        image_path = image_for_label(dataset, "train", label_path.stem)
        if image_path is None:
            continue
        score = sum(1.0 / max(box.area, 1e-8) for box in tiny_people)
        ranked.append((len(tiny_people), score, label_path, image_path, boxes))

    ranked.sort(key=lambda item: (item[0], item[1], item[2].stem), reverse=True)

    for _, _, label_path, image_path, boxes in ranked:
        if len(records) >= max_crops:
            break

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        image_h, image_w = image.shape[:2]
        tiny_people = sorted(
            [box for box in boxes if box.class_id == PERSON_ID and box.area <= max_person_area],
            key=lambda box: box.area,
        )
        selected_crops: list[tuple[int, int, int, int]] = []

        for person_box in tiny_people:
            if len(records) >= max_crops or len(selected_crops) >= max_crops_per_image:
                break

            crop = crop_for_person(
                person_box,
                image_w,
                image_h,
                min_crop_frac=min_crop_frac,
                max_crop_frac=max_crop_frac,
                context_scale=context_scale,
            )
            if any(crop_iou(crop, existing) > crop_iou_skip for existing in selected_crops):
                continue

            yolo_lines = [
                line
                for box in boxes
                if (
                    line := clipped_yolo_line(
                        box,
                        crop,
                        image_w,
                        image_h,
                        min_visibility=min_visibility,
                        min_box_px=min_box_px,
                    )
                )
            ]
            person_count = sum(line.startswith(f"{PERSON_ID} ") for line in yolo_lines)
            if person_count == 0:
                continue

            selected_crops.append(crop)
            crop_x1, crop_y1, crop_x2, crop_y2 = crop
            crop_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_index = len(records) + 1
            out_stem = f"{image_path.stem}_tinycrop_{crop_index:05d}"
            out_image = image_out / f"{out_stem}.jpg"
            out_label = label_out / f"{out_stem}.txt"
            cv2.imwrite(str(out_image), crop_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            out_label.write_text("\n".join(yolo_lines) + "\n")
            records.append(
                CropRecord(
                    image_name=out_image.name,
                    label_name=out_label.name,
                    source_image=image_path.name,
                    source_tiny_persons=len(tiny_people),
                    output_labels=len(yolo_lines),
                    output_persons=person_count,
                    crop_x1=crop_x1,
                    crop_y1=crop_y1,
                    crop_x2=crop_x2,
                    crop_y2=crop_y2,
                )
            )

    return records


def write_report(out: Path, base: Path, records: list[CropRecord], args: argparse.Namespace) -> str:
    rows = count_dataset(out)
    person_sizes = {split: person_size_counts(out, split) for split in ("train", "val", "test")}

    lines = [
        "# Step 3 Tiny-Person Crop Report",
        "",
        "Policy:",
        "- Base dataset copied from step 2.",
        "- Validation and test splits kept unchanged.",
        "- New images are added to train only.",
        "- Crops are centered on tiny `person` boxes.",
        "- All classes visible inside each crop are kept with clipped YOLO boxes.",
        "- CityPersons images are skipped by default because they were already added in step 2.",
        "",
        f"Base dataset: `{base}`",
        f"Output dataset: `{out}`",
        f"Max source person area selected: {args.max_person_area * 100:.3f}%",
        f"Added crop images: {len(records)}",
        f"Added crop labels: {len(records)}",
        f"Person instances inside added crops: {sum(record.output_persons for record in records)}",
        "",
        "| split | images | labels | bus | car | emergency_vehicle | motorcycle | person | truck |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    totals: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        counts = rows[split]
        for name in CLASS_NAMES:
            totals[name] += counts[name]
        values = [
            split,
            str(counts["images"]),
            str(counts["labels"]),
            *[str(counts[name]) for name in CLASS_NAMES],
        ]
        lines.append("| " + " | ".join(values) + " |")

    total_images = sum(rows[split]["images"] for split in ("train", "val", "test"))
    total_labels = sum(rows[split]["labels"] for split in ("train", "val", "test"))
    lines.append(
        "| total | "
        + str(total_images)
        + " | "
        + str(total_labels)
        + " | "
        + " | ".join(str(totals[name]) for name in CLASS_NAMES)
        + " |"
    )

    lines.extend(
        [
            "",
            "Person box size distribution after step 3:",
            "",
            "| split | tiny <0.05% | very small 0.05-0.10% | small 0.10-0.25% | medium 0.25-1.00% | large >=1.00% |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("train", "val", "test"):
        sizes = person_sizes[split]
        values = [
            split,
            str(sizes["tiny_lt_0_05_pct"]),
            str(sizes["very_small_0_05_0_10_pct"]),
            str(sizes["small_0_10_0_25_pct"]),
            str(sizes["medium_0_25_1_00_pct"]),
            str(sizes["large_ge_1_00_pct"]),
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "Recommended Colab experiment:",
            "- Start with YOLO26n at `imgsz=800` for the Pi 5 real-time constraint.",
            "- Use the step 3 zip as the dataset input.",
            "- Keep step 1/2 validation and test unchanged when reporting metrics.",
            "",
        ]
    )

    report = "\n".join(lines)
    (out / "tiny_person_crop_report.md").write_text(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a step3 dataset with tiny-person train crops.")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("data/dataset/dataset_step2_citypersons_person"),
        help="Base step2 dataset root.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/dataset/dataset_step3_tiny_person_crops"),
        help="Output step3 dataset root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    parser.add_argument(
        "--max-person-area",
        type=float,
        default=0.001,
        help="Select source person boxes up to this normalized area. 0.001 = 0.10%%.",
    )
    parser.add_argument("--max-crops", type=int, default=1600)
    parser.add_argument("--max-crops-per-image", type=int, default=2)
    parser.add_argument("--skip-prefix", default="citypersons_")
    parser.add_argument("--min-crop-frac", type=float, default=0.42)
    parser.add_argument("--max-crop-frac", type=float, default=0.65)
    parser.add_argument("--context-scale", type=float, default=18.0)
    parser.add_argument("--crop-iou-skip", type=float, default=0.65)
    parser.add_argument("--min-visibility", type=float, default=0.25)
    parser.add_argument("--min-box-px", type=float, default=2.0)
    args = parser.parse_args()

    copy_base_dataset(args.base, args.out, overwrite=args.overwrite)
    records = make_crops(
        args.out,
        max_person_area=args.max_person_area,
        max_crops=args.max_crops,
        max_crops_per_image=args.max_crops_per_image,
        skip_prefix=args.skip_prefix,
        min_crop_frac=args.min_crop_frac,
        max_crop_frac=args.max_crop_frac,
        context_scale=args.context_scale,
        crop_iou_skip=args.crop_iou_skip,
        min_visibility=args.min_visibility,
        min_box_px=args.min_box_px,
    )
    report = write_report(args.out, args.base, records, args)
    print(report)


if __name__ == "__main__":
    main()
