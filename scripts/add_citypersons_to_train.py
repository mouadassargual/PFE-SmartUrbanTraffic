#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = ["bus", "car", "emergency_vehicle", "motorcycle", "person", "truck"]
PERSON_CLASS_ID = 4
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class CityRecord:
    label_path: str
    image_path: str
    output_stem: str
    original_key: str
    city: str
    lines: list[str]
    person_count: int
    medium_count: int
    large_count: int
    score: float


def original_key(filename: str) -> str:
    stem = Path(filename).stem
    return stem.split(".rf.")[0]


def city_name(filename: str) -> str:
    return original_key(filename).split("_", 1)[0]


def parse_label(text: str) -> tuple[list[str], int, int, int, float]:
    remapped: list[str] = []
    person_count = 0
    medium_count = 0
    large_count = 0
    score = 0.0

    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO line: {line}")

        source_class = int(float(parts[0]))
        if source_class != 0:
            raise ValueError(f"Unexpected CityPersons class id {source_class}; expected 0")

        width = float(parts[3])
        height = float(parts[4])
        area = width * height
        person_count += 1
        medium_count += area >= 0.001
        large_count += area >= 0.0025
        score += min(area / 0.0025, 3.0)
        remapped.append(" ".join([str(PERSON_CLASS_ID), *parts[1:]]))

    return remapped, person_count, medium_count, large_count, score


def find_image_for_label(zip_names: set[str], label_path: str) -> str:
    stem = Path(label_path).stem
    for ext in IMAGE_EXTS:
        candidate = f"train/images/{stem}{ext}"
        if candidate in zip_names:
            return candidate
    raise FileNotFoundError(f"Missing image for {label_path}")


def collect_citypersons_records(zip_path: Path) -> list[CityRecord]:
    with zipfile.ZipFile(zip_path) as zf:
        zip_names = set(zf.namelist())
        labels = sorted(
            name for name in zip_names if name.startswith("train/labels/") and name.endswith(".txt")
        )

        records_by_key: dict[str, list[CityRecord]] = defaultdict(list)
        for label_path in labels:
            raw_label = zf.read(label_path).decode("utf-8", errors="replace").strip()
            remapped, person_count, medium_count, large_count, score = parse_label(raw_label)
            if person_count == 0:
                continue

            image_path = find_image_for_label(zip_names, label_path)
            label_name = Path(label_path).name
            key = original_key(label_name)
            record = CityRecord(
                label_path=label_path,
                image_path=image_path,
                output_stem=f"citypersons_{Path(image_path).stem}",
                original_key=key,
                city=city_name(label_name),
                lines=remapped,
                person_count=person_count,
                medium_count=medium_count,
                large_count=large_count,
                score=score,
            )
            records_by_key[key].append(record)

    selected = [
        max(records, key=lambda item: (item.score, item.medium_count, item.person_count, item.label_path))
        for records in records_by_key.values()
    ]
    selected.sort(key=lambda item: item.original_key)
    return selected


def copy_base_dataset(src: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    shutil.copytree(src, out)


def add_records(zip_path: Path, out: Path, records: list[CityRecord]) -> None:
    image_out = out / "images" / "train"
    label_out = out / "labels" / "train"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        for record in records:
            image_suffix = Path(record.image_path).suffix.lower()
            out_image = image_out / f"{record.output_stem}{image_suffix}"
            out_label = label_out / f"{record.output_stem}.txt"

            with zf.open(record.image_path) as src_image, out_image.open("wb") as dst_image:
                shutil.copyfileobj(src_image, dst_image)
            out_label.write_text("\n".join(record.lines) + "\n")


def write_data_yaml(out: Path) -> None:
    lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"- {name}" for name in CLASS_NAMES)
    lines.append(f"nc: {len(CLASS_NAMES)}")
    (out / "data.yaml").write_text("\n".join(lines) + "\n")


def count_dataset(root: Path) -> dict[str, Counter[str] | Counter[int]]:
    split_rows: dict[str, Counter[str] | Counter[int]] = {}
    for split in ("train", "val", "test"):
        counts: Counter[str] = Counter()
        image_count = sum(1 for path in (root / "images" / split).iterdir() if path.is_file())
        label_count = 0
        for label_path in (root / "labels" / split).glob("*.txt"):
            label_count += 1
            for line in label_path.read_text(errors="replace").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(float(parts[0]))
                counts[CLASS_NAMES[class_id]] += 1
        counts["images"] = image_count
        counts["labels"] = label_count
        split_rows[split] = counts
    return split_rows


def write_report(out: Path, records: list[CityRecord]) -> str:
    rows = count_dataset(out)
    added_people = sum(record.person_count for record in records)
    added_medium = sum(record.medium_count for record in records)
    added_large = sum(record.large_count for record in records)
    cities = Counter(record.city for record in records)

    lines = [
        "# Step 2 CityPersons Integration Report",
        "",
        "CityPersons policy:",
        "- Source split used: CityPersons train only.",
        "- Empty-label images skipped.",
        "- Augmented duplicate sources reduced to one image per original key.",
        "- Class remap: CityPersons `ped` -> project `person` class id 4.",
        "- Project validation and test splits kept unchanged from step 1.",
        "",
        f"Added CityPersons images: {len(records)}",
        f"Added person instances: {added_people}",
        f"Added medium person boxes (area >= 0.1%): {added_medium}",
        f"Added large person boxes (area >= 0.25%): {added_large}",
        f"Cities represented: {len(cities)}",
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

    report_text = "\n".join(lines) + "\n"
    (out / "citypersons_integration_report.md").write_text(report_text)
    return report_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Add CityPersons pedestrians to the train split.")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("data/dataset/dataset_step1_no_leak_split"),
        help="Base project dataset root.",
    )
    parser.add_argument(
        "--citypersons",
        type=Path,
        default=Path("data/dataset/Citypersons YOLOv8.zip"),
        help="CityPersons YOLOv8 Roboflow zip.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/dataset/dataset_step2_citypersons_person"),
        help="Output merged dataset root.",
    )
    args = parser.parse_args()

    records = collect_citypersons_records(args.citypersons)
    copy_base_dataset(args.base, args.out)
    add_records(args.citypersons, args.out, records)
    write_data_yaml(args.out)
    print(write_report(args.out, records))


if __name__ == "__main__":
    main()
