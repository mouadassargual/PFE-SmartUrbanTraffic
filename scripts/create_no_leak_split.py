#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = ["bus", "car", "emergency_vehicle", "motorcycle", "person", "truck"]
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Record:
    image: Path
    label: Path
    group: str
    counts: Counter[str]


def original_key(filename: str) -> str:
    stem = Path(filename).stem
    return stem.split(".rf.")[0]


def source_name(key: str) -> str:
    if key.startswith("Bellevue_"):
        return "Bellevue"
    if key.startswith("emergency_"):
        return "emergency_video"
    if key.startswith("image-"):
        return "image"
    if key.startswith("youtube-"):
        return "youtube"
    if key.startswith("bandicam-"):
        return "bandicam"
    return "other"


def group_key(filename: str) -> str:
    key = original_key(filename)

    emergency = re.match(r"(emergency_VID_[0-9_]+)_mp4-", key)
    if emergency:
        return f"emergency_video:{emergency.group(1)}"

    if key.startswith("Bellevue_") and "_t" in key:
        return f"Bellevue:{key.split('_t', 1)[0]}"

    bandicam = re.match(r"(bandicam-\d{4}-\d{2}-\d{2}-\d{2})-", key)
    if bandicam:
        return f"bandicam:{bandicam.group(1)}"

    return f"{source_name(key)}:{key}"


def label_counts(label_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in label_path.read_text(errors="replace").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        class_id = int(float(parts[0]))
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(f"Invalid class id {class_id} in {label_path}")
        counts[CLASS_NAMES[class_id]] += 1
    return counts


def collect_records(src: Path) -> list[Record]:
    records: list[Record] = []
    seen_images: set[Path] = set()

    for split in SPLITS:
        image_dir = src / "images" / split
        label_dir = src / "labels" / split
        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_EXTS:
                continue
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"Missing label for {image}")
            if image in seen_images:
                raise ValueError(f"Duplicate image path: {image}")
            seen_images.add(image)
            records.append(
                Record(
                    image=image,
                    label=label,
                    group=group_key(image.name),
                    counts=label_counts(label),
                )
            )

    return records


def split_records(records: list[Record]) -> dict[str, list[Record]]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.group].append(record)

    group_items = []
    for group, items in groups.items():
        counts: Counter[str] = Counter()
        for item in items:
            counts.update(item.counts)
        importance = sum(counts[c] for c in ("emergency_vehicle", "person", "truck", "bus"))
        group_items.append((group, items, counts, importance, len(items)))

    group_items.sort(key=lambda row: (row[3], row[4], row[0]), reverse=True)

    total_counts: Counter[str] = Counter()
    for record in records:
        total_counts.update(record.counts)

    total_images = len(records)
    target_images = {split: total_images * RATIOS[split] for split in SPLITS}
    target_counts = {
        split: {name: total_counts[name] * RATIOS[split] for name in CLASS_NAMES}
        for split in SPLITS
    }

    assigned: dict[str, list[Record]] = {split: [] for split in SPLITS}
    split_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}

    def score(candidate: str, counts: Counter[str], image_count: int) -> float:
        total_score = 0.0
        for split in SPLITS:
            new_image_count = len(assigned[split])
            if split == candidate:
                new_image_count += image_count
            total_score += ((new_image_count - target_images[split]) ** 2) / (
                target_images[split] + 1
            )

            for name in CLASS_NAMES:
                new_count = split_counts[split][name]
                if split == candidate:
                    new_count += counts[name]
                total_score += ((new_count - target_counts[split][name]) ** 2) / (
                    target_counts[split][name] + 1
                )

        return total_score

    for _, items, counts, _, image_count in group_items:
        chosen = min(SPLITS, key=lambda split: score(split, counts, image_count))
        assigned[chosen].extend(items)
        split_counts[chosen].update(counts)

    return assigned


def copy_split(assigned: dict[str, list[Record]], out: Path) -> None:
    for split, records in assigned.items():
        image_dir = out / "images" / split
        label_dir = out / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            shutil.copy2(record.image, image_dir / record.image.name)
            shutil.copy2(record.label, label_dir / record.label.name)


def write_yaml(out: Path) -> None:
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


def report(assigned: dict[str, list[Record]], out: Path) -> str:
    lines = ["# No-Leak Split Report", ""]
    lines.append("| split | images | bus | car | emergency_vehicle | motorcycle | person | truck |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    totals: Counter[str] = Counter()
    total_images = 0
    for split in SPLITS:
        counts: Counter[str] = Counter()
        for record in assigned[split]:
            counts.update(record.counts)
        total_images += len(assigned[split])
        totals.update(counts)
        values = [split, str(len(assigned[split]))] + [str(counts[name]) for name in CLASS_NAMES]
        lines.append("| " + " | ".join(values) + " |")

    lines.append("| total | " + str(total_images) + " | " + " | ".join(str(totals[name]) for name in CLASS_NAMES) + " |")
    lines.append("")

    group_splits: dict[str, set[str]] = defaultdict(set)
    for split, records in assigned.items():
        for record in records:
            group_splits[record.group].add(split)

    leaked = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    lines.append(f"Group leakage check: {len(leaked)} group(s) found in multiple splits.")
    if leaked:
        for group, splits in sorted(leaked.items())[:20]:
            lines.append(f"- {group}: {', '.join(sorted(splits))}")

    text = "\n".join(lines) + "\n"
    (out / "split_report.md").write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a source-grouped YOLO split.")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/dataset/dataset_clean_merged_bbox"),
        help="Extracted YOLO dataset root.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/dataset/dataset_step1_no_leak_split"),
        help="Output YOLO dataset root.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {args.out}. Use --overwrite to replace it.")
        shutil.rmtree(args.out)

    records = collect_records(args.src)
    assigned = split_records(records)
    copy_split(assigned, args.out)
    write_yaml(args.out)
    print(report(assigned, args.out))


if __name__ == "__main__":
    main()
