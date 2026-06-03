#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import ZONE_PROFILES


BELLEVUE_GROUPS = {
    "Bellevue_Bellevue_NE8th": "ne8th",
    "Bellevue_116th_NE12th": "116th",
    "Bellevue_150th_Eastgate": None,
    "Bellevue_150th_Newport": None,
}

ZONE_COLORS = {
    "N": (255, 120, 0),
    "E": (0, 220, 255),
    "S": (0, 180, 80),
    "W": (255, 0, 180),
}


def infer_group(path: Path) -> str | None:
    name = path.name
    for group in BELLEVUE_GROUPS:
        if name.startswith(group):
            return group
    return None


def collect_images(dataset: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for image_path in sorted((dataset / "images").glob("*/*")):
        if not image_path.is_file():
            continue
        group = infer_group(image_path)
        if group is not None:
            grouped[group].append(image_path)
    return grouped


def pick_representative(paths: list[Path]) -> Path:
    test_paths = [path for path in paths if path.parent.name == "test"]
    candidates = test_paths or paths
    return candidates[len(candidates) // 2]


def draw_zones(image: np.ndarray, zones: dict[str, np.ndarray] | None, title: str) -> np.ndarray:
    canvas = image.copy()
    overlay = canvas.copy()

    if zones:
        for direction, polygon in zones.items():
            color = ZONE_COLORS.get(direction, (255, 255, 255))
            cv2.fillPoly(overlay, [polygon], color)
            cv2.polylines(canvas, [polygon], True, color, 3)
            center = polygon.mean(axis=0).astype(int)
            cv2.putText(
                canvas,
                direction,
                tuple(center),
                cv2.FONT_HERSHEY_DUPLEX,
                1.4,
                (255, 255, 255),
                2,
            )
        cv2.addWeighted(overlay, 0.24, canvas, 0.76, 0, canvas)
    else:
        h, w = canvas.shape[:2]
        cv2.rectangle(canvas, (0, 0), (w, h), (0, 0, 220), 8)
        cv2.putText(
            canvas,
            "POLYGONES A DESIGNER",
            (40, 80),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            (0, 0, 255),
            2,
        )

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (15, 15, 15), -1)
    cv2.putText(canvas, title, (14, 28), cv2.FONT_HERSHEY_DUPLEX, 0.74, (245, 245, 245), 1)
    return canvas


def make_contact_sheet(images: list[np.ndarray], out_path: Path, thumb_width: int = 640) -> None:
    thumbs = []
    for image in images:
        h, w = image.shape[:2]
        scale = thumb_width / w
        thumbs.append(cv2.resize(image, (thumb_width, int(h * scale))))

    if not thumbs:
        return

    columns = 2
    rows = (len(thumbs) + columns - 1) // columns
    thumb_h = max(thumb.shape[0] for thumb in thumbs)
    sheet = np.full((rows * thumb_h, columns * thumb_width, 3), 235, dtype=np.uint8)

    for idx, thumb in enumerate(thumbs):
        row = idx // columns
        col = idx % columns
        y = row * thumb_h
        x = col * thumb_width
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def write_report(rows: list[dict[str, str]], out: Path) -> None:
    lines = [
        "# Bellevue Zone Polygon Illustration",
        "",
        "| group | images | profile | illustration | next action |",
        "|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {group} | {count} | {profile} | `{image}` | {action} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Design command for missing profiles:",
            "",
            "```bash",
            "python3 scripts/design_zone_polygons.py \\",
            "  --image <representative_image.jpg> \\",
            "  --profile <profile_name> \\",
            "  --output data/zones/<profile_name>_zones.json",
            "```",
            "",
        ]
    )
    (out / "bellevue_zone_profiles_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Bellevue representative frames with zone polygons.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/dataset/dataset_step2_citypersons_person"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/zones/illustrations"))
    args = parser.parse_args()

    grouped = collect_images(args.dataset)
    args.out.mkdir(parents=True, exist_ok=True)

    rendered = []
    report_rows = []
    for group, profile in BELLEVUE_GROUPS.items():
        paths = grouped.get(group, [])
        if not paths:
            continue

        representative = pick_representative(paths)
        image = cv2.imread(str(representative))
        if image is None:
            continue

        zones = ZONE_PROFILES.get(profile) if profile else None
        title_profile = profile if profile else "needs_custom_design"
        title = f"{group} | profile={title_profile} | {representative.parent.name}"
        canvas = draw_zones(image, zones, title)

        out_image = args.out / f"{group}_{title_profile}.jpg"
        cv2.imwrite(str(out_image), canvas)
        rendered.append(canvas)

        if profile:
            action = "OK: verifier visuellement"
        else:
            action = f"Designer un nouveau JSON depuis `{representative}`"

        report_rows.append(
            {
                "group": group,
                "count": str(len(paths)),
                "profile": title_profile,
                "image": str(out_image),
                "action": action,
            }
        )
        print(f"{group}: {len(paths)} images -> {out_image}")

    make_contact_sheet(rendered, args.out / "bellevue_zone_profiles_contact_sheet.jpg")
    write_report(report_rows, args.out)
    print(f"Contact sheet: {args.out / 'bellevue_zone_profiles_contact_sheet.jpg'}")
    print(f"Report: {args.out / 'bellevue_zone_profiles_report.md'}")


if __name__ == "__main__":
    main()
