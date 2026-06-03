#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="$ROOT_DIR/data/dataset/dataset_step3_tiny_person_crops"
OUT_ZIP="$ROOT_DIR/data/dataset/dataset_step3_tiny_person_crops.zip"

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset not found: $DATASET_DIR" >&2
  echo "Run scripts/create_tiny_person_step3.py first." >&2
  exit 1
fi

if [[ -f "$OUT_ZIP" ]]; then
  echo "Archive already exists: $OUT_ZIP" >&2
  echo "Move or delete it before creating a new archive." >&2
  exit 1
fi

cd "$ROOT_DIR/data/dataset"
zip -qr "dataset_step3_tiny_person_crops.zip" "dataset_step3_tiny_person_crops"
du -sh "$OUT_ZIP"
