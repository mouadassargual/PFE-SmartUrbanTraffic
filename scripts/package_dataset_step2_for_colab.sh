#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="$ROOT_DIR/data/dataset/dataset_step2_citypersons_person"
OUT_ZIP="$ROOT_DIR/data/dataset/dataset_step2_citypersons_person.zip"

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset not found: $DATASET_DIR" >&2
  exit 1
fi

if [[ -f "$OUT_ZIP" ]]; then
  echo "Archive already exists: $OUT_ZIP" >&2
  echo "Move or delete it before creating a new archive." >&2
  exit 1
fi

cd "$ROOT_DIR/data/dataset"
zip -qr "dataset_step2_citypersons_person.zip" "dataset_step2_citypersons_person"
du -sh "$OUT_ZIP"
