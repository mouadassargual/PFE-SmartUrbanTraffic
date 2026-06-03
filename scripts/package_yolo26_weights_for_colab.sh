#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="$ROOT_DIR/models/downloads"
OUT_ZIP="$ROOT_DIR/models/yolo26_weights_for_colab.zip"

for weight in YOLO26n_best.pt YOLO26s_best.pt; do
  if [[ ! -f "$WEIGHTS_DIR/$weight" ]]; then
    echo "Missing weight: $WEIGHTS_DIR/$weight" >&2
    exit 1
  fi
done

if [[ -f "$OUT_ZIP" ]]; then
  echo "Archive already exists: $OUT_ZIP" >&2
  echo "Move or delete it before creating a new archive." >&2
  exit 1
fi

cd "$WEIGHTS_DIR"
zip -q "$OUT_ZIP" YOLO26n_best.pt YOLO26s_best.pt
du -sh "$OUT_ZIP"
