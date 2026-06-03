#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DVC_SITE_CACHE_DIR="$ROOT_DIR/.dvc/tmp/site-cache"
export DVC_NO_ANALYTICS=1

exec "$ROOT_DIR/.venv/bin/dvc" "$@"
