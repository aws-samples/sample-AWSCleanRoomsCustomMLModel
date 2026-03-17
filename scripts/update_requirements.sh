#!/usr/bin/env bash
# Regenerate pinned requirements.txt files for both containers from uv.lock.
# Run this after any dependency change in pyproject.toml + uv lock.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

uv export --no-dev --format requirements-txt \
  > "$REPO_ROOT/containers/training/requirements.txt"

uv export --no-dev --format requirements-txt \
  > "$REPO_ROOT/containers/inference/requirements.txt"

echo "Updated containers/training/requirements.txt"
echo "Updated containers/inference/requirements.txt"
