#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Cloudflare supplies Python; bootstrap uv when it is not preinstalled.
if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --disable-pip-version-check uv==0.12.13
fi
uv run --locked python -m mkdocs build --strict
