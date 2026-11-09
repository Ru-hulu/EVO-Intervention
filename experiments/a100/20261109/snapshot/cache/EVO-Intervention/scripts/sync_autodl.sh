#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# AutoDL's GitHub/Hugging Face proxy slows down domestic package mirrors.
# Keep the UV cache on the data disk and reuse previously downloaded wheels.
exec env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY -u UV_EXTRA_INDEX_URL -u UV_INDEX_URL \
  -u UV_DEFAULT_INDEX -u VIRTUAL_ENV \
  UV_CACHE_DIR="/root/autodl-tmp/.cache/uv" \
  UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=5 \
  UV_CONCURRENT_DOWNLOADS=8 UV_CONCURRENT_BUILDS=2 \
  uv sync "$@"
