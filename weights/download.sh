#!/usr/bin/env bash
# Fetch the detector weights ONCE, with internet, before the offline run.
#
#   bash weights/download.sh
#
# The organizers run this before evaluation if the repo does not ship the
# weights directly (README.md lists it as an accepted pattern). The OFFLINE path
# must work without it: if weights/yolo11s.pt is already committed, this script
# sees it and exits without touching the network.
#
# Nothing here is required at inference time. src/perception.py loads by an
# explicit local path and never passes a bare model NAME to Ultralytics, which
# is the code path that would auto-download.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# name  sha256  url
MODELS=(
  "yolo11s.pt|https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
)

echo "weights dir: $DIR"

for entry in "${MODELS[@]}"; do
  name="${entry%%|*}"
  url="${entry##*|}"
  dest="$DIR/$name"

  if [ -f "$dest" ]; then
    size=$(wc -c < "$dest" | tr -d ' ')
    echo "  $name already present (${size} bytes) — skipping"
    continue
  fi

  echo "  fetching $name ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 -o "$dest.part" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$dest.part" "$url"
  else
    echo "ERROR: neither curl nor wget is available" >&2
    exit 1
  fi
  mv "$dest.part" "$dest"
  echo "  wrote $dest ($(wc -c < "$dest" | tr -d ' ') bytes)"
done

echo
echo "total: $(du -sh "$DIR" 2>/dev/null | cut -f1) (limit 5 GB)"
echo "verify the offline path with:  python scripts/offline_check.py"
