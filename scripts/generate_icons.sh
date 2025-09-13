#!/usr/bin/env bash
set -euo pipefail
SRC="$(dirname "$0")/../resources/icon.svg"
OUTDIR="$(dirname "$0")/../resources"
sizes=(16 24 32 48 64 128 256 512)
command -v convert >/dev/null || { echo "ImageMagick 'convert' nicht gefunden" >&2; exit 1; }
for s in "${sizes[@]}"; do
  convert -background none -resize ${s}x${s} "$SRC" "$OUTDIR/icon_${s}.png"
done
echo "Icons erzeugt in $OUTDIR"