#!/usr/bin/env bash
set -euo pipefail

# Build an AppImage for Audio Restorer.
# This script can produce a fully offline AppImage by bundling:
# - A Python runtime (virtual environment)
# - All Python dependencies (including heavy ML packages)
# - Prefetched ML model files (Demucs, Open-Unmix, TorchCrepe, optional AudioSR/Spleeter)
# Requirements: appimagetool (optional), ImageMagick (optional for icon conversion).

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
APPDIR="$ROOT_DIR/AppDir"
BUILD_VENV="$ROOT_DIR/.build-venv"
PYVER_MAJOR_MINOR="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
ARCH="$(uname -m)"

clean() {
  rm -rf "$APPDIR" "$BUILD_VENV"
  mkdir -p "$DIST_DIR"
}

msg() { echo "[build] $*"; }

clean
msg "Create build venv"
python3 -m venv "$BUILD_VENV"
source "$BUILD_VENV/bin/activate"
python -m pip install -U pip wheel

msg "Install base requirements"
pip install -r "$ROOT_DIR/requirements.txt"

# Flags (defaults optimized for fully offline build when INCLUDE_EXTRAS=1)
INCLUDE_EXTRAS=${INCLUDE_EXTRAS:-0}
INCLUDE_DEMUCS=${INCLUDE_DEMUCS:-$INCLUDE_EXTRAS}
INCLUDE_OPENUNMIX=${INCLUDE_OPENUNMIX:-$INCLUDE_EXTRAS}
INCLUDE_TORCHCREPE=${INCLUDE_TORCHCREPE:-$INCLUDE_EXTRAS}
INCLUDE_SPLEETER=${INCLUDE_SPLEETER:-$INCLUDE_EXTRAS}
INCLUDE_AUDIOSR=${INCLUDE_AUDIOSR:-$INCLUDE_EXTRAS}

if [[ "$INCLUDE_EXTRAS" == "1" ]]; then
  msg "Install optional extras (heavy)"
  # Use CPU-only wheels for torch/torchaudio
  TV=${TORCH_VERSION:-2.3.1}
  TAV=${TORCHAUDIO_VERSION:-2.3.1}
  pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TV}" "torchaudio==${TAV}" || true
  pip install -U pyloudnorm || true
  [[ "$INCLUDE_DEMUCS" == "1" ]] && pip install -U demucs || true
  [[ "$INCLUDE_OPENUNMIX" == "1" ]] && pip install -U open-unmix || true
  [[ "$INCLUDE_TORCHCREPE" == "1" ]] && pip install -U torchcrepe || true
  [[ "$INCLUDE_SPLEETER" == "1" ]] && pip install -U spleeter || true
  [[ "$INCLUDE_AUDIOSR" == "1" ]] && pip install -U audiosr || true
fi

# PREWARM=1 to pre-initialize model caches (if installed)
if [[ "${PREWARM:-0}" == "1" ]]; then
  python - <<'PY'
try:
    import demucs  # noqa
except Exception:
    pass
PY
fi

SITEPKG=$(python -c 'import site; print(site.getsitepackages()[0])')

msg "Prepare AppDir layout"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/lib" "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/256x256/apps" "$APPDIR/usr/src" \
         "$APPDIR/usr/python" "$APPDIR/usr/models" "$APPDIR/usr/cache" "$APPDIR/usr/licenses"

msg "Bundle Python venv runtime"
# Copy the entire build venv into AppDir/usr/python to provide an isolated runtime
cp -a "$BUILD_VENV/." "$APPDIR/usr/python/"
chmod +x "$APPDIR/usr/python/bin/python" || true

msg "Copy project sources and docs"
cp -a "$ROOT_DIR/src" "$APPDIR/usr/"
cp -a "$ROOT_DIR/run.py" "$APPDIR/usr/"
cp -a "$ROOT_DIR/LICENSE" "$ROOT_DIR/README.md" "$ROOT_DIR/THIRD_PARTY_LICENSES.md" "$APPDIR/usr/" || true
cp -a "$ROOT_DIR/resources" "$APPDIR/usr/" || true

# Pre-download model weights/caches into AppDir (so first run is offline)
if [[ "$INCLUDE_EXTRAS" == "1" ]]; then
  msg "Prefetching ML models into AppDir"
  export TORCH_HOME="$APPDIR/usr/models/torch"
  export XDG_CACHE_HOME="$APPDIR/usr/cache"
  mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME"
  # Use the bundled python to ensure correct environment
  BPY="$APPDIR/usr/python/bin/python"
  if [[ "$INCLUDE_DEMUCS" == "1" ]]; then
  "$BPY" - <<'PY' || true
import os
os.environ['TORCH_HOME']=os.environ.get('TORCH_HOME','')
try:
  from demucs.pretrained import get_model
  for name in ["htdemucs", "htdemucs_6s", "htdemucs_ft"]:
    try:
      get_model(name)
      print("[prewarm] demucs model cached:", name)
    except Exception as e:
      print("[prewarm] demucs skip", name, e)
except Exception as e:
  print("[prewarm] demucs unavailable:", e)
PY
  fi
  if [[ "$INCLUDE_OPENUNMIX" == "1" ]]; then
  "$BPY" - <<'PY' || true
import os, torch
os.environ['TORCH_HOME']=os.environ.get('TORCH_HOME','')
try:
  from openunmix import predict
  x = torch.zeros(1,2,44100)
  predict.separate(x, rate=44100)
  print("[prewarm] open-unmix weights cached")
except Exception as e:
  print("[prewarm] open-unmix skip:", e)
PY
  fi
  if [[ "$INCLUDE_TORCHCREPE" == "1" ]]; then
  "$BPY" - <<'PY' || true
try:
  import torch, torchcrepe
  audio = torch.zeros(1,1,16000)
  _ = torchcrepe.predict(audio, 16000, hop_length=80, fmin=50, fmax=550, model='full')
  print("[prewarm] torchcrepe cached")
except Exception as e:
  print("[prewarm] torchcrepe skip:", e)
PY
  fi
  if [[ "$INCLUDE_AUDIOSR" == "1" ]]; then
  "$BPY" - <<'PY' || true
try:
  from audiosr import build_model, super_resolution
  m = build_model('basic')
  import numpy as np
  x = np.zeros(8000, dtype=np.float32)
  _ = super_resolution(m, x, sr=16000, target_sr=32000, progress=False)
  print("[prewarm] audiosr cached")
except Exception as e:
  print("[prewarm] audiosr skip:", e)
PY
  fi
fi

msg "Create launcher script"
cat > "$APPDIR/usr/bin/audio-restorer" <<'SH'
#!/bin/bash
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
APPUSR="$(cd "$HERE/.." && pwd)"
# Use bundled Python runtime
PY="$APPUSR/python/bin/python"
# Direct caches to bundled dirs (no internet on first run)
export TORCH_HOME="$APPUSR/models/torch"
export XDG_CACHE_HOME="$APPUSR/cache"
export MPLCONFIGDIR="$APPUSR/cache/mpl"
# Make sure our usr/bin is preferred (ffmpeg, etc.)
export PATH="$APPUSR/bin:$PATH"
# User overlay for additional packages (installed via audio-restorer-pip)
OVERLAY_DEFAULT="${XDG_DATA_HOME:-$HOME/.local/share}/open-audio-stem-restorer/site-packages"
if [[ -d "$OVERLAY_DEFAULT" ]]; then
  export PYTHONPATH="$OVERLAY_DEFAULT:$PYTHONPATH"
fi
# Keep bundled source discoverable
export PYTHONPATH="$APPUSR:$PYTHONPATH"
exec "$PY" "$APPUSR/run.py" "$@"
SH
chmod +x "$APPDIR/usr/bin/audio-restorer"

# Helper to install extra Python packages into a writable overlay
cat > "$APPDIR/usr/bin/audio-restorer-pip" <<'SH'
#!/bin/bash
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
APPUSR="$(cd "$HERE/.." && pwd)"
PY="$APPUSR/python/bin/python"
OVERLAY_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/open-audio-stem-restorer/site-packages"
mkdir -p "$OVERLAY_DIR"
echo "[overlay] Installing to: $OVERLAY_DIR" >&2
exec "$PY" -m pip install --no-warn-script-location --upgrade --target "$OVERLAY_DIR" "$@"
SH
chmod +x "$APPDIR/usr/bin/audio-restorer-pip"

msg "Create desktop entry"
cat > "$APPDIR/usr/share/applications/audio-restorer.desktop" <<'DESK'
[Desktop Entry]
Type=Application
Name=Audio Restorer
Exec=audio-restorer
Icon=audio-restorer
Categories=AudioVideo;Audio;Music;Utility;
Terminal=false
DESK

# Also place a top-level desktop file for appimagetool to detect
cp "$APPDIR/usr/share/applications/audio-restorer.desktop" "$APPDIR/audio-restorer.desktop"

msg "Create AppRun"
cat > "$APPDIR/AppRun" <<'AR'
#!/bin/bash
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
export PYTHONPATH="$HERE/usr/lib:$PYTHONPATH"
# Ensure bundled tools like ffmpeg are preferred
export PATH="$HERE/usr/bin:$PATH"

# Support subcommand to install extra packages into user overlay
if [[ "${1-}" == "audio-restorer-pip" ]]; then
  shift
  exec "$HERE/usr/bin/audio-restorer-pip" "$@"
fi

exec "$HERE/usr/bin/audio-restorer" "$@"
AR
chmod +x "$APPDIR/AppRun"

ICON_SRC="$ROOT_DIR/resources/icon.svg"
if command -v convert >/dev/null 2>&1 && [ -f "$ICON_SRC" ]; then
  msg "Render icon"
  convert -background none -resize 256x256 "$ICON_SRC" "$APPDIR/usr/share/icons/hicolor/256x256/apps/audio-restorer.png" || true
fi

# Provide a top-level icon as well (fallback for some desktops/builders)
if [ -f "$APPDIR/usr/share/icons/hicolor/256x256/apps/audio-restorer.png" ]; then
  cp "$APPDIR/usr/share/icons/hicolor/256x256/apps/audio-restorer.png" "$APPDIR/audio-restorer.png" || true
fi

VERSION=$(python -c 'import pathlib; p=pathlib.Path("src/audio_restorer/__init__.py"); import re; m=re.search(r"__version__\s*=\s*\"([^\"]+)\"", p.read_text()); print(m.group(1) if m else "0.0.0")')

# Optionally bundle ffmpeg if requested (do this before packing)
if [[ "${BUNDLE_FFMPEG:-0}" == "1" ]]; then
  msg "Bundling ffmpeg (you are responsible for choosing an LGPL-compatible build)"
  mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/licenses"
  FFMPEG_SRC="${FFMPEG_PATH:-}"
  if [[ -z "$FFMPEG_SRC" ]]; then
    if command -v ffmpeg >/dev/null 2>&1; then
      FFMPEG_SRC="$(command -v ffmpeg)"
    fi
  fi
  if [[ -n "${FFMPEG_SRC}" && -x "${FFMPEG_SRC}" ]]; then
    cp -f "${FFMPEG_SRC}" "$APPDIR/usr/bin/ffmpeg"
    chmod +x "$APPDIR/usr/bin/ffmpeg"
    # Try to copy license/copyright information from common locations (Debian/Ubuntu)
    for CAND in \
      /usr/share/doc/ffmpeg/copyright \
      /usr/share/licenses/ffmpeg/* \
      /usr/share/doc/FFmpeg/copyright; do
      if [[ -f "$CAND" ]]; then
        cp -f "$CAND" "$APPDIR/usr/licenses/ffmpeg.COPYRIGHT" && break
      fi
    done
    # Always include a small notice with links
    cat > "$APPDIR/usr/licenses/NOTICE-FFMPEG.txt" << 'FFN'
This AppImage bundles an ffmpeg executable.

IMPORTANT: Ensure the bundled ffmpeg build is license-compatible (LGPL/GPL) with your distribution scenario.
For LGPL compliance, prefer an ffmpeg build configured without GPL-only components.

References:
  - FFmpeg Licensing: https://ffmpeg.org/legal.html
  - FFmpeg Project:   https://ffmpeg.org/

If you redistribute this AppImage, include this NOTICE and upstream copyright/license texts.
FFN
    msg "ffmpeg bundled from: ${FFMPEG_SRC}"
  else
    msg "BUNDLE_FFMPEG=1 requested but ffmpeg not found and FFMPEG_PATH not set. Skipping."
  fi
fi

if command -v appimagetool >/dev/null 2>&1; then
  msg "Build AppImage"
  APPIMAGE_OUT="$DIST_DIR/AudioRestorer-${VERSION}-${ARCH}.AppImage"
  appimagetool "$APPDIR" "$APPIMAGE_OUT"
  msg "Built $APPIMAGE_OUT"
else
  msg "appimagetool not found. AppDir prepared at $APPDIR. Install appimagetool to pack AppImage."
fi

msg "Done"
