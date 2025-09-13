#!/usr/bin/env bash
set -euo pipefail

# Build an AppImage for Audio Restorer, bundling Python deps.
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

# Optional extras: uncomment to bundle DL components (increases size significantly)
# pip install pyloudnorm demucs open-unmix spleeter torch torchcrepe audiosr

# Optionally pre-warm model caches here (no-op by default)
python - <<'PY'
try:
    import demucs  # noqa
except Exception:
    pass
PY

SITEPKG=$(python -c 'import site; print(site.getsitepackages()[0])')

msg "Prepare AppDir layout"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/lib" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps" "$APPDIR/usr/src"

msg "Copy site-packages"
cp -a "$SITEPKG" "$APPDIR/usr/lib/"

msg "Copy project sources and docs"
cp -a "$ROOT_DIR/src" "$APPDIR/usr/"
cp -a "$ROOT_DIR/run.py" "$APPDIR/usr/"
cp -a "$ROOT_DIR/LICENSE" "$ROOT_DIR/README.md" "$ROOT_DIR/THIRD_PARTY_LICENSES.md" "$APPDIR/usr/" || true
cp -a "$ROOT_DIR/resources" "$APPDIR/usr/" || true

msg "Create launcher script"
cat > "$APPDIR/usr/bin/audio-restorer" <<'SH'
#!/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"
export PYTHONPATH="$HERE/../lib:$(python3 -c 'import site; print(site.getsitepackages()[0])'):$PYTHONPATH"
exec /usr/bin/env python3 "$HERE/../run.py" "$@"
SH
chmod +x "$APPDIR/usr/bin/audio-restorer"

msg "Create desktop entry"
cat > "$APPDIR/usr/share/applications/audio-restorer.desktop" <<'DESK'
[Desktop Entry]
Type=Application
Name=Audio Restorer
Exec=audio-restorer
Icon=audio-restorer
Categories=Audio;Music;Utility;
Terminal=false
DESK

# Also place a top-level desktop file for appimagetool to detect
cp "$APPDIR/usr/share/applications/audio-restorer.desktop" "$APPDIR/audio-restorer.desktop"

msg "Create AppRun"
cat > "$APPDIR/AppRun" <<'AR'
#!/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"
export PYTHONPATH="$HERE/usr/lib:$PYTHONPATH"
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

if command -v appimagetool >/dev/null 2>&1; then
  msg "Build AppImage"
  APPIMAGE_OUT="$DIST_DIR/AudioRestorer-${VERSION}-${ARCH}.AppImage"
  appimagetool "$APPDIR" "$APPIMAGE_OUT"
  msg "Built $APPIMAGE_OUT"
else
  msg "appimagetool not found. AppDir prepared at $APPDIR. Install appimagetool to pack AppImage."
fi

msg "Done"
