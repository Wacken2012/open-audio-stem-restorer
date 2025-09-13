# Audio Restorer (Stem-based)

GUI-Tool zur Restaurierung historischer Aufnahmen (Schellack, Tape, Lo-Fi) und moderner Musik mittels Stem-Separation & mehrstufiger DSP-/KI‑Pipeline.

## Funktionsumfang (aktueller Stand)
Kernfunktionen:
- Laden & Export von Audio (WAV, FLAC, MP3) – MP3 in 320 kbps via `ffmpeg` (Fallback WAV)
- Mehrere Stem-Separation-Backends:
	- Eingebaut: HPSS (harmonic/percussive + Begleitung)
	- Optional: Demucs (mehrere Modelle), Open-Unmix, Spleeter (bereit für Integration)
- Individuelle Stem-Auswahl (vocals, drums, bass, other, harmonic, percussive, accompaniment)
- Option: Stems einzeln restaurieren (präzisere Korrektur, langsamer) oder gemischt verarbeiten

Restaurierung & Enhancement:
- Spektrales Denoise (Noise Profil aus erster Passage, Stärke regelbar)
- Declick & Decrackle (Median-/Band-Glätter)
- De-Clip (nicht-ML, geclipte Peaks rekonstruieren, Stärke regelbar)
- Transienten-Enhancement (Attack-gewichtete Hochton-/Klarheitsanhebung)
- Codec-Artefakt-Unterdrückung (zeitliche Median-Glättung im Spektrum)
- EQ-Presets: `shellac`, `tape`, `none`
- Air/Höhen-Anreicherung (HF-Shelf + sanfte harmonische Anreicherung)
- Adaptive Loudness-Normalisierung (integrierte LUFS + optionale zeitliche Glättung)
- Klarheits-Stabilisierung (dynamischer HF-Tilt zur Konstanz der Präsenz)
- Wow/Flutter-Reduktion (leichte Tonhöhenschwankungs-Glättung)
	- Pitch-Engine wählbar: `torch` (Hilbert-basiert, ohne Zusatzabhängigkeiten) oder `crepe` (mit `torchcrepe`)
- Stereo-Widening aus Mono (HF‑seitige Dekorrelation + Micro-Delay)
- Optionale generative Super-Resolution / High-Frequency-Reconstruction (AudioSR – falls installiert)
	- Engine: `audiosr`
	- Modi: `full` (Vollband) oder `highs` (nur Hochtonergänzung)
	- Mix-Regler (Dry/Wet) + Ziel-Sample-Rate (Standard 48 kHz, bis 96 kHz)
- Finale Normalisierung & Sicherheits-Limiter

Workflow / GUI:
- PySide6 GUI (progressive Statusmeldungen, abbrechbarer Worker-Thread)
- Per-Stem Export (optional) in wählbares Verzeichnis
- Fortschrittsbalken & detailierte Status-Texte
 - Optionaler Debug-Log (Checkbox „Debug-Log“): schreibt SR/Längen-Infos nach `work/debug.log`

Fallback & Robustheit:
- Alle optionalen KI-/Pitch-/Loudness-Funktionen haben sichere Fallbacks (kein Absturz bei fehlenden Paketen)
- Ohne optionale Pakete läuft Basis-Restaurierung (HPSS + klassische DSP) vollständig

## Optional installierbare Pakete
| Zweck | Paket(e) | Nutzen |
|-------|----------|--------|
| Erweiterte Lautheit (LUFS) | `pyloudnorm` | Präzise integrierte & adaptive Loudness |
| Fortgeschrittenes Pitch-Tracking | `torch`, `torchcrepe` | Genauere Wow/Flutter-Reduktion |
| Generative Super-Resolution | `audiosr`, `torch` | Hochfrequenz-/Bandbreitenrekonstruktion |
| Zusätzliche Separation | `demucs`, `open-unmix`, `spleeter` | Bessere/alternative Stem-Qualität |

Installation (optional), z.B. nur Loudness & Demucs:
```bash
pip install pyloudnorm demucs
```

## Systemanforderungen
Die tatsächliche Performance hängt stark von Modellwahl (z.B. Demucs) und optionalen KI-Funktionen ab.

### Minimal ("Classic Mode" ohne Deep-Learning, nur HPSS + klassische DSP)
- CPU: 2‑Kern x86_64 oder ARM (≥ 1.2 GHz)
- RAM: 2 GB (1 GB frei während Lauf)
- Speicher: ~300 MB frei
- Python: 3.10 – 3.12
- ffmpeg: Für MP3-Export (sonst nur WAV)
- Keine GPU nötig

### Empfohlen (mit Demucs / Open-Unmix & Loudness / Wow/Flutter)
- CPU: 4‑Kern (≥ 2.0 GHz) oder Apple Silicon M‑Serie (M1 oder besser)
- RAM: 8 GB (Demucs Modelle laden mehrere hundert MB; Batch-Verarbeitung profitiert)
- Python: 3.11 oder 3.12
- Optional GPU (CUDA ≥ 11) für Torch-Modelle (Demucs / AudioSR / torchcrepe) → starke Beschleunigung

### Generative Enhancement (AudioSR) – Empfohlen
- GPU: NVIDIA (≥6 GB VRAM) für vertretbare Laufzeiten; CPU-Only möglich aber langsam
- RAM: ≥ 8–12 GB

### Raspberry Pi / ARM Single Board
- Getestet Szenario: Raspberry Pi 4 / 5 (4–8 GB RAM)
- Empfohlen: Classic Mode (HPSS + Denoise/Declick/EQ/Air/Loudness/Clarity/Widen)
- Nicht empfohlen / sehr langsam: Demucs (kein offizielles vorgebautes Wheel mit AVX), AudioSR, torchcrepe
- Deaktivieren Sie generative & tiefe Modelle im GUI (Engine = `none` / Mix = 0)

### Apple Silicon (macOS)
- Läuft nativ (ARM64). Torch Wheels für M‑Serie verfügbar → gute Performance.
- ffmpeg via Homebrew: `brew install ffmpeg`

### Windows
- Getestet ab Windows 10/11 64‑bit
- Visual C++ Build Tools empfohlen (falls Torch nachinstalliert wird)
- ffmpeg in PATH (z.B. via `choco install ffmpeg`)

### Abhängigkeiten zur Laufzeit (Minimum)
- Python-Pakete aus `requirements.txt`
- Optional installierte Deep-Learning-Pakete nur wenn Features aktiv

## Installation (Basis)
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install -r requirements.txt
```

Optional (alles):
```bash
pip install pyloudnorm demucs open-unmix spleeter torch torchcrepe audiosr
```

Hinweis: Große Modelle (Demucs, AudioSR) können beim ersten Start automatisch Gewichte herunterladen.

## Start
```bash
python run.py
```

Optional: Debug-Log aktivieren (GUI → Checkbox „Debug-Log“). Der Log liegt unter `work/debug.log` und enthält u.a. Sample-Rate- und Längenangaben, hilfreich für Fehleranalysen (z.B. Tempo-/Längenprobleme).

## Nutzungshinweise
1. Audio laden
2. Backend & Stems wählen
3. Optional: Per-Stem Restaurierung aktivieren (langsamer, präziser)
4. Denoise/Declick/EQ/Air einstellen
5. Optional: Generative Enhancement aktivieren (Engine `audiosr`, Mix > 0)
6. Ziel-Lautheit & Glättung wählen
7. Clarity / Wow/Flutter / Stereo-Widening nach Bedarf
8. Start → Fortschritt beobachten → Exportdialog (WAV oder MP3)
9. Optional: Stems zusätzlich in Ordner schreiben lassen

## Verzeichnisstruktur
- `src/audio_restorer/gui/` – GUI (PySide6)
- `src/audio_restorer/separation/` – Backend-Registry & Implementierungen (HPSS, Hooks für Demucs/Open-Unmix/Spleeter)
- `src/audio_restorer/restoration/` – DSP & Enhancement Funktionen (Denoise, Declick, EQ, Loudness, Clarity, Wow/Flutter, Generativ)
 - `scripts/test_length_invariance.py` – Längeninvarianz-Schnelltest

## Bekannte Grenzen
- Kein destruktives Speichern – immer neuer Export
- Generative Modelle können Artefakte erzeugen; Mix-Regler konservativ nutzen
- Wow/Flutter-Reduktion ist heuristisch (kein vollständiges Pitch-Reconstructing)

## Fehlerbehebung (Kurz)
| Problem | Ursache | Lösung |
|---------|---------|--------|
| MP3 Export fehlt | ffmpeg nicht gefunden | ffmpeg installieren & in PATH |
| Sehr langsam | Deep-Learning Backend aktiv | Backend wechseln zu HPSS / Features deaktivieren |
| Kein Lautheitsabgleich | `pyloudnorm` fehlt | `pip install pyloudnorm` |
| Keine generative Optionen | `audiosr` fehlt | `pip install audiosr torch` |
| Absturz bei Pitch Engine `crepe` | `torchcrepe` fehlt | `pip install torchcrepe` oder Engine auf `torch` stellen |

## Lizenz
MIT (siehe `LICENSE`). Drittanbieter-Lizenzen: `THIRD_PARTY_LICENSES.md`.

## Distribution / AppImage
Das Skript `scripts/build_appimage.sh` erzeugt ein AppImage. Es kann (a) ein leichtes AppImage mit System‑Python oder (b) ein vollständig offline lauffähiges AppImage mit gebündeltem Python‑Runtime, allen Paketen und vorab heruntergeladenen ML‑Modellen bauen.

Schnellstart (Linux):
```bash
# optional: Abhängigkeiten für Builder
# sudo apt-get install -y imagemagick

# 1) Basis-AppImage (ohne schwere ML-Extras)
bash scripts/build_appimage.sh

# 2) Vollständig offline AppImage (inkl. Python-Runtime & ML-Extras)
#    - Bundelt eine virtuelle Python-Umgebung in AppDir/usr/python
#    - Installiert Torch (CPU), Torchaudio, Demucs, Open-Unmix, TorchCrepe, pyloudnorm
#    - Lädt Modellgewichte vor und legt sie in AppDir/usr/models bzw. AppDir/usr/cache ab
INCLUDE_EXTRAS=1 PREWARM=1 bash scripts/build_appimage.sh

# Optional: ffmpeg ins AppImage bündeln (achten Sie auf Lizenzkompatibilität)
# Variante A: systemweites ffmpeg verwenden
BUNDLE_FFMPEG=1 bash scripts/build_appimage.sh
# Variante B: expliziten Pfad angeben
FFMPEG_PATH=/opt/ffmpeg/bin/ffmpeg BUNDLE_FFMPEG=1 bash scripts/build_appimage.sh
```

Weitere Schalter:
- `INCLUDE_DEMUCS=1` (implizit bei `INCLUDE_EXTRAS=1`) – Demucs installieren und Modelle vorladen
- `INCLUDE_OPENUNMIX=1` (implizit bei `INCLUDE_EXTRAS=1`) – Open‑Unmix installieren/vorwärmen
- `INCLUDE_TORCHCREPE=1` (implizit bei `INCLUDE_EXTRAS=1`) – TorchCrepe installieren/vorwärmen
- `INCLUDE_SPLEETER=1` – Spleeter (TensorFlow‑basiert, sehr groß) zusätzlich einbinden
- `INCLUDE_AUDIOSR=1` – AudioSR (sehr groß/langsam ohne GPU) zusätzlich einbinden
- `TORCH_VERSION` / `TORCHAUDIO_VERSION` – exakte Torch/Torchaudio Versionen für CPU‑Wheels setzen (Default: 2.3.1)

Was das Skript tut:
- erstellt eine isolierte Build‑Venv,
- installiert Basis‑ und optionale Pakete (Torch CPU‑Wheels via `download.pytorch.org`),
- bündelt die komplette Venv als Laufzeit nach `AppDir/usr/python`,
- lädt (optional) ML‑Modelle vor und speichert sie unter `AppDir/usr/models`/`AppDir/usr/cache`,
- legt Desktop‑Datei, Icon und Wrapper an,
- packt mit `appimagetool` (falls vorhanden) zum AppImage.

Laufzeit: Der Starter setzt `TORCH_HOME` und `XDG_CACHE_HOME` auf die gebündelten Ordner, sodass beim ersten Start keine Internetverbindung nötig ist. Das AppImage bevorzugt außerdem gebündelte Tools (z.B. `ffmpeg`, wenn eingeschlossen).

Artefakt: `dist/AudioRestorer-<version>-<arch>.AppImage`. Falls `appimagetool` fehlt, bleibt ein gebrauchsfertiges `AppDir/` zurück.

Größe & CPU‑Support:
- Mit Extras und vorab geladenen Modellen kann das AppImage mehrere hundert MB groß werden.
- Torch wird als CPU‑Build eingebunden. GPU‑Beschleunigung ist in diesem Artefakt nicht enthalten.

Lizenzhinweise: PySide6/Qt wird dynamisch verwendet (LGPLv3). Bitte legen Sie bei Distribution `LICENSE`, `THIRD_PARTY_LICENSES.md` sowie relevante Upstream‑Lizenztexte bei. Für ffmpeg gilt: je nach Build ist LGPL oder GPL maßgeblich. Wenn Sie ffmpeg bündeln (BUNDLE_FFMPEG=1), stellen Sie sicher, dass der verwendete Build zur gewünschten Weitergabe passt (für LGPL: ohne GPL‑only Komponenten). Das Skript legt dafür eine `usr/licenses/NOTICE-FFMPEG.txt` im AppDir an; ergänzen Sie ggf. die vollständigen Lizenztexte.

### Nachinstallation nicht gebündelter Pakete (Overlay)
Das AppImage unterstützt eine Benutzer‑Overlay‑Installation zusätzlicher Python‑Pakete, ohne das AppImage selbst zu ändern. Verwenden Sie den mitgelieferten Helfer:

```bash
# Beispiel: AudioSR nachinstallieren
./AudioRestorer-<version>-<arch>.AppImage --appimage-extract-and-run audio-restorer-pip audiosr torchvision

# oder (AppImage gemountet im Hintergrund von Desktop) einfach:
audio-restorer-pip audiosr torchvision
```

Die Pakete landen unter `${XDG_DATA_HOME:-$HOME/.local/share}/open-audio-stem-restorer/site-packages` und werden beim Start automatisch via PYTHONPATH verwendet. Hinweis: Manche Pakete erfordern bestimmte Python/Numpy‑Versionen. AudioSR benötigt i.d.R. Python 3.10/3.11 mit `numpy<=1.23.x`. Falls die Installation fehlschlägt, prüfen Sie kompatible Versionen oder verwenden Sie die offline‑gebündelte Alternative ohne AudioSR.

## Tests / Qualitätssicherung
Schneller Längeninvarianz-Test (stellt sicher, dass Export/Verarbeitung die Länge nicht verändert):

```bash
python scripts/test_length_invariance.py
```

Das Skript generiert synthetisches Audio für mehrere Sample-Raten und Laufzeiten, nutzt verfügbare Backends (HPSS, optional Demucs/Open‑Unmix) und prüft, dass die Pipeline-Ausgabe dieselbe Länge wie der Eingang besitzt – sowohl im Mono‑ als auch im Stereo‑Fall (mit Widening). Bei fehlenden optionalen Paketen werden entsprechende Backends übersprungen.
