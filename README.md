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
Ein AppImage inkl. Python-Abhängigkeiten lässt sich mit dem bereitgestellten Skript bauen. Optional können auch DL‑Pakete (Demucs/Open‑Unmix/AudioSR/torchcrepe) eingebunden werden.

Schnellstart (Linux):
```bash
# optional: Abhängigkeiten für Builder
# sudo apt-get install -y imagemagick

bash scripts/build_appimage.sh
```

Das Skript:
- erstellt eine isolierte Build‑Venv,
- installiert Basis‑ und optional gewünschte Pakete,
- kopiert Site‑Packages und Projektcode nach `AppDir/usr`,
- legt Desktop‑Datei, Icon und Wrapper an,
- packt mit `appimagetool` (falls vorhanden) zum AppImage.

Das fertige Artefakt liegt als `dist/AudioRestorer-<version>-<arch>.AppImage` vor. Falls `appimagetool` fehlt, erzeugt das Skript ein gebrauchsfertiges `AppDir/` und gibt Hinweise zum Nachinstallieren.

Lizenzhinweise: PySide6/Qt wird dynamisch verwendet (LGPLv3). Bitte legen Sie bei Distribution `LICENSE`, `THIRD_PARTY_LICENSES.md` sowie relevante Upstream‑Lizenztexte bei und beachten Sie ggf. ffmpeg‑Lizenzbedingungen (je nach Build LGPL/GPL).
