# Third-Party Licenses

Dieses Projekt verwendet optionale und zwingende Abhängigkeiten. Unten eine Übersicht (ohne Gewähr; bitte jeweilige Upstream-Repos für aktuellste Lizenztexte prüfen).

| Paket | Zweck | Lizenz (Kurz) | Upstream |
|-------|-------|--------------|----------|
| numpy | Numerik | BSD 3-Clause | https://github.com/numpy/numpy |
| scipy | Signalverarbeitung | BSD 3-Clause | https://github.com/scipy/scipy |
| librosa | Audio Features / DSP | ISC | https://github.com/librosa/librosa |
| soundfile | WAV/FLAC IO (libsndfile) | BSD (Python), LGPL (libsndfile) | https://github.com/bastibe/python-soundfile |
| PySide6 | GUI (Qt for Python) | LGPLv3 / GPL / Kommerziell (Qt) | https://www.qt.io/ | 
| noisereduce | Denoise | MIT | https://github.com/timsainb/noisereduce |
| pyloudnorm (optional) | Loudness (EBU R128) | MIT | https://github.com/csteinmetz1/pyloudnorm |
| demucs (optional) | Separation DL | MIT | https://github.com/facebookresearch/demucs |
| open-unmix (optional) | Separation DL | MIT | https://github.com/sigsep/open-unmix-pytorch |
| spleeter (optional) | Separation DL | MIT | https://github.com/deezer/spleeter |
| torch / torchcrepe (optional) | DL / Pitch | BSD-style / MIT | https://pytorch.org / https://github.com/maxrmorrison/torchcrepe |
| audiosr (optional) | Generative SR | MIT | https://github.com/Audio-AGI/AudioSR |
| ffmpeg (extern) | Transcoding (MP3) | LGPL/GPL (je nach Build) | https://ffmpeg.org |

Weitere interne DSP-Primitiven (De-Clip, Transienten-Enhancement, Codec-Artefakt-Unterdrückung) sind Eigenimplementierungen (MIT, Teil dieses Repos). Sie orientieren sich an üblichen Signalverarbeitungsmustern (Inpainting/Glättung, Attack-Gewichtung, spektrale Medianfilter) ohne Code aus fremden Projekten zu übernehmen.

Hinweis zu PySide6 / Qt: Nutzung unter den Bedingungen der LGPLv3 (dynamisches Linking) vorgesehen. Eigenständige Weitergabe eines statisch gelinkten Builds würde zusätzliche Anforderungen auslösen.

Weitergabe-Hinweise:
- Legen Sie `LICENSE` und diese Datei bei.
- Falls `ffmpeg` mitgeliefert wird (Option `BUNDLE_FFMPEG=1` im Build-Skript): dessen Lizenz beachten (LGPL/GPL je nach Build-Optionen). Für LGPL-Verteilung ffmpeg ohne GPL-only Komponenten wählen. Fügen Sie die upstream Copyright/Lizenztexte bei.
- Falls Torch/Demucs/AudioSR/torchcrepe gebündelt werden (`INCLUDE_EXTRAS=1`): jeweilige Lizenzen der Wheels/Modelle beilegen. Vorsicht: Gewichte/Modelle können gesonderten Nutzungsbedingungen unterliegen.

Dieses Projekt selbst steht unter MIT (siehe `LICENSE`). Bei Distribution zusammen mit optionalen Modellen/Binaries müssen deren Lizenzen beigelegt werden.
