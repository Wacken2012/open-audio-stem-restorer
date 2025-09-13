from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QFileDialog,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QHBoxLayout,
    QProgressBar,
    QCheckBox,
    QSlider,
    QSpinBox,
    QMessageBox,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QSettings
from PySide6.QtGui import QIcon, QPalette
import sys
import soundfile as sf
import numpy as np

from ..separation.interface import list_backends, get_backend
from ..restoration.pipeline import restore_pipeline


class Worker(QObject):
    progress = Signal(int, str)  # percent, message
    finished = Signal(np.ndarray, dict)
    error = Signal(str)

    def __init__(
        self,
        audio: np.ndarray,
        sr: int,
        backend_name: str,
        eq: str,
        include_names: list[str],
        denoise_strength: float,
        declick_strength: float,
        declip_strength: float = 0.0,
        transient_strength: float = 0.0,
        codec_artifact_strength: float = 0.0,
        model_name: str | None = None,
        air_strength: float = 0.0,
        decrackle_strength: float = 0.0,
        process_per_stem: bool = False,
        widen_amount: float = 0.0,
        loudness_target_lufs: float = -16.0,
        loudness_smooth: float = 0.5,
        clarity_strength: float = 0.0,
        wowflutter_strength: float = 0.0,
        wowflutter_engine: str = "torch",
        gen_engine: str = "none",
        gen_mode: str = "full",
        gen_mix: float = 0.0,
        gen_target_sr: int = 48000,
    ):
        super().__init__()
        self.audio = audio
        self.sr = sr
        self.backend_name = backend_name
        self.eq = eq
        self.include_names = include_names
        self.denoise_strength = denoise_strength
        self.declick_strength = declick_strength
        self.decrackle_strength = decrackle_strength
        self.declip_strength = declip_strength
        self.transient_strength = transient_strength
        self.codec_artifact_strength = codec_artifact_strength
        self.model_name = model_name
        self.air_strength = air_strength
        self.process_per_stem = process_per_stem
        self.widen_amount = widen_amount
        self.loudness_target_lufs = loudness_target_lufs
        self.loudness_smooth = loudness_smooth
        self.clarity_strength = clarity_strength
        self.wowflutter_strength = wowflutter_strength
        self.wowflutter_engine = wowflutter_engine
        self.gen_engine = gen_engine
        self.gen_mode = gen_mode
        self.gen_mix = float(gen_mix)
        self.gen_target_sr = int(gen_target_sr)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            if self._cancel:
                return
            self.progress.emit(5, f"Trenne Stems mit {self.backend_name}…")
            backend = get_backend(self.backend_name)
            if self.model_name and hasattr(backend, "model_name"):
                try:
                    setattr(backend, "model_name", self.model_name)
                except Exception:
                    pass
            sep = getattr(backend, "separate", None)
            if sep is None:
                raise RuntimeError("Ausgewähltes Backend unterstützt 'separate' nicht.")
            stems = sep(self.audio, self.sr)

            if self._cancel:
                return
            self.progress.emit(30, "Restauriere…")

            def on_prog(p, msg):
                p2 = int(30 + 0.65 * p)
                self.progress.emit(p2, msg)

            def is_canceled():
                return self._cancel

            out = restore_pipeline(
                stems,
                self.sr,
                eq=self.eq,
                include=self.include_names,
                denoise_strength=self.denoise_strength,
                declick_strength=self.declick_strength,
                decrackle_strength=self.decrackle_strength,
                declip_strength=self.declip_strength,
                transient_strength=self.transient_strength,
                codec_artifact_strength=self.codec_artifact_strength,
                air_strength=self.air_strength,
                progress=on_prog,
                canceled=is_canceled,
                process_per_stem=self.process_per_stem,
                widen_amount=self.widen_amount,
                loudness_target_lufs=self.loudness_target_lufs,
                loudness_smooth=self.loudness_smooth,
                clarity_strength=self.clarity_strength,
                wowflutter_strength=self.wowflutter_strength,
                wowflutter_engine=self.wowflutter_engine,
                gen_engine=self.gen_engine,
                gen_mode=self.gen_mode,
                gen_mix=self.gen_mix,
                gen_target_sr=self.gen_target_sr,
            )

            if self._cancel:
                return
            self.progress.emit(95, "Abschließen…")
            self.finished.emit(out, stems)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Restorer (Stem-based)")
        try:
            from pathlib import Path
            root = Path(__file__).resolve().parents[3]
            icon_svg = root / "resources" / "icon.svg"
            icon_png = root / "resources" / "icon_256.png"
            if icon_png.exists():
                self.setWindowIcon(QIcon(str(icon_png)))
            elif icon_svg.exists():
                self.setWindowIcon(QIcon(str(icon_svg)))
        except Exception:
            pass

        self.audio_path = None
        self.sr = 44100
        self.audio = None
        self.worker = None
        self.worker_thread = None
        self.settings = QSettings("AudioRestorer", "AudioRestorer")

        vbox = QVBoxLayout()

        # File selection
        file_row = QHBoxLayout()
        self.file_label = QLabel("Keine Datei geladen")
        btn_load = QPushButton("Audio öffnen…")
        btn_load.clicked.connect(self.load_audio)
        file_row.addWidget(self.file_label)
        file_row.addWidget(btn_load)
        vbox.addLayout(file_row)

        # Backend selection
        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Stem-Backend:"))
        self.backend_cb = QComboBox()
        self.backend_cb.addItems(list_backends())
        for i in range(self.backend_cb.count()):
            if "Open-Unmix" in self.backend_cb.itemText(i):
                self.backend_cb.setCurrentIndex(i)
                break
        backend_row.addWidget(self.backend_cb)
        vbox.addLayout(backend_row)

        # Demucs model selector
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Demucs Modell:"))
        self.model_cb = QComboBox()
        self.model_cb.addItems(["htdemucs", "htdemucs_6s", "htdemucs_ft"])
        model_row.addWidget(self.model_cb)
        vbox.addLayout(model_row)

        def _update_model_enabled(txt: str):
            demucs_active = txt.startswith("Demucs") and not txt.endswith("(installieren)")
            self.model_cb.setEnabled(demucs_active)

        _update_model_enabled(self.backend_cb.currentText())
        self.backend_cb.currentTextChanged.connect(_update_model_enabled)

        # EQ preset
        eq_row = QHBoxLayout()
        eq_row.addWidget(QLabel("EQ-Preset:"))
        self.eq_cb = QComboBox()
        self.eq_cb.addItems(["shellac", "tape", "none"])
        eq_row.addWidget(self.eq_cb)
        vbox.addLayout(eq_row)

        # Stem selection
        stems_row = QHBoxLayout()
        stems_row.addWidget(QLabel("Stems:"))
        self.cb_voc = QCheckBox("vocals"); self.cb_voc.setChecked(True)
        self.cb_drm = QCheckBox("drums"); self.cb_drm.setChecked(True)
        self.cb_bas = QCheckBox("bass"); self.cb_bas.setChecked(True)
        self.cb_oth = QCheckBox("other"); self.cb_oth.setChecked(True)
        self.cb_har = QCheckBox("harmonic"); self.cb_har.setChecked(True)
        self.cb_per = QCheckBox("percussive"); self.cb_per.setChecked(True)
        self.cb_acc = QCheckBox("accompaniment"); self.cb_acc.setChecked(True)
        for w in [self.cb_voc, self.cb_drm, self.cb_bas, self.cb_oth, self.cb_har, self.cb_per, self.cb_acc]:
            stems_row.addWidget(w)
        vbox.addLayout(stems_row)

        # Denoise slider
        den_row = QHBoxLayout()
        den_row.addWidget(QLabel("Denoise:"))
        self.sl_denoise = QSlider(Qt.Orientation.Horizontal)
        self.sl_denoise.setRange(0, 100)
        self.sl_denoise.setValue(50)
        den_row.addWidget(self.sl_denoise)
        vbox.addLayout(den_row)

        # Declick slider
        dec_row = QHBoxLayout()
        dec_row.addWidget(QLabel("Declick:"))
        self.sl_declick = QSlider(Qt.Orientation.Horizontal)
        self.sl_declick.setRange(0, 100)
        self.sl_declick.setValue(50)
        dec_row.addWidget(self.sl_declick)
        vbox.addLayout(dec_row)

        # Decrackle slider (fine crackle)
        dcr_row = QHBoxLayout()
        dcr_row.addWidget(QLabel("Decrackle:"))
        self.sl_decrackle = QSlider(Qt.Orientation.Horizontal)
        self.sl_decrackle.setRange(0, 100)
        self.sl_decrackle.setValue(0)
        dcr_row.addWidget(self.sl_decrackle)
        vbox.addLayout(dcr_row)

        # De-Clip
        dclip_row = QHBoxLayout()
        dclip_row.addWidget(QLabel("De-Clip:"))
        self.sl_declip = QSlider(Qt.Orientation.Horizontal)
        self.sl_declip.setRange(0, 100)
        self.sl_declip.setValue(0)
        dclip_row.addWidget(self.sl_declip)
        vbox.addLayout(dclip_row)

        # Transienten
        trans_row = QHBoxLayout()
        trans_row.addWidget(QLabel("Transienten:"))
        self.sl_transient = QSlider(Qt.Orientation.Horizontal)
        self.sl_transient.setRange(0, 100)
        self.sl_transient.setValue(0)
        trans_row.addWidget(self.sl_transient)
        vbox.addLayout(trans_row)

        # Codec-Artefakte
        codec_row = QHBoxLayout()
        codec_row.addWidget(QLabel("Codec-Artefakte:"))
        self.sl_codec = QSlider(Qt.Orientation.Horizontal)
        self.sl_codec.setRange(0, 100)
        self.sl_codec.setValue(0)
        codec_row.addWidget(self.sl_codec)
        vbox.addLayout(codec_row)

        # Air
        air_row = QHBoxLayout()
        air_row.addWidget(QLabel("Air/Höhen:"))
        self.sl_air = QSlider(Qt.Orientation.Horizontal)
        self.sl_air.setRange(0, 100)
        self.sl_air.setValue(0)
        air_row.addWidget(self.sl_air)
        vbox.addLayout(air_row)

        # Stereo widen
        wide_row = QHBoxLayout()
        wide_row.addWidget(QLabel("Stereo-Breite:"))
        self.sl_widen = QSlider(Qt.Orientation.Horizontal)
        self.sl_widen.setRange(0, 100)
        self.sl_widen.setValue(0)
        wide_row.addWidget(self.sl_widen)
        vbox.addLayout(wide_row)

        # Loudness controls
        loud_row = QHBoxLayout()
        loud_row.addWidget(QLabel("Ziel-Lautheit (LUFS):"))
        self.sb_lufs = QSpinBox()
        self.sb_lufs.setRange(-35, -8)
        self.sb_lufs.setValue(-16)
        loud_row.addWidget(self.sb_lufs)
        loud_row.addWidget(QLabel("Glättung:"))
        self.sl_loud_smooth = QSlider(Qt.Orientation.Horizontal)
        self.sl_loud_smooth.setRange(0, 100)
        self.sl_loud_smooth.setValue(50)
        loud_row.addWidget(self.sl_loud_smooth)
        vbox.addLayout(loud_row)

        # Clarity
        clr_row = QHBoxLayout()
        clr_row.addWidget(QLabel("Klarheit-Stabilisierung:"))
        self.sl_clarity = QSlider(Qt.Orientation.Horizontal)
        self.sl_clarity.setRange(0, 100)
        self.sl_clarity.setValue(0)
        clr_row.addWidget(self.sl_clarity)
        vbox.addLayout(clr_row)

        # Wow/Flutter strength
        wf_row = QHBoxLayout()
        wf_row.addWidget(QLabel("Wow/Flutter-Reduktion:"))
        self.sl_wf = QSlider(Qt.Orientation.Horizontal)
        self.sl_wf.setRange(0, 100)
        self.sl_wf.setValue(0)
        wf_row.addWidget(self.sl_wf)
        vbox.addLayout(wf_row)

        # Wow/Flutter engine
        eng_row = QHBoxLayout()
        eng_row.addWidget(QLabel("Pitch-Engine (Wow/Flutter):"))
        self.cb_wf_engine = QComboBox()
        self.cb_wf_engine.addItems(["torch", "crepe"])
        self.cb_wf_engine.setCurrentText("torch")
        eng_row.addWidget(self.cb_wf_engine)
        vbox.addLayout(eng_row)

        # Generative enhancement controls
        gen_row1 = QHBoxLayout()
        gen_row1.addWidget(QLabel("Generativ (48 kHz):"))
        self.cb_gen_engine = QComboBox()
        self.cb_gen_engine.addItems(["none", "audiosr"])
        self.cb_gen_engine.setCurrentText("none")
        gen_row1.addWidget(self.cb_gen_engine)
        gen_row1.addWidget(QLabel("Modus:"))
        self.cb_gen_mode = QComboBox()
        self.cb_gen_mode.addItems(["full", "highs"])
        self.cb_gen_mode.setCurrentText("full")
        gen_row1.addWidget(self.cb_gen_mode)
        vbox.addLayout(gen_row1)

        gen_row2 = QHBoxLayout()
        gen_row2.addWidget(QLabel("Mix:"))
        self.sl_gen_mix = QSlider(Qt.Orientation.Horizontal)
        self.sl_gen_mix.setRange(0, 100)
        self.sl_gen_mix.setValue(0)
        gen_row2.addWidget(self.sl_gen_mix)
        gen_row2.addWidget(QLabel("Ziel-SR:"))
        self.sb_gen_sr = QSpinBox()
        self.sb_gen_sr.setRange(32000, 96000)
        self.sb_gen_sr.setSingleStep(1000)
        self.sb_gen_sr.setValue(48000)
        gen_row2.addWidget(self.sb_gen_sr)
        vbox.addLayout(gen_row2)

        # Theme & About row
        misc_row = QHBoxLayout()
        misc_row.addWidget(QLabel("Theme:"))
        self.cb_theme = QComboBox()
        self.cb_theme.addItems(["System", "Light", "Dark"])
        misc_row.addWidget(self.cb_theme)
        self.btn_about = QPushButton("Über…")
        self.btn_about.clicked.connect(self.show_about)
        misc_row.addWidget(self.btn_about)
        misc_row.addStretch(1)
        vbox.addLayout(misc_row)

        # Save stems
        save_row = QHBoxLayout()
        self.cb_save_stems = QCheckBox("Stems zusätzlich speichern")
        self.cb_save_stems.setChecked(False)
        self.btn_pick_dir = QPushButton("Ordner wählen…")
        self.btn_pick_dir.setEnabled(False)
        self.stems_dir = None

        def on_cb_save_changed(state):
            self.btn_pick_dir.setEnabled(self.cb_save_stems.isChecked())

        self.cb_save_stems.stateChanged.connect(on_cb_save_changed)

        def on_pick_dir():
            d = QFileDialog.getExistingDirectory(self, "Speicherordner wählen")
            if d:
                self.stems_dir = d

        self.btn_pick_dir.clicked.connect(on_pick_dir)
        save_row.addWidget(self.cb_save_stems)
        save_row.addWidget(self.btn_pick_dir)
        vbox.addLayout(save_row)

        # Per stem processing
        perstem_row = QHBoxLayout()
        self.cb_per_stem = QCheckBox("Stems einzeln restaurieren (langsamer)")
        self.cb_per_stem.setChecked(False)
        perstem_row.addWidget(self.cb_per_stem)
        vbox.addLayout(perstem_row)

        # Buttons
        self.status_label = QLabel("")
        btns = QHBoxLayout()
        self.btn_process = QPushButton("Restaurieren & Exportieren…")
        self.btn_process.clicked.connect(self.process)
        self.btn_cancel = QPushButton("Abbrechen")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        btns.addWidget(self.btn_process)
        btns.addWidget(self.btn_cancel)
        vbox.addLayout(btns)

        # Progress
        self.prog = QProgressBar()
        self.prog.setRange(0, 100)
        self.prog.setValue(0)
        vbox.addWidget(self.prog)
        vbox.addWidget(self.status_label)

        self.setLayout(vbox)
        self._init_tooltips()

        # After UI creation
        self.restore_settings()
        self.apply_theme(self.cb_theme.currentText())
        self.cb_theme.currentTextChanged.connect(self.apply_theme)

    def _init_tooltips(self):
        try:
            self.backend_cb.setToolTip("Auswahl des Stem-Separation-Backends (Open-Unmix empfohlen, Demucs optional)")
            self.model_cb.setToolTip("Demucs Modellvariante – nur aktiv wenn Demucs gewählt")
            self.eq_cb.setToolTip("EQ-Preset: Shellac (ältere Platten), Tape oder None")
            for cb in [self.cb_voc, self.cb_drm, self.cb_bas, self.cb_oth, self.cb_har, self.cb_per, self.cb_acc]:
                cb.setToolTip("Ein-/Ausschalten dieses Stems für Mischung/Verarbeitung")
            self.sl_denoise.setToolTip("Spektrales Rauschreduzieren – höher = stärker (kann Artefakte erzeugen)")
            self.sl_declick.setToolTip("Entfernt Klicks/Knackser – moderat wählen für natürliche Transienten")
            self.sl_decrackle.setToolTip("Feinknister (hochfrequente Impulse) reduzieren – behutsam einsetzen")
            self.sl_declip.setToolTip("Rekonstruiert geclipte Peaks (vorsichtig einsetzen, 0–40% typisch)")
            self.sl_transient.setToolTip("Transienten betonen (Attack-basiert, 0–30% subtil)")
            self.sl_codec.setToolTip("MP3/Codec-Artefakte glätten (zeitliche Median-Glättung im Spektrum)")
            self.sl_air.setToolTip("Hochton-/Air-Anreicherung: leichte Exciter & Shelf")
            self.sl_widen.setToolTip("Stereo-Verbreiterung aus Mono (HF-Dekorrelation) – zu hoch kann hohl klingen")
            self.sb_lufs.setToolTip("Ziel-Lautheit in LUFS (integriert)")
            self.sl_loud_smooth.setToolTip("Glättung der Lautheit – höhere Werte nivellieren Dynamik stärker")
            self.sl_clarity.setToolTip("Klarheits-Stabilisierung der Höhen (dynamischer Tilt)")
            self.sl_wf.setToolTip("Wow/Flutter Reduktion: glättet leichte Tonhöhen-Jitter")
            self.cb_wf_engine.setToolTip("Pitch-Engine: 'torch' ohne Zusatzabhängigkeiten, 'crepe' genauer (benötigt torchcrepe)")
            self.cb_gen_engine.setToolTip("Generative Super-Resolution Engine (AudioSR) – optional")
            self.cb_gen_mode.setToolTip("'full' = Vollband rekonstruieren, 'highs' = nur Hochton ergänzen")
            self.sl_gen_mix.setToolTip("Dry/Wet Mix generativer Ergänzung")
            self.sb_gen_sr.setToolTip("Ziel-Samplerate für generatives Upscaling")
            self.cb_save_stems.setToolTip("Ausgewählte Stems zusätzlich als einzelne WAV-Dateien speichern")
            self.cb_per_stem.setToolTip("Jedes Stem separat restaurieren (präziser, langsamer)")
            self.btn_process.setToolTip("Startet Verarbeitung & Export")
            self.btn_cancel.setToolTip("Verarbeitung abbrechen")
            self.prog.setToolTip("Verarbeitungsfortschritt")
            self.cb_theme.setToolTip("Darstellung: System (Standard), Light oder Dark Palette")
            self.btn_about.setToolTip("Info über Version, Lizenz und Bibliotheken")
        except Exception:
            pass

    def load_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Audio öffnen", filter="Audio Files (*.wav *.flac *.mp3 *.ogg)")
        if path:
            data, sr = sf.read(path, always_2d=False)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            self.audio = data.astype(np.float32)
            self.sr = sr
            self.audio_path = path
            self.file_label.setText(path)

    def process(self):
        if self.audio is None:
            self.status_label.setText("Bitte zuerst Audio laden.")
            return
        backend_name = self.backend_cb.currentText()
        eq = self.eq_cb.currentText()
        include: list[str] = []
        if self.cb_voc.isChecked():
            include.append("vocals")
        if self.cb_drm.isChecked():
            include.append("drums")
        if self.cb_bas.isChecked():
            include.append("bass")
        if self.cb_oth.isChecked():
            include.append("other")
        if self.cb_har.isChecked():
            include.append("harmonic")
        if self.cb_per.isChecked():
            include.append("percussive")
        if self.cb_acc.isChecked():
            include.append("accompaniment")

        denoise_strength = self.sl_denoise.value() / 100.0
        declick_strength = self.sl_declick.value() / 100.0
        decrackle_strength = self.sl_decrackle.value() / 100.0
        declip_strength = self.sl_declip.value() / 100.0
        transient_strength = self.sl_transient.value() / 100.0
        codec_artifact_strength = self.sl_codec.value() / 100.0
        air_strength = self.sl_air.value() / 100.0
        widen_amount = self.sl_widen.value() / 100.0
        loudness_target_lufs = float(self.sb_lufs.value())
        loudness_smooth = self.sl_loud_smooth.value() / 100.0
        clarity_strength = self.sl_clarity.value() / 100.0
        wowflutter_strength = self.sl_wf.value() / 100.0
        wowflutter_engine = self.cb_wf_engine.currentText()
        gen_engine = self.cb_gen_engine.currentText()
        gen_mode = self.cb_gen_mode.currentText()
        gen_mix = self.sl_gen_mix.value() / 100.0
        gen_target_sr = int(self.sb_gen_sr.value())
        model_name = self.model_cb.currentText()
        process_per_stem = self.cb_per_stem.isChecked()

        self._last_include = include[:]

        self.worker_thread = QThread()
        self.worker = Worker(
            self.audio,
            self.sr,
            backend_name,
            eq,
            include,
            denoise_strength,
            declick_strength,
            declip_strength=declip_strength,
            transient_strength=transient_strength,
            codec_artifact_strength=codec_artifact_strength,
            decrackle_strength=decrackle_strength,
            model_name=model_name,
            air_strength=air_strength,
            process_per_stem=process_per_stem,
            widen_amount=widen_amount,
            loudness_target_lufs=loudness_target_lufs,
            loudness_smooth=loudness_smooth,
            clarity_strength=clarity_strength,
            wowflutter_strength=wowflutter_strength,
            wowflutter_engine=wowflutter_engine,
            gen_engine=gen_engine,
            gen_mode=gen_mode,
            gen_mix=gen_mix,
            gen_target_sr=gen_target_sr,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.btn_process.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.prog.setValue(0)
        self.status_label.setText("Starte…")
        self.worker_thread.start()

    def cancel(self):
        if hasattr(self, "worker") and self.worker:
            self.worker.cancel()
        thread = getattr(self, "worker_thread", None)
        if isinstance(thread, QThread) and thread.isRunning():
            thread.quit()
            thread.wait(200)
        self.btn_process.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.status_label.setText("Abgebrochen")

    def on_progress(self, pct: int, msg: str):
        self.prog.setValue(pct)
        self.status_label.setText(msg)

    def on_finished(self, y: np.ndarray, stems: dict):
        self.prog.setValue(100)
        out_path, _ = QFileDialog.getSaveFileName(self, "Export", filter="Audio (*.wav *.mp3)")
        if out_path:
            if out_path.lower().endswith(".mp3"):
                import tempfile, subprocess, os
                with tempfile.TemporaryDirectory() as td:
                    tmpwav = os.path.join(td, "tmp.wav")
                    sf.write(tmpwav, y, self.sr)
                    try:
                        subprocess.run([
                            "ffmpeg", "-y", "-i", tmpwav, "-vn", "-ar", str(self.sr), "-ac", "2", "-b:a", "320k", out_path
                        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except Exception:
                        sf.write(out_path + ".wav", y, self.sr)
            else:
                sf.write(out_path, y, self.sr)
            self.status_label.setText(f"Exportiert: {out_path}")
        else:
            self.status_label.setText("Export abgebrochen")

        try:
            if getattr(self, 'cb_save_stems', None) and self.cb_save_stems.isChecked() and self.stems_dir:
                import os
                base = "output"
                if self.audio_path:
                    base = os.path.splitext(os.path.basename(self.audio_path))[0]
                names = self._last_include if hasattr(self, '_last_include') else list(stems.keys())
                for name in names:
                    if name in stems:
                        path = os.path.join(self.stems_dir, f"{base}_{name}.wav")
                        sf.write(path, stems[name].astype(np.float32), self.sr)
                self.status_label.setText(self.status_label.text() + "  • Stems gespeichert")
        except Exception as e:
            self.status_label.setText(self.status_label.text() + f"  • Stem-Speichern fehlgeschlagen: {e}")
        thread = getattr(self, "worker_thread", None)
        if isinstance(thread, QThread) and thread.isRunning():
            thread.quit()
            thread.wait(200)
        self.btn_process.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def on_error(self, err: str):
        self.status_label.setText(f"Fehler: {err}")
        thread = getattr(self, "worker_thread", None)
        if isinstance(thread, QThread) and thread.isRunning():
            thread.quit()
            thread.wait(200)
        self.btn_process.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def save_settings(self):
        s = self.settings
        s.setValue("backend", self.backend_cb.currentText())
        s.setValue("model", self.model_cb.currentText())
        s.setValue("eq", self.eq_cb.currentText())
        s.setValue("stems", {
            'voc': self.cb_voc.isChecked(), 'drm': self.cb_drm.isChecked(), 'bas': self.cb_bas.isChecked(),
            'oth': self.cb_oth.isChecked(), 'har': self.cb_har.isChecked(), 'per': self.cb_per.isChecked(), 'acc': self.cb_acc.isChecked()
        })
        s.setValue("denoise", self.sl_denoise.value())
        s.setValue("declick", self.sl_declick.value())
        s.setValue("decrackle", self.sl_decrackle.value())
        s.setValue("declip", self.sl_declip.value())
        s.setValue("transient", self.sl_transient.value())
        s.setValue("codec_artifact", self.sl_codec.value())
        s.setValue("air", self.sl_air.value())
        s.setValue("widen", self.sl_widen.value())
        s.setValue("lufs", self.sb_lufs.value())
        s.setValue("loud_smooth", self.sl_loud_smooth.value())
        s.setValue("clarity", self.sl_clarity.value())
        s.setValue("wowflutter", self.sl_wf.value())
        s.setValue("wf_engine", self.cb_wf_engine.currentText())
        s.setValue("gen_engine", self.cb_gen_engine.currentText())
        s.setValue("gen_mode", self.cb_gen_mode.currentText())
        s.setValue("gen_mix", self.sl_gen_mix.value())
        s.setValue("gen_sr", self.sb_gen_sr.value())
        s.setValue("save_stems", self.cb_save_stems.isChecked())
        s.setValue("per_stem", self.cb_per_stem.isChecked())
        s.setValue("theme", self.cb_theme.currentText())

    def restore_settings(self):
        s = self.settings
        def _get(key, default):
            return s.value(key, default)
        backend = _get("backend", None)
        if backend:
            idx = self.backend_cb.findText(str(backend))
            if idx >= 0:
                self.backend_cb.setCurrentIndex(idx)
        model = _get("model", None)
        if model:
            idx = self.model_cb.findText(str(model))
            if idx >= 0:
                self.model_cb.setCurrentIndex(idx)
        eqv = _get("eq", None)
        if eqv:
            idx = self.eq_cb.findText(str(eqv))
            if idx >= 0:
                self.eq_cb.setCurrentIndex(idx)
        stems = _get("stems", None)
        if isinstance(stems, dict):
            self.cb_voc.setChecked(stems.get('voc', True))
            self.cb_drm.setChecked(stems.get('drm', True))
            self.cb_bas.setChecked(stems.get('bas', True))
            self.cb_oth.setChecked(stems.get('oth', True))
            self.cb_har.setChecked(stems.get('har', True))
            self.cb_per.setChecked(stems.get('per', True))
            self.cb_acc.setChecked(stems.get('acc', True))
        for name, widget in [
            ("denoise", self.sl_denoise), ("declick", self.sl_declick), ("decrackle", self.sl_decrackle), ("declip", self.sl_declip), ("transient", self.sl_transient), ("codec_artifact", self.sl_codec), ("air", self.sl_air),
            ("widen", self.sl_widen), ("lufs", self.sb_lufs), ("loud_smooth", self.sl_loud_smooth),
            ("clarity", self.sl_clarity), ("wowflutter", self.sl_wf), ("gen_mix", self.sl_gen_mix), ("gen_sr", self.sb_gen_sr)
        ]:
            val = _get(name, None)
            if val is not None:
                try:
                    widget.setValue(int(str(val)))
                except Exception:
                    pass
        for name, cb in [("wf_engine", self.cb_wf_engine), ("gen_engine", self.cb_gen_engine), ("gen_mode", self.cb_gen_mode)]:
            val = _get(name, None)
            if val:
                idx = cb.findText(str(val))
                if idx >= 0:
                    cb.setCurrentIndex(idx)
        self.cb_save_stems.setChecked(bool(_get("save_stems", False)))
        self.cb_per_stem.setChecked(bool(_get("per_stem", False)))
        theme = _get("theme", None)
        if theme:
            idx = self.cb_theme.findText(str(theme))
            if idx >= 0:
                self.cb_theme.setCurrentIndex(idx)

    def apply_theme(self, name: str):
        name = (name or "System").lower()
        if name == "dark":
            pal = self.palette()
            pal.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)
            pal.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
            pal.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.black)
            pal.setColor(QPalette.ColorRole.AlternateBase, Qt.GlobalColor.black)
            pal.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
            pal.setColor(QPalette.ColorRole.Button, Qt.GlobalColor.black)
            pal.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
            pal.setColor(QPalette.ColorRole.Highlight, Qt.GlobalColor.cyan)
            pal.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
            self.setPalette(pal)
        elif name == "light":
            pass
        else:
            pass

    def show_about(self):
        import sys
        ver = getattr(sys.modules.get('audio_restorer'), '__version__', '?')
        text = (
            f"<b>Audio Restorer</b><br>Version: {ver}<br><br>"
            "Lizenz: MIT (siehe LICENSE)<br>"
            "Drittanbieter: numpy, scipy, librosa, PySide6, u.a. (siehe THIRD_PARTY_LICENSES.md)"
        )
        QMessageBox.information(self, "Über Audio Restorer", text)

    def closeEvent(self, event):  # type: ignore
        try:
            self.save_settings()
        except Exception:
            pass
        return super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(900, 520)
    w.show()
    sys.exit(app.exec())

