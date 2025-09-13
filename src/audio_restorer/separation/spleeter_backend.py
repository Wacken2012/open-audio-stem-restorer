import importlib
import numpy as np


def _is_available() -> bool:
    try:
        importlib.import_module("spleeter")
        return True
    except Exception:
        return False


class SpleeterBackend:
    def __init__(self):
        from spleeter.separator import Separator
        # 2 stems for speed; can be made configurable later
        self.sep = Separator("spleeter:2stems")

    def separate(self, audio: np.ndarray, sr: int):
        x = np.asarray(audio, dtype=np.float32)
        # Spleeter expects shape (n_samples, n_channels) typically; use mono
        y = self.sep.separate(x)
        voc = y.get("vocals")
        acc = y.get("accompaniment")
        if voc is None or acc is None:
            return {"mixture": x}
        def mono(a):
            a = np.squeeze(a)
            if a.ndim > 1:
                a = a.mean(axis=-1)
            return a.astype(np.float32)
        return {"vocals": mono(voc), "accompaniment": mono(acc)}
from dataclasses import dataclass
import numpy as np


def _is_available() -> bool:
    try:
        import spleeter  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class SpleeterBackend:
    name: str = "Spleeter"

    def separate(self, audio: np.ndarray, sr: int):
        from spleeter.separator import Separator
        import soundfile as sf
        import tempfile
        import os

        # Spleeter expects a file input in many simple scenarios; use temp WAV
        with tempfile.TemporaryDirectory() as d:
            in_wav = os.path.join(d, "in.wav")
            out_dir = os.path.join(d, "out")
            sf.write(in_wav, audio, sr)
            sep = Separator("spleeter:4stems")
            sep.separate_to_file(in_wav, out_dir)
            # Load stems (bass, drums, other, vocals)
            stems = {}
            for stem in ["bass", "drums", "other", "vocals"]:
                wav_path = os.path.join(out_dir, "in", f"{stem}.wav")
                y, _ = sf.read(wav_path)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                stems[stem] = y.astype(np.float32)
        return stems
