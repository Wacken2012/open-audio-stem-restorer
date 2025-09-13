from dataclasses import dataclass
import numpy as np
import librosa


@dataclass
class HPSSBackend:
    """Simple HPSS-based separation backend (time-domain).

    Uses librosa.effects.hpss to avoid Griffin-Lim artifacts.
    Produces two stems: 'harmonic' and 'percussive'.
    Returns a dict with keys: 'harmonic', 'percussive'.
    """

    def separate(self, audio: np.ndarray, sr: int):
        y_h, y_p = librosa.effects.hpss(audio)
        # Align lengths defensively
        n = min(len(y_h), len(y_p))
        return {"harmonic": y_h[:n].astype(np.float32), "percussive": y_p[:n].astype(np.float32)}
