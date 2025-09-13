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
        n_ref = len(audio)
        # Align lengths defensively to the original input length
        def _fix(y):
            y = np.asarray(y, dtype=np.float32)
            if len(y) == n_ref:
                return y
            if len(y) < 2:
                return np.resize(y, (n_ref,)).astype(np.float32)
            t_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False, dtype=np.float64)
            t_new = np.linspace(0.0, 1.0, num=n_ref, endpoint=False, dtype=np.float64)
            return np.interp(t_new, t_old, y.astype(np.float64)).astype(np.float32)
        return {"harmonic": _fix(y_h), "percussive": _fix(y_p)}
