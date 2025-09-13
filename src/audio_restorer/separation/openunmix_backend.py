from dataclasses import dataclass
import importlib
import numpy as np


def _is_available() -> bool:
    try:
        importlib.import_module("openunmix")
        return True
    except Exception:
        return False


@dataclass
class OpenUnmixBackend:
    name: str = "Open-Unmix"

    def separate(self, audio: np.ndarray, sr: int):
        # Minimal wrapper using openunmix inference for 4 stems
        import torch
        from openunmix import predict

        x = np.asarray(audio, dtype=np.float32)
        if x.ndim == 1:
            x = np.stack([x, x], axis=0)  # stereo expected
        elif x.shape[0] == 1:
            x = np.vstack([x, x])
        x = torch.from_numpy(x[None, ...])  # [1, 2, T]

        with torch.no_grad():
            est = predict.separate(x, rate=sr)
        # est is dict with keys 'vocals','drums','bass','other' each [1,2,T]
        T_ref = int(audio.shape[-1])

        def _match_len(y: np.ndarray, n: int) -> np.ndarray:
            y = np.asarray(y, dtype=np.float32)
            if y.ndim > 1:
                # collapse to mono if needed
                y = y.mean(axis=0)
            if len(y) == n:
                return y
            if len(y) < 2:
                return np.resize(y, (n,)).astype(np.float32)
            t_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False, dtype=np.float64)
            t_new = np.linspace(0.0, 1.0, num=n, endpoint=False, dtype=np.float64)
            return np.interp(t_new, t_old, y.astype(np.float64)).astype(np.float32)

        stems = {}
        for k, v in est.items():
            y = v[0].mean(dim=0)  # mono [T]
            y_np = y.detach().cpu().numpy().astype(np.float32)
            stems[k] = _match_len(y_np, T_ref)
        return stems
