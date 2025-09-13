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
        stems = {}
        for k, v in est.items():
            y = v[0].mean(dim=0)  # mono
            stems[k] = y.detach().cpu().numpy().astype(np.float32)
        return stems
