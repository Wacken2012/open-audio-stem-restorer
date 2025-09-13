from dataclasses import dataclass
import numpy as np
import importlib


def _is_available() -> bool:
    try:
        importlib.import_module("demucs.apply")
        importlib.import_module("demucs.pretrained")
        return True
    except Exception:
        return False


@dataclass
class DemucsBackend:
    name: str = "Demucs"
    model_name: str = "htdemucs"
    # Common pretrained variants; extend as needed
    AVAILABLE_MODELS = [
        "htdemucs",
        "htdemucs_ft",
        "htdemucs_6s",
    ]

    def separate(self, audio: np.ndarray, sr: int):
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
        import torch

        # Ensure float32 mono/stereo handling
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim == 1:
            # duplicate mono to stereo for Demucs
            x = np.stack([x, x], axis=0)  # [2, T]
        elif x.ndim == 2:
            # assume [channels, time] or [time, channels]
            if x.shape[0] < x.shape[1]:
                # shape [channels, time] expected
                pass
            else:
                # convert [time, channels] -> [channels, time]
                x = x.T
            if x.shape[0] == 1:
                x = np.vstack([x, x])

        wav = torch.from_numpy(x[None, ...])  # [1, 2, T]

        # Load selected model on CPU
        model_name = getattr(self, "model_name", "htdemucs")
        if model_name not in self.AVAILABLE_MODELS:
            model_name = "htdemucs"
        model = get_model(model_name)
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        with torch.no_grad():
            out = apply_model(model, wav.to(device), split=True, overlap=0.25, device=device)[0]
        # out: [num_sources, channels, time]
        names = getattr(model, "sources", ["drums", "bass", "other", "vocals"])  # default order fallback
        stems: dict[str, np.ndarray] = {}
        for i, name in enumerate(names):
            y = out[i]  # [channels, time]
            y = y.mean(dim=0)  # mono
            stems[name] = y.detach().cpu().numpy().astype(np.float32)
        return stems
