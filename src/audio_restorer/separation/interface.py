from typing import Dict, List

from .simple_hpss import HPSSBackend

# Optional backends (real or stub)
try:
    from .demucs_backend import DemucsBackend, _is_available as _demucs_avail
except Exception:
    DemucsBackend = None  # type: ignore
    _demucs_avail = lambda: False  # type: ignore

try:
    from .spleeter_backend import SpleeterBackend, _is_available as _spleeter_avail
except Exception:
    SpleeterBackend = None  # type: ignore
    _spleeter_avail = lambda: False  # type: ignore

try:
    from .openunmix_backend import OpenUnmixBackend, _is_available as _openunmix_avail
except Exception:
    OpenUnmixBackend = None  # type: ignore
    _openunmix_avail = lambda: False  # type: ignore


class _MissingBackend:
    def __init__(self, package: str, pip_name: str):
        self.package = package
        self.pip_name = pip_name

    def separate(self, audio, sr):
        raise RuntimeError(
            f"Backend '{self.package}' ist nicht installiert. Bitte im aktiven venv installieren: pip install {self.pip_name}"
        )


_REGISTRY: Dict[str, object] = {
    "HPSS (Fallback)": HPSSBackend(),
}

# Demucs entry
if DemucsBackend and _demucs_avail():
    _REGISTRY["Demucs"] = DemucsBackend()
else:
    _REGISTRY["Demucs (installieren)"] = _MissingBackend("demucs", "demucs")

# Spleeter entry
if SpleeterBackend and _spleeter_avail():
    _REGISTRY["Spleeter 2-Stems"] = SpleeterBackend()
else:
    _REGISTRY["Spleeter 2-Stems (installieren)"] = _MissingBackend("spleeter", "spleeter")

# Open-Unmix entry
if OpenUnmixBackend and _openunmix_avail():
    _REGISTRY["Open-Unmix"] = OpenUnmixBackend()
else:
    _REGISTRY["Open-Unmix (installieren)"] = _MissingBackend("openunmix", "openunmix")


def list_backends() -> List[str]:
    return list(_REGISTRY.keys())


def get_backend(name: str):
    return _REGISTRY[name]
