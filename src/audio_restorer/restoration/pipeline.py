from typing import Dict, Callable, Optional, Iterable, Set
import numpy as np
from .tools import (
    denoise_spectral_gate,
    simple_declick,
    decrackle_band_suppressor,
    declip_iterative_smooth,
    eq_preset,
    air_enhance,
    transient_enhance,
    stereo_widen_from_mono,
    loudness_normalize_adaptive,
    clarity_stabilization,
    wow_flutter_reduce,
    generative_enhance,
    codec_artifact_suppressor,
)


def restore_pipeline(
    stems: Dict[str, np.ndarray],
    sr: int,
    eq: str = "shellac",
    include: Optional[Iterable[str]] = None,
    denoise_strength: float = 0.5,
    declick_strength: float = 0.5,
    decrackle_strength: float = 0.0,
    declip_strength: float = 0.0,
    transient_strength: float = 0.0,
    codec_artifact_strength: float = 0.0,
    air_strength: float = 0.0,
    progress: Optional[Callable[[int, str], None]] = None,
    canceled: Optional[Callable[[], bool]] = None,
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
) -> np.ndarray:
    def is_canceled() -> bool:
        return bool(canceled and canceled())

    def apply_chain(y: np.ndarray) -> np.ndarray:
        # Pre-normalize to avoid clipping during processing
        peak_local = float(np.max(np.abs(y)) + 1e-9)
        y = (y / peak_local).astype(np.float32)

        if declip_strength and declip_strength > 1e-3:
            if is_canceled():
                return np.zeros(1, dtype=np.float32)
            if progress:
                progress(35, "Declip…")
            y = declip_iterative_smooth(y, sr, strength=float(declip_strength))

        if is_canceled():
            return np.zeros(1, dtype=np.float32)
        if progress:
            progress(40, "Denoise (Spektrales Gating)…")
        y = denoise_spectral_gate(y, sr, strength=denoise_strength)

        if is_canceled():
            return np.zeros(1, dtype=np.float32)
        if progress:
            progress(65, "Declick/Decrackle…")
        y = simple_declick(y, sr, strength=declick_strength)

        if decrackle_strength and decrackle_strength > 1e-3:
            if progress:
                progress(75, "Feinknister entfernen…")
            y = decrackle_band_suppressor(y, sr, strength=float(decrackle_strength))

        if codec_artifact_strength and codec_artifact_strength > 1e-3:
            if progress:
                progress(78, "Codec-Artefakte dämpfen…")
            y = codec_artifact_suppressor(y, sr, strength=float(codec_artifact_strength))

        if progress:
            progress(80, f"EQ-Preset: {eq}…")
        y = eq_preset(y, sr, preset=eq)

        if air_strength > 1e-3:
            if progress:
                progress(90, "Höhen anreichern…")
            y = air_enhance(y, sr, strength=air_strength)

        if transient_strength and transient_strength > 1e-3:
            if progress:
                progress(91, "Transienten wiederherstellen…")
            y = transient_enhance(y, sr, strength=float(transient_strength))
        return y.astype(np.float32)

    # Determine included stems
    allow: Optional[Set[str]] = set(include) if include is not None else None

    if process_per_stem:
        # Process each selected stem individually, then sum
        names = [n for n in stems.keys() if (allow is None or n in allow)]
        if not names:
            return np.zeros(1, dtype=np.float32)
        if progress:
            progress(5, "Restauriere Stems einzeln…")
        acc: Optional[np.ndarray] = None
        total = len(names)
        for i, name in enumerate(names, 1):
            if is_canceled():
                return np.zeros(1, dtype=np.float32)
            y = stems[name].astype(np.float32)
            y = apply_chain(y)
            if acc is None:
                acc = y
            else:
                n = min(len(acc), len(y))
                acc = acc[:n]
                acc[:n] += y[:n]
            if progress:
                pct = 5 + int(85 * i / max(1, total))
                progress(pct, f"Stem restauriert: {name}")
        x = acc if acc is not None else np.zeros(1, dtype=np.float32)
    else:
        # Mix selected stems back (simple sum) then apply restoration chain
        if progress:
            progress(5, "Mische Stems…")
        mix = None
        for name, y in stems.items():
            if allow is not None and name not in allow:
                continue
            if is_canceled():
                return np.zeros(1, dtype=np.float32)
            if mix is None:
                mix = y.astype(np.float32)
            else:
                # align lengths
                n = min(len(mix), len(y))
                mix = mix[:n]
                mix[:n] += y[:n].astype(np.float32)
        if mix is None:
            return np.zeros(1, dtype=np.float32)

        if progress:
            progress(10, "Normalisiere…")
        # Normalize to avoid clipping before processing
        peak = float(np.max(np.abs(mix)) + 1e-9)
        mix = (mix / peak).astype(np.float32)

        x = apply_chain(mix)

    # Optional generative enhancement (fullband or highs-only)
    if gen_engine and gen_engine != "none" and gen_mix and gen_mix > 1e-3:
        if progress:
            progress(91, "Generative Enhancement…")
        x = generative_enhance(
            x, sr,
            engine=str(gen_engine),
            mode=str(gen_mode),
            mix=float(gen_mix),
            target_sr=int(gen_target_sr),
        )

    # Adaptive loudness and clarity
    if progress:
        progress(92, "Lautheit angleichen…")
    x = loudness_normalize_adaptive(x, sr, target_lufs=float(loudness_target_lufs), smooth=float(loudness_smooth))

    if clarity_strength and clarity_strength > 1e-3:
        if progress:
            progress(93, "Klarheit stabilisieren…")
        x = clarity_stabilization(x, sr, strength=float(clarity_strength))

    if wowflutter_strength and wowflutter_strength > 1e-3:
        if progress:
            progress(93, "Wow/Flutter reduzieren…")
        x = wow_flutter_reduce(x, sr, strength=float(wowflutter_strength), engine=str(wowflutter_engine))

    # Optional stereo widening on the final mix (creates stereo from mono)
    if widen_amount and widen_amount > 1e-6:
        if progress:
            progress(94, "Stereo erweitern…")
        x_st = stereo_widen_from_mono(x, sr, width=float(widen_amount))
        # Normalize stereo safely
        if progress:
            progress(97, "Finale Normalisierung…")
        peak = float(np.max(np.abs(x_st)) + 1e-9)
        x_out = (x_st / peak * 0.98).astype(np.float32)
    else:
        # Final normalize (mono)
        if progress:
            progress(97, "Finale Normalisierung…")
        peak = float(np.max(np.abs(x)) + 1e-9)
        x_out = (x / peak * 0.98).astype(np.float32)
    if progress:
        progress(100, "Fertig")
    return x_out
