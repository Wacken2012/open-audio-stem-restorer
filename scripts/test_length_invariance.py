#!/usr/bin/env python3
"""
Length invariance smoke test for the restoration pipeline.

- Generates synthetic mono signals at multiple sample rates and lengths
- Runs available separation backends (skips placeholders "(installieren)")
- Passes stems through restore_pipeline with combinations of options
- Asserts the output length equals input length (for mono or stereo)

Exit codes:
- 0: All checks passed
- 1: A check failed

Run:
  python scripts/test_length_invariance.py
"""
from __future__ import annotations

import sys
import math
import traceback
from typing import Dict, List, Tuple

import numpy as np

# Ensure src/ is on sys.path for src-layout projects
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import project modules (src-layout)
from audio_restorer.separation.interface import list_backends, get_backend
from audio_restorer.restoration.pipeline import restore_pipeline


def _gen_signal(sr: int, n: int) -> np.ndarray:
    """Create a moderately rich synthetic test signal: sum of sines + clicks.
    Returns mono float32 of length n.
    """
    t = np.arange(n, dtype=np.float32) / float(sr)
    y = (
        0.45 * np.sin(2 * np.pi * 220.0 * t)
        + 0.30 * np.sin(2 * np.pi * 440.0 * (1 + 0.003 * np.sin(2 * np.pi * 1.3 * t)) * t)
        + 0.20 * np.sin(2 * np.pi * 3000.0 * t)
    ).astype(np.float32)
    # Add sparse clicks
    rng = np.random.default_rng(0)
    for idx in rng.choice(n, size=max(1, n // 10000), replace=False):
        y[idx:idx+2] += 0.9
    # Normalize to < 1
    peak = float(np.max(np.abs(y)) + 1e-9)
    y = (y / peak * 0.9).astype(np.float32)
    return y


def _len_ok(y: np.ndarray, n_ref: int) -> bool:
    arr = np.asarray(y)
    if arr.ndim == 1:
        return len(arr) == n_ref
    if arr.ndim == 2:
        return arr.shape[0] == n_ref
    return False


def main() -> int:
    # Sample rates and frame counts to test
    srs = [22050, 44100, 48000]
    seconds = [0.73, 3.37]
    results: List[Tuple[str, int, int, str]] = []  # (backend, sr, n, status)
    backends = [b for b in list_backends() if "(installieren)" not in b]
    if not backends:
        print("No available backends to test.")
        return 0

    print(f"Testing backends: {backends}")

    combos = [
        {"process_per_stem": False, "widen_amount": 0.0},
        {"process_per_stem": False, "widen_amount": 0.5},
        {"process_per_stem": True,  "widen_amount": 0.0},
    ]

    failed = 0
    for be_name in backends:
        be = get_backend(be_name)
        print(f"\nBackend: {be_name}")
        for sr in srs:
            for sec in seconds:
                n = int(round(sr * sec)) + (1 if sec < 1.0 else 0)  # include odd lengths
                x = _gen_signal(sr, n)
                try:
                    stems = be.separate(x, sr)
                except Exception as e:
                    print(f"  [SKIP] separate failed: sr={sr}, n={n}, err={e}")
                    continue
                for opts in combos:
                    try:
                        y = restore_pipeline(
                            stems, sr,
                            eq="shellac",
                            include=None,
                            denoise_strength=0.4,
                            declick_strength=0.4,
                            decrackle_strength=0.15,
                            declip_strength=0.0,
                            transient_strength=0.1,
                            codec_artifact_strength=0.0,
                            air_strength=0.0,
                            progress=None,
                            canceled=None,
                            process_per_stem=bool(opts["process_per_stem"]),
                            widen_amount=float(opts["widen_amount"]),
                            loudness_target_lufs=-16.0,
                            loudness_smooth=0.5,
                            clarity_strength=0.0,
                            wowflutter_strength=0.0,
                            wowflutter_engine="torch",
                            gen_engine="none",
                            gen_mode="full",
                            gen_mix=0.0,
                            gen_target_sr=48000,
                        )
                        ok = _len_ok(y, n)
                        status = "OK" if ok else f"FAIL(len={len(y) if np.asarray(y).ndim==1 else np.asarray(y).shape[0]} != {n})"
                        print(f"  sr={sr:5d} n={n:7d} per_stem={opts['process_per_stem']} widen={opts['widen_amount']}: {status}")
                        results.append((be_name, sr, n, status))
                        if not ok:
                            failed += 1
                    except Exception as e:
                        print(f"  ERROR at sr={sr}, n={n}, opts={opts}: {e}")
                        traceback.print_exc()
                        failed += 1

    print("\nSummary:")
    total = len(results)
    okc = sum(1 for _, _, _, st in results if st == "OK")
    print(f"  {okc}/{total} checks passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
