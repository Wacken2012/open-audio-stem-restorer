import numpy as np
import scipy.signal as sps
try:
    import pyloudnorm as pyln
except Exception:
    pyln = None  # type: ignore
try:
    import noisereduce as nr
except Exception:
    nr = None  # type: ignore

# Optional: generative audio super-resolution (AudioSR)
_AUDIOSR_MODEL = None

def _hz_cap(f_hz: float, sr: int, frac_nyq: float = 0.49) -> float:
    """Clamp frequency to a safe range below Nyquist for scipy iirfilter with fs=sr."""
    lo = 1.0
    hi = max(lo + 1.0, float(frac_nyq * sr))
    return float(min(max(lo, f_hz), hi))


def _hz_cap(freq: float, sr: int, min_hz: float = 1.0, max_ratio: float = 0.49) -> float:
    """Clamp frequency in Hz to a safe digital range (0 < Wn < sr/2).

    max_ratio defaults to 0.49 to leave margin below Nyquist.
    """
    return float(max(min_hz, min(freq, max_ratio * float(sr))))

def _resample_poly_safe(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    # Rational resample using scipy, fallback to linear interp if anything fails
    try:
        from math import gcd
        g = gcd(sr_in, sr_out)
        up = sr_out // g
        down = sr_in // g
        y = sps.resample_poly(x.astype(np.float32), up, down)
        return y.astype(np.float32)
    except Exception:
        t = np.arange(len(x), dtype=np.float64)
        n_out = int(round(len(x) * (sr_out / float(sr_in))))
        t_new = np.linspace(0, len(x) - 1, num=n_out, dtype=np.float64)
        y = np.interp(t_new, t, x.astype(np.float64))
        return y.astype(np.float32)

def _load_audiosr_model():
    global _AUDIOSR_MODEL
    if _AUDIOSR_MODEL is not None:
        return _AUDIOSR_MODEL
    try:
        # Lazy import; if unavailable, return None
        from audiosr import build_model  # type: ignore
        _AUDIOSR_MODEL = build_model('basic')
        return _AUDIOSR_MODEL
    except Exception:
        return None

def _audiosr_super_resolve(x: np.ndarray, sr: int, target_sr: int = 48000) -> np.ndarray:
    """Attempt AudioSR super-resolution; fallback to high-quality resample on failure."""
    model = _load_audiosr_model()
    if model is None:
        return _resample_poly_safe(x, sr, target_sr)
    try:
        from audiosr import super_resolution  # type: ignore
        y = super_resolution(model, x.astype(np.float32), sr=sr, target_sr=target_sr, progress=False)
        return np.asarray(y, dtype=np.float32)
    except Exception:
        return _resample_poly_safe(x, sr, target_sr)

def _highpass(x: np.ndarray, sr: int, fc: float = 7000.0) -> np.ndarray:
    try:
        fc_safe = _hz_cap(fc, sr)
        sos = sps.iirfilter(4, Wn=fc_safe, btype='highpass', ftype='butter', fs=sr, output='sos')
        return sps.sosfiltfilt(sos, x.astype(np.float32)).astype(np.float32)
    except Exception:
        return x.astype(np.float32)


def declip_iterative_smooth(x: np.ndarray, sr: int, strength: float = 0.0) -> np.ndarray:
    """Simple open-source declipping by detecting flattened peaks and inpainting.

    - Detect clipped samples by threshold near peak and flat-run heuristic.
    - Linearly interpolate across clipped runs using neighboring unclipped samples.
    - Apply light Tikhonov-like smoothing over the repaired regions.
    strength in [0,1] controls aggressiveness of detection and smoothing.
    """
    s = float(max(0.0, min(1.0, strength)))
    if s <= 1e-3:
        return x.astype(np.float32)

    y = x.astype(np.float32).copy()
    peak = float(np.max(np.abs(y)) + 1e-9)
    thr = 0.92 - 0.3 * s  # 0.92..0.62 of peak
    t = thr * peak

    # Candidate clipped mask: near peak magnitude and low local derivative
    mag = np.abs(y)
    d = np.abs(np.diff(y, prepend=y[:1]))
    d2 = np.abs(np.diff(y, n=2, prepend=[y[0], y[0]]))
    mask = (mag >= t) & (d < 0.005 + 0.02 * s) & (d2 < 0.01 + 0.05 * s)

    # Find runs and interpolate across them
    idx = np.where(mask)[0]
    if idx.size == 0:
        return y

    def iter_runs(m: np.ndarray):
        if not np.any(m):
            return
        starts = []
        ends = []
        i = 0
        n = len(m)
        while i < n:
            if m[i]:
                j = i
                while j < n and m[j]:
                    j += 1
                starts.append(i)
                ends.append(j - 1)
                i = j
            else:
                i += 1
        for a, b in zip(starts, ends):
            yield a, b

    for a, b in iter_runs(mask):
        L = a - 1
        R = b + 1
        while L >= 0 and mask[L]:
            L -= 1
        while R < len(y) and mask[R]:
            R += 1
        if L < 0 or R >= len(y):
            # Edge case: fade to edge
            valL = y[a-1] if a > 0 else 0.0
            valR = y[b+1] if b+1 < len(y) else 0.0
            span = max(1, b - a + 1)
            y[a:b+1] = np.linspace(valL, valR, span, dtype=np.float32)
        else:
            span = R - L
            if span <= 1:
                continue
            y[a:b+1] = np.interp(np.arange(a, b+1), [L, R], [y[L], y[R]]).astype(np.float32)

    # Light smoothing over repaired mask only
    win = int(0.0015 * sr * (1.0 + s))  # ~1.5ms base
    win = max(3, win | 1)  # odd
    ker = np.ones(win, dtype=np.float32) / float(win)
    y_s = sps.convolve(y, ker, mode='same').astype(np.float32)
    alpha = 0.35 + 0.45 * s
    y = np.where(mask, alpha * y_s + (1 - alpha) * y, y)
    # Final peak safe
    p = float(np.max(np.abs(y)) + 1e-9)
    if p > 1.0:
        y = y / p * 0.98
    return y.astype(np.float32)


def denoise_spectral_gate(x: np.ndarray, sr: int, strength: float = 0.5) -> np.ndarray:
    if nr is None:
        return x.astype(np.float32)
    # Clamp strength 0..1 and map to nr parameters
    s = float(max(0.0, min(1.0, strength)))
    n0 = min(len(x), int(0.5 * sr))
    noise_clip = x[:n0]
    # scale prop_decrease from light 0.3 to heavy 1.0
    prop = 0.3 + 0.7 * s
    y = nr.reduce_noise(y=x, sr=sr, y_noise=noise_clip, prop_decrease=prop)
    return y.astype(np.float32)


def simple_declick(x: np.ndarray, sr: int, strength: float = 0.5) -> np.ndarray:
    # Lightweight click removal: median filter the residual of a lowpass
    s = float(max(0.0, min(1.0, strength)))
    # adjust LPF cutoff slightly with strength (avoid over-smoothing)
    cutoff = 0.35 + 0.1 * (1.0 - s)  # 0.35..0.45 nyquist
    sos = sps.butter(4, cutoff, btype='lowpass', output='sos')  # normalized (Nyquist)
    low = sps.sosfiltfilt(sos, x)
    resid = x - low
    # median kernel 3..9 samples depending on strength
    k = int(3 + round(6 * s))
    if k % 2 == 0:
        k += 1
    resid_f = sps.medfilt(resid, kernel_size=k)
    y = low + resid_f
    return y.astype(np.float32)


def decrackle_band_suppressor(x: np.ndarray, sr: int, strength: float = 0.0) -> np.ndarray:
    """Reduce fine-grained crackle (tiny HF impulses) without dulling the signal.

    Approach:
    - High-pass to isolate HF band (3.5–8 kHz+)
    - Detect outliers via robust MAD threshold; replace only those using a local median
    - Recombine with original low/mid band
    strength in [0,1].
    """
    s = float(max(0.0, min(1.0, strength)))
    if s <= 1e-3:
        return x.astype(np.float32)

    # High-pass emphasize crackle region; adapt cutoff with strength
    cutoff = 3000.0 + 4000.0 * s
    cutoff = _hz_cap(cutoff, sr)
    try:
        sos_hp = sps.iirfilter(3, Wn=cutoff, btype='highpass', ftype='butter', fs=sr, output='sos')
        high = sps.sosfiltfilt(sos_hp, x.astype(np.float32))
    except Exception:
        return x.astype(np.float32)

    # Robust threshold using MAD
    med = float(np.median(high))
    mad = float(np.median(np.abs(high - med)) + 1e-9)
    # Lower k => more aggressive; map s: 0.0->8, 1.0->2.2
    k = 8.0 - 5.8 * s
    thr = k * mad
    mask = np.abs(high - med) > thr

    if not np.any(mask):
        return x.astype(np.float32)

    # Local median replacement only at masked positions
    ksize = int(5 + round(6 * s))  # 5..11
    if ksize % 2 == 0:
        ksize += 1
    high_med = sps.medfilt(high, kernel_size=ksize)
    # Soft blend for stability
    blend = 0.5 + 0.5 * s
    high_fixed = np.where(mask, blend * high_med + (1.0 - blend) * high, high)

    # Recombine with low/mid band by replacing high band content
    # Low/mid via complementary low-pass
    try:
        sos_lp = sps.iirfilter(3, Wn=cutoff, btype='lowpass', ftype='butter', fs=sr, output='sos')
        lowmid = sps.sosfiltfilt(sos_lp, x.astype(np.float32))
        y = lowmid + high_fixed
    except Exception:
        y = x - high + high_fixed

    # Safety normalization
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = y / peak * 0.99
    return y.astype(np.float32)


def eq_preset(x: np.ndarray, sr: int, preset: str = "shellac") -> np.ndarray:
    # Practical EQ presets; 'none' is passthrough
    if preset == "shellac":
        # Gentle HF roll-off (LPF ~5 kHz) to tame crackle but keep some air
        lpf_cut = _hz_cap(min(9000.0, max(3500.0, 5000.0)), sr)
        sos_lpf = sps.iirfilter(2, Wn=lpf_cut, btype='lowpass', ftype='butter', fs=sr, output='sos')
        y = sps.sosfiltfilt(sos_lpf, x)
        # Bass lift via low-shelf approximation (blend of lowpassed bass)
        sos_bass = sps.iirfilter(2, Wn=_hz_cap(120, sr), btype='lowpass', ftype='butter', fs=sr, output='sos')
        bass = sps.sosfiltfilt(sos_bass, x)
        y = 0.75 * y + 0.25 * bass
        # Presence region bump (~2.5–4.5 kHz) for clarity on vocals/instruments
        bp_lo = _hz_cap(2500, sr)
        bp_hi = _hz_cap(4500, sr)
        if bp_hi <= bp_lo:
            bp_hi = min(_hz_cap(4800, sr), bp_lo + 10.0)
        sos_pres = sps.iirfilter(2, Wn=[bp_lo, bp_hi], btype='bandpass', ftype='butter', fs=sr, output='sos')
        pres = sps.sosfiltfilt(sos_pres, x)
        y = y + 0.15 * pres
    elif preset == "tape":
        # Mild hiss reduction (LPF ~12k) and stronger presence boost (~3k)
        sos = sps.iirfilter(2, Wn=_hz_cap(12000, sr), btype='lowpass', ftype='butter', fs=sr, output='sos')
        y = sps.sosfiltfilt(sos, x)
        bp_lo = _hz_cap(2500, sr)
        bp_hi = _hz_cap(5000, sr)
        if bp_hi <= bp_lo:
            bp_hi = min(_hz_cap(5200, sr), bp_lo + 10.0)
        sos_bp = sps.iirfilter(2, Wn=[bp_lo, bp_hi], btype='bandpass', ftype='butter', fs=sr, output='sos')
        pres = sps.sosfiltfilt(sos_bp, x)
        y = y + 0.25 * pres
    else:
        y = x
    return y.astype(np.float32)


def air_enhance(x: np.ndarray, sr: int, strength: float = 0.0) -> np.ndarray:
    """Simple HF enhancement / exciter-like effect.

    strength in [0,1]: 0 disables, 1 is strong. We combine a gentle high-shelf
    and a very light harmonic excitation (soft nonlinearity on highpassed band).
    """
    s = float(max(0.0, min(1.0, strength)))
    if s <= 1e-3:
        return x.astype(np.float32)

    # High-shelf via peaking approximation: mix of original and highpassed
    # Choose cutoff ~4–8 kHz depending on strength
    cutoff = float(min(12000.0, max(3500.0, 4500.0 + 3500.0 * s)))
    cutoff = _hz_cap(cutoff, sr)
    sos_hp = sps.iirfilter(2, Wn=cutoff, btype='highpass', ftype='butter', fs=sr, output='sos')
    high = sps.sosfiltfilt(sos_hp, x)

    # Soft harmonic excitation on highs
    exc = np.tanh(3.5 * high) - 0.5 * np.tanh(1.8 * high)

    # Mix levels
    y = x + (0.35 + 0.9 * s) * high + 0.20 * s * exc

    # Gentle limiter to avoid clipping
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def transient_enhance(x: np.ndarray, sr: int, strength: float = 0.0) -> np.ndarray:
    """Upward transient enhancement via attack-weighted HF boost.

    - Extract HF band (>1 kHz), build attack envelope from rectified derivative of a smoothed signal.
    - Scale HF band by (1 + k * attack_env) and mix back.
    """
    s = float(max(0.0, min(1.0, strength)))
    if s <= 1e-3:
        return x.astype(np.float32)
    x = x.astype(np.float32)
    # HF band
    sos_hp = sps.iirfilter(2, Wn=_hz_cap(1000.0, sr), btype='highpass', ftype='butter', fs=sr, output='sos')
    hf = sps.sosfiltfilt(sos_hp, x)
    # Attack envelope
    smooth = sps.sosfiltfilt(sps.iirfilter(1, Wn=15.0, btype='lowpass', ftype='butter', fs=sr, output='sos'), np.abs(x))
    deriv = np.maximum(0.0, np.diff(smooth, prepend=smooth[:1]))
    env = sps.sosfiltfilt(sps.iirfilter(1, Wn=30.0, btype='lowpass', ftype='butter', fs=sr, output='sos'), deriv)
    env /= (np.percentile(env, 95) + 1e-9)
    env = np.clip(env, 0.0, 1.0)
    # Scale HF by attack
    k = 0.4 + 1.0 * s
    hf_boost = hf * (1.0 + k * env)
    y = x + (0.25 + 0.5 * s) * hf_boost
    # Safety
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def codec_artifact_suppressor(x: np.ndarray, sr: int, strength: float = 0.0) -> np.ndarray:
    """Reduce typical low-bitrate codec artifacts (pre-echo/warble) using STFT-domain median smoothing.

    - Compute STFT (scipy.signal.stft), magnitude median across small time window per bin.
    - Attenuate outlier bins below/above median depending on mask.
    """
    s = float(max(0.0, min(1.0, strength)))
    if s <= 1e-3:
        return x.astype(np.float32)
    x = x.astype(np.float32)
    nper = 1024
    nover = 768
    f, t, Z = sps.stft(x, fs=sr, nperseg=nper, noverlap=nover, boundary=None)
    mag = np.abs(Z)
    pha = np.angle(Z)
    # Median over ±W frames
    W = int(2 + round(4 * s))
    mag_pad = np.pad(mag, ((0, 0), (W, W)), mode='edge')
    med = np.empty_like(mag)
    for i in range(mag.shape[1]):
        med[:, i] = np.median(mag_pad[:, i:i + 2*W + 1], axis=1)
    # Build suppression mask for isolated spikes or holes
    upper = med * (1.0 + 1.5 * s)
    lower = med * (0.5 + 0.3 * s)
    mag2 = np.where(mag > upper, upper + (mag - upper) * (0.3 + 0.2 * s), mag)
    mag2 = np.where(mag2 < lower, lower + (mag2 - lower) * (0.3 + 0.2 * s), mag2)
    Z2 = mag2 * np.exp(1j * pha)
    _, y = sps.istft(Z2, fs=sr, nperseg=nper, noverlap=nover, boundary=None)
    y = y[:len(x)].astype(np.float32)
    # Small de-ringing LPF
    sos = sps.iirfilter(2, Wn=_hz_cap(min(0.48 * (sr/2), 18000.0), sr), btype='lowpass', ftype='butter', fs=sr, output='sos')
    y = sps.sosfiltfilt(sos, y).astype(np.float32)
    # Normalize
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def stereo_widen_from_mono(x: np.ndarray, sr: int, width: float = 0.0) -> np.ndarray:
    """Create a natural-sounding pseudo-stereo from mono with adjustable width.

    width in [0,1]: 0 = mono passthrough (returned as 2 channels with identical content),
    1 = strong widening. We combine:
      - small decorrelated high-frequency component via high-pass and tiny interaural delays
      - mid/side mix with side limited to HF band
    Returns array shape [N, 2].
    """
    w = float(max(0.0, min(1.0, width)))
    x = x.astype(np.float32)
    if w <= 1e-6:
        # return dual-mono
        return np.stack([x, x], axis=1)

    # High-pass to focus widening in presence/air region (2.5 kHz+)
    hp_cut = 2500.0
    sos_hp = sps.iirfilter(2, Wn=hp_cut, btype='highpass', ftype='butter', fs=sr, output='sos')
    high = sps.sosfiltfilt(sos_hp, x)

    # Tiny interaural delays (0.2..0.8 ms) scaled by width
    # Convert ms to samples
    dL = int(round((0.2 + 0.6 * w) * 1e-3 * sr))
    dR = int(round((0.8 - 0.6 * w) * 1e-3 * sr))

    def delay(sig: np.ndarray, d: int) -> np.ndarray:
        if d <= 0:
            return sig
        y = np.zeros_like(sig)
        y[d:] = sig[:-d]
        y[:d] = sig[0]
        return y

    hL = delay(high, dL)
    hR = delay(high, dR)

    # Build side component and mix
    side_gain = 0.15 + 0.5 * w
    L = x + side_gain * hL
    R = x - side_gain * hR

    # Gentle output normalization to prevent overload
    peak = float(max(np.max(np.abs(L)), np.max(np.abs(R))) + 1e-9)
    if peak > 1.0:
        L = L / peak
        R = R / peak
    y = np.stack([L, R], axis=1)
    return y.astype(np.float32)


def loudness_normalize_adaptive(x: np.ndarray, sr: int, target_lufs: float = -16.0, smooth: float = 0.5) -> np.ndarray:
    """Adaptive loudness leveling:
    - Measure integrated LUFS; normalize to target
    - Apply a slow-varying gain based on short-term loudness to smooth intra-song swings
    smooth in [0,1]: 0 = integrated only, 1 = stronger short-term correction.
    """
    if pyln is None:
        # Fallback: simple RMS normalize
        rms = float(np.sqrt(np.mean(x**2)) + 1e-9)
        y = x / rms * 0.1
        return y.astype(np.float32)

    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(x)
    gain_db = target_lufs - lufs
    lin = 10 ** (gain_db / 20.0)
    y = x * lin

    # Short-term correction: compute short-term loudness over windows
    s = float(max(0.0, min(1.0, smooth)))
    if s > 1e-3:
        win = int(0.400 * sr)  # 400 ms window (ITU BS.1770 short-term ~3s, but we do faster for responsiveness)
        hop = int(0.100 * sr)
        # Envelope via RMS approximation
        rms_env = np.sqrt(sps.convolve(y**2, np.ones(win)/win, mode='same') + 1e-12)
        # Convert to dB and compute delta vs median
        db_env = 20.0 * np.log10(rms_env + 1e-9)
        med = np.median(db_env)
        delta_db = med - db_env  # positive when current is quieter
        # Smooth delta with long attack/fast-ish release to avoid pumping
        alpha_a = np.exp(-hop / (sr * 1.0))    # ~1s attack
        alpha_r = np.exp(-hop / (sr * 0.2))    # ~200ms release
        out_db = np.zeros_like(db_env)
        last = 0.0
        for i in range(len(db_env)):
            d = delta_db[i]
            if d > last:
                last = alpha_a * last + (1 - alpha_a) * d
            else:
                last = alpha_r * last + (1 - alpha_r) * d
            out_db[i] = last
        g = 10 ** ((s * out_db) / 20.0)
        y = y * g

    # Final safety limiter
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def clarity_stabilization(x: np.ndarray, sr: int, strength: float = 0.0) -> np.ndarray:
    """Stabilize clarity over time by dynamically tilting EQ based on HF energy envelope.
    strength in [0,1]."""
    s = float(max(0.0, min(1.0, strength)))
    if s <= 1e-3:
        return x.astype(np.float32)
    # HF envelope 3–8 kHz
    hi = _hz_cap(8000.0, sr)
    lo = _hz_cap(3000.0, sr)
    if hi <= lo:
        hi = min(_hz_cap(0.49 * sr, sr), lo + 10.0)
    sos_bp = sps.iirfilter(2, Wn=[lo, hi], btype='bandpass', ftype='butter', fs=sr, output='sos')
    hf = sps.sosfiltfilt(sos_bp, x)
    env = np.sqrt(sps.convolve(hf**2, np.ones(int(0.1*sr))/(0.1*sr), mode='same') + 1e-12)
    # Normalize envelope to ~0..1
    env_n = env / (np.percentile(env, 95) + 1e-9)
    # Desired gain inversely related to envelope (more boost when HF low)
    gain = 1.0 + 0.6 * s * (1.0 - env_n)
    # Apply a smooth peaking EQ by mixing HF back proportionally
    y = x + (gain - 1.0) * hf
    # Safety
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def wow_flutter_reduce(x: np.ndarray, sr: int, strength: float = 0.0, engine: str = "torch") -> np.ndarray:
    """Very lightweight wow/flutter reduction using pitch envelope smoothing proxy.
    Not full pitch-correction: approximate by time-varying allpass-based detune compensation.
    For stronger correction, a dedicated model would be needed.
    strength in [0,1]."""
    s = float(max(0.0, min(1.0, strength)))
    if s <= 1e-3:
        return x.astype(np.float32)
    # Two modes:
    # - 'torch' (default): lightweight Hilbert-based frequency smoothing (no deps)
    # - 'crepe': use torchcrepe pitch tracking if available to estimate f0 and smooth it
    mode = (engine or "torch").lower()
    try:
        if mode == "crepe":
            try:
                import torch
                import torchcrepe
                # Prepare input tensor [batch, time]
                device = "cuda" if torch.cuda.is_available() else "cpu"
                x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
                # torchcrepe expects 16k; resample if needed
                target_sr = 16000
                if sr != target_sr:
                    try:
                        import torchaudio
                        x_t = torchaudio.functional.resample(x_t, sr, target_sr)
                    except Exception:
                        # numpy fallback
                        t = np.arange(len(x), dtype=np.float32)
                        t_new = np.linspace(0, len(x)-1, int(len(x) * target_sr / sr), dtype=np.float32)
                        x_np = np.interp(t_new, t, x).astype(np.float32)
                        x_t = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
                # Frame-level pitch
                f0, pd = torchcrepe.predict(x_t, sample_rate=target_sr, hop_length=160, fmin=50.0, fmax=8000.0,
                                            model='full', batch_size=1024, device=device, return_periodicity=True)
                f0 = torchcrepe.filter.median(torchcrepe.threshold.At(0.2)(f0, pd), 5)
                f0 = torchcrepe.filter.mean(f0, 5)
                f0_np = f0.squeeze(0).detach().cpu().numpy()
                # Upsample f0 track to sample rate timeline
                t_frames = np.linspace(0, len(x), num=len(f0_np), endpoint=False)
                t = np.arange(len(x))
                inst_freq = np.maximum(1.0, np.interp(t, t_frames, f0_np))
            except Exception:
                # Fallback to Hilbert if torch/crepe unavailable
                mode = "torch"
        if mode != "crepe":
            anal = sps.hilbert(x)
            phase = np.unwrap(np.angle(anal))
            inst_freq = np.diff(phase) * (sr / (2*np.pi))
            inst_freq = np.concatenate([[inst_freq[0]], inst_freq])

        # Smooth with low-pass (cutoff ~5 Hz)
        sos_lp = sps.iirfilter(2, Wn=_hz_cap(5.0, sr), btype='lowpass', ftype='butter', fs=sr, output='sos')
        smooth_freq = sps.sosfiltfilt(sos_lp, inst_freq)
        # Blend towards smoothed freq using strength
        target_freq = (1.0 - s) * inst_freq + s * smooth_freq
        # Reconstruct a monotonic time mapping by integrating target freq
        target_phase = np.cumsum(target_freq) * (2*np.pi/sr)
        # Normalize mapping to exactly cover the original duration [0, N-1]
        t = np.arange(len(x))
        target_t = target_phase - target_phase[0]
        target_t = target_t - np.min(target_t)
        denom = float(np.max(target_t) + 1e-9)
        target_t = target_t / denom * (len(x) - 1)
        target_t = np.clip(target_t, 0, len(x)-1)
        y = np.interp(t, target_t, x).astype(np.float32)
        # Final blend to avoid artifacts
        y = 0.7 * y + 0.3 * x
        # Normalize
        peak = float(np.max(np.abs(y)) + 1e-9)
        if peak > 1.0:
            y = y / peak
        return y.astype(np.float32)
    except Exception:
        return x.astype(np.float32)


def generative_enhance(
    x: np.ndarray,
    sr: int,
    engine: str = "none",
    mode: str = "full",      # 'full' or 'highs'
    mix: float = 0.0,         # 0..1
    target_sr: int = 48000,
) -> np.ndarray:
    """Optional generative super-resolution and blending.

    - engine: 'audiosr' to use AudioSR if available, otherwise 'none'.
    - mode: 'full' uses the full generated signal; 'highs' only adds high frequencies.
    - mix: dry/wet mix between original and generated component.
    Returns audio at original sample rate.
    """
    m = float(max(0.0, min(1.0, mix)))
    if m <= 1e-4 or engine not in ("audiosr",):
        return x.astype(np.float32)

    mono = False
    if x.ndim == 1:
        mono = True
        X = x[None, :]
    else:
        X = x

    outs = []
    for c in range(X.shape[0]):
        ch = X[c].astype(np.float32)
        gen_hi_sr = _audiosr_super_resolve(ch, sr=sr, target_sr=int(target_sr))
        gen = _resample_poly_safe(gen_hi_sr, target_sr, sr)
        n = min(len(ch), len(gen))
        ch = ch[:n]
        gen = gen[:n]
        if mode == "highs":
            gen = _highpass(gen, sr, fc=7000.0)
        y = (1.0 - m) * ch + m * gen
        # Safety limiter
        peak = float(np.max(np.abs(y)) + 1e-9)
        if peak > 1.0:
            y = y / peak * 0.99
        outs.append(y.astype(np.float32))
    if not outs:
        return x.astype(np.float32)
    if mono:
        return outs[0]
    return np.vstack(outs)
