"""Extract only the per-vowel measurements needed by the fixed eight features.

Input WAVs must already be segmented and labelled a/e/i by the data holder.
No patient audio or patient-level output belongs in the public repository.
"""

from pathlib import Path

import numpy as np


def _positive(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr) & (arr > 0)]


def _mean(values):
    valid = _positive(values)
    return float(valid.mean()) if len(valid) else float("nan")


def _cv(values):
    valid = _positive(values)
    return float(valid.std() / (valid.mean() + 1e-8)) if len(valid) else float("nan")


def _lpc(y, sr, librosa):
    start, end = int(len(y) * .2), int(len(y) * .8)
    segment = y[start:end] if end > start else y
    if len(segment) < int(.08 * sr):
        segment = y
    if len(segment) < int(.1 * sr):
        return float("nan"), float("nan")
    segment = np.append(segment[0], segment[1:] - .97 * segment[:-1])
    segment *= np.hamming(len(segment))
    try:
        order = max(10, min(18, len(segment) // 100))
        roots = np.roots(librosa.lpc(segment, order=order))
        roots = roots[np.imag(roots) >= .01]
        freqs = np.arctan2(np.imag(roots), np.real(roots)) * sr / (2 * np.pi)
        bandwidths = -.5 * sr / (2 * np.pi) * np.log(np.abs(roots))
        candidates = sorted(f for f, b in zip(freqs, bandwidths) if 90 < f < 5000 and 0 < b < 1000)
        return (float(candidates[0]) if len(candidates) > 0 else float("nan"),
                float(candidates[2]) if len(candidates) > 2 else float("nan"))
    except (ValueError, np.linalg.LinAlgError):
        return float("nan"), float("nan")


def _cpps(sound, parselmouth):
    try:
        cep = parselmouth.praat.call(sound, "To PowerCepstrogram...", 60., .002, 5000., 50.)
        return float(parselmouth.praat.call(
            cep, "Get CPPS...", 1, .02, .001, 60., 330., .05,
            "Parabolic", .001, .05, "Straight", "Robust slow"))
    except (ValueError, RuntimeError):
        return float("nan")


def _stable(sound, parselmouth):
    duration = float(sound.duration)
    if duration < .15:
        return float("nan"), float("nan")
    try:
        formant = parselmouth.praat.call(sound, "To Formant (burg)", 0., 5, 5500, .025, 50)
        pitch = parselmouth.praat.call(sound, "To Pitch", 0., 75, 300)
    except (ValueError, RuntimeError):
        return float("nan"), float("nan")
    window = min(.24, max(.16, duration * .25))
    half = window / 2
    lo, hi = max(half, duration * .2), min(duration - half, duration * .8)
    if hi <= lo:
        lo, hi = half, max(half, duration - half)
    centers = np.arange(lo, hi + 1e-8, .05)
    if len(centers) == 0:
        centers = np.array([duration / 2])
    best_score = float("inf")
    best_b1 = float("nan")
    for center in centers:
        times = np.clip(np.linspace(center - half, center + half, 5), 0, duration)
        try:
            f1 = [parselmouth.praat.call(formant, "Get value at time", 1, float(t), "Hertz", "Linear") for t in times]
            f2 = [parselmouth.praat.call(formant, "Get value at time", 2, float(t), "Hertz", "Linear") for t in times]
            f0 = [parselmouth.praat.call(pitch, "Get value at time", float(t), "Hertz", "Linear") for t in times]
            b1 = [parselmouth.praat.call(formant, "Get bandwidth at time", 1, float(t), "Hertz", "Linear") for t in times]
        except (ValueError, RuntimeError):
            continue
        valid = np.isfinite(f1) & (np.asarray(f1) > 0) & np.isfinite(f2) & (np.asarray(f2) > 0) & np.isfinite(f0) & (np.asarray(f0) > 0)
        ratio = float(np.mean(valid))
        if ratio < .55:
            continue
        terms = [v for v in (_cv(f1), _cv(f2), _cv(f0)) if np.isfinite(v)]
        if not terms:
            continue
        score = float(np.mean(terms) + 2 * (1 - ratio))
        if score < best_score:
            best_score, best_b1 = score, _mean(b1)
    return best_b1, best_score


def extract_vowel(path: str | Path) -> dict[str, float]:
    """Measure LPC F1/F3, spectral, harmonic, CPPS and stable-window terms."""
    try:
        import librosa
        import parselmouth
    except ImportError as exc:
        raise RuntimeError("Install requirements-audio.txt for WAV extraction") from exc
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    y, sr = librosa.load(path, sr=16000, mono=True)
    if len(y) < int(.15 * sr):
        raise ValueError(f"Audio segment shorter than 0.15 seconds: {path}")
    sound = parselmouth.Sound(str(path))
    y = librosa.util.normalize(y)
    f1, f3 = _lpc(y, sr, librosa)
    stable_b1, stable_score = _stable(sound, parselmouth)
    spectrum = np.abs(librosa.stft(y, n_fft=512, hop_length=160, win_length=320))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    def power(lo, hi):
        band = spectrum[(freqs >= lo) & (freqs < hi)]
        return float(np.mean(band ** 2)) if band.size else float("nan")
    high = power(3000, 8000)
    mid_high = power(1000, 3000) / high if high > 0 else float("nan")
    rolloff = float(np.nanmean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=.95)))
    h13 = float("nan")
    try:
        f0 = librosa.yin(y, fmin=70, fmax=400, sr=sr)
        f0 = f0[np.isfinite(f0)]
        if len(f0):
            target = float(np.median(f0))
            center = len(y) // 2
            half = int(.25 * sr / 2)
            window = y[max(0, center - half):min(len(y), center + half)]
            amp = np.abs(np.fft.rfft(window * np.hanning(len(window))))
            hz = np.fft.rfftfreq(len(window), 1 / sr)
            peaks = [amp[np.abs(hz - k * target) <= 30] for k in (1, 3)]
            if all(p.size and p.max() > 0 for p in peaks):
                h13 = float(20 * np.log10(peaks[0].max() / peaks[1].max()))
    except (ValueError, np.linalg.LinAlgError):
        pass
    return {"lpc_F1": f1, "lpc_F3": f3, "mid_high_energy_ratio": mid_high,
            "b1_stable": stable_b1, "stable_score": stable_score,
            "rolloff95_mean": rolloff, "h1_h3_db": h13,
            "cpps_full": _cpps(sound, parselmouth)}
