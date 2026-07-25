"""
WORLD vocoder front-end (pyworld).

WORLD decomposes speech into three independent streams:

    f0   -- fundamental frequency per frame      (the pitch)
    sp   -- spectral envelope per frame          (the vowel identity / formants)
    ap   -- aperiodicity per frame               (breath vs. tone)

Because the three are separable, a note can be re-pitched by *replacing f0* while
the spectral envelope is untouched -- the vowel keeps its identity and the voice
keeps its timbre, with none of the grain/splice artefacts of hand-rolled PSOLA.
Time is stretched by resampling the frame sequence, which is likewise artefact
free: holding a vowel just means holding its frames.

This is the same decomposition professional singing synthesisers are built on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME_PERIOD = 5.0        # ms between analysis frames


@dataclass
class Frames:
    """WORLD analysis of one utterance."""
    f0: np.ndarray            # (n_frames,)
    sp: np.ndarray            # (n_frames, n_bins)
    ap: np.ndarray            # (n_frames, n_bins)
    sr: int
    frame_period: float = FRAME_PERIOD

    @property
    def n(self) -> int:
        return self.f0.size

    def samples_to_frame(self, s):
        """Sample index -> (fractional) frame index."""
        return np.asarray(s) / (self.frame_period / 1000.0 * self.sr)

    def frames_to_samples(self, f):
        return np.asarray(f) * (self.frame_period / 1000.0 * self.sr)


def analyze(x, sr, frame_period=FRAME_PERIOD) -> Frames:
    """Full WORLD analysis (harvest → cheaptrick → d4c)."""
    import pyworld as pw

    x = np.asarray(x, dtype=np.float64)
    f0, t = pw.harvest(x, sr, frame_period=frame_period)
    f0 = pw.stonemask(x, f0, t, sr)                 # refine the f0 estimate
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    return Frames(f0, sp, ap, sr, frame_period)


def resample_frames(fr: Frames, frame_index: np.ndarray) -> Frames:
    """Read the analysis along an arbitrary (fractional) frame trajectory.

    ``frame_index[i]`` is which analysis frame output frame ``i`` comes from, so
    this performs the time-warp: slow the trajectory to sustain a vowel, keep it
    at 1.0 to play a consonant at natural speed.
    """
    idx = np.clip(np.asarray(frame_index, dtype=np.float64), 0, fr.n - 1)
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(lo + 1, fr.n - 1)
    w = (idx - lo)[:, None]
    sp = fr.sp[lo] * (1.0 - w) + fr.sp[hi] * w
    ap = np.clip(fr.ap[lo] * (1.0 - w) + fr.ap[hi] * w, 0.0, 1.0)
    # f0 is interpolated only where both endpoints are voiced, so voiced frames
    # never bleed into unvoiced ones (which would buzz through consonants).
    f0lo, f0hi = fr.f0[lo], fr.f0[hi]
    both = (f0lo > 0) & (f0hi > 0)
    f0 = np.where(both, f0lo * (1.0 - w[:, 0]) + f0hi * w[:, 0],
                  np.where(w[:, 0] < 0.5, f0lo, f0hi))
    return Frames(f0, sp, ap, fr.sr, fr.frame_period)


def synthesize(fr: Frames) -> np.ndarray:
    """WORLD synthesis back to a waveform."""
    import pyworld as pw

    y = pw.synthesize(np.ascontiguousarray(fr.f0, dtype=np.float64),
                      np.ascontiguousarray(fr.sp, dtype=np.float64),
                      np.ascontiguousarray(fr.ap, dtype=np.float64),
                      fr.sr, fr.frame_period)
    return np.asarray(y, dtype=np.float32)


def shift_formants(fr: Frames, ratio: float) -> Frames:
    """Scale the spectral envelope along frequency (vocal-tract size).

    ``ratio`` > 1 raises the formants -- a smaller, younger-sounding tract --
    independently of pitch.
    """
    if abs(ratio - 1.0) < 1e-3:
        return fr
    bins = fr.sp.shape[1]
    src = np.clip(np.arange(bins) / ratio, 0, bins - 1)
    lo = np.floor(src).astype(np.int64)
    hi = np.minimum(lo + 1, bins - 1)
    w = src - lo
    sp = fr.sp[:, lo] * (1.0 - w) + fr.sp[:, hi] * w
    return Frames(fr.f0, sp, fr.ap, fr.sr, fr.frame_period)


def breathiness(fr: Frames, amount: float) -> Frames:
    """Blend the aperiodicity toward more breath (``amount`` > 0) or less."""
    if abs(amount) < 1e-3:
        return fr
    ap = np.clip(fr.ap + amount * (1.0 - fr.ap), 0.0, 1.0) if amount > 0 \
        else np.clip(fr.ap * (1.0 + amount), 0.0, 1.0)
    return Frames(fr.f0, fr.sp, ap, fr.sr, fr.frame_period)
