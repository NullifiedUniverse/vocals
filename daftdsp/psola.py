"""
Time-Domain Pitch-Synchronous Overlap-Add (TD-PSOLA) hard auto-tune.

TD-PSOLA shifts pitch while *preserving formants*: it chops the voice into
pitch-synchronous, Hann-windowed grains and re-lays them at a new period taken
from the quantised (in-scale) target pitch.  Because each grain keeps its own
spectral envelope, the vocal-tract colour is unchanged -- only the perceived
pitch snaps.  ``retune`` sets correction depth (1 = hard snap) and
``retune_time_ms`` the glide time (near-zero = the instant Daft-Punk snap).
"""
from __future__ import annotations

import numpy as np

from .pitch import PitchTrack, quantize_track


def _grain(x, center, half):
    """Extract a Hann-windowed grain of half-length ``half`` centred at ``center``
    (zero-padded at the signal edges).  Returns (grain, start_index)."""
    n = x.size
    start = center - half
    stop = center + half
    lo = max(0, start)
    hi = min(n, stop)
    length = 2 * half
    grain = np.zeros(length, dtype=np.float64)
    if hi > lo:
        grain[lo - start:hi - start] = x[lo:hi]
    # Hann window (period 2*half).
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / max(1, length - 1))
    return grain * w, start


def psola_correct(x, sr, track: PitchTrack, *, key_root=0, scale="chromatic",
                  retune=1.0, retune_time_ms=1.0, min_f0=70.0, max_f0=500.0):
    """Return ``x`` with its pitch snapped toward ``scale``.

    ``retune`` in [0, 1] blends between the original pitch (0) and the fully
    quantised pitch (1).  ``retune_time_ms`` smooths the correction trajectory.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0 or retune <= 0.0:
        return x.astype(np.float32)

    f0_s, v_s = track.to_per_sample(n)
    quant_frames = quantize_track(track.f0, key_root, scale)
    quant_s = np.interp(np.arange(n), track.frame_centers, quant_frames)

    # Correction ratio per sample, then smoothed over `retune_time_ms`.
    ratio = np.ones(n, dtype=np.float64)
    good = v_s & (f0_s > 0) & (quant_s > 0)
    ratio[good] = (quant_s[good] / f0_s[good]) ** float(np.clip(retune, 0.0, 1.0))
    ratio = _one_pole_smooth(ratio, sr, max(0.1, retune_time_ms))

    target_f0 = np.where(good, f0_s * ratio, f0_s)

    per_min = sr / max_f0
    per_max = sr / min_f0

    def local_period(i, hz):
        if hz > 0:
            return float(np.clip(sr / hz, per_min, per_max))
        return float(np.clip(sr / 150.0, per_min, per_max))

    # Analysis pitch marks span the whole signal (so consonants get grains too).
    marks = []
    periods = []
    t = 0.0
    i = 0
    while i < n:
        hz = f0_s[i] if v_s[i] else 0.0
        p = local_period(i, hz)
        marks.append(i)
        periods.append(p)
        t += p
        i = int(round(t))
    marks = np.array(marks, dtype=np.int64)
    periods = np.array(periods, dtype=np.float64)
    if marks.size < 2:
        return x.astype(np.float32)

    out = np.zeros(n, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)

    # Synthesis: place grains at the target (possibly re-pitched) spacing.
    s = float(marks[0])
    while s < n:
        si = int(round(s))
        # Nearest analysis mark to this output position (no time-stretch).
        k = int(np.searchsorted(marks, si))
        if k >= marks.size:
            k = marks.size - 1
        elif k > 0 and abs(marks[k - 1] - si) <= abs(marks[k] - si):
            k -= 1
        a = int(marks[k])
        pa = periods[k]
        half = max(2, int(round(pa)))
        grain, gstart = _grain(x, a, half)
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(grain.size) /
                               max(1, grain.size - 1))

        # OLA the grain at the synthesis position.
        dst0 = si - half
        lo = max(0, dst0)
        hi = min(n, dst0 + grain.size)
        if hi > lo:
            out[lo:hi] += grain[lo - dst0:hi - dst0]
            norm[lo:hi] += w[lo - dst0:hi - dst0]

        # Advance by the target period (voiced) or analysis period (unvoiced).
        if v_s[min(si, n - 1)] and target_f0[min(si, n - 1)] > 0:
            ps = float(np.clip(sr / target_f0[min(si, n - 1)], per_min, per_max))
        else:
            ps = pa
        s += ps

    # Normalise overlap; keep dry signal where nothing was written.
    mask = norm > 1e-6
    out[mask] /= norm[mask]
    out[~mask] = x[~mask]
    return out.astype(np.float32)


def _one_pole_smooth(sig, sr, time_ms):
    """Symmetric-ish one-pole low-pass used to slow the retune trajectory."""
    tau = max(1e-4, time_ms / 1000.0)
    a = np.exp(-1.0 / (tau * sr))
    y = np.empty_like(sig)
    acc = sig[0]
    for i in range(sig.size):
        acc = a * acc + (1.0 - a) * sig[i]
        y[i] = acc
    return y
