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
from .util import midi_to_freq


def melody_target(track: PitchTrack, sr, n, melody_midi, gap_ms=70.0,
                  vibrato_rate=5.5, vibrato_depth=0.0):
    """Build a per-sample target-frequency line that assigns successive melody
    notes to successive *sung syllables*.

    Voiced runs (short unvoiced gaps bridged) are treated as syllables; note
    ``i`` of the melody drives syllable ``i`` (cycling if the melody is short).
    A little vibrato can be layered on for a less machine-flat delivery."""
    if not len(melody_midi):
        return np.zeros(n)
    _, v_s = track.to_per_sample(n)

    runs = []
    i = 0
    while i < n:
        if v_s[i]:
            j = i
            while j < n and v_s[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    gap = int(gap_ms / 1000.0 * sr)
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < gap:
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))

    target = np.zeros(n)
    for idx, (s, e) in enumerate(merged):
        target[s:e] = midi_to_freq(melody_midi[idx % len(melody_midi)])
    if vibrato_depth > 0.0:
        t = np.arange(n) / sr
        vib = 2.0 ** (vibrato_depth * np.sin(2.0 * np.pi * vibrato_rate * t) / 12.0)
        target = target * vib
    return target


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


def psola_correct(x, sr, track: PitchTrack, target_hz=None, *, key_root=0,
                  scale="chromatic", retune=1.0, retune_time_ms=1.0,
                  min_f0=70.0, max_f0=700.0):
    """Pitch-shift ``x`` toward a target, preserving formants.

    If ``target_hz`` (a per-sample target-frequency array) is given the voice is
    sung onto that line -- this is how a melody drives the voice, Vocaloid-style.
    Otherwise the pitch is snapped to ``scale``.  ``retune`` in [0, 1] blends
    original->target; ``retune_time_ms`` sets the glide (portamento) time.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0 or retune <= 0.0:
        return x.astype(np.float32)

    f0_s, v_s = track.to_per_sample(n)
    if target_hz is not None:
        tgt = np.asarray(target_hz, dtype=np.float64)
        if tgt.size < n:
            tgt = np.pad(tgt, (0, n - tgt.size))
        else:
            tgt = tgt[:n]
    else:
        quant_frames = quantize_track(track.f0, key_root, scale)
        tgt = np.interp(np.arange(n), track.frame_centers, quant_frames)

    # Correction ratio per sample, then smoothed over `retune_time_ms` (glide).
    ratio = np.ones(n, dtype=np.float64)
    good = v_s & (f0_s > 0) & (tgt > 0)
    ratio[good] = (tgt[good] / f0_s[good]) ** float(np.clip(retune, 0.0, 1.0))
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
