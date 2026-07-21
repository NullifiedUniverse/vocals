"""
Post-processing effects chain (all hand-written):

  * :func:`saturate`   -- tanh soft-clip wave-shaper.
  * :func:`phaser`     -- 4-8 LFO-swept first-order all-passes with feedback.
  * :func:`make_kick` / :func:`sidechain` -- four-on-the-floor kick + ducking
                          compressor for the Daft-Punk "pump".
  * :func:`eq`         -- 150 Hz high-pass + 6 kHz high-shelf.
  * :func:`stereoize`  -- 20 ms Haas delay + mid/side width -> stereo output.
"""
from __future__ import annotations

import numpy as np

from .biquad import biquad_fft, high_shelf, highpass, lowpass


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------

def saturate(x, drive=2.0, mix=1.0):
    """tanh soft-clip.  ``drive`` >= 1 adds harmonics; output is level-matched."""
    x = np.asarray(x, dtype=np.float64)
    if drive <= 0:
        return x.astype(np.float32)
    norm = np.tanh(drive)
    wet = np.tanh(drive * x) / (norm if norm > 1e-6 else 1.0)
    out = (1.0 - mix) * x + mix * wet
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Phaser
# ---------------------------------------------------------------------------

def phaser(x, sr, *, rate=0.4, depth=0.7, stages=6, feedback=0.4,
           f_min=250.0, f_max=1800.0):
    """Cascade of first-order all-passes swept by an LFO, mixed with the dry
    signal to create moving notches."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0 or depth <= 0.0:
        return x.astype(np.float32)
    stages = int(max(2, min(8, stages)))

    # Precompute the per-sample all-pass coefficient from the swept centre freq.
    t = np.arange(n) / sr
    lfo = 0.5 - 0.5 * np.cos(2.0 * np.pi * rate * t)
    fc = f_min * (f_max / f_min) ** lfo
    tan = np.tan(np.pi * np.clip(fc, 20.0, sr * 0.45) / sr)
    g = (tan - 1.0) / (tan + 1.0)

    fb = float(np.clip(feedback, 0.0, 0.95))
    mix = float(np.clip(depth, 0.0, 1.0))
    xs = np.zeros(stages)
    ys = np.zeros(stages)
    out = np.empty(n)
    zm1 = 0.0
    for i in range(n):
        gi = g[i]
        s = x[i] + fb * zm1
        for k in range(stages):
            y = -gi * s + xs[k] + gi * ys[k]
            xs[k] = s
            ys[k] = y
            s = y
        zm1 = s
        out[i] = (1.0 - 0.5 * mix) * x[i] + mix * s
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Sidechain compressor
# ---------------------------------------------------------------------------

def make_kick(n, sr, bpm=120.0, level=0.9):
    """Generate a four-on-the-floor kick trigger track of ``n`` samples."""
    out = np.zeros(n, dtype=np.float32)
    beat = int(sr * 60.0 / max(40.0, bpm))
    if beat <= 0:
        return out
    dur = int(0.18 * sr)
    idx = np.arange(dur)
    # Pitch-dropping sine thump with a fast decay.
    pitch = 120.0 * np.exp(-idx / (0.03 * sr)) + 45.0
    phase = np.cumsum(2.0 * np.pi * pitch / sr)
    thump = (np.sin(phase) * np.exp(-idx / (0.05 * sr))).astype(np.float32)
    pos = 0
    while pos < n:
        end = min(n, pos + dur)
        out[pos:end] += level * thump[:end - pos]
        pos += beat
    return out


def sidechain(x, trigger, sr, *, threshold_db=-24.0, ratio=6.0, attack_ms=5.0,
              release_ms=180.0, amount=1.0):
    """Duck ``x`` (mono or stereo) whenever ``trigger`` (kick) is loud."""
    x = np.asarray(x, dtype=np.float64)
    if amount <= 0.0 or trigger is None or len(trigger) == 0:
        return x.astype(np.float32)
    trig = np.abs(np.asarray(trigger, dtype=np.float64))
    n = x.shape[0]
    if trig.size < n:
        trig = np.pad(trig, (0, n - trig.size))
    else:
        trig = trig[:n]

    aa = np.exp(-1.0 / (max(0.1, attack_ms) / 1000.0 * sr))
    ar = np.exp(-1.0 / (max(1.0, release_ms) / 1000.0 * sr))

    # Detector envelope of the trigger.
    env = 0.0
    det = np.empty(n)
    for i in range(n):
        v = trig[i]
        coeff = aa if v > env else ar
        env = coeff * env + (1.0 - coeff) * v
        det[i] = env

    det_db = 20.0 * np.log10(np.maximum(det, 1e-6))
    over = np.maximum(0.0, det_db - threshold_db)
    gain_db = -over * (1.0 - 1.0 / max(1.0, ratio)) * amount
    gain = 10.0 ** (gain_db / 20.0)

    if x.ndim == 1:
        out = x * gain
    else:
        out = x * gain[:, None]
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# EQ + stereo
# ---------------------------------------------------------------------------

def eq(x, sr, *, hp_freq=150.0, shelf_freq=6000.0, shelf_gain_db=4.0):
    """150 Hz high-pass + high-shelf boost (both applied in one FFT pass)."""
    chain = [highpass(hp_freq, sr, 0.707)]
    if abs(shelf_gain_db) > 0.01:
        chain.append(high_shelf(shelf_freq, sr, shelf_gain_db))
    return biquad_fft(chain, x)


def _comb(x, delay, g):
    """Feedback comb y[n] = x[n] + g*y[n-D], vectorised a delay-block at a time
    (within one block of length ``delay`` there is no self-dependency)."""
    n = x.size
    d = max(1, int(delay))
    y = x.astype(np.float64).copy()
    for s in range(d, n, d):
        e = min(n, s + d)
        m = e - s
        y[s:e] += g * y[s - d:s - d + m]
    return y


def _allpass(x, delay, g):
    n = x.size
    d = max(1, int(delay))
    y = np.zeros(n)
    xd = np.zeros(n)
    xd[d:] = x[:n - d]
    for s in range(0, n, d):
        e = min(n, s + d)
        m = e - s
        yprev = y[s - d:s - d + m] if s >= d else np.zeros(m)
        y[s:e] = -g * x[s:e] + xd[s:e] + g * yprev
    return y


def reverb(stereo, sr, *, mix=0.25, size=0.6, damp=0.5, width=1.0):
    """Schroeder/Moorer stereo reverb (four combs + two all-passes per side),
    all hand-written.  Adds the space/tail that makes the voice sit and breathe."""
    if mix <= 0.0:
        return stereo
    x = np.asarray(stereo, dtype=np.float64)
    send = x.mean(axis=1)                         # mono reverb send
    scale = 0.7 + 1.1 * float(np.clip(size, 0.0, 1.0))
    g = 0.72 + 0.24 * float(np.clip(size, 0.0, 1.0))
    dmp = float(np.clip(damp, 0.0, 0.95))

    comb_ms = np.array([29.7, 37.1, 41.1, 43.7])
    ap_ms = np.array([5.0, 1.7])
    # Damping = a low-pass on the wet tail (darker as `damp` rises).
    cutoff = 12000.0 * (1.0 - dmp) + 1800.0 * dmp

    def one_side(offset):
        wet = np.zeros(send.size)
        for cm in comb_ms:
            wet += _comb(send, (cm + offset) * 1e-3 * sr * scale, g)
        wet /= len(comb_ms)
        for am in ap_ms:
            wet = _allpass(wet, am * 1e-3 * sr * scale, 0.7)
        return biquad_fft(lowpass(cutoff, sr, 0.6), wet)

    left = one_side(0.0)
    right = one_side(0.9)                          # detune delays -> stereo spread
    mono = 0.5 * (left + right)
    sidew = 0.5 * (left - right) * width
    wl = mono + sidew
    wr = mono - sidew
    m = max(float(np.max(np.abs([wl, wr]))), 1e-9)
    wet = np.stack([wl, wr], axis=1) / m
    return ((1.0 - mix) * x + mix * wet).astype(np.float32)


def stereoize(x, sr, *, haas_ms=20.0, width=1.0, level=1.0):
    """Turn mono ``x`` into a stereo pair using a Haas delay + mid/side width."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    delay = int(max(0.0, haas_ms) / 1000.0 * sr)
    right = np.zeros(n)
    if delay > 0 and delay < n:
        right[delay:] = x[:n - delay]
    else:
        right[:] = x
    left = x

    mid = 0.5 * (left + right)
    side = 0.5 * (left - right) * width
    l = (mid + side) * level
    r = (mid - side) * level
    return np.stack([l, r], axis=1).astype(np.float32)
