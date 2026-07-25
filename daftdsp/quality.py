"""
Objective audio-quality metrics.

The singing/robot voices are tuned against measurements rather than opinion:
these functions turn "does it sound broken?" into numbers that tests can assert
on and that a report tool can print.  Pure functions, numpy only, no side effects.

Metrics and the failure each one catches:

===========================  =================================================
metric                       catches
===========================  =================================================
:func:`discontinuity`        clicks / glitches / splice pops
:func:`headroom`             clipping, DC offset, over-compression
:func:`pitch_errors`         out-of-tune or mis-assigned notes
:func:`sustain_flatness`     held notes that decay, pump or re-trigger
:func:`dropout_ratio`        words/syllables that vanish mid-phrase
:func:`spectral_balance`     tone that is boxy, harsh or thin
===========================  =================================================
"""
from __future__ import annotations

import numpy as np

from . import pitch as _pitch
from .util import midi_to_freq

BANDS = ((80, 250), (250, 500), (500, 1000), (1000, 2000),
         (2000, 3500), (3500, 5000), (5000, 8000), (8000, 14000))


def to_mono(x):
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=1) if x.ndim == 2 else x


# ---------------------------------------------------------------------------
# waveform integrity
# ---------------------------------------------------------------------------

def discontinuity(x, factor=25.0):
    """Count samples whose step is far larger than the signal's typical step.

    A click/splice pop shows up as a single huge sample-to-sample jump, so
    ``count`` should be ~0 for clean audio (a few is normal for real plosives).
    """
    x = to_mono(x)
    if x.size < 3:
        return {"count": 0, "rate": 0.0}
    d = np.abs(np.diff(x))
    nz = d[d > 0]
    med = float(np.median(nz)) if nz.size else 0.0
    if med <= 0:
        return {"count": 0, "rate": 0.0}
    count = int(np.sum(d > factor * med))
    return {"count": count, "rate": count / d.size}


def headroom(x):
    """Peak/RMS/crest, clipped-sample count and DC offset."""
    m = to_mono(x)
    peak = float(np.max(np.abs(m))) if m.size else 0.0
    rms = float(np.sqrt(np.mean(m ** 2))) if m.size else 0.0
    crest = float(peak / rms) if rms > 1e-9 else 0.0
    return {"peak": peak, "rms": rms, "crest": crest,
            "clipped": int(np.sum(np.abs(m) >= 0.999)),
            "dc": float(np.mean(m)) if m.size else 0.0,
            "finite": bool(np.all(np.isfinite(m)))}


# ---------------------------------------------------------------------------
# musical accuracy
# ---------------------------------------------------------------------------

def pitch_errors(x, sr, timeline, max_f0=1200.0, lo=0.45, hi=0.85):
    """Per-note pitch error in cents.

    ``timeline`` is a list of ``(start_s, dur_s, midi)``.  Each note is sampled
    over the ``lo``..``hi`` fraction of its slot (skipping the attack/release)
    and compared with the intended pitch.
    """
    m = to_mono(x)
    track = _pitch.track_pitch(m, sr, max_f0=max_f0)
    f0s, _ = track.to_per_sample(m.size)
    rows = []
    for start, dur, midi in timeline:
        a = int((start + dur * lo) * sr)
        b = int((start + dur * hi) * sr)
        seg = f0s[max(0, a):max(0, b)]
        seg = seg[seg > 0]
        want = midi_to_freq(midi)
        if seg.size:
            det = float(np.median(seg))
            cents = 1200.0 * np.log2(det / want)
        else:
            det, cents = 0.0, float("nan")
        rows.append({"midi": midi, "target_hz": want, "detected_hz": det,
                     "cents": cents})
    valid = np.array([r["cents"] for r in rows if np.isfinite(r["cents"])])
    stats = {
        "notes": len(rows),
        "measured": int(valid.size),
        "mean_abs_cents": float(np.mean(np.abs(valid))) if valid.size else float("nan"),
        "median_abs_cents": float(np.median(np.abs(valid))) if valid.size else float("nan"),
        "within_50c": int(np.sum(np.abs(valid) < 50)),
        "within_100c": int(np.sum(np.abs(valid) < 100)),
    }
    return rows, stats


# ---------------------------------------------------------------------------
# sustain / continuity
# ---------------------------------------------------------------------------

def _frame_rms(x, sr, frame_ms=20.0):
    n = max(1, int(frame_ms / 1000.0 * sr))
    if x.size < n:
        return np.array([np.sqrt(np.mean(x ** 2))]) if x.size else np.array([0.0])
    trim = x[:x.size - x.size % n].reshape(-1, n)
    return np.sqrt((trim ** 2).mean(axis=1))


def sustain_flatness(x, sr, skip=0.2):
    """Coefficient of variation of the loudness over the sustained middle.

    Low = a steady held note; high = decay, tremolo or re-triggering.
    """
    m = to_mono(x)
    r = _frame_rms(m, sr)
    if r.size < 6:
        return float("nan")
    k = max(1, int(r.size * skip))
    body = r[k:r.size - k]
    body = body[body > 0.05 * (r.max() or 1.0)]
    if body.size < 3:
        return float("nan")
    return float(np.std(body) / (np.mean(body) + 1e-12))


def dropout_ratio(x, sr, timeline=None, floor=0.08, frame_ms=20.0):
    """Fraction of *sung* audio that is near-silent — syllables that vanish.

    With a ``timeline`` only the note slots are examined, so scored rests (which
    are supposed to be silent) are not counted as defects; this is the meaningful
    reading.  Without one it falls back to the whole active region, which
    over-reports on music that contains rests.
    """
    m = to_mono(x)
    r = _frame_rms(m, sr, frame_ms)
    if r.size == 0:
        return 1.0
    peak = float(r.max()) or 1.0
    if timeline:
        fps = 1000.0 / frame_ms
        keep = np.zeros(r.size, dtype=bool)
        for start, dur, _ in timeline:
            a = int(start * fps)
            b = min(r.size, int((start + dur) * fps))
            if b > a:
                keep[a:b] = True
        span = r[keep]
    else:
        active = np.where(r > 0.05 * peak)[0]
        if active.size < 2:
            return 1.0
        span = r[active[0]:active[-1] + 1]
    live = span[span > 0.05 * peak]
    if live.size < 3:
        return 1.0
    ref = float(np.median(live))
    return float(np.mean(span < floor * ref))


# ---------------------------------------------------------------------------
# tone
# ---------------------------------------------------------------------------

def spectral_balance(x, sr, bands=BANDS):
    """Percentage of spectral energy per band, plus the centroid in Hz."""
    m = to_mono(x)
    if m.size < 1024:
        return {"centroid": 0.0, "bands": {}}
    seg = m[int(m.size * 0.2):int(m.size * 0.8)]
    if seg.size < 1024:
        seg = m
    sp = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    fr = np.fft.rfftfreq(seg.size, 1.0 / sr)
    tot = float(sp.sum()) or 1.0
    out = {f"{lo}-{hi}": round(100.0 * float(sp[(fr >= lo) & (fr < hi)].sum()) / tot, 1)
           for lo, hi in bands}
    return {"centroid": float((fr * sp).sum() / tot), "bands": out}


# ---------------------------------------------------------------------------
# roll-up
# ---------------------------------------------------------------------------

def summarize(x, sr, timeline=None):
    """Full metric set as one dict (``timeline`` enables the pitch metrics)."""
    rep = {
        "duration_s": round(to_mono(x).size / sr, 2),
        "headroom": headroom(x),
        "discontinuity": discontinuity(x),
        "sustain_cv": sustain_flatness(x, sr),
        "dropout": dropout_ratio(x, sr, timeline),
        "spectrum": spectral_balance(x, sr),
    }
    if timeline:
        _, stats = pitch_errors(x, sr, timeline)
        rep["pitch"] = stats
    return rep


def format_report(rep, name=""):
    """Human-readable one-block summary of :func:`summarize` output."""
    h, d, s = rep["headroom"], rep["discontinuity"], rep["spectrum"]
    lines = [f"── {name} ({rep['duration_s']}s)"]
    lines.append(f"   level    peak {h['peak']:.2f}  rms {h['rms']:.3f}  "
                 f"crest {h['crest']:.1f}  clipped {h['clipped']}  dc {h['dc']:+.4f}")
    lines.append(f"   clean    clicks {d['count']}  sustain-cv {rep['sustain_cv']:.3f}  "
                 f"dropout {rep['dropout']:.3f}")
    if "pitch" in rep:
        p = rep["pitch"]
        lines.append(f"   pitch    {p['within_50c']}/{p['notes']} within 50c  "
                     f"mean |err| {p['mean_abs_cents']:.0f}c")
    lines.append(f"   tone     centroid {s['centroid']:.0f} Hz  " +
                 " ".join(f"{k}:{v}%" for k, v in list(s["bands"].items())[:5]))
    return "\n".join(lines)
