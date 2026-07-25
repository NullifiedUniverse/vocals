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
    # Reference the typical step of the *loud* part of the signal.  Taking it
    # over everything lets long near-silences drive the median to ~0, which makes
    # ordinary speech look like it is full of clicks.
    amp = np.abs(x[:-1])
    active = d[amp > 0.05 * (float(amp.max()) or 1.0)]
    ref = active if active.size > 32 else d[d > 0]
    med = float(np.median(ref)) if ref.size else 0.0
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

    Detected pitches are **octave-folded** toward the target before the error is
    computed, and the number of folds is reported as ``octave_flips``.  A sung
    vowel can have a weak fundamental with strong upper harmonics ("missing
    fundamental"), where every pitch tracker reads an octave high even though the
    waveform really is periodic at the intended note and is heard that way.
    Folding measures the perceived pitch; the flip count keeps it visible.
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
            raw_cents = 1200.0 * np.log2(det / want)
            octaves = int(round(raw_cents / 1200.0))
            cents = raw_cents - 1200.0 * octaves
        else:
            det, raw_cents, octaves, cents = 0.0, float("nan"), 0, float("nan")
        rows.append({"midi": midi, "target_hz": want, "detected_hz": det,
                     "cents": cents, "raw_cents": raw_cents, "octaves": octaves})
    valid = np.array([r["cents"] for r in rows if np.isfinite(r["cents"])])
    flips = int(sum(1 for r in rows if r["octaves"]))
    stats = {
        "notes": len(rows),
        "measured": int(valid.size),
        "mean_abs_cents": float(np.mean(np.abs(valid))) if valid.size else float("nan"),
        "median_abs_cents": float(np.median(np.abs(valid))) if valid.size else float("nan"),
        "within_50c": int(np.sum(np.abs(valid) < 50)),
        "within_100c": int(np.sum(np.abs(valid) < 100)),
        "octave_flips": flips,
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

def sung_note(x, sr, midi=None):
    """Measure one sustained note against how a human singer behaves.

    Returned values and the human range each should fall in:

    ``vibrato_hz``      5.5-7.5   rate of the pitch modulation
    ``vibrato_cents``   30-60     its extent (peak deviation)
    ``pitch_drift``     < 25      steadiness of the note's centre, in cents
    ``amp_ripple``      0.03-0.12 how much the loudness moves during the sustain
    ``decay_ratio``     ~1.0      end-of-sustain level / start-of-sustain level;
                                  well below 1 means the note faded out

    Pitch statistics are taken from the *voiced, in-range* part only, because a
    tracker's octave slips on the unvoiced tail otherwise dominate the numbers.
    """
    m = to_mono(x)
    tr = _pitch.track_pitch(m, sr, max_f0=1200.0)
    f0, _ = tr.to_per_sample(m.size)
    v = f0[f0 > 0]
    if v.size < sr // 50:
        return {}
    centre = float(np.median(v))
    keep = v[(v > centre * 0.7) & (v < centre * 1.4)]        # drop octave slips
    cents = 1200.0 * np.log2(keep / centre) if keep.size else np.zeros(1)

    c = cents - cents.mean()
    spec = np.abs(np.fft.rfft(c * np.hanning(c.size))) if c.size > 64 else None
    vib_hz = vib_cents = 0.0
    if spec is not None:
        fr = np.fft.rfftfreq(c.size, 1.0 / sr)
        band = (fr >= 3.0) & (fr <= 9.0)
        if band.any():
            vib_hz = float(fr[band][np.argmax(spec[band])])
            vib_cents = float(np.sqrt(2.0) * np.std(c))      # peak of a sinusoid

    env = _frame_rms(m, sr, 25.0)
    live = env[env > 0.2 * (env.max() or 1.0)]
    ripple = float(np.std(live) / (np.mean(live) + 1e-12)) if live.size else 0.0
    if live.size >= 6:
        head = float(np.mean(live[:max(1, live.size // 4)]))
        tail = float(np.mean(live[-max(1, live.size // 4):]))
        decay = tail / (head + 1e-12)
    else:
        decay = 1.0

    out = {"vibrato_hz": vib_hz, "vibrato_cents": vib_cents,
           "pitch_drift": float(np.std(cents)), "amp_ripple": ripple,
           "decay_ratio": float(decay), "centre_hz": centre}
    if midi is not None:
        out["cents_off"] = float(1200.0 * np.log2(centre / midi_to_freq(midi)))
    return out


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

def note_levels(x, sr, timeline, voiced_only=True):
    """Loudness of each note, measured over its **voiced** part.

    Consonants are legitimately quieter than vowels, so measuring a whole slot
    penalises notes that happen to carry a long consonant.  What matters for
    evenness is that the sung part of every note sits at the same level.
    """
    m = to_mono(x)
    if voiced_only:
        tr = _pitch.track_pitch(m, sr, max_f0=1200.0)
        f0, _ = tr.to_per_sample(m.size)
    else:
        f0 = np.ones(m.size)
    out = []
    for start, dur, _ in timeline:
        a, b = int(start * sr), min(int((start + dur) * sr), m.size)
        if b - a < 32:
            out.append(0.0)
            continue
        seg, v = m[a:b], f0[a:b]
        live = seg[(v > 0) & (np.abs(seg) > 0.02 * (np.abs(seg).max() or 1.0))]
        out.append(float(np.sqrt(np.mean(live ** 2))) if live.size > 16 else 0.0)
    return np.array(out)


def evenness(x, sr, timeline):
    """How consistent the notes are in loudness (1.0 = identical).

    Reported as the ratio between the 90th and 10th percentile note level, which
    ignores one odd note but catches a line that genuinely swings.
    """
    lv = note_levels(x, sr, timeline)
    lv = lv[lv > 1e-6]
    if lv.size < 2:
        return {"spread": 1.0, "cv": 0.0}
    if lv.size < 5:
        # Too few notes for percentiles to mean anything -- compare directly.
        hi, lo = float(lv.max()), float(lv.min())
        return {"spread": hi / max(lo, 1e-9),
                "cv": float(np.std(lv) / (np.mean(lv) + 1e-12))}
    hi = float(np.percentile(lv, 90))
    lo = float(np.percentile(lv, 10))
    return {"spread": hi / max(lo, 1e-9),
            "cv": float(np.std(lv) / (np.mean(lv) + 1e-12))}


def flow(x, sr, timeline, floor=0.06, min_gap_ms=40.0):
    """Silence *inside* a sung phrase -- gaps break the legato line.

    Only the span covered by notes is examined, and only gaps longer than
    ``min_gap_ms`` count, so ordinary stop closures are not mistaken for breaks.
    """
    m = to_mono(x)
    if not timeline:
        return {"gap_ratio": 0.0, "gaps": 0, "longest_ms": 0.0}
    frame = 0.01
    env = _frame_rms(m, sr, frame * 1000.0)
    peak = float(env.max()) or 1.0
    # Each note is examined on its own.  Rests between phrases are supposed to be
    # silent, and runs must never be measured across a removed region -- doing
    # that joins frames either side of a rest and invents gaps that aren't there.
    gaps, total = [], 0
    for start, dur, _ in timeline:
        a = max(0, int(start / frame))
        b = min(env.size, int((start + dur) / frame))
        if b - a < 2:
            continue
        quiet = env[a:b] < floor * peak
        total += quiet.size
        run = 0
        for q in quiet:
            if q:
                run += 1
            elif run:
                gaps.append(run)
                run = 0
        if run:
            gaps.append(run)
    if total < 8:
        return {"gap_ratio": 0.0, "gaps": 0, "longest_ms": 0.0}
    long_gaps = [g for g in gaps if g * frame * 1000.0 >= min_gap_ms]
    return {"gap_ratio": float(sum(long_gaps) / total),
            "gaps": len(long_gaps),
            "longest_ms": float(max(long_gaps, default=0) * frame * 1000.0)}


def _band(value, good, bad):
    """Map a measurement onto 0-100, where ``good`` scores 100 and ``bad`` 0."""
    if good == bad:
        return 100.0
    t = (value - bad) / (good - bad)
    return float(np.clip(t, 0.0, 1.0) * 100.0)


def score(x, sr, timeline):
    """Score a sung render out of 100, with a breakdown.

    The weights reflect how audible each fault is: being out of tune or having
    notes drop out is far worse than a slightly uneven phrase.
    """
    m = to_mono(x)
    rep = summarize(m, sr, timeline)
    ev = evenness(m, sr, timeline)
    fl = flow(m, sr, timeline)
    p = rep.get("pitch", {})

    parts = {
        # in tune: 0 cents is perfect, 50 cents (a quarter tone) is a fail
        "tuning": (_band(p.get("mean_abs_cents", 99.0), 0.0, 50.0), 25),
        # every note audible for its whole length
        "sustain": (0.5 * _band(rep["dropout"], 0.0, 0.25)
                    + 0.5 * _band(rep["sustain_cv"], 0.15, 0.9), 20),
        # notes at a consistent level
        "evenness": (_band(ev["spread"], 1.1, 3.0), 20),
        # no clicks, clipping or DC
        "cleanliness": (0.6 * _band(rep["discontinuity"]["rate"], 0.0, 2e-3)
                        + 0.2 * _band(rep["headroom"]["clipped"], 0, 200)
                        + 0.2 * _band(abs(rep["headroom"]["dc"]), 0.0, 0.02), 20),
        # a connected line, no holes
        "flow": (_band(fl["gap_ratio"], 0.0, 0.15), 15),
    }
    total = sum(v * w for v, w in parts.values()) / sum(w for _, w in parts.values())
    return {"total": round(total, 1),
            "parts": {k: round(v, 1) for k, (v, _) in parts.items()},
            "detail": {"evenness": ev, "flow": fl,
                       "pitch": p, "dropout": rep["dropout"],
                       "clicks": rep["discontinuity"], "level": rep["headroom"]}}


def grade(total):
    """Letter grade for a score, for quick reading."""
    for cut, g in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
        if total >= cut:
            return g
    return "F"


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
                     f"mean |err| {p['mean_abs_cents']:.0f}c  "
                     f"octave-folds {p['octave_flips']}")
    lines.append(f"   tone     centroid {s['centroid']:.0f} Hz  " +
                 " ".join(f"{k}:{v}%" for k, v in list(s["bands"].items())[:5]))
    return "\n".join(lines)
