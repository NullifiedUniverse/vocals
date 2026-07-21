"""
Aria -- a from-scratch singing voice.

This revision is built entirely around **keeping espeak's real, correctly
pronounced, coarticulated speech** and only re-timing and re-pitching it onto the
melody -- nothing is resynthesised, so pronunciation and flow are whatever espeak
already produces (which is good), just sung.

Per phrase (the words between rests):

  1. espeak the whole phrase as ONE natural utterance.
  2. Detect its vowel nuclei (one per sung syllable) and the syllable boundaries.
  3. Build a continuous time-warp that stretches each syllable's VOWEL to fill its
     note while keeping the consonants at natural speed, and a pitch track that
     follows the melody with legato glides.
  4. Apply one continuous **TD-PSOLA** pass: real espeak grains (Hann, pitch-
     synchronous) are overlap-added at the melody's period for voiced vowels and
     copied at natural speed for unvoiced consonants.  Real waveform in, real
     waveform out -> intelligible words that flow into one another, click-free.
"""
from __future__ import annotations

import numpy as np

from . import effects, pitch, tts, util
from .biquad import (biquad_fft, high_shelf, highpass, low_shelf,
                     one_pole_lp_fft, peaking)
from .util import midi_to_freq, note_to_midi


def _hann(n):
    n = max(2, int(n))
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


# ---------------------------------------------------------------------------
# analysis: epochs, vowel nuclei, syllable spans
# ---------------------------------------------------------------------------

def _epochs(sr, f0s, voiced, n):
    """Pitch marks at local-period spacing across the signal."""
    out = []
    i = 0.0
    while i < n - 1:
        ii = int(i)
        p = (sr / f0s[ii]) if (voiced[ii] and f0s[ii] > 0) else sr / 150.0
        p = float(np.clip(p, 40.0, 900.0))
        out.append(ii)
        i += p
    if len(out) < 2:
        out = [0, max(1, n - 1)]
    return np.array(out, dtype=np.int64)


def _nuclei(x, sr, voiced, nsyl):
    """Return ``nsyl`` vowel-nucleus positions (sorted) in ``x``."""
    e = one_pole_lp_fft(np.abs(x) * voiced.astype(np.float64), sr, 20.0)
    emax = float(e.max()) or 1.0
    vi = np.where(voiced)[0]
    if vi.size < 2:
        return np.linspace(0, x.size - 1, nsyl).astype(np.int64), e
    span = vi[-1] - vi[0]
    mind = max(int(0.05 * sr), int(0.45 * span / max(1, nsyl)))
    peaks = []
    for i in range(1, e.size - 1):
        if e[i] > e[i - 1] and e[i] >= e[i + 1] and e[i] > 0.12 * emax:
            if not peaks or i - peaks[-1] >= mind:
                peaks.append(i)
            elif e[i] > e[peaks[-1]]:
                peaks[-1] = i
    if len(peaks) >= nsyl:
        peaks = sorted(sorted(peaks, key=lambda p: -e[p])[:nsyl])
    else:                                           # fall back to even spacing
        peaks = [int(vi[0] + span * (k + 0.5) / nsyl) for k in range(nsyl)]
    return np.array(peaks, dtype=np.int64), e


def _vowel_around(e, voiced, nuc, lo, hi):
    """Contiguous voiced high-energy span around a nucleus, within [lo, hi)."""
    thr = 0.5 * e[nuc]
    a = nuc
    while a > lo and voiced[a - 1] and e[a - 1] > thr:
        a -= 1
    b = nuc
    while b + 1 < hi and voiced[b + 1] and e[b + 1] > thr:
        b += 1
    return a, b + 1


# ---------------------------------------------------------------------------
# warp construction
# ---------------------------------------------------------------------------

def _build_warp(x, sr, f0s, voiced, e, nuclei, notes, beat, glide_ms):
    """Continuous input-time trajectory + target pitch for a phrase."""
    n = x.size
    nsyl = len(nuclei)
    bounds = [0]
    for k in range(nsyl - 1):
        a, b = int(nuclei[k]), int(nuclei[k + 1])
        bounds.append(a + int(np.argmin(e[a:b])) if b > a else a)
    bounds.append(n)

    ins_parts, f0_parts, slots = [], [], []
    pf = 0.0
    opos = 0
    for k in range(nsyl):
        s0, s1 = bounds[k], bounds[k + 1]
        v0, v1 = _vowel_around(e, voiced, int(nuclei[k]), s0, s1)
        note, beats = notes[k]
        slot = max(int(beats * beat * sr), int(0.11 * sr))
        onset = min(max(0, v0 - s0), int(0.13 * sr))
        coda = min(max(0, s1 - v1), int(0.12 * sr))
        vowel_out = max(1, slot - onset - coda)

        ins = np.empty(slot)
        if onset > 0:
            ins[:onset] = np.linspace(s0, v0, onset)
        # Stretch the vowel across the note; near the end move toward v1 so a
        # diphthong resolves (real waveform, so it stays the correct vowel).
        vspan = max(1, v1 - v0)
        hold = v0 + 0.35 * vspan
        tail = min(vowel_out // 3, int(0.16 * sr))
        core = vowel_out - tail
        vv = np.empty(vowel_out)
        vv[:core] = np.linspace(v0 + 0.2 * vspan, hold, core)
        if tail > 0:
            vv[core:] = np.linspace(hold, v0 + 0.9 * vspan, tail)
        ins[onset:onset + vowel_out] = np.clip(vv, v0, v1)
        if coda > 0:
            ins[onset + vowel_out:] = np.linspace(v1, s1, slot - onset - vowel_out)
        ins_parts.append(ins)

        f0 = midi_to_freq(note_to_midi(note))
        tgt = np.full(slot, f0)
        if pf > 0 and glide_ms > 0:
            g = min(int(glide_ms / 1000.0 * sr), slot // 2)
            if g > 1:
                tgt[:g] = pf * (f0 / pf) ** np.linspace(0.0, 1.0, g)
        f0_parts.append(tgt)
        slots.append((opos, opos + slot, note_to_midi(note)))
        opos += slot
        pf = f0
    return np.concatenate(ins_parts), np.concatenate(f0_parts), slots


def _f0_vibrato(f0_out, sr, seed):
    n = f0_out.size
    t = np.arange(n)
    venv = np.clip((t / sr - 0.35) / 0.4, 0.0, 1.0)
    cents = 18.0 * venv * np.sin(2.0 * np.pi * 5.6 * t / sr)
    fl = np.random.default_rng(seed).standard_normal(n)
    w = max(1, int(sr * 0.15))
    fl = np.convolve(fl, np.ones(w) / w, mode="same")
    cents += 2.5 * (fl / (np.std(fl) or 1.0))
    return f0_out * 2.0 ** (cents / 1200.0)


# ---------------------------------------------------------------------------
# TD-PSOLA time + pitch warp
# ---------------------------------------------------------------------------

def _psola(x, sr, marks, f0s, voiced, in_of_out, f0_out):
    """Overlap-add real grains onto the warped, re-pitched timeline."""
    n_out = in_of_out.size
    pad = 2048
    out = np.zeros(n_out + pad)
    nrm = np.zeros(n_out + pad)
    nx = x.size
    o = 0.0
    while o < n_out:
        oi = int(o)
        ti = float(in_of_out[oi])
        si = int(np.clip(ti, 0, nx - 1))
        vc = voiced[si] and f0s[si] > 0
        pin = float(np.clip(sr / f0s[si], 40.0, 900.0)) if vc else sr / 160.0
        half = max(2, int(round(pin)))
        k = int(np.searchsorted(marks, ti))
        if k >= marks.size:
            k = marks.size - 1
        elif k > 0 and abs(marks[k - 1] - ti) < abs(marks[k] - ti):
            k -= 1
        a = int(marks[k])
        g0, g1 = a - half, a + half
        lo, hi = max(0, g0), min(nx, g1)
        grain = np.zeros(2 * half)
        grain[lo - g0:hi - g0] = x[lo:hi]
        w = _hann(2 * half)
        grain *= w
        d0 = oi - half
        dlo, dhi = max(0, d0), min(out.size, d0 + 2 * half)
        if dhi > dlo:
            out[dlo:dhi] += grain[dlo - d0:dhi - d0]
            nrm[dlo:dhi] += w[dlo - d0:dhi - d0]
        # Voiced vowels advance at the melody period (pitched); unvoiced
        # consonants advance at their natural period (copied, unpitched).
        if vc:
            o += float(np.clip(sr / f0_out[oi], 40.0, 900.0))
        else:
            o += pin
    m = nrm > 1e-6
    out[m] /= nrm[m]
    return out[:n_out]


def _render_phrase(phrase, sr, voice, base_pitch, wpm, beat, glide_ms, seed):
    text = " ".join(word for word, _ in phrase)
    notes = [nt for _, wnotes in phrase for nt in wnotes]
    raw = tts.text_to_vocal(text, sr, voice=voice, pitch=base_pitch, wpm=wpm)
    raw = util.normalize_peak(raw, 0.95)
    track = pitch.track_pitch(raw, sr, max_f0=1000.0)
    f0s, voiced = track.to_per_sample(raw.size)
    marks = _epochs(sr, f0s, voiced, raw.size)
    nuclei, e = _nuclei(raw, sr, voiced, len(notes))
    in_of_out, f0_out, slots = _build_warp(raw, sr, f0s, voiced, e, nuclei,
                                           notes, beat, glide_ms)
    f0_out = _f0_vibrato(f0_out, sr, seed)
    out = _psola(raw, sr, marks, f0s, voiced, in_of_out, f0_out)

    # Gentle per-note dynamics + phrase edges.
    mids = [m for _, _, m in slots]
    lo, hi = min(mids), max(mids)
    gain = np.ones(out.size)
    for j, (o0, o1, m) in enumerate(slots):
        pn = (m - lo) / (hi - lo) if hi > lo else 0.5
        arc = np.sin(np.pi * (j + 0.5) / len(slots))
        gain[o0:min(o1, out.size)] = 0.82 + 0.10 * pn + 0.10 * arc
    out = out * one_pole_lp_fft(gain, sr, 10.0)
    at = min(int(0.02 * sr), out.size)
    rel = min(int(0.13 * sr), out.size)
    out[:at] *= np.linspace(0.0, 1.0, at)
    out[-rel:] *= np.linspace(1.0, 0.0, rel) ** 1.3
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# song assembly + mastering
# ---------------------------------------------------------------------------

def sing(score, sr=44100, bpm=100, voice="en+f4", base_pitch=64, wpm=150,
         glide_ms=40.0, seed=5):
    """Render a word-based score to a continuous, phrased sung mono line."""
    beat = 60.0 / bpm
    phrases, t, cur, cur_t = [], 0.0, [], 0.0
    for it in score:
        if it[0] == "rest":
            if cur:
                phrases.append((cur_t, cur))
                cur = []
            t += it[1] * beat
        else:
            if not cur:
                cur_t = t
            cur.append(it)
            t += sum(b for _, b in it[1]) * beat
    if cur:
        phrases.append((cur_t, cur))

    out = np.zeros(int(t * sr) + sr)
    sd = seed
    for start, phrase in phrases:
        buf = _render_phrase(phrase, sr, voice, base_pitch, wpm, beat, glide_ms, sd)
        sd += 1
        p = int(start * sr)
        e = min(out.size, p + buf.size)
        out[p:e] += buf[:e - p]
    return out[:int(t * sr) + int(0.3 * sr)].astype(np.float32)


def _voice_timbre(x, sr):
    chain = [
        highpass(105.0, sr, 0.7),
        low_shelf(320.0, sr, 2.5),
        peaking(3000.0, sr, 1.3, 2.5),
        high_shelf(8500.0, sr, 3.0),
    ]
    return biquad_fft(chain, x)


def _deess(x, sr, amount=0.55):
    hi = biquad_fft(highpass(6500.0, sr, 0.7), x)
    lo = np.asarray(x, dtype=np.float64) - hi
    env = one_pole_lp_fft(np.abs(hi), sr, 55.0)
    nz = env[env > 1e-5]
    thr = 1.6 * float(np.median(nz)) if nz.size else 1.0
    gain = one_pole_lp_fft(np.clip(thr / (env + 1e-6), 1.0 - amount, 1.0), sr, 80.0)
    return (lo + hi * gain).astype(np.float32)


def render_song(score, sr=44100, bpm=100, voice="en+f4", base_pitch=64,
                reverb_mix=0.18, width=1.2):
    """Full render: sing -> timbre -> de-ess -> stereo -> reverb."""
    dry = sing(score, sr, bpm, voice, base_pitch)
    dry = _voice_timbre(dry, sr)
    dry = _deess(dry, sr)
    dry = util.normalize_peak(dry, 0.92)
    stereo = effects.stereoize(dry, sr, haas_ms=9.0, width=width)
    stereo = effects.reverb(stereo, sr, mix=reverb_mix, size=0.7, damp=0.48,
                            width=1.2)
    stereo = util.normalize_percentile(stereo, target=0.85)
    stereo = util.soft_limit(stereo, 0.98)
    return stereo
