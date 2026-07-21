"""
Aria -- a from-scratch singing voice.

Design (one clean pipeline):

  * espeak speaks whole **words** naturally (correct pronunciation).
  * Each word is split into syllables at its vowel nuclei and warped onto its
    notes in **one continuous pass** -- no per-syllable clips crossfaded together,
    so consonants are never faded away and words stay intelligible.
  * Voiced vowels are rebuilt by **harmonic resynthesis** (harmonic.py): clean
    phase-dispersed sinusoids at the exact target pitch -> a smooth, stable,
    non-buzzy vocal that holds without decaying.
  * Unvoiced consonants (s/t/k/f/sh ...) are taken from **espeak's real
    waveform**, time-warped to the same clock -> crisp, natural articulation.
  * Pitch gets legato glides and a vibrato that swells in; a light timbre / de-ess
    / reverb chain finishes it.

The vowels come from the synthesiser (stable, tuneful) and the consonants from
real speech (intelligible) -- the best of both.
"""
from __future__ import annotations

import numpy as np

from . import effects, harmonic, pitch, tts, util
from .biquad import (biquad_fft, high_shelf, highpass, low_shelf,
                     one_pole_lp_fft, peaking)
from .util import midi_to_freq, note_to_midi


# ---------------------------------------------------------------------------
# analysis helpers
# ---------------------------------------------------------------------------

def _energy(x, win):
    p = np.asarray(x, dtype=np.float64) ** 2
    k = np.ones(max(1, win)) / max(1, win)
    return np.convolve(p, k, mode="same")


def _split_syllables(x, sr, track, nsyl):
    """Split a word into ``nsyl`` (start, end) syllable spans at vowel nuclei."""
    n = x.size
    f0s, voiced = track.to_per_sample(n)
    if nsyl <= 1:
        return [(0, n)], voiced
    e = one_pole_lp_fft(np.abs(x) * voiced.astype(np.float64), sr, 24.0)
    emax = float(e.max()) or 1.0
    mind = int(0.07 * sr)
    peaks = []
    for i in range(1, e.size - 1):
        if e[i] > e[i - 1] and e[i] >= e[i + 1] and e[i] > 0.22 * emax:
            if not peaks or i - peaks[-1] >= mind:
                peaks.append(i)
            elif e[i] > e[peaks[-1]]:
                peaks[-1] = i
    if len(peaks) > nsyl:
        peaks = sorted(sorted(peaks, key=lambda p: -e[p])[:nsyl])
    if len(peaks) < nsyl:                                   # fallback: even split
        vi = np.where(voiced)[0]
        lo, hi = (vi[0], vi[-1]) if vi.size >= 2 else (0, n)
        spans = [(int(lo + (hi - lo) * k / nsyl),
                  int(lo + (hi - lo) * (k + 1) / nsyl)) for k in range(nsyl)]
        return spans, voiced
    bounds = [0]
    for a, b in zip(peaks[:-1], peaks[1:]):
        bounds.append(a + int(np.argmin(e[a:b])))
    bounds.append(n)
    return [(bounds[k], bounds[k + 1]) for k in range(nsyl)], voiced


def _vowel_region(x, sr, span, voiced):
    s0, s1 = span
    seg = np.abs(x[s0:s1]) * voiced[s0:s1]
    if seg.size < 8:
        return s0, s1
    e = one_pole_lp_fft(seg, sr, 45.0)
    thr = 0.4 * (float(e.max()) or 1.0)
    idx = np.where(e > thr)[0]
    if idx.size == 0:
        return s0, s1
    return s0 + int(idx[0]), s0 + int(idx[-1]) + 1


# ---------------------------------------------------------------------------
# per-word continuous warp + hybrid synthesis
# ---------------------------------------------------------------------------

def _f0_expression(f0_out, sr, seed):
    """Add legato-friendly vibrato that swells in + small pitch flutter, straight
    into the synthesis frequency."""
    n = f0_out.size
    t = np.arange(n)
    venv = np.clip((t / sr - 0.22) / 0.35, 0.0, 1.0)
    cents = 26.0 * venv * np.sin(2.0 * np.pi * 5.6 * t / sr)
    fl = np.random.default_rng(seed).standard_normal(n)
    w = max(1, int(sr * 0.13))
    fl = np.convolve(fl, np.ones(w) / w, mode="same")
    cents += 4.5 * (fl / (np.std(fl) or 1.0))
    return f0_out * 2.0 ** (cents / 1200.0)


def _warp_word(raw, sr, hop, spans, notes, voiced, beat, prev_f0, glide_ms, tail):
    """Build the continuous input-sample trajectory and target-pitch track that
    lay this word's syllables onto its notes (vowels stretched, consonants at
    natural speed)."""
    ins_parts, f0_parts = [], []
    pf = prev_f0
    last = len(notes) - 1
    for j, (span, (note, beats)) in enumerate(zip(spans, notes)):
        s0, s1 = span
        v0, v1 = _vowel_region(raw, sr, span, voiced)
        slot = max(int(beats * beat * sr), int(0.11 * sr))
        if j == last:
            slot += tail
        onset_out = min(max(0, v0 - s0), int(0.12 * sr))
        coda_out = min(max(0, s1 - v1), int(0.11 * sr))
        vowel_out = max(1, slot - onset_out - coda_out)

        ins = np.empty(slot)
        if onset_out > 0:
            ins[:onset_out] = np.linspace(s0, v0, onset_out)
        cvow = 0.5 * (v0 + v1)
        half = 0.5 * (v1 - v0)
        st = np.arange(vowel_out)
        drift = 0.10 * half * np.sin(2.0 * np.pi * 0.6 * st / sr)   # gentle life
        ins[onset_out:onset_out + vowel_out] = np.clip(cvow + drift, v0, v1)
        if coda_out > 0:
            ins[onset_out + vowel_out:] = np.linspace(v1, s1, slot - onset_out - vowel_out)
        ins_parts.append(ins)

        f0 = midi_to_freq(note_to_midi(note))
        tgt = np.full(slot, f0)
        if pf > 0 and glide_ms > 0:
            g = min(int(glide_ms / 1000.0 * sr), slot // 2)
            if g > 1:
                tgt[:g] = pf * (f0 / pf) ** np.linspace(0.0, 1.0, g)
        f0_parts.append(tgt)
        pf = f0
    return np.concatenate(ins_parts), np.concatenate(f0_parts), pf


def render_word(word, notes, sr, voice, base_pitch, wpm, beat, prev_f0,
                glide_ms, tail, seed):
    """Render one word (all its syllables/notes) as one continuous sung buffer:
    HNM for the voiced vowels, espeak's real waveform for unvoiced consonants."""
    raw = tts.text_to_vocal(word, sr, voice=voice, pitch=base_pitch, wpm=wpm)
    raw = util.normalize_peak(raw, 0.92)
    track = pitch.track_pitch(raw, sr, max_f0=1000.0)
    ana = harmonic.analyze(raw, sr)
    spans, voiced = _split_syllables(raw, sr, track, len(notes))
    insamp, f0_out, end_f0 = _warp_word(raw, sr, ana["hop"], spans, notes,
                                        voiced, beat, prev_f0, glide_ms, tail)
    f0_out = _f0_expression(f0_out, sr, seed)

    frame_of_out = np.clip(insamp / ana["hop"], 0, ana["env"].shape[0] - 1)
    harm = harmonic._harmonic(ana, frame_of_out, f0_out, sr)
    harm = harm / (float(np.max(np.abs(harm))) or 1.0)

    esp = np.interp(insamp, np.arange(raw.size, dtype=np.float64), raw)
    esp = esp / (float(np.max(np.abs(esp))) or 1.0)

    # Voicing weight per output sample -> HNM on voiced vowels, espeak on
    # unvoiced consonants, cross-faded smoothly at the boundaries.
    vfr = voiced[np.clip(insamp.astype(np.int64), 0, raw.size - 1)].astype(np.float64)
    vw = np.clip(one_pole_lp_fft(vfr, sr, 70.0), 0.0, 1.0)
    out = vw * harm + (1.0 - vw) * esp
    return out.astype(np.float32), end_f0


# ---------------------------------------------------------------------------
# phrase / song assembly
# ---------------------------------------------------------------------------

def sing(score, sr=44100, bpm=100, voice="en+f4", base_pitch=64, wpm=150,
         glide_ms=55.0, crossfade_ms=16.0, seed=5):
    """Render a word-based score to a continuous sung mono line.

    Score items: ``("rest", beats)`` or ``(word, [(note, beats), ...])`` with one
    note per syllable of the word."""
    beat = 60.0 / bpm
    xf = int(crossfade_ms / 1000.0 * sr)
    words = []
    t = 0.0
    prev_f0 = 0.0
    phrase = 0
    sd = seed
    for it in score:
        if it[0] == "rest":
            t += it[1] * beat
            if prev_f0 != 0.0:
                phrase += 1
            prev_f0 = 0.0
            continue
        word, notes = it
        dur = sum(b for _, b in notes) * beat
        audio, end_f0 = render_word(word, notes, sr, voice, base_pitch, wpm,
                                    beat, prev_f0, glide_ms, tail=xf, seed=sd)
        sd += 1
        words.append({"a": audio, "pos": int(t * sr), "phrase": phrase,
                      "midi": note_to_midi(notes[0][0]), "start": prev_f0 == 0.0})
        t += dur
        prev_f0 = end_f0

    for i, w in enumerate(words):
        w["end"] = (i + 1 == len(words)) or (words[i + 1]["phrase"] != w["phrase"])

    # Gentle per-phrase dynamics: a little louder toward the phrase's high point.
    by_phrase = {}
    for i, w in enumerate(words):
        by_phrase.setdefault(w["phrase"], []).append(i)
    gains = [1.0] * len(words)
    for idxs in by_phrase.values():
        mids = [words[i]["midi"] for i in idxs]
        lo, hi = min(mids), max(mids)
        for j, i in enumerate(idxs):
            pn = (words[i]["midi"] - lo) / (hi - lo) if hi > lo else 0.5
            arc = np.sin(np.pi * (j + 0.5) / len(idxs))
            gains[i] = 0.78 + 0.12 * pn + 0.12 * arc

    out = np.zeros(int(t * sr) + sr)
    ramp = np.linspace(0.0, 1.0, xf)
    for i, w in enumerate(words):
        a = w["a"].copy() * gains[i]
        if not w["start"]:
            a[:xf] *= np.sin(0.5 * np.pi * ramp)
        if not w["end"]:
            a[-xf:] *= np.cos(0.5 * np.pi * ramp)
        else:
            rel = min(int(0.14 * sr), a.size)
            a[-rel:] *= np.linspace(1.0, 0.0, rel) ** 1.3
        p = w["pos"]
        e = min(out.size, p + a.size)
        out[p:e] += a[:e - p]
    return out[:int(t * sr) + int(0.3 * sr)].astype(np.float32)


# ---------------------------------------------------------------------------
# voicing / mastering + full render
# ---------------------------------------------------------------------------

def _voice_timbre(x, sr):
    """Gentle, natural voicing: low-mid warmth, a touch of singer's-formant ring,
    soft air."""
    chain = [
        highpass(95.0, sr, 0.7),
        low_shelf(340.0, sr, 3.0),
        peaking(2900.0, sr, 1.3, 2.0),
        high_shelf(9000.0, sr, 2.0),
    ]
    return biquad_fft(chain, x)


def _deess(x, sr, amount=0.55):
    """Tame sibilants so the air/consonants don't get harsh."""
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
    stereo = effects.reverb(stereo, sr, mix=reverb_mix, size=0.7, damp=0.5,
                            width=1.2)
    stereo = util.normalize_percentile(stereo, target=0.85)
    stereo = util.soft_limit(stereo, 0.98)
    return stereo
