"""
Aria -- a from-scratch singing voice (bright, Miku-leaning).

One clean pipeline, rendered a **phrase at a time** (a phrase = the words between
rests) so syllables and words flow into one another:

  * espeak speaks the phrase's words (correct pronunciation); their audio is
    concatenated into one phrase signal.
  * The phrase is split into syllables at its vowel nuclei and warped onto its
    notes -- vowels stretched, the diphthong off-glide done near each note's end,
    consonants kept at natural speed.
  * The whole phrase is synthesised in ONE continuous pass: voiced vowels from
    harmonic resynthesis (clean sinusoids at the exact pitch, phase carried across
    the entire phrase -> no per-word resets), unvoiced consonants spliced from
    espeak's real waveform.  Continuous phase + continuous pitch = legato flow.
  * Formants are shifted up for a younger/brighter (Miku-ish) tone; pitch gets
    glides + a vibrato that swells in; light timbre / de-ess / reverb finish it.
"""
from __future__ import annotations

import numpy as np

from . import effects, harmonic, pitch, tts, util
from .biquad import (biquad_fft, high_shelf, highpass, low_shelf,
                     one_pole_lp_fft, peaking)
from .util import midi_to_freq, note_to_midi


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------

def _syllable_spans(x, sr, voiced, w0, w1, nsyl):
    """Split region ``[w0, w1)`` of ``x`` into ``nsyl`` (start, end) spans at
    vowel-energy nuclei (indices in ``x``)."""
    if nsyl <= 1:
        return [(w0, w1)]
    reg = np.abs(x[w0:w1]) * voiced[w0:w1].astype(np.float64)
    e = one_pole_lp_fft(reg, sr, 24.0)
    emax = float(e.max()) or 1.0
    mind = int(0.06 * sr)
    peaks = []
    for i in range(1, e.size - 1):
        if e[i] > e[i - 1] and e[i] >= e[i + 1] and e[i] > 0.2 * emax:
            if not peaks or i - peaks[-1] >= mind:
                peaks.append(i)
            elif e[i] > e[peaks[-1]]:
                peaks[-1] = i
    if len(peaks) > nsyl:
        peaks = sorted(sorted(peaks, key=lambda p: -e[p])[:nsyl])
    if len(peaks) < nsyl:
        m = w1 - w0
        return [(w0 + m * k // nsyl, w0 + m * (k + 1) // nsyl) for k in range(nsyl)]
    bounds = [0]
    for a, b in zip(peaks[:-1], peaks[1:]):
        bounds.append(a + int(np.argmin(e[a:b])))
    bounds.append(w1 - w0)
    return [(w0 + bounds[k], w0 + bounds[k + 1]) for k in range(nsyl)]


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
# per-phrase warp + continuous synthesis
# ---------------------------------------------------------------------------

def _f0_expression(f0_out, sr, seed):
    """Vibrato that holds back until a note settles (so the melody reads first),
    plus small pitch flutter -- straight into the synthesis frequency."""
    n = f0_out.size
    t = np.arange(n)
    venv = np.clip((t / sr - 0.33) / 0.35, 0.0, 1.0)
    cents = 20.0 * venv * np.sin(2.0 * np.pi * 5.6 * t / sr)
    fl = np.random.default_rng(seed).standard_normal(n)
    w = max(1, int(sr * 0.14))
    fl = np.convolve(fl, np.ones(w) / w, mode="same")
    cents += 3.0 * (fl / (np.std(fl) or 1.0))
    return f0_out * 2.0 ** (cents / 1200.0)


def _concat_words(phrase, sr, voice, base_pitch, wpm):
    """espeak each word, butt-join into one phrase signal; return (raw, spans)
    where spans are ``(word_start, word_end, notes)``."""
    parts, spans, pos = [], [], 0
    for word, notes in phrase:
        r = tts.text_to_vocal(word, sr, voice=voice, pitch=base_pitch, wpm=wpm)
        r = util.normalize_peak(r, 0.92)
        if r.size < int(0.05 * sr):
            r = np.pad(r, (0, int(0.05 * sr) - r.size))
        parts.append(r)
        spans.append((pos, pos + r.size, notes))
        pos += r.size
    return np.concatenate(parts), spans


def _warp_seq(raw, sr, hop, spans, notes, voiced, beat, prev_f0, glide_ms):
    """Continuous input-sample trajectory + target pitch for a whole phrase."""
    ins_parts, f0_parts, slots = [], [], []
    pf = prev_f0
    opos = 0
    for span, (note, beats) in zip(spans, notes):
        s0, s1 = span
        v0, v1 = _vowel_region(raw, sr, span, voiced)
        slot = max(int(beats * beat * sr), int(0.11 * sr))
        onset_out = min(max(0, v0 - s0), int(0.12 * sr))
        coda_out = min(max(0, s1 - v1), int(0.11 * sr))
        vowel_out = max(1, slot - onset_out - coda_out)

        ins = np.empty(slot)
        if onset_out > 0:
            ins[:onset_out] = np.linspace(s0, v0, onset_out)
        span_v = v1 - v0
        main = v0 + 0.30 * span_v
        end = v0 + 0.82 * span_v
        vpath = np.full(vowel_out, main)
        gl = min(vowel_out // 3, int(0.16 * sr))
        if gl > 1:
            vpath[-gl:] = np.linspace(main, end, gl)
        st = np.arange(vowel_out)
        vpath += 0.06 * span_v * np.sin(2.0 * np.pi * 0.5 * st / sr)
        ins[onset_out:onset_out + vowel_out] = np.clip(vpath, v0, v1)
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
        slots.append((opos, opos + slot, note_to_midi(note)))
        opos += slot
        pf = f0
    return np.concatenate(ins_parts), np.concatenate(f0_parts), slots, pf


def _render_phrase(phrase, sr, voice, base_pitch, wpm, beat, prev_f0, glide_ms,
                   seed, formant_shift):
    """Render one phrase as a single continuous sung buffer."""
    raw, wspans = _concat_words(phrase, sr, voice, base_pitch, wpm)
    track = pitch.track_pitch(raw, sr, max_f0=1000.0)
    ana = harmonic.analyze(raw, sr)
    _, voiced = track.to_per_sample(raw.size)
    hop = ana["hop"]

    spans, notes = [], []
    for w0, w1, wnotes in wspans:
        for span, note in zip(_syllable_spans(raw, sr, voiced, w0, w1, len(wnotes)),
                              wnotes):
            spans.append(span)
            notes.append(note)

    insamp, f0_out, slots, end_f0 = _warp_seq(raw, sr, hop, spans, notes, voiced,
                                              beat, prev_f0, glide_ms)
    f0_out = _f0_expression(f0_out, sr, seed)

    frame_of_out = np.clip(insamp / hop, 0, ana["env"].shape[0] - 1)
    harm = harmonic._harmonic(ana, frame_of_out, f0_out, sr,
                              formant_shift=formant_shift)
    harm = harm / (float(np.max(np.abs(harm))) or 1.0)
    esp = np.interp(insamp, np.arange(raw.size, dtype=np.float64), raw)
    esp = esp / (float(np.max(np.abs(esp))) or 1.0)
    vfr = voiced[np.clip(insamp.astype(np.int64), 0, raw.size - 1)].astype(np.float64)
    vw = np.clip(one_pole_lp_fft(vfr, sr, 70.0), 0.0, 1.0)
    out = vw * harm + (1.0 - vw) * esp

    # Per-note dynamics (higher notes / phrase peak a touch louder) + phrase edges.
    mids = [m for _, _, m in slots]
    lo, hi = min(mids), max(mids)
    gain = np.ones(out.size)
    for j, (o0, o1, m) in enumerate(slots):
        pn = (m - lo) / (hi - lo) if hi > lo else 0.5
        arc = np.sin(np.pi * (j + 0.5) / len(slots))
        gain[o0:o1] = 0.80 + 0.11 * pn + 0.11 * arc
    gain = one_pole_lp_fft(gain, sr, 10.0)
    out = out[:gain.size] * gain[:out.size]
    at = min(int(0.02 * sr), out.size)
    rel = min(int(0.14 * sr), out.size)
    out[:at] *= np.linspace(0.0, 1.0, at)
    out[-rel:] *= np.linspace(1.0, 0.0, rel) ** 1.3
    return out.astype(np.float32), end_f0


def sing(score, sr=44100, bpm=100, voice="en+f4", base_pitch=64, wpm=150,
         glide_ms=38.0, formant_shift=1.08, seed=5):
    """Render a word-based score to a continuous, phrased sung mono line."""
    beat = 60.0 / bpm
    # Group into phrases (runs of words between rests) with their start times.
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
        buf, _ = _render_phrase(phrase, sr, voice, base_pitch, wpm, beat, 0.0,
                                glide_ms, sd, formant_shift)
        sd += 1
        p = int(start * sr)
        e = min(out.size, p + buf.size)
        out[p:e] += buf[:e - p]
    return out[:int(t * sr) + int(0.3 * sr)].astype(np.float32)


# ---------------------------------------------------------------------------
# voicing / mastering + full render
# ---------------------------------------------------------------------------

def _voice_timbre(x, sr):
    """Bright, young (Miku-ish) voicing: a little warmth, presence around 3 kHz,
    and airy top."""
    chain = [
        highpass(110.0, sr, 0.7),
        low_shelf(320.0, sr, 2.0),
        peaking(3000.0, sr, 1.2, 3.0),       # presence / ring
        high_shelf(8000.0, sr, 4.0),         # air / sparkle
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
                formant_shift=1.08, reverb_mix=0.2, width=1.25):
    """Full render: sing -> timbre -> de-ess -> stereo -> reverb."""
    dry = sing(score, sr, bpm, voice, base_pitch, formant_shift=formant_shift)
    dry = _voice_timbre(dry, sr)
    dry = _deess(dry, sr)
    dry = util.normalize_peak(dry, 0.92)
    stereo = effects.stereoize(dry, sr, haas_ms=9.0, width=width)
    stereo = effects.reverb(stereo, sr, mix=reverb_mix, size=0.72, damp=0.45,
                            width=1.25)
    stereo = util.normalize_percentile(stereo, target=0.85)
    stereo = util.soft_limit(stereo, 0.98)
    return stereo
