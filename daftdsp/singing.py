"""
Note-timed singing synthesis -- "make it sing", with musical flow.

A ``(syllable, note, beats)`` score becomes a sung vocal.  What makes it sound
like phrasing rather than beeps:

  * **Phoneme input** -- each syllable is given to espeak as ``[[...]]`` so it is
    pronounced correctly out of word context.
  * **Vowel-nucleus sustain** -- long notes hold the vowel's steady centre with
    pitch-synchronous overlap-add (smooth, no decay, no syllable-repeat).
  * **Continuous phrases** -- within a breath (between rests) notes are cross-
    faded into one continuous line, mid-phrase syllable-final consonants are
    dropped so the vowels *connect*, and a single dynamic envelope arcs over the
    phrase (crescendo toward the peak, taper at the end) for expression.
  * **Legato glides**, **vibrato that swells in**, and light humanised timing.

All from-scratch (PSOLA, biquads, modulated fractional delays with cubic interp).
"""
from __future__ import annotations

import numpy as np

from . import effects, pitch, psola, tts, util
from .biquad import (biquad_fft, high_shelf, highpass, low_shelf, lowpass,
                     one_pole_lp_fft, peaking)
from .util import midi_to_freq, note_to_midi


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _xfade(a, b, xf):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return b.copy()
    if b.size == 0:
        return a.copy()
    xf = int(min(xf, a.size, b.size))
    if xf <= 1:
        return np.concatenate([a, b])
    ramp = np.linspace(0.0, 1.0, xf)
    mid = a[-xf:] * np.cos(0.5 * np.pi * ramp) + b[:xf] * np.sin(0.5 * np.pi * ramp)
    return np.concatenate([a[:-xf], mid, b[xf:]])


def _energy(x, win):
    p = np.asarray(x, dtype=np.float64) ** 2
    k = np.ones(max(1, win)) / max(1, win)
    return np.convolve(p, k, mode="same")


def _read_cubic(x, read):
    """Catmull-Rom cubic interpolation read at fractional positions."""
    n = x.size
    i = np.floor(read).astype(np.int64)
    f = read - i

    def tap(o):
        return x[np.clip(i + o, 0, n - 1)]

    p0, p1, p2, p3 = tap(-1), tap(0), tap(1), tap(2)
    return p1 + 0.5 * f * (p2 - p0 + f * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
                          + f * (3.0 * (p1 - p2) + p3 - p0)))


# ---------------------------------------------------------------------------
# vowel sustain (pitch-synchronous overlap-add, holds the nucleus)
# ---------------------------------------------------------------------------

def _sustain(x, sr, f0, length, coda=True):
    """Hold ``x`` to ``length`` samples.  Grains (two target-periods, Hann) are
    laid at exact target-period spacing; the onset plays 1:1, then the vowel
    nucleus is held (steady, no decay).  ``coda`` keeps the syllable-final
    consonant (used at phrase ends); dropped mid-phrase so vowels connect."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if length <= n and not coda:
        return x[:length].copy()

    period = max(4, int(round(sr / max(1.0, f0))))
    win = 2 * period
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win) / win)

    e = _energy(x, period)
    peak = int(np.argmax(e[:max(2 * period, int(n * 0.7))]))
    onset = int(np.clip(peak - 2 * period, 0, n))
    vend = int(np.clip(int(n * 0.86), onset + 2 * period, n - 1))
    coda_seg = x[vend:]
    coda_len = int(min(coda_seg.size, 0.13 * sr)) if coda else 0
    body_len = max(int(0.05 * sr), length - coda_len)
    hold = int(np.clip(round(peak / period) * period, 0, n - win))

    out = np.zeros(body_len + win)
    nrm = np.zeros(body_len + win)
    # Micro-movement: the read point wanders a few periods around the nucleus
    # (period-snapped, so phase-coherent) so the held vowel's timbre isn't frozen.
    wander = 3 * period
    lo_hold = max(onset, hold - wander)
    hi_hold = min(n - win, hold + wander)
    k = 0
    while k * period < body_len:
        op = k * period
        if op < onset:
            ai = op
        else:
            ph = np.sin(2.0 * np.pi * 1.4 * op / sr
                        + 0.6 * np.sin(2.0 * np.pi * 0.27 * op / sr))
            ai = hold + int(round(ph * wander / period)) * period
            ai = int(np.clip(ai, lo_hold, hi_hold))
        ai = int(np.clip(ai, 0, n - win))
        out[op:op + win] += x[ai:ai + win] * w
        nrm[op:op + win] += w
        k += 1
    m = nrm > 1e-6
    out[m] /= nrm[m]
    body = out[:body_len]

    if coda_len > 0:
        note = _xfade(body, coda_seg[:coda_len],
                      min(period, body_len // 4, max(1, coda_len // 2)))
    else:
        note = body
    if note.size < length:
        note = np.pad(note, (0, length - note.size))
    return note[:length]


def _smooth_noise(rng, n, sr, ms):
    """Zero-mean, unit-std smooth random signal (a slow drift)."""
    x = rng.standard_normal(n)
    w = max(1, int(sr * ms / 1000.0))
    x = np.convolve(x, np.ones(w) / w, mode="same")
    s = np.std(x)
    return x / s if s > 1e-9 else x


def _expression(x, sr, rng, vib_rate, vib_depth, vib_delay, ramp_s,
                flutter_cents=6.0, shimmer=0.045):
    """Add human pitch/amplitude life: vibrato that swells in, plus small random
    pitch *jitter* (flutter) and amplitude *shimmer* -- what separates a real
    voice from a frozen synth tone.  Pitch is modulated exactly via a resampling
    read-index, not an approximate delay."""
    n = x.size
    t = np.arange(n)
    cents = np.zeros(n)
    if vib_depth > 0:
        env = np.clip((t / sr - vib_delay) / max(1e-3, ramp_s), 0.0, 1.0)
        cents += vib_depth * env * np.sin(2.0 * np.pi * vib_rate * t / sr)
    if flutter_cents > 0:
        cents += flutter_cents * _smooth_noise(rng, n, sr, 120.0)

    ratio = 2.0 ** (cents / 1200.0) - 1.0
    ratio -= float(np.mean(ratio))                 # zero-mean -> no time drift
    read = np.cumsum(1.0 + ratio)
    read = np.clip(read - read[0], 0.0, n - 1)
    y = _read_cubic(x.astype(np.float64), read)
    if shimmer > 0:
        y = y * (1.0 + shimmer * _smooth_noise(rng, n, sr, 90.0))
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# note rendering
# ---------------------------------------------------------------------------

def render_note(syllable, midi, dur_s, sr, voice="en+f4", base_pitch=62,
                phoneme=True, wpm=150, prev_midi=None, glide_ms=55.0,
                vib_rate=5.8, vib_depth=34.0, vib_delay=0.30, coda=True,
                attack_ms=6.0):
    """Render one note: pronounce the syllable, glide/pitch it to ``midi``, hold
    it for ``dur_s`` seconds, add swelling vibrato.  No release fade -- phrase
    assembly handles joins and dynamics."""
    text = f"[[{syllable}]]" if phoneme else syllable
    raw = tts.text_to_vocal(text, sr, voice=voice, pitch=base_pitch, wpm=wpm)
    if raw.size < int(0.05 * sr):
        raw = np.pad(raw, (0, int(0.05 * sr) - raw.size))
    raw = util.normalize_peak(raw, 0.9)
    track = pitch.track_pitch(raw, sr, max_f0=1000.0)
    f0 = midi_to_freq(midi)

    target = np.full(raw.size, f0)
    if prev_midi is not None and glide_ms > 0:
        # Glide length grows a little with interval, capped -- legato portamento.
        semis = abs(midi - prev_midi)
        g = int(min(glide_ms * (1.0 + 0.05 * semis), 120.0) / 1000.0 * sr)
        g = min(g, raw.size // 2)
        if g > 1:
            pf = midi_to_freq(prev_midi)
            target[:g] = pf * (f0 / pf) ** np.linspace(0.0, 1.0, g)

    pitched = psola.psola_correct(raw, sr, track, target_hz=target, retune=1.0,
                                  retune_time_ms=6.0, max_f0=1300.0)
    length = max(int(dur_s * sr), int(0.09 * sr))
    note = _sustain(pitched, sr, f0, length, coda=coda)

    a = int(attack_ms / 1000.0 * sr)
    if 0 < a < note.size:
        note[:a] *= np.linspace(0.0, 1.0, a)      # anti-click attack only
    rng = np.random.default_rng(int(abs(midi) * 131 + length) % (2 ** 31))
    note = _expression(note, sr, rng, vib_rate, vib_depth, vib_delay, 0.32)
    return note.astype(np.float32)


# ---------------------------------------------------------------------------
# phrase assembly (continuous, expressive)
# ---------------------------------------------------------------------------

def _render_phrase(phrase, sr, voice, base_pitch, phoneme, xf, rng, note_kw):
    """Render one breath-phrase (list of (syllable, midi, dur, start)) into a
    continuous, dynamically-shaped buffer.  Returns (buffer, phrase_start)."""
    n = len(phrase)
    pstart = phrase[0][3]
    pitches = [m for _, m, _, _ in phrase]
    pmin, pmax = min(pitches), max(pitches)

    audios, centers, louds = [], [], []
    prev = None
    for k, (syl, midi, dur, st) in enumerate(phrase):
        last = k == n - 1
        a = render_note(syl, midi, dur + xf / sr, sr, voice, base_pitch,
                        phoneme=phoneme, prev_midi=prev, coda=last, **note_kw)
        ramp = np.linspace(0.0, 1.0, xf)
        if k > 0:
            a[:xf] *= np.sin(0.5 * np.pi * ramp)          # crossfade in
        if not last:
            a[-xf:] *= np.cos(0.5 * np.pi * ramp)         # crossfade out
        else:
            rel = min(int(0.16 * sr), a.size)
            a[-rel:] *= np.linspace(1.0, 0.0, rel) ** 1.4  # phrase release
        audios.append(a)
        # Expressive dynamics: higher notes and the phrase middle sing louder.
        pn = (midi - pmin) / (pmax - pmin) if pmax > pmin else 0.5
        arc = np.sin(np.pi * (k + 0.5) / n)
        louds.append(0.66 + 0.17 * pn + 0.17 * arc)
        centers.append((st - pstart + dur * 0.5) * sr)
        prev = midi

    end = phrase[-1][3] - pstart + phrase[-1][2]
    buf = np.zeros(int(end * sr) + xf + int(0.25 * sr))
    for k, (syl, midi, dur, st) in enumerate(phrase):
        jitter = int(rng.normal(0.0, 0.004) * sr) if k > 0 else 0   # ~4ms timing
        pos = max(0, int((st - pstart) * sr) + jitter)
        a = audios[k]
        e = min(buf.size, pos + a.size)
        buf[pos:e] += a[:e - pos]

    # Smooth dynamic envelope from per-note loudness + phrase fade-in.
    xs = np.arange(buf.size)
    dyn = np.interp(xs, np.array(centers), np.array(louds),
                    left=louds[0], right=louds[-1])
    at = min(int(0.03 * sr), buf.size)
    dyn[:at] *= np.linspace(0.0, 1.0, at)
    buf *= dyn
    return buf.astype(np.float32), pstart


def sing(score, sr=44100, bpm=100, voice="en+f4", base_pitch=62, phoneme=True,
         crossfade_ms=24.0, seed=7, **note_kw):
    """Render a whole score to a continuous, phrased mono sung line."""
    beat = 60.0 / bpm
    notes, t = [], 0.0
    for syllable, note, beats in score:
        dur = beats * beat
        if note is None or syllable in (None, "", "rest", "-"):
            notes.append(("rest", None, dur, t))
        else:
            notes.append((syllable, note_to_midi(note), dur, t))
        t += dur
    total = t

    out = np.zeros(int(total * sr) + sr)
    xf = int(crossfade_ms / 1000.0 * sr)
    rng = np.random.default_rng(seed)
    i = 0
    while i < len(notes):
        if notes[i][0] == "rest":
            i += 1
            continue
        j = i
        while j < len(notes) and notes[j][0] != "rest":
            j += 1
        buf, pstart = _render_phrase(notes[i:j], sr, voice, base_pitch, phoneme,
                                     xf, rng, note_kw)
        pos = int(pstart * sr)
        e = min(out.size, pos + buf.size)
        out[pos:e] += buf[:e - pos]
        i = j
    return out[:int(total * sr) + int(0.3 * sr)].astype(np.float32)


# ---------------------------------------------------------------------------
# timbre, breath, compression (learning from vocal-synth voicing)
# ---------------------------------------------------------------------------

def _voice_timbre(x, sr):
    """Reshape espeak's thin/buzzy tone toward a natural sung voice: add low-mid
    warmth, fill the 3 kHz 'singer's formant' ring, tame the ~4 kHz buzz spike,
    and open up air on top.  (Frequencies chosen from espeak's measured spectrum.)"""
    chain = [
        highpass(90.0, sr, 0.7),
        low_shelf(330.0, sr, 5.0),          # body / warmth (espeak is thin here)
        peaking(2900.0, sr, 1.5, 3.5),      # singer's-formant ring (fills the dip)
        peaking(4200.0, sr, 2.4, -4.5),     # tame the harsh buzzy resonance
        high_shelf(7500.0, sr, 4.0),        # air / breathiness sheen
    ]
    return biquad_fft(chain, x)


def _breath(dry, sr, amount=0.14, seed=3):
    """Aspiration/breath layer -- the signature of modern vocal synths.  Band-
    limited noise (a breath spectrum) modulated by the voice envelope, with extra
    on note attacks."""
    if amount <= 0:
        return np.zeros_like(dry)
    n = dry.size
    env = one_pole_lp_fft(np.abs(dry), sr, 22.0)
    env = env / (float(np.max(env)) + 1e-9)
    rise = np.diff(env, prepend=env[0])
    attack = one_pole_lp_fft(np.maximum(rise, 0.0), sr, 40.0)
    attack = attack / (float(np.max(attack)) + 1e-9)
    mod = 0.75 * env + 0.9 * attack

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    noise = biquad_fft([highpass(1600.0, sr, 0.7), lowpass(7500.0, sr, 0.7)], noise)
    breath = noise * mod
    breath = breath / (float(np.max(np.abs(breath))) + 1e-9)
    return (amount * breath).astype(np.float32)


def _compress(x, sr, thresh_db=-20.0, ratio=2.5, makeup_db=3.0):
    """Gentle 'produced' bus compression (smooth FFT-envelope detector)."""
    env = one_pole_lp_fft(np.abs(x).astype(np.float64), sr, 26.0)
    env_db = 20.0 * np.log10(np.maximum(env, 1e-6))
    over = np.maximum(0.0, env_db - thresh_db)
    gain_db = -over * (1.0 - 1.0 / max(1.0, ratio)) + makeup_db
    gain = one_pole_lp_fft(10.0 ** (gain_db / 20.0), sr, 45.0)
    return (x * gain).astype(np.float32)


def chorus(x, sr, mix=0.12, voices=2, depth_ms=4.0, rate=0.45):
    """Light detuned doubling for a thicker, produced sheen."""
    if mix <= 0:
        return x
    n = x.size
    t = np.arange(n)
    out = x.astype(np.float64) * (1.0 - mix)
    for k in range(voices):
        d = (7.0 + 4.0 * k + depth_ms * np.sin(2.0 * np.pi * rate * (1.0 + 0.3 * k)
             * t / sr + 1.7 * k)) * sr / 1000.0
        out += (mix / voices) * _read_cubic(x.astype(np.float64),
                                            np.clip(t - d, 0, n - 1))
    return out.astype(np.float32)


def render_song(score, sr=44100, bpm=100, voice="en+f4", base_pitch=62,
                phoneme=True, breath=0.14, chorus_mix=0.12, reverb_mix=0.2,
                width=1.25):
    """Full render: sing -> timbre-shape -> +breath -> compress -> chorus ->
    stereo -> reverb.  Aims for a warm, breathy, produced vocal-synth tone."""
    dry = sing(score, sr, bpm, voice, base_pitch, phoneme=phoneme)
    dry = _voice_timbre(dry, sr)
    dry = util.normalize_peak(dry, 0.9)
    dry = dry + _breath(dry, sr, amount=breath)
    dry = util.normalize_peak(dry, 0.9)
    dry = _compress(dry, sr)
    dry = chorus(dry, sr, mix=chorus_mix)
    dry = util.normalize_peak(dry, 0.92)

    stereo = effects.stereoize(dry, sr, haas_ms=10.0, width=width)
    stereo = effects.reverb(stereo, sr, mix=reverb_mix, size=0.7, damp=0.5,
                            width=1.2)
    stereo = util.normalize_percentile(stereo, target=0.85)
    stereo = util.soft_limit(stereo, 0.98)
    return stereo
