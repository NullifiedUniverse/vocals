"""
Note-timed singing synthesis -- "make it sing".

Turns a score of ``(syllable, note, beats)`` into a sung vocal.  Two things make
it sound like a voice rather than chopped speech:

  * **Phoneme input.**  Each syllable is given to espeak as phonemes (``[[...]]``)
    so it is pronounced correctly out of word context -- no more "kle"/"der"
    mispronunciations.
  * **Vowel-nucleus sustain.**  To hold a long note, a *few pitch periods* of the
    vowel's steady centre are cross-fade-looped and spliced back into the vowel,
    keeping the natural consonant onset and tail.  The note sustains as one held
    vowel instead of the whole syllable repeating.

Pitch comes from formant-preserving PSOLA; vibrato and chorus are modulated
fractional delays.  All from-scratch.
"""
from __future__ import annotations

import numpy as np

from . import effects, pitch, psola, tts, util
from .util import midi_to_freq, note_to_midi


# ---------------------------------------------------------------------------
# Cross-fade / loop helpers (equal-power)
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
    fade_out = np.cos(0.5 * np.pi * ramp)
    fade_in = np.sin(0.5 * np.pi * ramp)
    mid = a[-xf:] * fade_out + b[:xf] * fade_in
    return np.concatenate([a[:-xf], mid, b[xf:]])


def _energy(x, win):
    """Smoothed short-time energy (box filter of x^2)."""
    p = np.asarray(x, dtype=np.float64) ** 2
    k = np.ones(max(1, win)) / max(1, win)
    return np.convolve(p, k, mode="same")


def _sustain(x, sr, f0, length):
    """Hold ``x`` to ``length`` samples with pitch-synchronous overlap-add.

    Hann grains (two target-periods long) are laid down at exact target-period
    spacing, so the output pitch is exact and the sustain is click-free.  The
    read pointer copies the consonant onset 1:1, then scans *slowly* through the
    vowel (a natural sustain, not a repeat); the consonant tail is appended at
    the end."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if length <= n:
        return x[:length].copy()

    period = max(4, int(round(sr / max(1.0, f0))))
    win = 2 * period
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win) / win)

    # Vowel nucleus (peak energy in the first ~70%): loud, steady point to hold.
    e = _energy(x, period)
    peak = int(np.argmax(e[:max(2 * period, int(n * 0.7))]))
    onset = int(np.clip(peak - 2 * period, 0, n))
    vend = int(np.clip(int(n * 0.86), onset + 2 * period, n - 1))
    coda = x[vend:]
    coda_len = int(min(coda.size, 0.14 * sr))
    body_len = max(int(0.06 * sr), length - coda_len)
    hold = int(np.clip(round(peak / period) * period, 0, n - win))

    out = np.zeros(body_len + win)
    nrm = np.zeros(body_len + win)
    k = 0
    while k * period < body_len:
        op = k * period
        # Onset consonant plays 1:1; then the vowel nucleus is held (period-
        # aligned so grains stay phase-coherent) -> a steady, non-decaying vowel.
        ai = op if op < onset else hold
        ai = int(np.clip(ai, 0, n - win))
        out[op:op + win] += x[ai:ai + win] * w
        nrm[op:op + win] += w
        k += 1
    m = nrm > 1e-6
    out[m] /= nrm[m]
    body = out[:body_len]

    if coda_len > 0:
        note = _xfade(body, coda[:coda_len], min(period, body_len // 4,
                                                 max(1, coda_len // 2)))
    else:
        note = body
    if note.size < length:
        note = np.pad(note, (0, length - note.size))
    return note[:length]


def _adsr(x, sr, a=0.016, d=0.06, s=0.88, r=0.06):
    n = x.size
    env = np.full(n, s, dtype=np.float64)
    ai, di, ri = int(a * sr), int(d * sr), int(r * sr)
    if ai > 0:
        env[:min(ai, n)] = np.linspace(0.0, 1.0, min(ai, n))
    if di > 0 and ai + di <= n:
        env[ai:ai + di] = np.linspace(1.0, s, di)
    if ri > 0 and ri < n:
        env[-ri:] = env[-ri:] * np.linspace(1.0, 0.0, ri)
    return (x * env).astype(np.float32)


# ---------------------------------------------------------------------------
# Note / score rendering
# ---------------------------------------------------------------------------

def _read_cubic(x, read):
    """Catmull-Rom cubic interpolation read of ``x`` at fractional positions
    ``read`` (lower distortion than linear for the modulated delays)."""
    n = x.size
    i = np.floor(read).astype(np.int64)
    f = read - i

    def tap(o):
        return x[np.clip(i + o, 0, n - 1)]

    p0, p1, p2, p3 = tap(-1), tap(0), tap(1), tap(2)
    return p1 + 0.5 * f * (p2 - p0 + f * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
                          + f * (3.0 * (p1 - p2) + p3 - p0)))


def _vibrato(x, sr, rate, depth_cents, delay_s, ramp_s):
    """Pitch vibrato that swells in: silent for ``delay_s``, then ramps to full
    ``depth_cents`` over ``ramp_s`` -- how a singer actually adds vibrato."""
    if depth_cents <= 0:
        return x
    n = x.size
    t = np.arange(n)
    env = np.clip((t / sr - delay_s) / max(1e-3, ramp_s), 0.0, 1.0)
    amp_s = depth_cents * np.log(2.0) / 1200.0 / (2.0 * np.pi * max(0.1, rate))
    d = amp_s * sr * env * np.sin(2.0 * np.pi * rate * t / sr)
    return _read_cubic(x, np.clip(t - d, 0.0, n - 1)).astype(np.float32)


def render_note(syllable, midi, dur_s, sr, voice="en+f4", base_pitch=62,
                phoneme=True, wpm=150, prev_midi=None, glide_ms=38.0,
                vib_rate=5.7, vib_depth=32.0, vib_delay=0.32):
    """Synthesize one sung note: pronounce the syllable, glide/pitch it to
    ``midi``, hold it for ``dur_s`` seconds, and add swelling vibrato."""
    text = f"[[{syllable}]]" if phoneme else syllable
    raw = tts.text_to_vocal(text, sr, voice=voice, pitch=base_pitch, wpm=wpm)
    if raw.size < int(0.05 * sr):
        raw = np.pad(raw, (0, int(0.05 * sr) - raw.size))
    raw = util.normalize_peak(raw, 0.9)
    track = pitch.track_pitch(raw, sr, max_f0=1000.0)
    f0 = midi_to_freq(midi)

    # Legato: glide from the previous note's pitch over the first few ms.
    target = np.full(raw.size, f0)
    if prev_midi is not None and glide_ms > 0:
        g = min(int(glide_ms / 1000.0 * sr), raw.size // 2)
        if g > 1:
            pf = midi_to_freq(prev_midi)
            target[:g] = pf * (f0 / pf) ** np.linspace(0.0, 1.0, g)  # log glide

    pitched = psola.psola_correct(raw, sr, track, target_hz=target, retune=1.0,
                                  retune_time_ms=6.0, max_f0=1300.0)
    length = max(int(dur_s * sr), int(0.09 * sr))
    note = _adsr(_sustain(pitched, sr, f0, length), sr)
    return _vibrato(note, sr, vib_rate, vib_depth, vib_delay, 0.32)


def sing(score, sr=44100, bpm=100, voice="en+f4", base_pitch=62, phoneme=True,
         legato=0.05, **note_kw):
    """Render a whole score to a dry mono sung line (with legato glides)."""
    beat = 60.0 / bpm
    total = sum(b for _, _, b in score) * beat
    out = np.zeros(int(total * sr) + int(0.5 * sr), dtype=np.float64)
    t = 0.0
    prev_midi = None
    for syllable, note, beats in score:
        dur = beats * beat
        if note is None or syllable in (None, "", "rest", "-"):
            t += dur
            prev_midi = None                     # rest breaks the legato
            continue
        midi = note_to_midi(note)
        audio = render_note(syllable, midi, dur + legato * beat, sr, voice,
                            base_pitch, phoneme=phoneme, prev_midi=prev_midi,
                            **note_kw)
        pos = int(t * sr)
        end = min(out.size, pos + audio.size)
        out[pos:end] += audio[:end - pos]
        t += dur
        prev_midi = midi
    return out[:int(t * sr) + int(0.4 * sr)].astype(np.float32)


# ---------------------------------------------------------------------------
# Chorus (modulated fractional delays) + full render
# ---------------------------------------------------------------------------

def chorus(x, sr, mix=0.14, voices=2, depth_ms=4.0, rate=0.45):
    """Light detuned doubling for a thicker, synthetic-diva sheen."""
    if mix <= 0:
        return x
    n = x.size
    t = np.arange(n)
    out = x.astype(np.float64) * (1.0 - mix)
    for k in range(voices):
        d = (7.0 + 4.0 * k + depth_ms * np.sin(2.0 * np.pi * rate * (1.0 + 0.3 * k)
             * t / sr + 1.7 * k)) * sr / 1000.0
        out += (mix / voices) * _read_cubic(x.astype(np.float64), np.clip(t - d, 0, n - 1))
    return out.astype(np.float32)


def render_song(score, sr=44100, bpm=100, voice="en+f4", base_pitch=62,
                phoneme=True, chorus_mix=0.13, reverb_mix=0.2, bright_db=3.5,
                width=1.25):
    """Full render: sing (with per-note vibrato + legato) -> chorus -> bright EQ
    -> stereo -> reverb."""
    dry = sing(score, sr, bpm, voice, base_pitch, phoneme=phoneme)
    dry = chorus(dry, sr, mix=chorus_mix)
    dry = util.normalize_peak(dry, 0.9)
    dry = effects.eq(dry, sr, hp_freq=110.0, shelf_freq=5500.0,
                     shelf_gain_db=bright_db)
    stereo = effects.stereoize(dry, sr, haas_ms=10.0, width=width)
    stereo = effects.reverb(stereo, sr, mix=reverb_mix, size=0.68, damp=0.5,
                            width=1.2)
    stereo = util.normalize_percentile(stereo, target=0.85)
    stereo = util.soft_limit(stereo, 0.98)
    return stereo
