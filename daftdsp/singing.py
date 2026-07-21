"""
Note-timed singing synthesis -- "make it sing".

Turns a score of ``(syllable, note, beats)`` into a sung vocal: each syllable is
spoken by espeak, pitch-shifted onto its note with formants preserved (so it
stays a *voice*), and its vowel is loop-sustained to fill the note's duration.
A bright, vibrato + chorus + reverb chain gives a clean Vocaloid / Miku-style
synthetic-diva voice.

Everything is built on the same from-scratch DSP as the rest of the engine
(PSOLA, biquads, reverb); vibrato and chorus are modulated fractional delays.
"""
from __future__ import annotations

import numpy as np

from . import effects, pitch, psola, tts, util
from .util import midi_to_freq, note_to_midi


def _adsr(x, sr, a=0.014, d=0.05, s=0.86, r=0.07):
    """Simple attack/decay/sustain/release amplitude envelope for one note."""
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


def _sustain_to(x, sr, f0, length):
    """Loop the steady vowel of ``x`` (integer pitch periods, crossfaded) so the
    note lasts ``length`` samples."""
    n = x.size
    if length <= n:
        return x[:length].astype(np.float64)
    period = max(4, int(round(sr / max(1.0, f0))))
    k0 = int(n * 0.5)
    klen = ((int(n * 0.92) - k0) // period) * period
    if klen < 2 * period:
        k0 = max(0, n - 4 * period)
        klen = ((n - k0) // period) * period
    if klen < period:
        return np.pad(x.astype(np.float64), (0, length - n))
    kernel = x[k0:k0 + klen].astype(np.float64)
    xf = min(2 * period, klen // 2)
    wfade = np.linspace(0.0, 1.0, xf)
    out = x[:k0 + klen].astype(np.float64)
    while out.size < length:
        a = out[-xf:]
        b = kernel[:xf]
        out = np.concatenate([out[:-xf], a * (1 - wfade) + b * wfade, kernel[xf:]])
    return out[:length]


def render_note(syllable, midi, dur_s, sr, voice="en+f4", base_pitch=62, wpm=150):
    """Synthesize one sung note: speak the syllable, pitch it to ``midi``, hold it
    for ``dur_s`` seconds."""
    raw = tts.text_to_vocal(syllable, sr, voice=voice, pitch=base_pitch, wpm=wpm)
    if raw.size < int(0.04 * sr):
        raw = np.pad(raw, (0, int(0.04 * sr) - raw.size))
    raw = util.normalize_peak(raw, 0.9)
    track = pitch.track_pitch(raw, sr, max_f0=1000.0)
    f0 = midi_to_freq(midi)
    target = np.full(raw.size, f0)
    pitched = psola.psola_correct(raw, sr, track, target_hz=target, retune=1.0,
                                  retune_time_ms=6.0, max_f0=1300.0)
    length = max(int(dur_s * sr), int(0.08 * sr))
    note = _sustain_to(pitched, sr, f0, length)
    return _adsr(note, sr)


def sing(score, sr=44100, bpm=100, voice="en+f4", base_pitch=62, legato=0.06):
    """Render a whole score to a dry mono sung line.

    ``score`` items are ``(syllable, note, beats)``; ``note`` is a name like
    ``"C4"`` (or None / syllable "rest" for a rest)."""
    beat = 60.0 / bpm
    total = sum(b for _, _, b in score) * beat
    out = np.zeros(int(total * sr) + int(0.5 * sr), dtype=np.float64)
    t = 0.0
    for syllable, note, beats in score:
        dur = beats * beat
        if note is None or syllable in (None, "", "rest", "-"):
            t += dur
            continue
        # A touch of overlap into the next slot gives legato rather than clicks.
        audio = render_note(syllable, note_to_midi(note), dur + legato * beat,
                            sr, voice, base_pitch)
        pos = int(t * sr)
        end = min(out.size, pos + audio.size)
        out[pos:end] += audio[:end - pos]
        t += dur
    return out[:int(t * sr) + int(0.4 * sr)].astype(np.float32)


def _mod_delay(x, sr, base_ms, depth_ms, rate, phase=0.0):
    """Read ``x`` through a sine-modulated fractional delay (pitch modulation)."""
    n = x.size
    t = np.arange(n)
    d = (base_ms + depth_ms * np.sin(2.0 * np.pi * rate * t / sr + phase)) * sr / 1000.0
    read = np.clip(t - d, 0.0, n - 1)
    i0 = np.floor(read).astype(np.int64)
    frac = read - i0
    return x[i0] * (1.0 - frac) + x[np.clip(i0 + 1, 0, n - 1)] * frac


def vibrato(x, sr, rate=5.6, depth_cents=28.0):
    """Pitch vibrato via a zero-mean modulated delay."""
    if depth_cents <= 0:
        return x
    amp_s = depth_cents * np.log(2.0) / 1200.0 / (2.0 * np.pi * max(0.1, rate))
    return _mod_delay(x, sr, 0.0, amp_s * 1000.0, rate).astype(np.float32)


def chorus(x, sr, mix=0.25, voices=2, depth_ms=5.0, rate=0.5):
    """Light detuned doubling for a thicker, synthetic-diva sheen."""
    if mix <= 0:
        return x
    out = x.astype(np.float64) * (1.0 - mix)
    for k in range(voices):
        v = _mod_delay(x, sr, 6.0 + 4.0 * k, depth_ms, rate * (1.0 + 0.3 * k),
                       phase=1.7 * k)
        out += (mix / voices) * v
    return out.astype(np.float32)


def render_song(score, sr=44100, bpm=100, voice="en+f4", base_pitch=62,
                vib_rate=5.6, vib_depth=26.0, chorus_mix=0.24, reverb_mix=0.22,
                bright_db=4.0, width=1.3):
    """Full Miku-style render: sing the score, then vibrato -> chorus -> bright
    EQ -> stereo -> reverb.  Returns an ``(N, 2)`` float32 buffer."""
    dry = sing(score, sr, bpm, voice, base_pitch)
    dry = vibrato(dry, sr, vib_rate, vib_depth)
    dry = chorus(dry, sr, mix=chorus_mix)
    dry = util.normalize_peak(dry, 0.9)
    dry = effects.eq(dry, sr, hp_freq=120.0, shelf_freq=5000.0,
                     shelf_gain_db=bright_db)
    stereo = effects.stereoize(dry, sr, haas_ms=12.0, width=width)
    stereo = effects.reverb(stereo, sr, mix=reverb_mix, size=0.7, damp=0.45,
                            width=1.2)
    stereo = util.normalize_percentile(stereo, target=0.85)
    stereo = util.soft_limit(stereo, 0.98)
    return stereo
