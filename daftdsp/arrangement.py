"""
Instrumental backing: chords, bass and drums under the sung line.

Everything here is synthesised from the project's own from-scratch DSP -- the
polyphonic oscillator bank in :mod:`daftdsp.synth`, the filters in
:mod:`daftdsp.biquad` and the effects in :mod:`daftdsp.effects` -- so the backing
needs no samples and no extra dependencies.

An arrangement is a chord chart: ``[("Am", 4), ("F", 4), ("C", 4), ("G", 4)]``
meaning four beats of each.  From that it builds

  * **keys**   -- a sustained pad voicing the chord, filtered and soft,
  * **bass**   -- the chord root an octave or two down, played on the beat,
  * **drums**  -- kick, snare and hats on a simple pattern,

and mixes them under the voice with the vocal kept in front.
"""
from __future__ import annotations

import numpy as np

from . import effects, synth, util
from .biquad import biquad_fft, bandpass, highpass, low_shelf, lowpass, peaking
from .util import midi_to_freq, note_to_midi

# Chord spellings as semitone offsets from the root.
QUALITIES = {
    "": [0, 4, 7], "maj": [0, 4, 7], "m": [0, 3, 7], "min": [0, 3, 7],
    "7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11], "m7": [0, 3, 7, 10],
    "min7": [0, 3, 7, 10], "sus2": [0, 2, 7], "sus4": [0, 5, 7],
    "dim": [0, 3, 6], "aug": [0, 4, 8], "6": [0, 4, 7, 9], "m6": [0, 3, 7, 9],
    "add9": [0, 4, 7, 14], "m9": [0, 3, 7, 10, 14], "9": [0, 4, 7, 10, 14],
}


def parse_chord(name, octave=3):
    """``"Am7"`` -> (root midi, [midi notes]).  Root defaults to ``octave``."""
    name = name.strip()
    if not name or name in ("-", "N.C."):
        return None, []
    i = 1
    if len(name) > 1 and name[1] in "#b":
        i = 2
    root = note_to_midi(f"{name[:i]}{octave}")
    quality = name[i:]
    return root, [root + s for s in QUALITIES.get(quality, QUALITIES[""])]


def chart_timeline(chart, bpm):
    """``[(chord, beats)]`` -> ``[(start_s, dur_s, chord)]``."""
    beat = 60.0 / bpm
    out, t = [], 0.0
    for chord, beats in chart:
        out.append((t, beats * beat, chord))
        t += beats * beat
    return out, t


# ---------------------------------------------------------------------------
# instruments
# ---------------------------------------------------------------------------

def keys(chart, bpm, sr, level=0.22, octave=3):
    """A sustained pad: the chord held, filtered soft, with a gentle swell."""
    spans, total = chart_timeline(chart, bpm)
    out = np.zeros(int(total * sr) + sr, dtype=np.float64)
    for start, dur, name in spans:
        _, notes = parse_chord(name, octave)
        if not notes:
            continue
        n = int(dur * sr)
        voice = synth.render_carrier(
            n, sr, notes, saw_level=0.5, pulse_level=0.25, detune_cents=9.0,
            detune_voices=3, octave_layer=False, vibrato_depth=0.0, level=1.0)
        # A pad sits behind the voice, so most of its top end is removed.
        voice = biquad_fft([lowpass(2200.0, sr, 0.7), highpass(160.0, sr, 0.7)],
                           voice)
        env = np.ones(n)
        a = min(int(0.06 * sr), n // 2)
        r = min(int(0.12 * sr), n // 2)
        env[:a] = np.linspace(0.0, 1.0, a) ** 0.7
        env[-r:] = np.linspace(1.0, 0.0, r) ** 1.2
        p = int(start * sr)
        out[p:p + n] += voice[:n] * env
    return (out * level).astype(np.float32)


def bass(chart, bpm, sr, level=0.30, octave=2):
    """Root notes on the beat, rounded off so they stay under everything."""
    beat = 60.0 / bpm
    spans, total = chart_timeline(chart, bpm)
    out = np.zeros(int(total * sr) + sr, dtype=np.float64)
    for start, dur, name in spans:
        root, _ = parse_chord(name, octave)
        if root is None:
            continue
        t = start
        while t < start + dur - 1e-6:
            step = min(beat, start + dur - t)
            n = int(step * sr)
            if n < 32:
                break
            f = midi_to_freq(root)
            k = np.arange(n) / sr
            wave = (np.sin(2 * np.pi * f * k)
                    + 0.3 * np.sin(4 * np.pi * f * k)).astype(np.float64)
            env = np.exp(-k / (0.28 * step)) * (1 - np.exp(-k / 0.006))
            p = int(t * sr)
            out[p:p + n] += wave * env
            t += beat
    out = biquad_fft(lowpass(420.0, sr, 0.7), out.astype(np.float32))
    return (np.asarray(out, dtype=np.float64) * level).astype(np.float32)


def drums(bars_beats, bpm, sr, level=0.5, swing=0.0):
    """Kick / snare / hats on a straight pop pattern."""
    beat = 60.0 / bpm
    total = bars_beats * beat
    out = np.zeros(int(total * sr) + sr, dtype=np.float64)
    rng = np.random.default_rng(3)

    def place(at, sig, gain=1.0):
        p = int(at * sr)
        e = min(out.size, p + sig.size)
        if e > p:
            out[p:e] += gain * sig[:e - p]

    kd = int(0.22 * sr)
    ki = np.arange(kd) / sr
    kick = (np.sin(2 * np.pi * (48 + 90 * np.exp(-ki / 0.03)) * ki)
            * np.exp(-ki / 0.09))
    sd = int(0.18 * sr)
    si = np.arange(sd) / sr
    snare_noise = biquad_fft(bandpass(1900.0, sr, 1.0),
                             rng.standard_normal(sd).astype(np.float32))
    snare = (np.asarray(snare_noise, dtype=np.float64) * np.exp(-si / 0.055)
             + 0.5 * np.sin(2 * np.pi * 185 * si) * np.exp(-si / 0.04))
    hd = int(0.06 * sr)
    hi = np.arange(hd) / sr
    hat = (np.asarray(biquad_fft(highpass(7000.0, sr, 0.7),
                                 rng.standard_normal(hd).astype(np.float32)),
                      dtype=np.float64) * np.exp(-hi / 0.018))

    for b in range(int(bars_beats)):
        at = b * beat
        if b % 4 in (0, 2):
            place(at, kick, 1.0)
        if b % 4 in (1, 3):
            place(at, snare, 0.7)
        place(at, hat, 0.35)
        place(at + beat * (0.5 + swing), hat, 0.22)
    return (out * level).astype(np.float32)


# ---------------------------------------------------------------------------
# arrangement + mix
# ---------------------------------------------------------------------------

def render_backing(chart, bpm, sr, use_drums=True, use_bass=True, use_keys=True,
                   level=1.0):
    """Full instrumental bed for a chord chart."""
    _, total = chart_timeline(chart, bpm)
    beats = sum(b for _, b in chart)
    parts = []
    if use_keys:
        parts.append(keys(chart, bpm, sr))
    if use_bass:
        parts.append(bass(chart, bpm, sr))
    if use_drums:
        parts.append(drums(beats, bpm, sr))
    if not parts:
        return np.zeros(int(total * sr), dtype=np.float32)
    n = max(p.size for p in parts)
    mixed = np.zeros(n, dtype=np.float64)
    for p in parts:
        mixed[:p.size] += p
    # Trim to the chart, keeping a short tail so the last chord and cymbal can
    # ring out instead of being cut off.
    tail = int(0.35 * sr)
    mixed = mixed[:int(total * sr) + tail]
    mixed = util.soft_limit(util.normalize_percentile(
        mixed.astype(np.float32), target=0.7), 0.9)
    return (np.asarray(mixed, dtype=np.float64) * level).astype(np.float32)


def mix(vocal, backing, sr, vocal_level=1.0, backing_level=0.42, duck=0.3):
    """Sit the backing under the voice.

    The band is ducked by the vocal (the same side-chain idea a mix engineer
    uses) so the words stay in front instead of competing with the chords, and a
    small dip is carved around 1-3 kHz where the voice lives.
    """
    n = max(vocal.size, backing.size)
    v = np.zeros(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    v[:vocal.size] = vocal
    b[:backing.size] = backing

    if duck > 0 and vocal.size:
        env = np.abs(v)
        k = max(1, int(0.05 * sr))
        env = np.convolve(env, np.ones(k) / k, mode="same")
        env /= (float(env.max()) or 1.0)
        b *= 1.0 - duck * np.clip(env * 2.5, 0.0, 1.0)
    # Make room for the voice where it lives.
    b = np.asarray(biquad_fft(peaking(1800.0, sr, 1.1, -3.0),
                              b.astype(np.float32)), dtype=np.float64)
    b = np.asarray(biquad_fft(low_shelf(120.0, sr, 1.5),
                              b.astype(np.float32)), dtype=np.float64)

    out = vocal_level * v + backing_level * b
    out = util.normalize_percentile(out.astype(np.float32), target=0.85)
    return util.soft_limit(out, 0.97)
