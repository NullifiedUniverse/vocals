"""
Polyphonic carrier synthesiser.

Generates the harmonic "carrier" that the vocoder sculpts with the voice.  Built
from scratch: detuned super-saws and pulse waves with variable pulse-width
modulation, PolyBLEP band-limiting, full chord polyphony with automatic +/-1
octave layering, optional vibrato, and -- crucially for expression -- support
for chord *progressions* so the robot's harmony moves through the phrase instead
of sitting on one static pad.
"""
from __future__ import annotations

import numpy as np

from .util import midi_to_freq, note_to_midi


def _poly_blep(t: np.ndarray, dt: float) -> np.ndarray:
    """Vectorised PolyBLEP residual for a unit-step discontinuity."""
    out = np.zeros_like(t)
    if dt <= 0:
        return out
    m = t < dt
    tt = t[m] / dt
    out[m] = tt + tt - tt * tt - 1.0
    m = t > 1.0 - dt
    tt = (t[m] - 1.0) / dt
    out[m] = tt * tt + tt + tt + 1.0
    return out


def _phase(freq, n, sr, phase0, vib):
    """Phase ramp in [0,1); constant-rate, or integrated when vibrato is on."""
    dt = freq / sr
    if vib is None:
        ph = (phase0 + dt * np.arange(n, dtype=np.float64)) % 1.0
    else:
        ph = (phase0 + np.cumsum(dt * vib)) % 1.0
    return ph, dt


def _osc(phase, dt, saw_level, pulse_level, duty):
    """One band-limited oscillator = saw + PWM pulse for a given phase ramp."""
    out = np.zeros_like(phase)
    if saw_level != 0.0:
        out += saw_level * ((2.0 * phase - 1.0) - _poly_blep(phase, dt))
    if pulse_level != 0.0:
        pulse = np.where(phase < duty, 1.0, -1.0)
        pulse += _poly_blep(phase, dt)
        pulse -= _poly_blep((phase - duty) % 1.0, dt)
        out += pulse_level * pulse
    return out


def _render_chord(chord_midi, n, sr, duty, vib, *, saw_level, pulse_level,
                  detune_cents, detune_voices, octave_layer, sub_level, rng):
    """Full-length carrier for a single chord (all voices, phase-continuous)."""
    voices = []
    for m in chord_midi:
        voices.append((m, 1.0))
        if octave_layer:
            voices.append((m + 12, 0.42))
            voices.append((m - 12, sub_level))

    if detune_voices < 1:
        detune_voices = 1
    spread = [0.0] if detune_voices == 1 else \
        np.linspace(-detune_cents, detune_cents, detune_voices)

    out = np.zeros(n, dtype=np.float64)
    for midi, weight in voices:
        base = midi_to_freq(midi)
        if base <= 0 or base > sr * 0.45:
            continue
        for cents in spread:
            f = base * (2.0 ** (cents / 1200.0))
            phase, dt = _phase(f, n, sr, rng.random(), vib)
            out += weight * _osc(phase, dt, saw_level, pulse_level, duty)
    return out


def _seg_env(n, i, k, seg, xf):
    """Raised-cosine crossfade window for segment ``i`` of ``k`` (edges sum to 1)."""
    idx = np.arange(n, dtype=np.float64)
    s, e, h = i * seg, (i + 1) * seg, xf / 2.0
    env = np.zeros(n)
    left = (s + h) if i > 0 else -1.0
    right = (e - h) if i < k - 1 else n + 1.0
    env[(idx >= left) & (idx < right)] = 1.0
    if i > 0:
        m = (idx >= s - h) & (idx < s + h)
        env[m] = 0.5 - 0.5 * np.cos(np.pi * (idx[m] - (s - h)) / xf)
    if i < k - 1:
        m = (idx >= e - h) & (idx < e + h)
        env[m] = 0.5 + 0.5 * np.cos(np.pi * (idx[m] - (e - h)) / xf)
    return env


def render_carrier(
    n: int,
    sr: int,
    chord,
    *,
    saw_level: float = 0.8,
    pulse_level: float = 0.5,
    detune_cents: float = 16.0,
    detune_voices: int = 3,
    pulse_width: float = 0.5,
    pwm_rate: float = 0.5,
    pwm_depth: float = 0.25,
    octave_layer: bool = True,
    sub_level: float = 0.5,
    vibrato_rate: float = 5.5,
    vibrato_depth: float = 0.0,
    level: float = 1.0,
) -> np.ndarray:
    """Render a polyphonic carrier of ``n`` samples.

    ``chord`` is either a list of MIDI notes (static chord) or a list of such
    lists (a progression that is crossfaded across the duration)."""
    if n <= 0 or not len(chord):
        return np.zeros(max(n, 0), dtype=np.float32)

    t = np.arange(n, dtype=np.float64) / sr
    duty = np.clip(pulse_width + pwm_depth * np.sin(2.0 * np.pi * pwm_rate * t),
                   0.05, 0.95)
    vib = None
    if vibrato_depth > 0.0:
        vib = 2.0 ** (vibrato_depth * np.sin(2.0 * np.pi * vibrato_rate * t) / 12.0)

    rng = np.random.default_rng(1234)
    kw = dict(saw_level=saw_level, pulse_level=pulse_level, detune_cents=detune_cents,
              detune_voices=detune_voices, octave_layer=octave_layer,
              sub_level=sub_level, rng=rng)

    is_prog = isinstance(chord[0], (list, tuple, np.ndarray))
    if not is_prog:
        out = _render_chord(chord, n, sr, duty, vib, **kw)
    else:
        prog = [list(c) for c in chord if len(c)]
        k = len(prog)
        seg = n / k
        xf = int(min(0.14 * sr, seg * 0.45))
        xf = max(xf, 1)
        out = np.zeros(n, dtype=np.float64)
        for i, ch in enumerate(prog):
            out += _render_chord(ch, n, sr, duty, vib, **kw) * _seg_env(n, i, k, seg, xf)

    m = float(np.max(np.abs(out))) or 1.0
    return (out / m * 0.9 * level).astype(np.float32)


def parse_chord(root_name: str, quality: str) -> list[int]:
    """Return a list of MIDI notes for ``root_name`` (e.g. 'A3') and ``quality``."""
    intervals = {
        "major": [0, 4, 7], "minor": [0, 3, 7], "maj7": [0, 4, 7, 11],
        "min7": [0, 3, 7, 10], "dom7": [0, 4, 7, 10], "sus2": [0, 2, 7],
        "sus4": [0, 5, 7], "power": [0, 7], "min9": [0, 3, 7, 10, 14],
        "add9": [0, 4, 7, 14],
    }.get(quality, [0, 4, 7])
    try:
        root = note_to_midi(root_name)
    except Exception:
        root = 57
    return [root + i for i in intervals]


def parse_progression(spec) -> list[list[int]]:
    """Turn a progression spec into a list of MIDI-note chords.

    Each item may be ``"A3:min7"`` / ``"C4"`` (root[:quality]) or an explicit
    list of MIDI numbers."""
    out = []
    for item in spec:
        if isinstance(item, (list, tuple)):
            out.append([int(x) for x in item])
        else:
            root, _, qual = str(item).partition(":")
            out.append(parse_chord(root, qual or "min7"))
    return out
