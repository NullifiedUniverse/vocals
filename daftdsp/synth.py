"""
Polyphonic carrier synthesiser.

Generates the harmonic "carrier" that the vocoder sculpts with the voice.  Built
from scratch: detuned super-saws and pulse waves with variable pulse-width
modulation, PolyBLEP band-limiting to keep the aliasing down, full chord
polyphony and automatic +/-1 octave layering.
"""
from __future__ import annotations

import numpy as np

from .util import midi_to_freq


def _poly_blep(t: np.ndarray, dt: float) -> np.ndarray:
    """Vectorised PolyBLEP residual for a unit-step discontinuity.

    ``t`` is the phase in [0, 1); ``dt`` the per-sample phase increment.  Adding
    this residual around each discontinuity removes most of the harsh aliasing a
    naive saw/pulse would produce.
    """
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


def _osc(freq: float, n: int, sr: int, saw_level: float, pulse_level: float,
         duty: np.ndarray, phase0: float = 0.0) -> np.ndarray:
    """One band-limited oscillator = saw + PWM pulse mix at a fixed frequency."""
    dt = freq / sr
    phase = (phase0 + dt * np.arange(n, dtype=np.float64)) % 1.0
    out = np.zeros(n, dtype=np.float64)
    if saw_level != 0.0:
        saw = (2.0 * phase - 1.0) - _poly_blep(phase, dt)
        out += saw_level * saw
    if pulse_level != 0.0:
        pulse = np.where(phase < duty, 1.0, -1.0)
        pulse += _poly_blep(phase, dt)
        pulse -= _poly_blep((phase - duty) % 1.0, dt)
        out += pulse_level * pulse
    return out


def render_carrier(
    n: int,
    sr: int,
    chord_midi,
    *,
    saw_level: float = 0.8,
    pulse_level: float = 0.5,
    detune_cents: float = 14.0,
    detune_voices: int = 3,
    pulse_width: float = 0.5,
    pwm_rate: float = 0.6,
    pwm_depth: float = 0.25,
    octave_layer: bool = True,
    sub_level: float = 0.5,
    level: float = 1.0,
) -> np.ndarray:
    """Render a polyphonic carrier of ``n`` samples.

    ``chord_midi`` is a list of MIDI note numbers.  Each note becomes a stack of
    ``detune_voices`` slightly detuned oscillators; with ``octave_layer`` the note
    is also doubled one octave up and a sub one octave down.
    """
    if n <= 0 or not len(chord_midi):
        return np.zeros(max(n, 0), dtype=np.float32)

    # Time-varying pulse width (PWM via a sine LFO), clamped away from 0/1.
    t = np.arange(n, dtype=np.float64) / sr
    duty = pulse_width + pwm_depth * np.sin(2.0 * np.pi * pwm_rate * t)
    duty = np.clip(duty, 0.05, 0.95)

    # Build the list of (midi, weight) voices with octave layering.
    voices = []
    for m in chord_midi:
        voices.append((m, 1.0))
        if octave_layer:
            voices.append((m + 12, 0.45))
            voices.append((m - 12, sub_level))

    # Detune spread in cents across the super-saw voices.
    if detune_voices < 1:
        detune_voices = 1
    if detune_voices == 1:
        spread = [0.0]
    else:
        spread = np.linspace(-detune_cents, detune_cents, detune_voices)

    out = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(1234)  # deterministic per-voice phase offsets
    for midi, weight in voices:
        base = midi_to_freq(midi)
        if base <= 0 or base > sr * 0.45:
            continue
        for cents in spread:
            f = base * (2.0 ** (cents / 1200.0))
            phase0 = rng.random()
            out += weight * _osc(f, n, sr, saw_level, pulse_level, duty, phase0)

    # Normalise by an estimate of the number of summed oscillators.
    denom = max(1.0, len(voices) * len(spread) * (saw_level + pulse_level) * 0.6)
    out = out / denom * level
    return out.astype(np.float32)


def parse_chord(root_name: str, quality: str) -> list[int]:
    """Return a list of MIDI notes for ``root_name`` (e.g. 'A3') and ``quality``."""
    from .util import note_to_midi

    intervals = {
        "major": [0, 4, 7],
        "minor": [0, 3, 7],
        "maj7": [0, 4, 7, 11],
        "min7": [0, 3, 7, 10],
        "dom7": [0, 4, 7, 10],
        "sus2": [0, 2, 7],
        "sus4": [0, 5, 7],
        "power": [0, 7],
        "min9": [0, 3, 7, 10, 14],
        "add9": [0, 4, 7, 14],
    }.get(quality, [0, 4, 7])
    try:
        root = note_to_midi(root_name)
    except Exception:
        root = 57  # A3
    return [root + i for i in intervals]
