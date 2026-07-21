"""
Formant / talkbox stage.

Chains resonant peaking biquads at vowel formant frequencies (F1-F3) to impose a
vocal-tract "vowel" colour on the signal, the way a talkbox does.  ``formant_shift``
scales all formant centres together, resizing the perceived vocal tract
independently of pitch.  Two vowels can be cross-faded by an LFO to get the
talkbox "mouth-moving" motion.
"""
from __future__ import annotations

import numpy as np

from .biquad import biquad_fft, peaking

# Approximate formant centre frequencies (Hz) for sustained vowels.
VOWELS = {
    "a": (730.0, 1090.0, 2440.0),
    "e": (530.0, 1840.0, 2480.0),
    "i": (270.0, 2290.0, 3010.0),
    "o": (570.0, 840.0, 2410.0),
    "u": (300.0, 870.0, 2240.0),
    "ah": (640.0, 1190.0, 2390.0),
    "oo": (300.0, 870.0, 2240.0),
}


def _apply_formants(x, sr, formants, resonance, gain_db):
    # Whole formant chain applied as one FFT (responses multiplied).
    coeffs = [peaking(f, sr, resonance, gain_db) for f in formants]
    return biquad_fft(coeffs, x)


def talkbox(x, sr, *, vowel="a", vowel2="", morph_rate=0.0, formant_shift=1.0,
            resonance=8.0, gain_db=10.0, amount=1.0):
    """Apply the talkbox formant filter.  Returns a signal the length of ``x``."""
    x = np.asarray(x, dtype=np.float64)
    if amount <= 0.0 or x.size == 0:
        return x.astype(np.float32)

    base1 = VOWELS.get(vowel, VOWELS["a"])
    f1 = [np.clip(f * formant_shift, 80.0, sr * 0.45) for f in base1]

    if vowel2 and vowel2 in VOWELS and morph_rate > 0.0:
        base2 = VOWELS[vowel2]
        f2 = [np.clip(f * formant_shift, 80.0, sr * 0.45) for f in base2]
        wet1 = _apply_formants(x, sr, f1, resonance, gain_db)
        wet2 = _apply_formants(x, sr, f2, resonance, gain_db)
        t = np.arange(x.size) / sr
        lfo = 0.5 - 0.5 * np.cos(2.0 * np.pi * morph_rate * t)
        wet = (1.0 - lfo) * wet1 + lfo * wet2
    else:
        wet = _apply_formants(x, sr, f1, resonance, gain_db)

    out = (1.0 - amount) * x + amount * wet
    return out.astype(np.float32)
