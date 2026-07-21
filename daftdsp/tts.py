"""
Text -> vocal PCM (the modulator source for the vocoder / talkbox).

Primary path shells out to ``espeak-ng`` (a formant synth that already sounds
robotic and monotone -- ideal raw material for vocoding).  If no espeak binary is
present, a small from-scratch vowel-formant babble synthesiser keeps the whole
pipeline runnable so the DSP can still be demonstrated.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile

import numpy as np

from . import util


def espeak_path():
    return shutil.which("espeak-ng") or shutil.which("espeak")


def text_to_vocal(text, sr, *, wpm=150, pitch=35, voice="en", gap_ms=2):
    """Render ``text`` to a mono float32 array at ``sr``.  Falls back to the
    built-in synth when espeak is unavailable."""
    text = (text or "").strip()
    if not text:
        text = "we are the robots"

    exe = espeak_path()
    if exe:
        try:
            return _espeak(exe, text, sr, wpm, pitch, voice, gap_ms)
        except Exception:
            pass  # fall through to the built-in synth
    return _fallback_synth(text, sr, pitch)


def _espeak(exe, text, sr, wpm, pitch, voice, gap_ms):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        cmd = [exe, "-v", str(voice), "-s", str(int(wpm)),
               "-p", str(int(np.clip(pitch, 0, 99))),
               "-g", str(int(gap_ms)), "-w", f.name, text]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=30)
        with open(f.name, "rb") as fh:
            data = fh.read()
    mono, file_sr = util.read_wav(data)
    if file_sr != sr:
        mono = util.resample(mono, file_sr, sr)
    # Trim leading/trailing silence for tighter timing.
    return _trim_silence(mono)


def _trim_silence(x, thresh=2e-3):
    if x.size == 0:
        return x
    idx = np.where(np.abs(x) > thresh)[0]
    if idx.size == 0:
        return x
    lo = max(0, idx[0] - 256)
    hi = min(x.size, idx[-1] + 256)
    return x[lo:hi].astype(np.float32)


# ---------------------------------------------------------------------------
# Fallback formant-babble synth (no external dependencies)
# ---------------------------------------------------------------------------

_LETTER_VOWEL = {
    "a": "a", "e": "e", "i": "i", "o": "o", "u": "u", "y": "i",
}


def _fallback_synth(text, sr, pitch):
    from .formant import VOWELS, _apply_formants

    f0 = 90.0 * (2.0 ** ((pitch - 35) / 50.0))
    seg = int(0.10 * sr)
    silence = int(0.05 * sr)
    chunks = []
    vowels = ["a", "e", "i", "o", "u"]
    vi = 0
    for ch in text.lower():
        if ch == " ":
            chunks.append(np.zeros(silence, dtype=np.float64))
            continue
        if not ch.isalpha():
            continue
        # Buzzy glottal source: band-limited saw-ish pulse train.
        t = np.arange(seg) / sr
        phase = (f0 * t) % 1.0
        buzz = 2.0 * phase - 1.0
        v = _LETTER_VOWEL.get(ch, vowels[vi % len(vowels)])
        vi += 1
        shaped = _apply_formants(buzz, sr, VOWELS[v], 9.0, 12.0)
        env = np.hanning(seg)
        chunks.append(shaped * env)
    if not chunks:
        chunks = [np.zeros(seg, dtype=np.float64)]
    out = np.concatenate(chunks)
    m = float(np.max(np.abs(out))) or 1.0
    return (out / m * 0.7).astype(np.float32)
