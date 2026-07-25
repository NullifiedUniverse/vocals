"""
Neural speech front-end (Piper) with phoneme alignments.

This replaces espeak as the *source* of the singing voice.  espeak is a formant
synthesiser -- it generates speech from a rule-based resonator model, which is
why it sounds buzzy and robotic no matter what is done downstream.  Piper is a
neural TTS (VITS) trained on recordings of a real person, so its output is
human-like to begin with.

Just as important, Piper reports **phoneme alignments**: the exact number of
samples each phoneme occupies.  That removes the guesswork from deciding where
the vowels are, which is what a singing synthesiser needs in order to put the
right sound on the right note.

espeak remains available as a fallback so the package still runs (at lower
quality) when no neural model is present.
"""
from __future__ import annotations

import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Default voice: a clear, neutral US-English female voice.
DEFAULT_VOICE = "en_US-lessac-medium"
_HF = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
       "en/en_US/lessac/medium")

# IPA symbols Piper emits for vowel nuclei (plus the length mark, which extends
# the preceding vowel, and the stress marks, which attach to the next phoneme).
VOWELS = set("aeiouɑɐɒæɓɔəɚɛɜɞɘɵɪɨʉʊʌʏyøœɶɤɯɵ")
LENGTH = "ː"
STRESS = "ˈˌ"

_CACHE: dict[str, "NeuralVoice"] = {}


def models_dir() -> Path:
    """Where voice models live (override with ``DAFTDSP_MODELS``)."""
    d = os.environ.get("DAFTDSP_MODELS")
    if d:
        return Path(d)
    return Path(__file__).resolve().parent.parent / "models"


def model_path(name: str = DEFAULT_VOICE) -> Path:
    return models_dir() / f"{name}.onnx"


def ensure_model(name: str = DEFAULT_VOICE, download: bool = True) -> Path | None:
    """Return the local path to a voice model, downloading it if needed."""
    onnx = model_path(name)
    cfg = onnx.with_suffix(".onnx.json")
    if onnx.exists() and cfg.exists():
        return onnx
    if not download:
        return None
    onnx.parent.mkdir(parents=True, exist_ok=True)
    try:
        for url, dest in ((f"{_HF}/{name}.onnx.json", cfg),
                          (f"{_HF}/{name}.onnx", onnx)):
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            tmp.replace(dest)
        return onnx
    except Exception:
        for p in (onnx, cfg):
            part = p.with_suffix(p.suffix + ".part")
            if part.exists():
                part.unlink()
        return None


@dataclass
class Phone:
    """One phoneme with its span in samples."""
    symbol: str
    start: int
    end: int

    @property
    def is_vowel(self) -> bool:
        return any(c in VOWELS for c in self.symbol)


@dataclass
class Utterance:
    """Synthesised speech plus its phoneme segmentation."""
    audio: np.ndarray          # float32 mono
    sr: int
    phones: list[Phone]

    def vowel_groups(self) -> list[tuple[int, int]]:
        """Contiguous vowel runs (diphthongs stay one group) as sample spans."""
        groups, cur = [], None
        for p in self.phones:
            if p.is_vowel or (p.symbol == LENGTH and cur is not None):
                cur = [p.start, p.end] if cur is None else [cur[0], p.end]
            elif cur is not None:
                groups.append((cur[0], cur[1]))
                cur = None
        if cur is not None:
            groups.append((cur[0], cur[1]))
        return groups


class NeuralVoice:
    """Thin wrapper over a loaded Piper model."""

    def __init__(self, name: str = DEFAULT_VOICE):
        from piper import PiperVoice  # imported lazily: optional dependency

        path = ensure_model(name)
        if path is None:
            raise RuntimeError(f"voice model {name!r} unavailable")
        self.name = name
        self._voice = PiperVoice.load(str(path),
                                      config_path=str(path.with_suffix(".onnx.json")),
                                      include_alignments=True)
        self.sr = int(self._voice.config.sample_rate)

    def speak(self, text: str) -> Utterance:
        chunks = list(self._voice.synthesize(text, include_alignments=True))
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        phones, pos = [], 0
        for c in chunks:
            for al in (c.phoneme_alignments or []):
                n = int(al.num_samples)
                sym = al.phoneme
                if sym not in STRESS:           # stress marks carry no audio span
                    phones.append(Phone(sym, pos, pos + n))
                pos += n
        return Utterance(audio, self.sr, phones)


def get_voice(name: str = DEFAULT_VOICE) -> NeuralVoice | None:
    """Load (and cache) a neural voice, or ``None`` if unavailable."""
    if name in _CACHE:
        return _CACHE[name]
    try:
        v = NeuralVoice(name)
    except Exception:
        return None
    _CACHE[name] = v
    return v


def available(name: str = DEFAULT_VOICE) -> bool:
    return get_voice(name) is not None


def speak(text: str, name: str = DEFAULT_VOICE, sr: int | None = None) -> Utterance:
    """Synthesise ``text``.  Falls back to espeak (lower quality, phoneme spans
    estimated) when no neural model is available."""
    v = get_voice(name)
    if v is not None:
        utt = v.speak(text)
    else:
        utt = _espeak_fallback(text)
    if sr and sr != utt.sr:
        from .util import resample
        ratio = sr / utt.sr
        utt = Utterance(resample(utt.audio, utt.sr, sr), sr,
                        [Phone(p.symbol, int(p.start * ratio), int(p.end * ratio))
                         for p in utt.phones])
    return utt


def _espeak_fallback(text: str) -> Utterance:
    """espeak audio with vowel spans estimated from voiced energy."""
    from . import pitch as _pitch
    from . import tts, util
    from .biquad import one_pole_lp_fft

    sr = 22050
    audio = util.normalize_peak(tts.text_to_vocal(text, sr, voice="en+f4",
                                                  pitch=64, wpm=150), 0.95)
    track = _pitch.track_pitch(audio, sr, max_f0=1000.0)
    _, voiced = track.to_per_sample(audio.size)
    e = one_pole_lp_fft(np.abs(audio) * voiced.astype(np.float64), sr, 24.0)
    thr = 0.35 * (float(e.max()) or 1.0)
    phones, run = [], None
    for i in range(audio.size):
        hot = e[i] > thr and voiced[i]
        if hot and run is None:
            run = i
        elif not hot and run is not None:
            phones.append(Phone("a", run, i))       # treat the run as a vowel
            run = None
    if run is not None:
        phones.append(Phone("a", run, audio.size))
    return Utterance(audio, sr, phones)
