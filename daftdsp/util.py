"""
Low-level utilities for the Daft Punk vocal DSP engine.

Contains: WAV I/O (16-bit PCM read + 16-bit / 32-bit float write), a from-scratch
band-limited resampler, musical note helpers, and small array helpers.

No DSP libraries are used; the WAV reader/writer is a hand-rolled RIFF chunk
walker and everything else is plain numpy array math.
"""
from __future__ import annotations

import io
import struct

import numpy as np

# ---------------------------------------------------------------------------
# Musical helpers
# ---------------------------------------------------------------------------

# Semitone offsets (within an octave) for common scales, relative to the key root.
SCALES = {
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],           # natural minor
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_to_midi(name: str) -> int:
    """'C4' / 'A#3' / 'Eb2' -> MIDI note number (C4 = 60)."""
    name = name.strip()
    accidental = 0
    idx = 1
    root = name[0].upper()
    if len(name) > 1 and name[1] in "#b":
        accidental = 1 if name[1] == "#" else -1
        idx = 2
    octave = int(name[idx:])
    semitone = NOTE_NAMES.index(root if root in NOTE_NAMES else root.upper())
    return 12 * (octave + 1) + semitone + accidental


def midi_to_freq(midi: float) -> float:
    """MIDI note number -> frequency in Hz (A4 = 69 = 440 Hz)."""
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def freq_to_midi(freq: float) -> float:
    """Frequency in Hz -> (fractional) MIDI note number."""
    if freq <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def quantize_freq_to_scale(freq: float, key_root: int, scale: str) -> float:
    """Snap a frequency to the nearest pitch in ``scale`` (rooted at ``key_root``).

    ``key_root`` is a pitch-class 0..11 (0 = C).  Returns the quantised frequency.
    """
    if freq <= 0:
        return 0.0
    degrees = SCALES.get(scale, SCALES["chromatic"])
    midi = freq_to_midi(freq)
    # Build candidate MIDI notes across nearby octaves and pick the closest.
    base = int(round(midi))
    best = base
    best_dist = 1e9
    for octave in range(-1, 2):
        anchor = 12 * ((base // 12) + octave)
        for d in degrees:
            cand = anchor + ((key_root + d) % 12) + 12 * ((key_root + d) // 12)
            dist = abs(midi - cand)
            if dist < best_dist:
                best_dist = dist
                best = cand
    return midi_to_freq(best)


# ---------------------------------------------------------------------------
# WAV I/O
# ---------------------------------------------------------------------------

def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Parse a WAV byte string -> (mono float32 in [-1, 1], sample_rate).

    Hand-rolled RIFF chunk walker: supports PCM (8/16/24/32-bit) and IEEE-float
    (32/64-bit), including WAVE_FORMAT_EXTENSIBLE.  Multi-channel is down-mixed.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")

    fmt_tag = n_channels = sr = bits = None
    audio = None
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        csize = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + csize]
        if cid == b"fmt ":
            fmt_tag, n_channels, sr, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if fmt_tag == 0xFFFE and len(body) >= 26:
                # EXTENSIBLE: the true format tag is the first 2 bytes of the
                # 16-byte SubFormat GUID.
                fmt_tag = struct.unpack("<H", body[24:26])[0]
        elif cid == b"data":
            audio = body
        pos += 8 + csize + (csize & 1)  # chunks are word-aligned

    if fmt_tag is None or audio is None:
        raise ValueError("missing fmt/data chunk")

    if fmt_tag == 1:
        if bits == 16:
            arr = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
        elif bits == 8:
            arr = (np.frombuffer(audio, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif bits == 32:
            arr = np.frombuffer(audio, dtype="<i4").astype(np.float32) / 2147483648.0
        elif bits == 24:
            raw = np.frombuffer(audio, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            val = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
            val = np.where(val & 0x800000, val - 0x1000000, val)
            arr = val.astype(np.float32) / 8388608.0
        else:
            raise ValueError(f"unsupported PCM bit depth: {bits}")
    elif fmt_tag == 3:  # IEEE float
        dt = "<f4" if bits == 32 else "<f8"
        arr = np.frombuffer(audio, dtype=dt).astype(np.float32)
    else:
        raise ValueError(f"unsupported WAV format tag: {fmt_tag}")

    if n_channels and n_channels > 1:
        arr = arr[:arr.size - (arr.size % n_channels)].reshape(-1, n_channels).mean(axis=1)
    return arr.astype(np.float32), sr


def write_wav(signal: np.ndarray, sr: int, float32: bool = False) -> bytes:
    """Encode a signal to a WAV byte string.

    ``signal`` is either mono ``(N,)`` or stereo ``(N, 2)``.  When ``float32`` is
    True the samples are written as IEEE-float (format 3); otherwise 16-bit PCM
    (format 1), which every browser can play through an ``<audio>`` element.
    """
    sig = np.asarray(signal, dtype=np.float32)
    if sig.ndim == 1:
        sig = sig[:, None]
    n_frames, n_channels = sig.shape

    if float32:
        audio_bytes = sig.astype("<f4").tobytes()
        bits = 32
        fmt = 3
    else:
        clipped = np.clip(sig, -1.0, 1.0)
        audio_bytes = (clipped * 32767.0).astype("<i2").tobytes()
        bits = 16
        fmt = 1

    block_align = n_channels * bits // 8
    byte_rate = sr * block_align
    out = io.BytesIO()
    out.write(b"RIFF")
    out.write(struct.pack("<I", 36 + len(audio_bytes)))
    out.write(b"WAVE")
    out.write(b"fmt ")
    out.write(struct.pack("<IHHIIHH", 16, fmt, n_channels, sr, byte_rate,
                          block_align, bits))
    out.write(b"data")
    out.write(struct.pack("<I", len(audio_bytes)))
    out.write(audio_bytes)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Resampling (band-limited-ish, from scratch)
# ---------------------------------------------------------------------------

def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample ``x`` from ``sr_in`` to ``sr_out`` using cubic (Catmull-Rom)
    interpolation.  When downsampling, a short moving-average pre-filter tames
    the worst of the aliasing.  This is deliberately dependency-free."""
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32)

    x = x.astype(np.float32)
    if sr_out < sr_in:
        # Anti-alias pre-filter: box filter whose width ~ downsample ratio.
        ratio = sr_in / sr_out
        width = max(1, int(round(ratio)))
        if width > 1:
            kernel = np.ones(width, dtype=np.float32) / width
            x = np.convolve(x, kernel, mode="same").astype(np.float32)

    n_out = int(round(x.size * sr_out / sr_in))
    # Positions in the input signal for each output sample.
    pos = np.arange(n_out, dtype=np.float64) * (sr_in / sr_out)
    i0 = np.floor(pos).astype(np.int64)
    frac = (pos - i0).astype(np.float32)

    def tap(offset):
        idx = np.clip(i0 + offset, 0, x.size - 1)
        return x[idx]

    p0, p1, p2, p3 = tap(-1), tap(0), tap(1), tap(2)
    # Catmull-Rom cubic interpolation.
    t = frac
    t2 = t * t
    t3 = t2 * t
    out = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Small array helpers
# ---------------------------------------------------------------------------

def normalize_peak(x: np.ndarray, peak: float = 0.97) -> np.ndarray:
    """Scale so the maximum absolute sample equals ``peak`` (no-op on silence)."""
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m < 1e-9:
        return x
    return (x * (peak / m)).astype(np.float32)


def normalize_percentile(x: np.ndarray, target: float = 0.85,
                         pct: float = 99.5) -> np.ndarray:
    """Scale so the ``pct``-th percentile of |x| hits ``target``.

    Unlike peak normalisation this ignores a handful of stray transients, so the
    perceived loudness is consistent from render to render and the signal isn't
    dragged down (and then noise-amplified) by one spike."""
    if x.size == 0:
        return x
    ref = float(np.percentile(np.abs(x), pct))
    if ref < 1e-9:
        return x
    return (x * (target / ref)).astype(np.float32)


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def soft_limit(x: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    """Gentle tanh brick-wall so nothing digitally clips at the very end."""
    return (np.tanh(x / ceiling) * ceiling).astype(np.float32)
