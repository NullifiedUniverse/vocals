"""
daftdsp -- from-scratch vocal DSP: a Daft-Punk-style robot voice and the Aria
singing voice, sharing one hand-written DSP core.

All filters, pitch trackers and synthesis stages are implemented with plain numpy
array math (no scipy / librosa / high-level DSP libraries); ``numpy.fft`` is used
only as a math primitive.  See DESIGN.md for the architecture.

    from daftdsp import EngineParams, process          # robot voice
    from daftdsp import sing, render_song, SONGS       # singing voice
"""
from .engine import EngineParams, process
from .singer import render_song, sing
from .songs import SONGS, note_timeline, syllable_count

__all__ = [
    # robot voice
    "EngineParams", "process",
    # singing voice
    "sing", "render_song", "SONGS", "note_timeline", "syllable_count",
]
__version__ = "2.0.0"
