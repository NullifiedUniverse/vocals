"""
daftdsp -- a from-scratch Daft Punk style vocal DSP engine.

All filters, pitch trackers and synthesis stages are implemented by hand with
plain numpy array math (no scipy / librosa / high-level DSP libraries).
"""
from .engine import EngineParams, process

__all__ = ["EngineParams", "process"]
__version__ = "1.0.0"
