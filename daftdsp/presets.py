"""
A handful of ready-made parameter presets covering the main Daft-Punk vocal
flavours (pure vocoder, auto-tune + talkbox, monotone robot, etc.).  Each value
is an override applied on top of :class:`~daftdsp.engine.EngineParams` defaults.
"""

PRESETS = {
    "Around The World": {
        "scale": "minor", "key_root": 9, "chord_root": "A3", "chord_quality": "min7",
        "retune": 1.0, "talkbox_amount": 0.4, "dry_voice_mix": 0.0,
        "formant_shift": 1.0, "sat_drive": 2.0, "phaser_depth": 0.5,
        "sc_amount": 0.85, "vowel": "a",
    },
    "Robot Anthem (One More Time)": {
        "scale": "major", "key_root": 0, "chord_root": "C3", "chord_quality": "maj7",
        "retune": 1.0, "retune_time_ms": 1.0, "talkbox_amount": 0.7,
        "dry_voice_mix": 0.55, "vocoder_mix": 0.8, "formant_shift": 1.05,
        "sat_drive": 3.0, "eq_shelf_db": 6.0, "vowel": "a", "vowel2": "e",
        "morph_rate": 0.0,
    },
    "We Are The Robots": {
        "scale": "minor", "key_root": 2, "chord_root": "D2", "chord_quality": "minor",
        "tts_pitch": 20, "wpm": 130, "retune": 1.0, "talkbox_amount": 0.35,
        "formant_shift": 0.9, "sat_drive": 3.5, "sat_mix": 0.9, "sibilance": 0.3,
        "phaser_depth": 0.35, "vowel": "o", "width": 1.0,
    },
    "Harder Better (Chop)": {
        "scale": "minor", "key_root": 7, "chord_root": "G3", "chord_quality": "min7",
        "retune": 1.0, "talkbox_amount": 0.4, "sc_amount": 1.0, "sc_ratio": 10.0,
        "sc_release_ms": 120.0, "kick_bpm": 123.0, "sat_drive": 2.5,
        "phaser_depth": 0.4, "vowel": "a",
    },
    "Talkbox Sweep": {
        "scale": "pentatonic_minor", "key_root": 9, "chord_root": "A3",
        "chord_quality": "min9", "retune": 1.0, "talkbox_amount": 0.85,
        "vowel": "a", "vowel2": "i", "morph_rate": 2.5, "dry_voice_mix": 0.4,
        "formant_resonance": 10.0, "formant_gain_db": 11.0, "phaser_depth": 0.6,
    },
    "Deep Space Choir": {
        "scale": "minor", "key_root": 4, "chord_root": "E2", "chord_quality": "min7",
        "octave_layer": True, "sub_level": 0.7, "detune_cents": 22.0,
        "detune_voices": 5, "talkbox_amount": 0.5, "formant_shift": 1.1,
        "phaser_rate": 0.2, "phaser_depth": 0.7, "phaser_feedback": 0.5,
        "haas_ms": 24.0, "width": 1.4, "vowel": "o", "vowel2": "u",
        "morph_rate": 0.6,
    },
}
