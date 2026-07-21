"""
Voice presets -- a wide palette of Daft-Punk-style robot characters, from lush
choirs to gritty rock bots.  Each entry is a set of overrides on top of
:class:`~daftdsp.engine.EngineParams` defaults.  The musical ones drive a chord
*progression* (``chord_prog``) so the robot sings a moving harmony.

Grouped loosely so you can tune within a family:
  choir / uplifting / sad / dreamy / aggressive / intimate / retro / funk.
"""

PRESETS = {
    # ---- Lush & uplifting -------------------------------------------------
    "Angelic Choir": {
        "scale": "major", "key_root": 0, "tts_pitch": 55, "wpm": 140,
        "chord_prog": ["C3:maj7", "G3:maj7", "A3:min7", "F3:maj7"],
        "vowel": "a", "vowel2": "o", "morph_rate": 0.5, "talkbox_amount": 0.3,
        "formant_shift": 1.08, "sat_drive": 1.2, "sat_mix": 0.3,
        "reverb_mix": 0.3, "reverb_size": 0.75, "reverb_damp": 0.4,
        "sibilance": 0.16, "eq_shelf_db": 3.0, "width": 1.3, "vibrato_depth": 0.09,
    },
    "Uplifting Anthem": {
        "scale": "major", "key_root": 2, "tts_pitch": 45, "wpm": 150,
        "chord_prog": ["D3:major", "A2:major", "B2:min7", "G2:major"],
        "vowel": "a", "talkbox_amount": 0.32, "sat_drive": 1.6, "sat_mix": 0.4,
        "reverb_mix": 0.2, "sc_amount": 0.5, "kick_bpm": 123.0,
        "eq_shelf_db": 3.0, "width": 1.25, "vibrato_depth": 0.08,
    },

    # ---- Sad & tender -----------------------------------------------------
    "Melancholy Android": {
        "scale": "minor", "key_root": 9, "tts_pitch": 30, "wpm": 125,
        "chord_prog": ["A2:minor", "F2:maj7", "C3:maj7", "G2:major"],
        "vowel": "o", "vowel2": "u", "morph_rate": 0.4, "talkbox_amount": 0.4,
        "formant_shift": 0.95, "sat_drive": 1.2, "sat_mix": 0.3,
        "reverb_mix": 0.34, "reverb_size": 0.8, "reverb_damp": 0.6,
        "warmth_hz": 8500.0, "sibilance": 0.14, "width": 1.2, "vibrato_depth": 0.11,
    },
    "Lonely Transmission": {
        "scale": "minor", "key_root": 4, "tts_pitch": 38, "wpm": 120,
        "chord_prog": ["E3:min7", "C3:maj7"],
        "vowel": "u", "talkbox_amount": 0.45, "formant_shift": 1.0,
        "sat_drive": 1.1, "sat_mix": 0.25, "reverb_mix": 0.4, "reverb_size": 0.85,
        "reverb_damp": 0.55, "warmth_hz": 7500.0, "phaser_depth": 0.5,
        "width": 1.15, "vibrato_depth": 0.13, "dry_voice_mix": 0.2,
    },

    # ---- Dreamy & ethereal ------------------------------------------------
    "Dreamy Nebula": {
        "scale": "major", "key_root": 4, "tts_pitch": 50, "wpm": 135,
        "chord_prog": ["E3:maj7", "C3:add9"],
        "vowel": "o", "vowel2": "a", "morph_rate": 0.3, "talkbox_amount": 0.3,
        "detune_cents": 22.0, "detune_voices": 5, "reverb_mix": 0.42,
        "reverb_size": 0.9, "reverb_damp": 0.45, "phaser_rate": 0.18,
        "phaser_depth": 0.55, "width": 1.45, "haas_ms": 24.0, "vibrato_depth": 0.1,
    },
    "Cosmic Cathedral": {
        "scale": "major", "key_root": 2, "tts_pitch": 42, "wpm": 130,
        "chord_prog": ["D3:maj7", "A2:maj7", "G2:maj7"],
        "vowel": "a", "vowel2": "o", "morph_rate": 0.25, "talkbox_amount": 0.35,
        "octave_layer": True, "sub_level": 0.6, "reverb_mix": 0.45,
        "reverb_size": 0.95, "reverb_damp": 0.4, "width": 1.4, "vibrato_depth": 0.09,
    },

    # ---- Classic Daft Punk ------------------------------------------------
    "Neon Disco": {
        "scale": "minor", "key_root": 9, "tts_pitch": 40, "wpm": 150,
        "chord_prog": ["A3:min7", "D3:min7"],
        "vowel": "a", "talkbox_amount": 0.35, "sat_drive": 1.8, "sat_mix": 0.45,
        "sc_amount": 0.8, "kick_bpm": 120.0, "phaser_depth": 0.4,
        "reverb_mix": 0.16, "eq_shelf_db": 3.5, "width": 1.2,
    },
    "Robot Anthem (One More Time)": {
        "scale": "major", "key_root": 0, "tts_pitch": 45, "wpm": 145,
        "chord_prog": ["C3:maj7", "A2:min7", "F2:maj7", "G2:dom7"],
        "vowel": "a", "vowel2": "e", "morph_rate": 0.3, "talkbox_amount": 0.55,
        "dry_voice_mix": 0.4, "vocoder_mix": 0.8, "formant_shift": 1.05,
        "sat_drive": 1.8, "sat_mix": 0.4, "reverb_mix": 0.2, "eq_shelf_db": 4.0,
    },

    # ---- Deep & powerful --------------------------------------------------
    "Deep Bass Oracle": {
        "scale": "minor", "key_root": 4, "tts_pitch": 18, "wpm": 120,
        "chord_prog": ["E2:min7", "C2:maj7"],
        "vowel": "o", "talkbox_amount": 0.35, "octave_layer": True,
        "sub_level": 0.8, "formant_shift": 0.9, "band_hi": 8000.0,
        "warmth_hz": 7000.0, "reverb_mix": 0.3, "reverb_size": 0.85,
        "width": 1.1, "vibrato_depth": 0.06,
    },
    "Robot Rock Grit": {
        "scale": "minor", "key_root": 4, "tts_pitch": 30, "wpm": 155,
        "chord_prog": ["E2:power", "G2:power", "A2:power"],
        "vowel": "a", "talkbox_amount": 0.3, "sat_drive": 2.6, "sat_mix": 0.55,
        "sc_amount": 0.7, "kick_bpm": 128.0, "phaser_depth": 0.4,
        "reverb_mix": 0.12, "eq_shelf_db": 3.5, "sibilance": 0.22, "width": 1.15,
    },

    # ---- Intimate & retro / funk -----------------------------------------
    "Whisper Circuit": {
        "scale": "minor", "key_root": 5, "tts_pitch": 48, "wpm": 130,
        "chord_root": "F3", "chord_quality": "min9", "chord_prog": [],
        "vowel": "u", "talkbox_amount": 0.4, "sat_drive": 1.1, "sat_mix": 0.2,
        "sibilance": 0.3, "reverb_mix": 0.28, "reverb_size": 0.7,
        "enable_sidechain": False, "width": 1.1, "output_gain": 0.9,
        "vibrato_depth": 0.08,
    },
    "Vintage Vocoder": {
        "scale": "minor", "key_root": 7, "tts_pitch": 35, "wpm": 140,
        "chord_root": "G3", "chord_quality": "min7", "chord_prog": [],
        "n_bands": 20, "band_hi": 6500.0, "band_q": 5.0, "vowel": "a",
        "talkbox_amount": 0.3, "sat_drive": 1.6, "sat_mix": 0.4,
        "warmth_hz": 8000.0, "reverb_mix": 0.15, "width": 1.0, "haas_ms": 8.0,
    },
    "Funk Machine": {
        "scale": "dorian", "key_root": 4, "tts_pitch": 42, "wpm": 158,
        "chord_prog": ["E3:dom7", "A3:dom7", "D3:dom7", "E3:min7"],
        "vowel": "a", "vowel2": "e", "morph_rate": 1.2, "talkbox_amount": 0.4,
        "sat_drive": 1.7, "sat_mix": 0.4, "sc_amount": 0.85, "sc_ratio": 8.0,
        "sc_release_ms": 130.0, "kick_bpm": 112.0, "reverb_mix": 0.16, "width": 1.2,
    },
}
