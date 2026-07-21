"""
Voice presets for the (revised, intelligibility-first) engine.

Each preset pairs a **melody** (notes placed on successive syllables, sung by the
formant-preserving PSOLA voice) with a matching **chord** or **progression** for
the carrier, so the clear lead voice and the robot layer agree harmonically.
``dry_voice_mix`` vs ``vocoder_mix`` sets how human vs robotic each one is;
``whiten`` keeps the vocoder layer articulate.

Melodies stay near the source pitch (~A3/220 Hz) so the pitch-shift is small and
the words stay crisp.
"""

PRESETS = {
    # ---- Clean & singing (Vocaloid-like) ---------------------------------
    "Vocaloid Diva": {
        "voice": "en+f4", "tts_pitch": 68, "wpm": 128, "scale": "major",
        "melody": ["C4", "E4", "G4", "E4", "F4", "D4", "E4", "C4"],
        "chord_prog": ["C3:maj7", "G2:major", "A2:min7", "F2:maj7"],
        "dry_voice_mix": 0.74, "vocoder_mix": 0.32, "whiten": 0.85,
        "sat_drive": 1.1, "sat_mix": 0.2, "reverb_mix": 0.16, "eq_shelf_db": 2.5,
        "vibrato_depth": 0.12, "width": 1.15,
    },
    "Crystal Android": {
        "voice": "en+f3", "tts_pitch": 64, "wpm": 130, "scale": "major",
        "melody": ["E4", "D4", "C4", "D4", "E4", "G4"],
        "chord_prog": ["C3:maj7", "A2:min7", "F2:maj7", "G2:dom7"],
        "dry_voice_mix": 0.68, "vocoder_mix": 0.4, "whiten": 0.8,
        "reverb_mix": 0.2, "vibrato_depth": 0.11, "width": 1.2,
    },

    # ---- Emotive lead ----------------------------------------------------
    "Melancholy Android": {
        "voice": "en+f3", "tts_pitch": 58, "wpm": 122, "scale": "minor",
        "melody": ["A3", "C4", "B3", "A3", "G3", "E3"],
        "chord_prog": ["A2:minor", "F2:maj7", "C3:maj7", "E2:min7"],
        "dry_voice_mix": 0.7, "vocoder_mix": 0.38, "whiten": 0.78,
        "reverb_mix": 0.24, "reverb_size": 0.8, "warmth_hz": 9500.0,
        "vibrato_depth": 0.14, "retune_time_ms": 40.0, "width": 1.15,
    },
    "Uplifting Anthem": {
        "voice": "en+f3", "tts_pitch": 62, "wpm": 138, "scale": "major",
        "melody": ["D4", "E4", "F#4", "A4", "F#4", "E4"],
        "chord_prog": ["D3:major", "A2:major", "B2:min7", "G2:major"],
        "dry_voice_mix": 0.62, "vocoder_mix": 0.5, "whiten": 0.75,
        "sc_amount": 0.5, "kick_bpm": 123.0, "reverb_mix": 0.16,
        "eq_shelf_db": 3.0, "vibrato_depth": 0.09, "width": 1.25,
    },

    # ---- Classic Daft Punk blends ----------------------------------------
    "Neon Disco": {
        "voice": "en+f3", "tts_pitch": 60, "wpm": 140, "scale": "minor",
        "melody": ["A3", "A3", "C4", "E4", "D4", "C4"],
        "chord_prog": ["A2:min7", "D3:min7"],
        "dry_voice_mix": 0.5, "vocoder_mix": 0.6, "whiten": 0.68,
        "sat_drive": 1.6, "sat_mix": 0.35, "sc_amount": 0.8, "kick_bpm": 120.0,
        "reverb_mix": 0.14, "eq_shelf_db": 3.0, "width": 1.2,
    },
    "One More Time": {
        "voice": "en+f3", "tts_pitch": 62, "wpm": 132, "scale": "major",
        "melody": ["G3", "C4", "E4", "D4", "C4", "G3"],
        "chord_prog": ["C3:maj7", "A2:min7", "F2:maj7", "G2:dom7"],
        "dry_voice_mix": 0.6, "vocoder_mix": 0.55, "whiten": 0.72,
        "enable_talkbox": True, "talkbox_amount": 0.28, "vowel": "a", "vowel2": "e",
        "morph_rate": 0.4, "sat_drive": 1.6, "sat_mix": 0.35, "reverb_mix": 0.16,
        "eq_shelf_db": 3.5,
    },

    # ---- Deep / heavy ----------------------------------------------------
    "Deep Bass Oracle": {
        "voice": "en+m3", "tts_pitch": 40, "wpm": 120, "scale": "minor",
        "melody": ["A2", "C3", "E3", "C3", "A2", "G2"],
        "chord_prog": ["A1:min7", "F1:maj7"],
        "dry_voice_mix": 0.6, "vocoder_mix": 0.5, "whiten": 0.7,
        "sub_level": 0.8, "band_hi": 8500.0, "warmth_hz": 8000.0,
        "reverb_mix": 0.22, "reverb_size": 0.85, "width": 1.1, "vibrato_depth": 0.07,
    },
    "Robot Rock Grit": {
        "voice": "en+m2", "tts_pitch": 48, "wpm": 150, "scale": "minor",
        "melody": ["E3", "E3", "G3", "E3", "A3", "G3"],
        "chord_prog": ["E2:power", "G2:power", "A2:power"],
        "dry_voice_mix": 0.5, "vocoder_mix": 0.55, "whiten": 0.6,
        "sat_drive": 2.2, "sat_mix": 0.45, "sc_amount": 0.6, "kick_bpm": 128.0,
        "enable_phaser": True, "phaser_depth": 0.35, "reverb_mix": 0.12,
        "eq_shelf_db": 3.0, "sibilance": 0.4, "width": 1.15,
    },

    # ---- Ethereal / intimate / retro -------------------------------------
    "Dreamy Nebula": {
        "voice": "en+f4", "tts_pitch": 66, "wpm": 126, "scale": "major",
        "melody": ["E4", "G4", "B4", "A4", "G4", "E4"],
        "chord_prog": ["E3:maj7", "C3:add9"],
        "dry_voice_mix": 0.58, "vocoder_mix": 0.48, "whiten": 0.78,
        "detune_cents": 22.0, "detune_voices": 5, "reverb_mix": 0.3,
        "reverb_size": 0.9, "enable_phaser": True, "phaser_rate": 0.18,
        "phaser_depth": 0.45, "width": 1.4, "haas_ms": 22.0, "vibrato_depth": 0.12,
    },
    "Whisper Circuit": {
        "voice": "en+f3", "tts_pitch": 60, "wpm": 126, "scale": "minor",
        "melody": ["F3", "A3", "C4", "A3", "G3", "F3"],
        "chord_root": "F3", "chord_quality": "min9", "chord_prog": [],
        "dry_voice_mix": 0.8, "vocoder_mix": 0.28, "whiten": 0.85,
        "sat_drive": 1.05, "sat_mix": 0.15, "sibilance": 0.42,
        "enable_sidechain": False, "reverb_mix": 0.22, "width": 1.1,
        "output_gain": 0.92, "vibrato_depth": 0.1,
    },
    "Vintage Vocoder": {
        "voice": "en+m3", "tts_pitch": 52, "wpm": 138, "scale": "minor",
        "melody": ["G3", "A3", "C4", "A3", "G3", "D3"],
        "chord_root": "G3", "chord_quality": "min7", "chord_prog": [],
        "dry_voice_mix": 0.4, "vocoder_mix": 0.66, "whiten": 0.5,
        "n_bands": 22, "band_hi": 6800.0, "sat_drive": 1.5, "sat_mix": 0.35,
        "warmth_hz": 8500.0, "reverb_mix": 0.14, "haas_ms": 8.0, "width": 1.0,
    },

    # ---- Monotone robot (no melody -> scale snap) -------------------------
    "Monotone Kraftwerk": {
        "voice": "en+m2", "tts_pitch": 30, "wpm": 128, "scale": "minor",
        "melody": [], "key_root": 9, "chord_root": "A2", "chord_quality": "min7",
        "chord_prog": [], "dry_voice_mix": 0.45, "vocoder_mix": 0.6,
        "whiten": 0.65, "formant_shift": 0.95, "sat_drive": 1.8, "sat_mix": 0.4,
        "warmth_hz": 8500.0, "reverb_mix": 0.12, "vibrato_depth": 0.0, "width": 1.05,
    },
}
