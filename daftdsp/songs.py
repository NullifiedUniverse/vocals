"""
Songs for the singing synth, as ``(syllable, note, beats)`` scores.

Melodies are public domain (folk / classical) or original, so nothing here is a
copyrighted composition.  Syllables are spelled the way espeak reads them best.
``None`` note = a rest.
"""

# Solfege scale up and down -- the cleanest test of pitch accuracy.
SCALE = {
    "bpm": 130, "voice": "en+f4", "base_pitch": 62,
    "score": [
        ("do", "C4", 1), ("re", "D4", 1), ("mi", "E4", 1), ("fa", "F4", 1),
        ("sol", "G4", 1), ("la", "A4", 1), ("ti", "B4", 1), ("do", "C5", 2),
        ("do", "C5", 1), ("ti", "B4", 1), ("la", "A4", 1), ("sol", "G4", 1),
        ("fa", "F4", 1), ("mi", "E4", 1), ("re", "D4", 1), ("do", "C4", 2),
    ],
}

# "Twinkle Twinkle Little Star" (traditional, public domain).
TWINKLE = {
    "bpm": 112, "voice": "en+f4", "base_pitch": 62,
    "score": [
        ("twin", "C4", 1), ("kle", "C4", 1), ("twin", "G4", 1), ("kle", "G4", 1),
        ("lit", "A4", 1), ("tle", "A4", 1), ("star", "G4", 2),
        ("how", "F4", 1), ("I", "F4", 1), ("won", "E4", 1), ("der", "E4", 1),
        ("what", "D4", 1), ("you", "D4", 1), ("are", "C4", 2),
        ("up", "G4", 1), ("a", "G4", 1), ("bove", "F4", 1), ("the", "F4", 1),
        ("world", "E4", 1), ("so", "E4", 1), ("high", "D4", 2),
        ("like", "G4", 1), ("a", "G4", 1), ("dia", "F4", 1), ("mond", "F4", 1),
        ("in", "E4", 1), ("the", "E4", 1), ("sky", "D4", 2),
        ("twin", "C4", 1), ("kle", "C4", 1), ("twin", "G4", 1), ("kle", "G4", 1),
        ("lit", "A4", 1), ("tle", "A4", 1), ("star", "G4", 2),
        ("how", "F4", 1), ("I", "F4", 1), ("won", "E4", 1), ("der", "E4", 1),
        ("what", "D4", 1), ("you", "D4", 1), ("are", "C4", 2),
    ],
}

# "Ode to Joy" (Beethoven, public domain), sung on "la".
ODE = {
    "bpm": 120, "voice": "en+f4", "base_pitch": 62,
    "score": [
        ("la", "E4", 1), ("la", "E4", 1), ("la", "F4", 1), ("la", "G4", 1),
        ("la", "G4", 1), ("la", "F4", 1), ("la", "E4", 1), ("la", "D4", 1),
        ("la", "C4", 1), ("la", "C4", 1), ("la", "D4", 1), ("la", "E4", 1),
        ("la", "E4", 1), ("la", "D4", 1), ("la", "D4", 2),
        ("la", "E4", 1), ("la", "E4", 1), ("la", "F4", 1), ("la", "G4", 1),
        ("la", "G4", 1), ("la", "F4", 1), ("la", "E4", 1), ("la", "D4", 1),
        ("la", "C4", 1), ("la", "C4", 1), ("la", "D4", 1), ("la", "E4", 1),
        ("la", "D4", 1), ("la", "C4", 1), ("la", "C4", 2),
    ],
}

# Original tune + lyrics -- a little digital-diva song.
DIGITAL_HEART = {
    "bpm": 100, "voice": "en+f4", "base_pitch": 62,
    "score": [
        ("I", "C4", 1), ("am", "E4", 1), ("a", "G4", 1), ("voice", "G4", 1),
        ("of", "A4", 1), ("light", "G4", 2), ("rest", None, 1),
        ("sing", "E4", 1), ("ing", "F4", 1), ("through", "E4", 1),
        ("the", "D4", 1), ("night", "C4", 2), ("rest", None, 1),
        ("a", "G4", 1), ("heart", "A4", 1), ("of", "G4", 1), ("code", "E4", 1),
        ("and", "F4", 1), ("dreams", "G4", 2), ("rest", None, 1),
        ("for", "F4", 1), ("ev", "E4", 1), ("er", "D4", 1), ("I", "D4", 1),
        ("will", "F4", 1), ("shine", "E4", 2), ("rest", None, 1),
        ("I", "C4", 1), ("am", "E4", 1), ("a", "G4", 1), ("voice", "A4", 1),
        ("of", "B4", 1), ("light", "C5", 3),
    ],
}

SONGS = {
    "scale": SCALE,
    "twinkle": TWINKLE,
    "ode_to_joy": ODE,
    "digital_heart": DIGITAL_HEART,
}
