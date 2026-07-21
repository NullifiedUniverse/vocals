"""
Songs for the singing synth, as ``(syllable, note, beats)`` scores.

Syllables are written in **espeak phoneme notation** (fed to espeak as
``[[...]]``) so each note is pronounced correctly out of word context.  Get the
phonemes for any text with ``espeak-ng -q -x "your words"``.

Melodies are public domain (folk / classical) or original.  ``None`` note = rest.
Each song sets ``phoneme: True`` so the renderer uses phoneme mode.
"""

# Solfege scale, up and down -- cleanest pitch/tone test.
SCALE = {
    "bpm": 128, "voice": "en+f4", "base_pitch": 62, "phoneme": True,
    "score": [
        ("d'oU", "C4", 1), ("r'eI", "D4", 1), ("m'i:", "E4", 1), ("f'A:", "F4", 1),
        ("s'oU", "G4", 1), ("l'A:", "A4", 1), ("t'i:", "B4", 1), ("d'oU", "C5", 2),
        ("d'oU", "C5", 1), ("t'i:", "B4", 1), ("l'A:", "A4", 1), ("s'oU", "G4", 1),
        ("f'A:", "F4", 1), ("m'i:", "E4", 1), ("r'eI", "D4", 1), ("d'oU", "C4", 2),
    ],
}

# "Twinkle Twinkle Little Star" (traditional, public domain).
TWINKLE = {
    "bpm": 108, "voice": "en+f4", "base_pitch": 62, "phoneme": True,
    "score": [
        ("tw'IN", "C4", 1), ("k@L", "C4", 1), ("tw'IN", "G4", 1), ("k@L", "G4", 1),
        ("l'It", "A4", 1), ("t@L", "A4", 1), ("st'A@", "G4", 2),
        ("h'aU", "F4", 1), ("aI", "F4", 1), ("w'Vn", "E4", 1), ("d3", "E4", 1),
        ("w0t", "D4", 1), ("ju:", "D4", 1), ("A@", "C4", 2),
        ("'Vp", "G4", 1), ("@", "G4", 1), ("b'Vv", "F4", 1), ("D@", "F4", 1),
        ("w'3:ld", "E4", 1), ("s'oU", "E4", 1), ("h'aI", "D4", 2),
        ("l'aIk", "G4", 1), ("@", "G4", 1), ("d'aI", "F4", 1), ("m@nd", "F4", 1),
        ("In", "E4", 1), ("D@", "E4", 1), ("sk'aI", "D4", 2),
        ("tw'IN", "C4", 1), ("k@L", "C4", 1), ("tw'IN", "G4", 1), ("k@L", "G4", 1),
        ("l'It", "A4", 1), ("t@L", "A4", 1), ("st'A@", "G4", 2),
        ("h'aU", "F4", 1), ("aI", "F4", 1), ("w'Vn", "E4", 1), ("d3", "E4", 1),
        ("w0t", "D4", 1), ("ju:", "D4", 1), ("A@", "C4", 2),
    ],
}

# "Ode to Joy" (Beethoven, public domain), sung on "la".
ODE = {
    "bpm": 118, "voice": "en+f4", "base_pitch": 62, "phoneme": True,
    "score": [
        ("l'A:", "E4", 1), ("l'A:", "E4", 1), ("l'A:", "F4", 1), ("l'A:", "G4", 1),
        ("l'A:", "G4", 1), ("l'A:", "F4", 1), ("l'A:", "E4", 1), ("l'A:", "D4", 1),
        ("l'A:", "C4", 1), ("l'A:", "C4", 1), ("l'A:", "D4", 1), ("l'A:", "E4", 1),
        ("l'A:", "E4", 1), ("l'A:", "D4", 1), ("l'A:", "D4", 2),
        ("l'A:", "E4", 1), ("l'A:", "E4", 1), ("l'A:", "F4", 1), ("l'A:", "G4", 1),
        ("l'A:", "G4", 1), ("l'A:", "F4", 1), ("l'A:", "E4", 1), ("l'A:", "D4", 1),
        ("l'A:", "C4", 1), ("l'A:", "C4", 1), ("l'A:", "D4", 1), ("l'A:", "E4", 1),
        ("l'A:", "D4", 1), ("l'A:", "C4", 1), ("l'A:", "C4", 2),
    ],
}

# Original tune + lyrics -- a little digital-diva song.
DIGITAL_HEART = {
    "bpm": 96, "voice": "en+f4", "base_pitch": 62, "phoneme": True,
    "score": [
        ("aI", "C4", 1), ("am", "E4", 1), ("@", "G4", 1), ("v'OIs", "G4", 1),
        ("0v", "A4", 1), ("l'aIt", "G4", 2), ("rest", None, 1),
        ("s'IN", "E4", 1), ("IN", "F4", 1), ("Tr'u:", "E4", 1), ("D@", "D4", 1),
        ("n'aIt", "C4", 2), ("rest", None, 1),
        ("@", "G4", 1), ("h'A@t", "A4", 1), ("0v", "G4", 1), ("k'oUd", "E4", 1),
        ("and", "F4", 1), ("dr'i:mz", "G4", 2), ("rest", None, 1),
        ("f3", "F4", 1), ("Ev", "E4", 1), ("3", "D4", 1), ("aI", "D4", 1),
        ("w'Il", "F4", 1), ("S'aIn", "E4", 2), ("rest", None, 1),
        ("aI", "C4", 1), ("am", "E4", 1), ("@", "G4", 1), ("v'OIs", "A4", 1),
        ("0v", "B4", 1), ("l'aIt", "C5", 3),
    ],
}

SONGS = {
    "scale": SCALE,
    "twinkle": TWINKLE,
    "ode_to_joy": ODE,
    "digital_heart": DIGITAL_HEART,
}


# ---------------------------------------------------------------------------
# Word-based scores for the Aria singer (daftdsp/singer.py).
# Items are ("rest", beats) or (word, [(note, beats), ...]) with one note per
# syllable.  espeak pronounces whole words, so these are plain spellings.
# ---------------------------------------------------------------------------

ARIA_SCALE = {
    "bpm": 120, "voice": "en+f4", "base_pitch": 64,
    "score": [
        ("doe", [("C4", 1)]), ("ray", [("D4", 1)]), ("me", [("E4", 1)]),
        ("fah", [("F4", 1)]), ("soul", [("G4", 1)]), ("la", [("A4", 1)]),
        ("tea", [("B4", 1)]), ("doe", [("C5", 2)]), ("rest", 1),
        ("doe", [("C5", 1)]), ("tea", [("B4", 1)]), ("la", [("A4", 1)]),
        ("soul", [("G4", 1)]), ("fah", [("F4", 1)]), ("me", [("E4", 1)]),
        ("ray", [("D4", 1)]), ("doe", [("C4", 2)]),
    ],
}

ARIA_TWINKLE = {
    "bpm": 108, "voice": "en+f4", "base_pitch": 64,
    "score": [
        ("twinkle", [("C4", 1), ("C4", 1)]), ("twinkle", [("G4", 1), ("G4", 1)]),
        ("little", [("A4", 1), ("A4", 1)]), ("star", [("G4", 2)]), ("rest", 1),
        ("how", [("F4", 1)]), ("I", [("F4", 1)]),
        ("wonder", [("E4", 1), ("E4", 1)]), ("what", [("D4", 1)]),
        ("you", [("D4", 1)]), ("are", [("C4", 2)]), ("rest", 1),
        ("up", [("G4", 1)]), ("above", [("G4", 1), ("F4", 1)]), ("the", [("F4", 1)]),
        ("world", [("E4", 1)]), ("so", [("E4", 1)]), ("high", [("D4", 2)]), ("rest", 1),
        ("like", [("G4", 1)]), ("a", [("G4", 1)]), ("diamond", [("F4", 1), ("F4", 1)]),
        ("in", [("E4", 1)]), ("the", [("E4", 1)]), ("sky", [("D4", 2)]), ("rest", 1),
        ("twinkle", [("C4", 1), ("C4", 1)]), ("twinkle", [("G4", 1), ("G4", 1)]),
        ("little", [("A4", 1), ("A4", 1)]), ("star", [("G4", 2)]),
    ],
}

ARIA_ODE = {
    "bpm": 118, "voice": "en+f4", "base_pitch": 64,
    "score": [
        ("lah", [("E4", 1)]), ("lah", [("E4", 1)]), ("lah", [("F4", 1)]),
        ("lah", [("G4", 1)]), ("lah", [("G4", 1)]), ("lah", [("F4", 1)]),
        ("lah", [("E4", 1)]), ("lah", [("D4", 1)]), ("lah", [("C4", 1)]),
        ("lah", [("C4", 1)]), ("lah", [("D4", 1)]), ("lah", [("E4", 1)]),
        ("lah", [("E4", 1)]), ("lah", [("D4", 1)]), ("lah", [("D4", 2)]), ("rest", 1),
        ("lah", [("E4", 1)]), ("lah", [("E4", 1)]), ("lah", [("F4", 1)]),
        ("lah", [("G4", 1)]), ("lah", [("G4", 1)]), ("lah", [("F4", 1)]),
        ("lah", [("E4", 1)]), ("lah", [("D4", 1)]), ("lah", [("C4", 1)]),
        ("lah", [("C4", 1)]), ("lah", [("D4", 1)]), ("lah", [("E4", 1)]),
        ("lah", [("D4", 1)]), ("lah", [("C4", 1)]), ("lah", [("C4", 2)]),
    ],
}

ARIA_DIGITAL_HEART = {
    "bpm": 96, "voice": "en+f4", "base_pitch": 64,
    "score": [
        ("I", [("C4", 1)]), ("am", [("E4", 1)]), ("a", [("G4", 1)]),
        ("voice", [("G4", 1)]), ("of", [("A4", 1)]), ("light", [("G4", 2)]),
        ("rest", 1),
        ("singing", [("E4", 1), ("F4", 1)]), ("through", [("E4", 1)]),
        ("the", [("D4", 1)]), ("night", [("C4", 2)]), ("rest", 1),
        ("a", [("G4", 1)]), ("heart", [("A4", 1)]), ("of", [("G4", 1)]),
        ("code", [("E4", 1)]), ("and", [("F4", 1)]), ("dreams", [("G4", 2)]),
        ("rest", 1),
        ("forever", [("F4", 1), ("E4", 1), ("D4", 1)]), ("I", [("D4", 1)]),
        ("will", [("F4", 1)]), ("shine", [("E4", 2)]), ("rest", 1),
        ("I", [("C4", 1)]), ("am", [("E4", 1)]), ("a", [("G4", 1)]),
        ("voice", [("A4", 1)]), ("of", [("B4", 1)]), ("light", [("C5", 3)]),
    ],
}

ARIA_SONGS = {
    "scale": ARIA_SCALE,
    "twinkle": ARIA_TWINKLE,
    "ode_to_joy": ARIA_ODE,
    "digital_heart": ARIA_DIGITAL_HEART,
}
