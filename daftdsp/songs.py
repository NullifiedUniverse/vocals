"""
Songs for the Aria singer (daftdsp/singer.py), as word-based scores.

Each item is ``("rest", beats)`` or ``(word, [(note, beats), ...])`` with one
note per syllable of the word.  espeak pronounces whole words, so these are plain
spellings.  Melodies are public domain (folk / classical) or original.
"""

SCALE = {
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

TWINKLE = {
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

ODE = {
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

MARY = {
    "bpm": 112, "voice": "en+f4", "base_pitch": 64,
    "score": [
        ("Mary", [("E4", 1), ("D4", 1)]), ("had", [("C4", 1)]), ("a", [("D4", 1)]),
        ("little", [("E4", 1), ("E4", 1)]), ("lamb", [("E4", 2)]), ("rest", 1),
        ("little", [("D4", 1), ("D4", 1)]), ("lamb", [("D4", 2)]), ("rest", 1),
        ("little", [("E4", 1), ("G4", 1)]), ("lamb", [("G4", 2)]), ("rest", 1),
        ("Mary", [("E4", 1), ("D4", 1)]), ("had", [("C4", 1)]), ("a", [("D4", 1)]),
        ("little", [("E4", 1), ("E4", 1)]), ("lamb", [("E4", 1)]), ("its", [("E4", 1)]),
        ("rest", 1),
        ("fleece", [("D4", 1)]), ("was", [("D4", 1)]), ("white", [("E4", 1)]),
        ("as", [("D4", 1)]), ("snow", [("C4", 3)]),
    ],
}

ROW = {
    "bpm": 100, "voice": "en+f4", "base_pitch": 64,
    "score": [
        ("row", [("C4", 1)]), ("row", [("C4", 1)]), ("row", [("C4", 1)]),
        ("your", [("D4", 1)]), ("boat", [("E4", 2)]), ("rest", 1),
        ("gently", [("E4", 1), ("D4", 1)]), ("down", [("E4", 1)]),
        ("the", [("F4", 1)]), ("stream", [("G4", 3)]), ("rest", 1),
        ("merrily", [("C5", 1), ("C5", 1), ("C5", 1)]),
        ("merrily", [("G4", 1), ("G4", 1), ("G4", 1)]),
        ("merrily", [("E4", 1), ("E4", 1), ("E4", 1)]),
        ("merrily", [("C4", 1), ("C4", 1), ("C4", 1)]), ("rest", 1),
        ("life", [("G4", 1)]), ("is", [("F4", 1)]), ("but", [("E4", 1)]),
        ("a", [("D4", 1)]), ("dream", [("C4", 3)]),
    ],
}

JINGLE = {
    "bpm": 120, "voice": "en+f4", "base_pitch": 64,
    "score": [
        ("jingle", [("E4", 1), ("E4", 1)]), ("bells", [("E4", 2)]),
        ("jingle", [("E4", 1), ("E4", 1)]), ("bells", [("E4", 2)]),
        ("jingle", [("E4", 1), ("G4", 1)]), ("all", [("C4", 1)]), ("the", [("D4", 1)]),
        ("way", [("E4", 3)]), ("rest", 1),
        ("oh", [("F4", 1)]), ("what", [("F4", 1)]), ("fun", [("F4", 1)]),
        ("it", [("F4", 1)]), ("is", [("F4", 1)]), ("to", [("E4", 1)]),
        ("ride", [("E4", 2)]), ("rest", 1),
        ("in", [("E4", 1)]), ("a", [("G4", 1)]), ("one", [("G4", 1)]),
        ("horse", [("F4", 1)]), ("open", [("D4", 1), ("D4", 1)]), ("sleigh", [("C4", 3)]),
    ],
}

# Original ballad -- emotive, sustained.
STARLIGHT = {
    "bpm": 82, "voice": "en+f4", "base_pitch": 64,
    "score": [
        ("when", [("E4", 1)]), ("the", [("G4", 1)]), ("stars", [("A4", 2)]),
        ("come", [("G4", 1)]), ("out", [("E4", 2)]), ("rest", 1),
        ("I", [("D4", 1)]), ("will", [("E4", 1)]), ("sing", [("G4", 2)]),
        ("for", [("E4", 1)]), ("you", [("D4", 3)]), ("rest", 1),
        ("through", [("C4", 1)]), ("the", [("D4", 1)]), ("night", [("E4", 2)]),
        ("we", [("G4", 1)]), ("will", [("A4", 1)]), ("glow", [("A4", 3)]), ("rest", 1),
        ("never", [("G4", 1), ("E4", 1)]), ("let", [("D4", 1)]), ("me", [("E4", 1)]),
        ("go", [("C4", 4)]),
    ],
}

DIGITAL_HEART = {
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

def parse_score(text):
    """Parse the human/editable score format into a score list.

    One entry per line -- a word followed by one ``NOTE:BEATS`` per syllable, or
    a rest::

        twinkle C4:1 C4:1
        star    G4:2
        rest    1

    ``BEATS`` may be omitted (defaults to 1).  Blank lines and ``#`` comments are
    ignored.  Raises ``ValueError`` with a line number on bad input.
    """
    score = []
    for lineno, raw in enumerate(str(text).splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        head = parts[0]
        if head.lower() == "rest":
            try:
                beats = float(parts[1]) if len(parts) > 1 else 1.0
            except ValueError:
                raise ValueError(f"line {lineno}: bad rest length {parts[1]!r}")
            score.append(("rest", beats))
            continue
        if len(parts) < 2:
            raise ValueError(f"line {lineno}: {head!r} has no notes")
        notes = []
        for tok in parts[1:]:
            name, _, beats = tok.partition(":")
            try:
                notes.append((name, float(beats) if beats else 1.0))
            except ValueError:
                raise ValueError(f"line {lineno}: bad note {tok!r}")
        score.append((head, notes))
    if not score:
        raise ValueError("empty score")
    return score


def format_score(score):
    """Render a score back into the editable text format."""
    out = []
    for item in score:
        if item[0] == "rest":
            out.append(f"rest {_num(item[1])}")
        else:
            notes = " ".join(f"{n}:{_num(b)}" for n, b in item[1])
            out.append(f"{item[0]} {notes}")
    return "\n".join(out)


def _num(v):
    return str(int(v)) if float(v).is_integer() else str(v)


def note_timeline(score, bpm):
    """Expand a score into ``[(start_s, dur_s, midi), ...]`` (rests skipped).

    Used to check a render against what it was supposed to sing.
    """
    from .util import note_to_midi

    beat = 60.0 / bpm
    out, t = [], 0.0
    for item in score:
        if item[0] == "rest":
            t += item[1] * beat
            continue
        for note, beats in item[1]:
            dur = beats * beat
            out.append((t, dur, note_to_midi(note)))
            t += dur
    return out


def syllable_count(score):
    """Total sung syllables (notes) in a score."""
    return sum(len(notes) for item, notes in
               ((i[0], i[1]) for i in score) if item != "rest")


SONGS = {
    "scale": SCALE,
    "twinkle": TWINKLE,
    "ode_to_joy": ODE,
    "mary": MARY,
    "row_row": ROW,
    "jingle_bells": JINGLE,
    "starlight": STARLIGHT,
    "digital_heart": DIGITAL_HEART,
}
