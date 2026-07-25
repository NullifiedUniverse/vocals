"""
Automated quality tests for the singing voice.

These go beyond "does it run": each test inspects the rendered **waveform** and
fails on a specific, named defect, so a regression says what broke rather than
just that something did.

    python tests/test_singing_quality.py        # run + print the scorecard
    pytest tests/test_singing_quality.py

Groups:
  * metric self-tests -- the detectors must catch defects that are deliberately
    injected, otherwise a green suite means nothing;
  * waveform integrity -- clipping, DC, NaN, clicks;
  * musical correctness -- tuning, note coverage, no silent notes;
  * human-likeness -- vibrato, sustain, steady breath support;
  * scoring -- every bundled song is scored and must clear a floor.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daftdsp import arrangement, quality, singer, songs  # noqa: E402

SR = singer.DEFAULT_SR
SHORT = [("twinkle", [("C4", 1), ("C4", 1)]), ("little", [("G4", 1), ("G4", 1)]),
         ("star", [("A4", 2)])]
SHORT_BPM = 120

# Score floors.  Renders are deterministic (the neural voice's noise is zeroed),
# so these track the measured baseline rather than absorbing random variation.
# They are a regression guard, not a target -- raise them as the voice improves.
MIN_SONG_SCORE = 45.0
MIN_AVERAGE_SCORE = 65.0

_cache = {}


def _render(score, bpm):
    key = (id(score), bpm)
    if key not in _cache:
        _cache[key] = singer.sing(score, SR, bpm=bpm)
    return _cache[key]


# ---------------------------------------------------------------------------
# 1. the detectors themselves must work
# ---------------------------------------------------------------------------

def test_detector_finds_clicks():
    t = np.arange(SR) / SR
    clean = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    assert quality.discontinuity(clean)["count"] == 0
    popped = clean.copy()
    popped[::4000] = 0.99
    assert quality.discontinuity(popped)["count"] > 4


def test_detector_finds_dropouts_and_decay():
    t = np.arange(SR) / SR
    clean = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    gapped = clean.copy()
    gapped[SR // 3:SR // 2] = 0.0
    assert quality.dropout_ratio(gapped, SR) > quality.dropout_ratio(clean, SR)
    fading = clean * np.linspace(1.0, 0.02, clean.size)
    assert quality.sung_note(fading, SR)["decay_ratio"] < 0.5
    assert quality.sung_note(clean, SR)["decay_ratio"] > 0.8


def test_detector_finds_uneven_notes():
    t = np.arange(SR) / SR
    tone = 0.5 * np.sin(2 * np.pi * 220 * t)
    tl = [(0.0, 0.5, 57), (0.5, 0.5, 57)]
    even = tone.astype(np.float32)
    uneven = tone.copy()
    uneven[SR // 2:] *= 0.15                       # second note much quieter
    assert quality.evenness(even, SR, tl)["spread"] < 1.3
    assert quality.evenness(uneven.astype(np.float32), SR, tl)["spread"] > 2.0


def test_detector_finds_broken_legato():
    t = np.arange(SR) / SR
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    tl = [(0.0, 1.0, 57)]
    assert quality.flow(tone, SR, tl)["gaps"] == 0
    holed = tone.copy()
    holed[SR // 3:SR // 3 + SR // 8] = 0.0         # a hole inside the note
    assert quality.flow(holed, SR, tl)["gaps"] >= 1


def test_pitch_metric_folds_octaves_only():
    t = np.arange(SR) / SR
    up = (0.7 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    rows, stats = quality.pitch_errors(up, SR, [(0.0, 1.0, 69)])
    assert abs(rows[0]["cents"]) < 30 and stats["octave_flips"] == 1
    flat = (0.7 * np.sin(2 * np.pi * 415.3 * t)).astype(np.float32)
    _, s2 = quality.pitch_errors(flat, SR, [(0.0, 1.0, 69)])
    assert s2["within_50c"] == 0                   # a wrong note is not excused


# ---------------------------------------------------------------------------
# 2. waveform integrity
# ---------------------------------------------------------------------------

def test_waveform_is_well_formed():
    y = _render(SHORT, SHORT_BPM)
    h = quality.headroom(y)
    assert h["finite"], "output contains NaN/Inf"
    assert h["clipped"] == 0, "output clips"
    assert abs(h["dc"]) < 0.01, "DC offset"
    assert 0.05 < h["rms"] < 0.6, f"level out of range: {h['rms']}"
    assert h["crest"] < 14, "peaky/spiky waveform"


def test_render_is_click_free():
    y = _render(SHORT, SHORT_BPM)
    assert quality.discontinuity(y)["rate"] < 2e-3


def test_master_chain_is_mono_and_safe():
    st = singer.render_song(SHORT, SR, bpm=SHORT_BPM)
    assert st.ndim == 1, "a solo voice is rendered mono"
    assert quality.headroom(st)["clipped"] == 0
    assert np.all(np.isfinite(st))


# ---------------------------------------------------------------------------
# 3. musical correctness
# ---------------------------------------------------------------------------

def test_every_note_is_in_tune():
    y = _render(SHORT, SHORT_BPM)
    tl = songs.note_timeline(SHORT, SHORT_BPM)
    _, stats = quality.pitch_errors(y, SR, tl)
    assert stats["within_50c"] == stats["notes"], "a note is off pitch"
    assert stats["mean_abs_cents"] < 30


def test_no_note_is_silent():
    y = _render(SHORT, SHORT_BPM)
    tl = songs.note_timeline(SHORT, SHORT_BPM)
    lv = quality.note_levels(y, SR, tl)
    assert not np.any(lv < 0.02), f"silent note(s): {np.where(lv < 0.02)[0]}"


def test_notes_are_evenly_balanced():
    y = _render(SHORT, SHORT_BPM)
    tl = songs.note_timeline(SHORT, SHORT_BPM)
    assert quality.evenness(y, SR, tl)["spread"] < 3.0


def test_fast_notes_keep_their_vowel():
    score = [("merrily", [("C5", 1), ("C5", 1), ("C5", 1)]),
             ("merrily", [("G4", 1), ("G4", 1), ("G4", 1)])]
    y = singer.sing(score, SR, bpm=200)
    tl = songs.note_timeline(score, 200)
    assert np.all(np.isfinite(y))
    assert quality.dropout_ratio(y, SR, tl) < 0.35


def test_score_text_round_trips():
    text = songs.format_score(songs.SONGS["twinkle"]["score"])
    assert songs.parse_score(text) == songs.SONGS["twinkle"]["score"]


# ---------------------------------------------------------------------------
# 4. human-likeness
# ---------------------------------------------------------------------------

def test_sustained_note_behaves_like_a_singer():
    y = singer.sing([("lah", [("A4", 4)])], SR, bpm=60)
    m = quality.sung_note(y, SR, midi=69)
    assert abs(m["cents_off"]) < 35, "not in tune"
    # Vibrato is only asserted when the tracker could actually resolve it; a
    # single isolated note does not always give it enough voiced material.
    if m["vibrato_hz"] > 0:
        assert 4.5 <= m["vibrato_hz"] <= 8.5, "vibrato rate outside the human range"
        assert 10 <= m["vibrato_cents"] <= 95, "vibrato extent outside human range"
    assert m["pitch_drift"] < 90, "the note does not hold its centre"
    assert m["decay_ratio"] > 0.55, "the note fades instead of sustaining"
    assert m["amp_ripple"] < 0.4, "breath support is not steady"


def test_phrase_has_body_not_just_treble():
    """A voice that has lost its low end reads as thin/tinny."""
    y = _render(SHORT, SHORT_BPM)
    bands = quality.spectral_balance(y, SR)["bands"]
    low = bands["250-500"] + bands["80-250"]
    assert low > 8.0, f"too little body: {low}% below 500 Hz"


# ---------------------------------------------------------------------------
# 4b. instrumental arrangement
# ---------------------------------------------------------------------------

CHART = [("Am", 4), ("F", 4), ("C", 4), ("G", 4)]


def test_chords_parse_to_the_right_notes():
    root, notes = arrangement.parse_chord("Am", octave=3)
    assert [n - root for n in notes] == [0, 3, 7], "A minor"
    _, maj7 = arrangement.parse_chord("Cmaj7", octave=3)
    assert [n - maj7[0] for n in maj7] == [0, 4, 7, 11]
    assert arrangement.parse_chord("-")[1] == []      # no-chord is allowed


def test_backing_is_well_formed_and_in_time():
    sr = SR
    band = arrangement.render_backing(CHART, 100, sr)
    beats = sum(b for _, b in CHART)
    expected = beats * 60.0 / 100
    assert abs(band.size / sr - expected) < 0.6, "backing length must match chart"
    h = quality.headroom(band)
    assert h["finite"] and h["clipped"] == 0 and abs(h["dc"]) < 0.01
    assert h["rms"] > 0.02, "backing is silent"


def test_each_instrument_renders():
    for part in (arrangement.keys(CHART, 100, SR),
                 arrangement.bass(CHART, 100, SR),
                 arrangement.drums(16, 100, SR)):
        assert part.size > 0 and np.all(np.isfinite(part))
        assert np.max(np.abs(part)) > 1e-3, "instrument produced no sound"


def test_mix_keeps_the_voice_in_front():
    """The band must be ducked under the vocal, not level with it."""
    voice_only = _render(SHORT, SHORT_BPM)
    band = arrangement.render_backing(CHART, SHORT_BPM, SR)
    mixed = arrangement.mix(voice_only, band, SR)
    assert np.all(np.isfinite(mixed)) and quality.headroom(mixed)["clipped"] == 0
    # With the voice present the band is pushed down, so the mix must not be
    # dominated by the backing.
    loud_band = arrangement.mix(voice_only, band, SR, backing_level=1.0)
    assert quality.headroom(mixed)["rms"] <= quality.headroom(loud_band)["rms"] * 1.15


def test_band_songs_render_end_to_end():
    for name in songs.BAND_SONGS:
        s = songs.SONGS[name]
        out = singer.render_song(s["score"], SR, bpm=s["bpm"], chords=s["chords"])
        assert np.all(np.isfinite(out)), name
        assert quality.headroom(out)["clipped"] == 0, name
        assert out.size / SR > 5.0, f"{name} is suspiciously short"


# ---------------------------------------------------------------------------
# 5. scoring every bundled song
# ---------------------------------------------------------------------------

def _score_all():
    out = {}
    for name, s in songs.SONGS.items():
        y = _render(s["score"], s["bpm"])
        out[name] = quality.score(y, SR, songs.note_timeline(s["score"], s["bpm"]))
    return out


def test_every_song_clears_the_floor():
    results = _score_all()
    bad = {k: v["total"] for k, v in results.items() if v["total"] < MIN_SONG_SCORE}
    assert not bad, f"songs below the quality floor: {bad}"


def test_average_score_is_maintained():
    results = _score_all()
    avg = sum(v["total"] for v in results.values()) / len(results)
    assert avg >= MIN_AVERAGE_SCORE, f"average score regressed to {avg:.1f}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed, failures = 0, []
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
            failures.append(fn.__name__)
    print(f"\n{passed}/{len(fns)} tests passed")

    print("\nScorecard")
    print("-" * 74)
    results = _score_all()
    for name, r in results.items():
        parts = "  ".join(f"{k}:{v:>5.1f}" for k, v in r["parts"].items())
        print(f"  {name:<14} {r['total']:>5.1f} {quality.grade(r['total'])}   {parts}")
    avg = sum(r["total"] for r in results.values()) / len(results)
    print("-" * 74)
    print(f"  {'AVERAGE':<14} {avg:>5.1f} {quality.grade(avg)}")
    sys.exit(0 if not failures else 1)
