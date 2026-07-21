"""
Self-contained test suite for the daftdsp engine.

Run with either:
    python tests/test_engine.py
    pytest tests/
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daftdsp import EngineParams, process, util  # noqa: E402
from daftdsp import biquad, effects, pitch, psola, singing, synth, vocoder  # noqa: E402

SR = 44100


def test_biquad_lowpass_attenuates_highs():
    t = np.arange(SR // 4) / SR
    hi = np.sin(2 * np.pi * 15000 * t)
    lo = np.sin(2 * np.pi * 200 * t)
    c = biquad.lowpass(500, SR)
    assert np.sqrt(np.mean(biquad.df2t(c, hi) ** 2)) < 0.05
    assert np.sqrt(np.mean(biquad.df2t(c, lo) ** 2)) > 0.5


def test_fft_filter_matches_time_domain():
    x = np.random.default_rng(0).standard_normal(8000).astype(np.float32)
    c = biquad.bandpass(1000, SR, 4.0)
    y_iir = biquad.df2t(c, x)
    y_fft = biquad.biquad_fft(c, x)
    # Ignore the first samples (transient / circular edge) and compare.
    a, b = y_iir[500:-500], y_fft[500:-500]
    corr = np.corrcoef(a, b)[0, 1]
    assert corr > 0.999, corr


def test_yin_detects_pitch():
    t = np.arange(SR // 2) / SR
    x = (0.7 * np.sin(2 * np.pi * 220 * t) +
         0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    tr = pitch.track_pitch(x, SR)
    med = float(np.median(tr.f0[tr.voiced]))
    assert abs(med - 220) < 3, med


def test_scale_quantization():
    # 452.9 Hz is the midpoint A4/A#4; just above snaps up, just below snaps down.
    assert abs(util.quantize_freq_to_scale(448, 0, "chromatic") - 440.0) < 0.1
    assert abs(util.quantize_freq_to_scale(460, 0, "chromatic") - 466.16) < 0.5


def test_psola_snaps_and_preserves_length():
    t = np.arange(SR // 2) / SR
    x = (0.7 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)  # ~D4 sharp
    tr = pitch.track_pitch(x, SR)
    out = psola.psola_correct(x, SR, tr, key_root=0, scale="chromatic",
                              retune=1.0, retune_time_ms=1.0)
    assert out.size == x.size
    assert np.all(np.isfinite(out))
    med = float(np.median(pitch.track_pitch(out, SR).f0[pitch.track_pitch(out, SR).voiced]))
    # D4=293.66, D#4=311.13 -> 300 should snap to one of them.
    assert min(abs(med - 293.66), abs(med - 311.13)) < 6, med


def test_melody_psola_sings_target():
    # A steady 200 Hz tone sung onto D4 (~294 Hz) via a melody target.
    t = np.arange(SR) / SR
    x = (0.6 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    tr = pitch.track_pitch(x, SR)
    tgt = psola.melody_target(tr, SR, x.size, [62])   # D4
    out = psola.psola_correct(x, SR, tr, target_hz=tgt, retune=1.0,
                              retune_time_ms=5.0)
    med = float(np.median(pitch.track_pitch(out, SR).f0[
        pitch.track_pitch(out, SR).voiced]))
    assert abs(med - 293.66) < 8, med


def test_vocoder_whiten_finite():
    rng = np.random.default_rng(5)
    mod = rng.standard_normal(6000).astype(np.float32)
    car = synth.render_carrier(6000, SR, [57, 60, 64])
    out = vocoder.vocode(mod, car, SR, whiten=0.8, sibilance=0.3)
    assert out.size == 6000 and np.all(np.isfinite(out))


def test_synth_polyphony_length_and_finite():
    car = synth.render_carrier(SR, SR, [57, 60, 64], detune_voices=3)
    assert car.size == SR
    assert np.all(np.isfinite(car))
    assert np.max(np.abs(car)) <= 1.5


def test_vocoder_fft_iir_agree():
    rng = np.random.default_rng(1)
    mod = rng.standard_normal(6000).astype(np.float32)
    car = synth.render_carrier(6000, SR, [57, 64])
    vf = vocoder.vocode(mod, car, SR, method="fft", sibilance=0.0)
    vi = vocoder.vocode(mod, car, SR, method="iir", sibilance=0.0)
    assert vf.size == vi.size == 6000
    assert np.corrcoef(vf, vi)[0, 1] > 0.9


def test_effects_finite():
    x = np.random.default_rng(2).standard_normal(4000).astype(np.float32) * 0.3
    assert np.all(np.isfinite(effects.saturate(x, 3.0)))
    assert np.all(np.isfinite(effects.phaser(x, SR)))
    kick = effects.make_kick(x.size, SR, 120)
    assert np.all(np.isfinite(effects.sidechain(x, kick, SR)))
    st = effects.stereoize(x, SR)
    assert st.shape == (x.size, 2)
    rv = effects.reverb(st, SR, mix=0.3, size=0.6, damp=0.5)
    assert rv.shape == st.shape and np.all(np.isfinite(rv))


def test_carrier_progression():
    prog = synth.parse_progression(["A3:min7", "F3:maj7", "C4:maj7"])
    assert len(prog) == 3 and all(isinstance(c, list) for c in prog)
    car = synth.render_carrier(SR, SR, prog, vibrato_depth=0.1)
    assert car.size == SR and np.all(np.isfinite(car))


def test_wav_roundtrip():
    x = np.random.default_rng(3).standard_normal(1000).astype(np.float32) * 0.5
    for f32 in (False, True):
        data = util.write_wav(x, SR, float32=f32)
        r, sr = util.read_wav(data)
        assert sr == SR and r.size == x.size


def test_params_from_dict_coercion():
    p = EngineParams.from_dict({"n_bands": "24", "enable_phaser": "false",
                                "sat_drive": 3.5, "unknown_key": 1})
    assert p.n_bands == 24 and p.enable_phaser is False and p.sat_drive == 3.5


def test_singing_note_pitch_and_duration():
    # Sing "la" on A4 for ~0.6 s; check pitch and length.
    a4 = 69
    note = singing.render_note("la", a4, 0.6, SR, voice="en+f4", base_pitch=62)
    assert abs(note.size / SR - 0.6) < 0.12
    tr = pitch.track_pitch(note, SR, max_f0=1000.0)
    voiced = tr.f0[tr.voiced]
    med = float(np.median(voiced)) if voiced.size else 0.0
    cents = 1200 * np.log2(med / 440.0) if med > 0 else 999
    assert abs(cents) < 70, (med, cents)


def test_singing_long_note_flat_and_intune():
    # A 1.4 s held note must stay in tune and NOT decay or re-trigger.
    note = singing.render_note("l'A:", 69, 1.4, SR, phoneme=True, vib_depth=0.0)
    tr = pitch.track_pitch(note, SR, max_f0=1000.0)
    voiced = tr.f0[tr.voiced]
    med = float(np.median(voiced)) if voiced.size else 0.0
    assert abs(1200 * np.log2(med / 440.0)) < 40           # in tune
    # Sustain body energy should be roughly flat (no big decay/tremolo).
    ch = int(0.05 * SR)
    rms = np.array([np.sqrt(np.mean(note[i:i + ch] ** 2))
                    for i in range(0, note.size - ch, ch)])
    body = rms[3:-3][rms[3:-3] > 1e-3]
    assert body.size and np.std(body) / np.mean(body) < 0.3   # flat sustain


def test_singing_song_renders():
    score = [("do", "C4", 1), ("mi", "E4", 1), ("sol", "G4", 2)]
    out = singing.render_song(score, sr=SR, bpm=140)
    assert out.ndim == 2 and out.shape[1] == 2 and np.all(np.isfinite(out))


def test_full_engine_stereo_finite():
    for ov in ({}, {"enable_vocoder": False}, {"enable_autotune": False},
               {"scale": "major", "vowel2": "o", "morph_rate": 2.0}):
        stereo, meta = process(EngineParams.from_dict({"text": "robot voice", **ov}))
        assert stereo.ndim == 2 and stereo.shape[1] == 2
        assert np.all(np.isfinite(stereo))
        assert np.max(np.abs(stereo)) <= 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
