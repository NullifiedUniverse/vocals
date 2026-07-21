"""
Top-level Daft Punk vocal engine.

Wires the modules into the signal flow from the spec:

    text -> TTS voice ─► PSOLA auto-tune ─┐
                                          ▼
    chord -> poly synth ───────────► channel vocoder ─► formant/talkbox
                                          │
                                          ▼
    output ◄─ EQ/stereo ◄─ sidechain ◄─ phaser ◄─ tanh saturation

Every stage is individually toggleable and exposes runtime parameters via
:class:`EngineParams`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import numpy as np

from . import effects, formant, pitch, psola, synth, tts, util, vocoder


@dataclass
class EngineParams:
    sr: int = 44100

    # --- Voice / TTS ---
    text: str = "we are the robots"
    wpm: int = 150
    tts_pitch: int = 35
    voice: str = "en"

    # --- Auto-tune (PSOLA) ---
    enable_autotune: bool = True
    retune: float = 1.0
    retune_time_ms: float = 1.5
    key_root: int = 9           # 0=C ... 9=A
    scale: str = "minor"

    # --- Carrier synth ---
    chord_root: str = "A3"
    chord_quality: str = "min7"
    saw_level: float = 0.8
    pulse_level: float = 0.5
    detune_cents: float = 16.0
    detune_voices: int = 3
    pulse_width: float = 0.5
    pwm_rate: float = 0.5
    pwm_depth: float = 0.25
    octave_layer: bool = True
    sub_level: float = 0.5
    synth_level: float = 1.0

    # --- Vocoder ---
    enable_vocoder: bool = True
    n_bands: int = 32
    band_lo: float = 100.0
    band_hi: float = 10000.0
    band_q: float = 7.0
    voc_attack_ms: float = 3.0
    voc_release_ms: float = 18.0
    formant_shift: float = 1.0
    sibilance: float = 0.4
    vocoder_mix: float = 1.0
    dry_voice_mix: float = 0.0

    # --- Formant / talkbox ---
    enable_talkbox: bool = True
    vowel: str = "a"
    vowel2: str = ""
    morph_rate: float = 0.0
    formant_resonance: float = 7.0
    formant_gain_db: float = 8.0
    talkbox_amount: float = 0.45

    # --- Saturation ---
    enable_saturation: bool = True
    sat_drive: float = 2.0
    sat_mix: float = 0.8

    # --- Phaser ---
    enable_phaser: bool = True
    phaser_rate: float = 0.3
    phaser_depth: float = 0.5
    phaser_stages: int = 6
    phaser_feedback: float = 0.35

    # --- Sidechain ---
    enable_sidechain: bool = True
    sc_threshold_db: float = -24.0
    sc_ratio: float = 6.0
    sc_attack_ms: float = 5.0
    sc_release_ms: float = 180.0
    sc_amount: float = 0.8
    kick_bpm: float = 120.0

    # --- EQ / stereo ---
    enable_eq: bool = True
    eq_hp: float = 150.0
    eq_shelf_freq: float = 6000.0
    eq_shelf_db: float = 4.0
    haas_ms: float = 18.0
    width: float = 1.15
    output_gain: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "EngineParams":
        """Build params from a (partial) dict, ignoring unknown keys and
        coercing each value to its declared type."""
        valid = {f.name: f for f in fields(cls)}
        kwargs = {}
        for k, v in (d or {}).items():
            if k not in valid:
                continue
            typ = valid[k].type
            try:
                if typ == "bool" or typ is bool:
                    kwargs[k] = bool(v) if not isinstance(v, str) \
                        else v.lower() in ("1", "true", "yes", "on")
                elif typ == "int" or typ is int:
                    kwargs[k] = int(round(float(v)))
                elif typ == "float" or typ is float:
                    kwargs[k] = float(v)
                else:
                    kwargs[k] = v
            except (TypeError, ValueError):
                continue
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)


def process(params: EngineParams, vocal=None, chord_midi=None, kick=None):
    """Run the full engine.  Returns ``(stereo_float32 (N,2), meta)``.

    Optional overrides let a host inject a raw vocal PCM array, an explicit chord
    (list of MIDI notes) and/or an external kick trigger, per the I/O spec.
    """
    sr = int(params.sr)
    meta = {}

    # 1) Voice source (modulator).
    if vocal is None:
        vocal = tts.text_to_vocal(params.text, sr, wpm=params.wpm,
                                  pitch=params.tts_pitch, voice=params.voice)
    vocal = np.asarray(vocal, dtype=np.float32)
    if vocal.size < sr // 8:  # guarantee a minimum length
        vocal = np.pad(vocal, (0, sr // 8 - vocal.size))
    vocal = util.normalize_peak(vocal, 0.9)
    n = vocal.size
    meta["duration_s"] = round(n / sr, 3)

    # 2) Pitch tracking + PSOLA hard auto-tune.
    corrected = vocal
    if params.enable_autotune and params.retune > 0:
        track = pitch.track_pitch(vocal, sr)
        voiced = track.f0[track.voiced]
        meta["median_f0"] = round(float(np.median(voiced)), 1) if voiced.size else 0.0
        corrected = psola.psola_correct(
            vocal, sr, track, key_root=params.key_root, scale=params.scale,
            retune=params.retune, retune_time_ms=params.retune_time_ms)

    # 3) Carrier synth.
    if chord_midi is None:
        chord_midi = synth.parse_chord(params.chord_root, params.chord_quality)
    meta["chord_midi"] = list(chord_midi)
    carrier = synth.render_carrier(
        n, sr, chord_midi,
        saw_level=params.saw_level, pulse_level=params.pulse_level,
        detune_cents=params.detune_cents, detune_voices=params.detune_voices,
        pulse_width=params.pulse_width, pwm_rate=params.pwm_rate,
        pwm_depth=params.pwm_depth, octave_layer=params.octave_layer,
        sub_level=params.sub_level, level=params.synth_level)

    # 4) Channel vocoder (voice modulates synth).
    if params.enable_vocoder:
        voice = vocoder.vocode(
            corrected, carrier, sr, n_bands=params.n_bands, f_lo=params.band_lo,
            f_hi=params.band_hi, band_q=params.band_q,
            attack_ms=params.voc_attack_ms, release_ms=params.voc_release_ms,
            formant_shift=params.formant_shift, sibilance=params.sibilance)
    else:
        voice = util.normalize_peak(corrected, 0.9)

    # 5) Formant / talkbox colour.
    if params.enable_talkbox:
        voice = formant.talkbox(
            voice, sr, vowel=params.vowel, vowel2=params.vowel2,
            morph_rate=params.morph_rate, formant_shift=params.formant_shift,
            resonance=params.formant_resonance, gain_db=params.formant_gain_db,
            amount=params.talkbox_amount)

    # Blend in the dry auto-tuned voice (One-More-Time-style talkbox mix).
    voice = util.normalize_peak(voice, 0.9)
    if params.dry_voice_mix > 0:
        dry = util.normalize_peak(corrected, 0.9)
        voice = (params.vocoder_mix * voice + params.dry_voice_mix * dry)
        voice = util.normalize_peak(voice, 0.9)

    # 6) Saturation -> phaser.
    if params.enable_saturation:
        voice = effects.saturate(voice, drive=params.sat_drive, mix=params.sat_mix)
    if params.enable_phaser:
        voice = effects.phaser(
            voice, sr, rate=params.phaser_rate, depth=params.phaser_depth,
            stages=params.phaser_stages, feedback=params.phaser_feedback)

    # 7) Sidechain (mono), then EQ, then Haas stereo.
    if params.enable_sidechain and params.sc_amount > 0:
        if kick is None:
            kick = effects.make_kick(n, sr, bpm=params.kick_bpm)
        voice = effects.sidechain(
            voice, kick, sr, threshold_db=params.sc_threshold_db,
            ratio=params.sc_ratio, attack_ms=params.sc_attack_ms,
            release_ms=params.sc_release_ms, amount=params.sc_amount)

    if params.enable_eq:
        voice = effects.eq(voice, sr, hp_freq=params.eq_hp,
                           shelf_freq=params.eq_shelf_freq,
                           shelf_gain_db=params.eq_shelf_db)

    stereo = effects.stereoize(voice, sr, haas_ms=params.haas_ms,
                               width=params.width, level=params.output_gain)

    # Final safety: normalise then gentle limit.
    stereo = util.normalize_peak(stereo, 0.97)
    stereo = util.soft_limit(stereo, 0.99)
    meta["peak"] = round(float(np.max(np.abs(stereo))), 3)
    return stereo, meta
