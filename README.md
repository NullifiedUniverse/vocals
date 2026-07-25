# 🎤 Aria — a lifelike singing voice

Give it words and a melody; it sings them.

A neural voice speaks each phrase, and the WORLD vocoder places that real human
timbre onto the tune. Every stage is modelled on what human singers measurably
do — vowels land on the beat, sustained notes hold steady breath support, vibrato
enters after the note settles — and each of those is asserted by a test.

```
 words + notes ─► [neural voice speaks the phrase, with phoneme timings]
               ─► [vowels aligned to beats, consonants placed before them]
               ─► [WORLD: hold the vowel, replace the pitch]
               ─► [sung dynamics: steady sustain, vibrato, phrase arch]
               ─► [restore chest body, a touch of room]  ─► voice
```

Measured against human singing (see `tools/report.py`):

| | Aria | human singers |
|---|---|---|
| vibrato rate | 6.1 Hz | 5.5–7.5 Hz |
| vibrato extent | 33 cents | 30–60 cents |
| pitch steadiness | 20 cents drift | < 25 cents |
| loudness ripple | 0.10 | 0.03–0.12 |
| sustain decay | 0.92 | ~1.0 (no fade) |
| tuning | within a few cents of every note | — |

There is also a **🤖 Robot** tab: the original Daft-Punk-style vocoder engine,
still written entirely from scratch in NumPy (no DSP libraries). It is kept
intact but is no longer where the effort goes.

## Quick start

```bash
# 1. (recommended) install espeak-ng for real speech; without it a built-in
#    vowel-babble synth is used instead.
sudo apt-get install -y espeak-ng

# 2. install python deps
python3 -m pip install -r requirements.txt

# 3. run the web app
./run.sh                 # or:  PORT=8000 python3 server/app.py
```

Open **http://localhost:8000**. It opens on the **Sing** tab, which shows a live
quality readout (notes in tune, mean cents error, clicks, dropout) with every
render; the **Robot** tab holds the older vocoder engine.

### Command line (no server)

```bash
python examples/render_demo.py "we are the robots"          # -> renders/default.wav
python examples/render_demo.py "harder better" --all        # render every preset
```

### Make it sing — the Aria voice

`daftdsp/singer.py` turns a word-based score into singing. It is built on a
**neural voice** rather than a rule-based one, because that was the real ceiling:
espeak is a formant synthesiser and sounds buzzy by construction, no matter what
DSP follows it.

1. **Piper** (neural TTS trained on a real person) speaks the whole phrase — so
   pronunciation, and the flow from word to word, are natural to begin with.
2. Piper also reports **phoneme alignments**, so the vowels are *known*, not
   guessed from an energy envelope. Each vowel is matched to its note.
3. **WORLD** splits the phrase into pitch / spectral envelope / aperiodicity.
   Replacing the pitch therefore never disturbs the vowel or the timbre.
4. A continuous, corner-free **time-warp** holds each vowel across its note while
   consonants keep their natural length, so words stay intelligible.
5. **Sung dynamics** give every syllable full voice (speech stresses words;
   singing does not), then a light EQ / de-ess / stereo / reverb finish.

Measured on the dry signal: pitch within a few cents of every note, no clipping,
no vanished syllables. Rendering runs at the voice's native 22.05 kHz.

If no neural model is present the package still runs, falling back to espeak at
lower quality. The model (~63 MB) downloads on first use into `models/`.

Scores are plain text (`word NOTE:beats` per line, or `rest N`), editable in the
web app and stored in `daftdsp/songs.py`:

```
twinkle C4:1 C4:1
star    G4:2
rest    1
```

```bash
python examples/sing_demo.py            # render every song -> renders/aria_*.wav
python examples/sing_demo.py twinkle    # just one
```

### Quality is measured, not guessed

`daftdsp/quality.py` provides objective metrics — click/discontinuity rate,
per-note pitch error in cents, sustain flatness, within-note dropout,
headroom/clipping/DC and spectral balance. Tests assert on them, the web app
displays them, and a report tool prints them:

```bash
python -m tools.report            # every song
python -m tools.report twinkle --master
```

Pitch is always measured on the **dry** signal — reverb tails overlap
neighbouring notes and smear any pitch tracker.

### Tests

```bash
python tests/test_engine.py       # or:  pytest tests/
```

## Architecture

The package `daftdsp/` is one module per DSP stage:

| Module | What it does |
|---|---|
| `util.py`     | RIFF WAV read/write (PCM + 32-bit float), cubic resampler, note/scale math |
| `biquad.py`   | RBJ-cookbook biquad coefficients; Direct-Form-II-Transposed recursion; **exact transfer-function filtering via FFT** for speed; parallel band-bank |
| `synth.py`    | Polyphonic carrier: detuned super-saws + PWM pulses, PolyBLEP band-limiting, ±1 octave layering, vibrato, and crossfaded chord **progressions** |
| `pitch.py`    | **YIN** fundamental-frequency tracking + musical scale quantisation |
| `psola.py`    | **TD-PSOLA** formant-preserving pitch shift — sings the voice onto a **melody** (a note per syllable, Vocaloid-style) or snaps it to a scale |
| `vocoder.py`  | 32-band channel vocoder: band-pass split, rectify + low-pass envelope, carrier gating, carrier **whitening** (intelligibility) + consonant/air path |
| `formant.py`  | Talkbox: resonant vowel formant peaks (F1–F3), formant shift, vowel morphing |
| `effects.py`  | `tanh` saturation · 4–8 all-pass LFO phaser · kick + sidechain compressor · Schroeder/Moorer stereo **reverb** · 150 Hz HP / 6 kHz shelf EQ · Haas stereo widener |
| `tts.py`      | Text → vocal PCM via `espeak-ng`, with a from-scratch formant-babble fallback |
| `engine.py`   | Wires the whole chain together; `EngineParams` holds every runtime control |
| `presets.py`  | Ready-made parameter sets (Vocaloid Diva, Melancholy Android, …) |
| `voice.py`    | Neural speech front-end (**Piper**) with phoneme alignments, model download/caching, and an espeak fallback |
| `world.py`    | **WORLD** vocoder front-end: analysis, frame time-warp, formant shift, breathiness, resynthesis |
| `singer.py`   | **Aria** singer: whole-phrase neural speech → phoneme-exact alignment → continuous vowel warp → melody re-pitch → WORLD resynthesis → sung dynamics → master |

| `songs.py`    | Word-based scores (`SONGS`), the editable text score format (`parse_score`/`format_score`) and score helpers |
| `quality.py`  | Objective quality metrics (clicks, cents error, sustain, dropout, headroom, spectrum) used by tests, the report tool and the web app |

### A note on speed

Linear filters (band-passes, EQ, talkbox peaks) are *designed* from scratch as
biquads. They can be *applied* two equivalent ways: the literal time-domain
sample-by-sample recursion (`biquad.df2t`, `BandBank`, `vocode(method="iir")`),
or by evaluating each biquad's exact frequency response `H(e^jω)` — derived by
hand from the coefficients — and applying it with a single FFT (`biquad.biquad_fft`,
`vocode(method="fft")`, the default). For an LTI filter the two are
mathematically identical; the FFT path just removes the Python per-sample loop so
a full render stays interactive on the server. `numpy.fft` is used purely as a
math primitive.

## Intelligibility & melody

To keep the words clear and human-sounding (rather than a smeared robot), the
default signal path leads with the **formant-preserving PSOLA voice**: it takes
real synthesized speech and pitch-shifts it onto a **melody** — one note per
syllable — so it *sings* while staying articulate. The channel vocoder is mixed
*underneath* as a robot layer, with its carrier **whitened** (`whiten`) so the
voice's spectral shape, not the synth's, drives the output. The static
vowel-formant *talkbox* and the phaser are **off by default** because they blur
consonants; turn them on per taste. Blend the two layers with `dry_voice_mix`
(clear voice) vs `vocoder_mix` (robot).

## HTTP API

| Route | Method | Description |
|---|---|---|
| `/` | GET | the slider UI |
| `/api/meta` | GET | robot: default params, presets, option lists |
| `/api/synthesize` | POST | robot: JSON params → `{audio_b64 (16-bit WAV), meta}` |
| `/api/synthesize.wav` | POST | robot: same params → WAV download (`?float32=1` for 32-bit float) |
| `/api/sing/meta` | GET | singing: songs (as editable score text), voices, modes, defaults |
| `/api/sing` | POST | singing: `{score, bpm, voice, base_pitch, voice_mode, …}` → `{audio_b64, meta}` where `meta` carries the quality metrics |
| `/api/sing.wav` | POST | singing: same params → WAV download |

```bash
curl -X POST localhost:8000/api/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"one more time","scale":"major","talkbox_amount":0.7}' | jq .meta
```

## Runtime parameters

All of these are exposed as sliders/toggles and accepted by the API (see
`daftdsp/engine.py::EngineParams` for defaults and ranges):

- **Voice/TTS** — `text`, `voice` (e.g. `en+f3` female, `en+m3` male), `wpm`, `tts_pitch`
- **Auto-tune / melody** — `enable_autotune`, `melody` (e.g. `["A3","C4","E4"]`,
  blank = scale-snap), `melody_gap_ms`, `key_root`, `scale`, `retune`,
  `retune_time_ms` (glide)
- **Carrier synth** — `chord_root`, `chord_quality`, `chord_prog` (progression,
  e.g. `["A3:min7","F3:maj7"]`), `saw_level`, `pulse_level`, `pulse_width`,
  `pwm_rate`, `pwm_depth`, `detune_cents`, `detune_voices`, `octave_layer`,
  `sub_level`, `vibrato_rate`, `vibrato_depth`, `synth_level`
- **Vocoder** — `enable_vocoder`, `n_bands`, `band_lo`, `band_hi`, `band_q`,
  `whiten` (clarity), `voc_attack_ms`, `voc_release_ms`, `formant_shift`,
  `sibilance`, `vocoder_mix`, `dry_voice_mix`
- **Formant/talkbox** — `enable_talkbox`, `vowel`, `vowel2`, `morph_rate`,
  `formant_shift`, `formant_resonance`, `formant_gain_db`, `talkbox_amount`
- **Saturation** — `enable_saturation`, `sat_drive`, `sat_mix`
- **Phaser** — `enable_phaser`, `phaser_rate`, `phaser_depth`, `phaser_stages`,
  `phaser_feedback`
- **Sidechain** — `enable_sidechain`, `sc_amount`, `sc_threshold_db`, `sc_ratio`,
  `sc_attack_ms`, `sc_release_ms`, `kick_bpm`
- **Reverb** — `enable_reverb`, `reverb_mix`, `reverb_size`, `reverb_damp`,
  `reverb_width`
- **EQ/stereo** — `enable_eq`, `eq_hp`, `eq_shelf_freq`, `eq_shelf_db`,
  `haas_ms`, `width`, `warmth_hz` (output low-pass), `output_gain`

## I/O

- **Inputs:** raw vocal PCM float array (or generated from text), external chord
  (MIDI note list), external kick trigger array, sample-rate config.
- **Output:** interleaved 32-bit float stereo buffer (`process()` returns an
  `(N, 2)` float32 array; the server can emit 16-bit or 32-bit-float WAV).

## Dependencies

- **Runtime:** `numpy`, `flask`. Optional `espeak-ng` (system package) for
  speech; a from-scratch fallback synth runs without it.
- **Dev/test:** `playwright` (UI smoke test only).
