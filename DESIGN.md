# Architecture & design notes

## Two subsystems, one shared DSP core

```
                 ┌──────────────── shared DSP core ────────────────┐
                 │ util (WAV/resample/notes) · biquad (filters)    │
                 │ pitch (YIN) · psola (grain OLA) · effects (FX)  │
                 └───────────────┬────────────────┬───────────────┘
                                 │                │
        ┌────────────────────────┴───┐      ┌─────┴─────────────────────────┐
        │  ROBOT VOICE (Daft Punk)   │      │  SINGING VOICE (Aria)         │
        │  engine · vocoder · synth  │      │  singer · voice (Piper)       │
        │  formant · presets · tts   │      │  world (WORLD) · songs        │
        │  100% from scratch         │      │  neural source + vocoder      │
        └────────────────────────────┘      └───────────────────────────────┘
                       │                                   │
                       └──────────── server + web ─────────┘
                                  (one app, two tabs)
```

Nothing in the DSP core depends on the subsystems; the subsystems don't depend on
each other. `quality.py` is a leaf used only by tests/tools.

## Root cause of the old singing quality (and what replaced it)

The singing voice was rebuilt after diagnosing why it stayed robotic no matter how
much DSP was layered on:

| Symptom | Root cause | Fix |
|---|---|---|
| never sounded lifelike | **espeak is a formant synthesiser** — it generates speech from a rule-based resonator model, so it is buzzy *by construction* | **Piper**, a neural (VITS) TTS trained on recordings of a real person |
| pronunciation errors | espeak's own pronunciation, plus syllable boundaries **guessed from energy** | Piper's pronunciation + its **phoneme alignments** (exact sample span per phoneme) |
| words ran together / unintelligible | vowel nuclei were inferred from an energy envelope, so notes could land on the wrong sound | vowels are read directly from the phoneme alignment |
| not smooth | hand-rolled TD-PSOLA: grains, splices, clicks | the **WORLD vocoder**: pitch, spectral envelope and aperiodicity are separated, so re-pitching never touches the vowel |

The lesson: the ceiling was the *source*, not the processing.

### Libraries (singing voice only)

`piper-tts` + `onnxruntime` (neural speech), `pyworld` (WORLD vocoder), `numpy`.
The **robot voice remains 100% from-scratch** — that was its original brief — and
nothing in the shared DSP core depends on these libraries. If no neural model is
present, `voice.py` falls back to espeak so the package still runs.

## Aria singing pipeline

Rendered **one phrase at a time** (a phrase = the words between rests) so that
pronunciation and flow come from real, continuous speech:

| Stage | What happens | Why |
|---|---|---|
| 1. Speak | Piper says the whole phrase as one natural utterance | correct pronunciation + coarticulation between words |
| 2. Align | vowel phonemes (from Piper's alignments) are the syllable nuclei; matched to the score's notes | the right sound lands on the right note, with no guessing |
| 3. Analyse | WORLD splits the phrase into f0 / spectral envelope / aperiodicity | pitch becomes independent of vowel identity and timbre |
| 4. Warp | a frame trajectory holds each vowel across its note; consonants keep their natural length (compressed only if a note is too short) | intelligible words, the score's rhythm |
| 5. Re-pitch | f0 is replaced by the melody + legato glides + vibrato that swells in | the tune, sung, with the envelope untouched |
| 6. Synthesise | WORLD resynthesis | no grains, no splices, no clicks |
| 7. Shape | sung dynamics (every syllable gets full voice) + phrase arc | speech stresses words; singing does not |
| 8. Master | timbre EQ → de-ess → stereo → reverb → normalise → limit | finished sound |

The warp trajectory is built to be **continuous and corner-free**: a jump in read
position is an abrupt spectral change (a click), and a step in its *slope* is a
sudden change of read-rate (a tick), so the sections join at shared knots and the
result is lightly smoothed.

### Sample rate

Rendering happens at the neural voice's native 22.05 kHz. It has no content above
~11 kHz, so resampling every phrase to 44.1 kHz would cost time and add
interpolation error for nothing.

## Quality is measured, not guessed

`daftdsp/quality.py` provides objective metrics — discontinuity (click) rate,
per-note pitch error in cents, sustain flatness, dropout ratio, headroom/clipping,
spectral balance, and syllable-alignment checks. Tests assert on them, so
refinements are verifiable and regressions are caught rather than argued about.

```bash
python -m tools.report            # quality report for every song
python -m tools.report twinkle    # one song
```

## Conventions

- The **robot voice and the shared DSP core are 100% from scratch**: filters,
  trackers and synthesis are hand-written numpy, and `numpy.fft` is used only as
  a math primitive. That was the original brief and it still holds.
- The **singing voice** uses best-in-class libraries where they decide quality —
  a neural TTS for the voice itself and the WORLD vocoder for re-pitching —
  because the source, not the processing, was the ceiling.
- Modules are single-purpose and side-effect free; rendering functions take
  explicit parameters and return arrays (no globals, no hidden state).
- Public API is re-exported from `daftdsp/__init__.py`.
