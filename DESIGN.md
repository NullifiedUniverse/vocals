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

## Aria singing pipeline — modelled on human singers

Rendered one **phrase** at a time (the words between rests). Each stage encodes a
measured behaviour of human singing rather than whatever is easiest to compute:

| Stage | What happens | The human behaviour it encodes |
|---|---|---|
| 1. Speak | the neural voice says the whole phrase, reporting phoneme spans | pronunciation and word-to-word flow come from real continuous speech |
| 2. Plan | each **vowel onset is placed on its beat**; the consonant occupies the time just before it, borrowed from the previous note | singers carry rhythm with vowels and put consonants ahead of the beat |
| 3. Analyse | WORLD splits the phrase into f0 / spectral envelope / aperiodicity | pitch becomes independent of vowel identity and timbre |
| 4. Hold | the **steadiest** stretch of each vowel (lowest spectral flux) is found and sustained | a singer settles into a vowel posture and holds it, rather than freezing a transitional moment |
| 5. Pitch | melody with ~70 ms legato glides, vibrato at 6 Hz / ±58 cents entering after 250 ms, plus small jitter | measured vibrato rate, extent and onset delay; phonation is never perfectly periodic |
| 6. Synthesise | WORLD resynthesis | no grains, no splices |
| 7. Loudness | the sustain is levelled **after** synthesis, with attack, phrase arch, tremolo and shimmer | steady subglottal pressure gives a steady note; phrases arch and are released at the end |
| 8. Output | restore chest body, optional small room, mono | a solo singer is a point source |

Loudness has to be corrected after synthesis, not in the envelope: WORLD's output
level depends on the f0 it is given as well as on the spectral envelope, so
flattening `sp` alone still let a sustained note decay from 0.18 to 0.004.

Effects that make a solo voice sound *processed* — chorus, stereo widening, large
reverb — were removed.

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
