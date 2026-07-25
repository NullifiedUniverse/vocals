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
        │  engine · vocoder · synth  │      │  singer · harmonic · songs    │
        │  formant · presets · tts   │      │  tts                          │
        └────────────────────────────┘      └───────────────────────────────┘
                       │                                   │
                       └──────────── server + web ─────────┘
                                  (one app, two tabs)
```

Nothing in the DSP core depends on the subsystems; the subsystems don't depend on
each other. `quality.py` is a leaf used only by tests/tools.

## Aria singing pipeline

Rendered **one phrase at a time** (a phrase = the words between rests) so that
pronunciation and flow come from real, continuous speech:

| Stage | What happens | Why |
|---|---|---|
| 1. Speak | espeak says the whole phrase as one natural utterance | correct pronunciation + coarticulation between words |
| 2. Analyse | YIN pitch track, voicing, pitch epochs, vowel-nucleus detection | find one nucleus per sung syllable |
| 3. Align | nuclei → notes; syllable bounds at energy minima between nuclei | syllables land on their notes |
| 4. Warp | continuous input-time trajectory: vowels stretched to fill their note, consonants at natural speed (compressed proportionally if a note is too short) | intelligible words, correct rhythm |
| 5. Pitch | melody + legato glides + vibrato that swells in | the tune, sung |
| 6. Synthesise | one continuous pass over the whole phrase | flow: no per-word resets |
| 7. Shape | per-note dynamics, phrase edges | musical phrasing |
| 8. Master | timbre EQ → de-ess → stereo → reverb → normalise → limit | finished sound |

### The one strategy seam: `voice_mode`

Only **stage 6** varies; everything else is shared.

- `voice_mode="natural"` (default) — **TD-PSOLA** on espeak's real waveform.
  Real grains in, real grains out, so the words are exactly espeak's
  pronunciation. Best intelligibility and flow.
- `voice_mode="synth"` — **harmonic + noise resynthesis** (`harmonic.py`).
  Rebuilds voiced vowels from phase-dispersed sinusoids at the exact pitch
  (smoother, more synthetic-diva) and splices espeak's real unvoiced consonants.

This is a strategy seam, not a duplicated pipeline: both modes share stages 1–5
and 7–8.

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

- No scipy / librosa / DSP libraries: filters, trackers and synthesis are
  hand-written numpy. `numpy.fft` is used only as a math primitive.
- Modules are single-purpose and side-effect free; rendering functions take
  explicit parameters and return arrays (no globals, no hidden state).
- Public API is re-exported from `daftdsp/__init__.py`.
