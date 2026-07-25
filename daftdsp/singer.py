"""
Aria -- the singing voice.

Pipeline (see DESIGN.md).  Rendered one **phrase** at a time (the words between
rests), so pronunciation and word-to-word flow come from one continuous, natural
utterance:

  1. **Speak**      Piper (neural TTS) says the whole phrase, and reports the
                    exact sample span of every phoneme.
  2. **Align**      vowel phonemes are the syllable nuclei -- read from the
                    alignment, not guessed from energy -- and are matched to the
                    score's notes.
  3. **Analyse**    WORLD splits the phrase into f0 / spectral envelope /
                    aperiodicity.
  4. **Warp**       a frame trajectory holds each vowel across its note while
                    consonants run at natural speed, so words stay intelligible
                    and the rhythm is the score's.
  5. **Re-pitch**   f0 is replaced by the melody (legato glides, vibrato that
                    swells in, light drift); the envelope is untouched, so the
                    vowel and the timbre survive exactly.
  6. **Synthesise** WORLD resynthesis -- no grains, no splices, no clicks.
  7. **Shape**      sung dynamics (every syllable gets full voice) + phrase arc.
  8. **Master**     gentle EQ, de-ess, stereo, reverb.
"""
from __future__ import annotations

import numpy as np

from . import effects, util, voice, world
from .biquad import (biquad_fft, high_shelf, highpass, low_shelf,
                     one_pole_lp_fft, peaking)
from .util import midi_to_freq, note_to_midi

# The neural voice is natively 22.05 kHz and has no content above ~11 kHz, so
# rendering at its own rate avoids a pointless resample of every phrase.
DEFAULT_SR = 22050


# ---------------------------------------------------------------------------
# alignment: phonemes -> syllables -> notes
# ---------------------------------------------------------------------------

def _syllables(utt, n_notes):
    """Return ``n_notes`` syllable spans ``(start, vowel_start, vowel_end, end)``
    in samples, derived from the phoneme alignment.

    Spans are clamped to the utterance's speech region: the surrounding silence
    is not part of any syllable, and letting a note read it makes the note fade
    away to nothing.
    """
    groups = utt.vowel_groups()
    lo, hi = utt.speech_span()

    if not groups:                                   # no vowels detected at all
        step = (hi - lo) / max(1, n_notes)
        return [(int(lo + k * step), int(lo + k * step), int(lo + (k + 1) * step),
                 int(lo + (k + 1) * step)) for k in range(n_notes)]

    if len(groups) > n_notes:
        # Keep the longest nuclei (unstressed schwas get absorbed).
        groups = sorted(sorted(groups, key=lambda g: g[0] - g[1])[:n_notes])
    while len(groups) < n_notes:
        # Split the longest nucleus so every note still gets a vowel to sing.
        i = int(np.argmax([b - a for a, b in groups]))
        a, b = groups[i]
        mid = (a + b) // 2
        groups[i:i + 1] = [(a, mid), (mid, b)]
        groups = sorted(groups)

    spans = []
    for k, (v0, v1) in enumerate(groups):
        start = lo if k == 0 else (groups[k - 1][1] + v0) // 2
        end = hi if k == len(groups) - 1 else (v1 + groups[k + 1][0]) // 2
        v0 = min(max(v0, lo), hi)
        v1 = min(max(v1, v0 + 1), hi)
        spans.append((min(start, v0), v0, v1, max(v1, min(end, hi))))
    return spans


# ---------------------------------------------------------------------------
# warp + melody
# ---------------------------------------------------------------------------

def _build_trajectory(fr, spans, notes, beat, sr, glide_ms, vib_depth,
                      vib_rate, seed):
    """Frame trajectory + target f0 for a phrase, one entry per output frame."""
    fps = 1000.0 / fr.frame_period                       # frames per second
    traj, f0, slots = [], [], []
    prev_hz = 0.0
    pos = 0
    rng = np.random.default_rng(seed)

    for (s0, v0, v1, s1), (note, beats) in zip(spans, notes):
        n_out = max(int(beats * beat * fps), int(0.10 * fps))
        f_s0, f_v0 = fr.samples_to_frame(s0), fr.samples_to_frame(v0)
        f_v1, f_s1 = fr.samples_to_frame(v1), fr.samples_to_frame(s1)

        # Consonants keep their natural length -- racing through a cluster like
        # "str" makes it chirp -- but are capped so a long tail can never take
        # over the note, and compressed further if the note is short.
        on = int(min(max(0.0, f_v0 - f_s0), 0.20 * fps))
        co = int(min(max(0.0, f_s1 - f_v1), 0.18 * fps))
        budget = int(0.5 * n_out)
        if on + co > budget and on + co > 0:
            f = budget / (on + co)
            on, co = int(on * f), int(co * f)
        hold = max(1, n_out - on - co)

        # The trajectory must be continuous: any jump in read position is an
        # abrupt spectral change, i.e. an audible click.  Build it from knots
        # that join exactly, collapsing a knot when its section has no frames.
        vow_a = f_v0 if on else f_s0
        vow_b = f_v1 if co else f_s1
        seg = np.empty(n_out)
        if on:
            seg[:on] = np.linspace(f_s0, vow_a, on)
        # Across the vowel, move quickly through the on-glide, dwell on the
        # steady middle, then resolve the off-glide -- monotone, so it never
        # jumps, and slow in the centre, so the note is a held vowel.  A slow
        # wander is added on top: a perfectly frozen frame sounds synthetic
        # ("tinny"), whereas a real held vowel keeps drifting slightly.
        u = np.linspace(0.0, 1.0, hold)
        ease = u + 0.85 * np.sin(2.0 * np.pi * u) / (2.0 * np.pi)
        path = vow_a + (vow_b - vow_a) * ease
        if hold > 4:
            wander = 0.06 * (vow_b - vow_a) * np.sin(2.0 * np.pi * 1.7 * u * hold / fps)
            path = path + wander * np.clip(np.sin(np.pi * u) * 2.0, 0.0, 1.0)
        seg[on:on + hold] = path
        if co:
            seg[on + hold:] = np.linspace(vow_b, f_s1, n_out - on - hold)
        traj.append(seg)

        hz = midi_to_freq(note_to_midi(note))
        line = np.full(n_out, hz)
        if prev_hz > 0 and glide_ms > 0:
            g = min(int(glide_ms / 1000.0 * fps), n_out // 2)
            if g > 1:
                line[:g] = prev_hz * (hz / prev_hz) ** np.linspace(0.0, 1.0, g)
        f0.append(line)
        slots.append((pos, pos + n_out, note_to_midi(note)))
        pos += n_out
        prev_hz = hz

    traj = np.concatenate(traj)
    f0 = np.concatenate(f0)
    # The trajectory joins its sections without gaps, but its *slope* still
    # steps at each knot, and a sudden change of read-rate is heard as a tick.
    # A short moving average rounds those corners (C1) at negligible timing cost.
    k = 5
    traj = np.convolve(np.pad(traj, (k, k), mode="edge"),
                       np.ones(k) / k, mode="same")[k:-k]

    # Expression: vibrato that swells in after a note settles, plus a slow drift
    # so sustained notes are never mathematically static.
    t = np.arange(f0.size) / fps
    cents = np.zeros(f0.size)
    for o0, o1, _ in slots:
        loc = t[o0:o1] - t[o0]
        env = np.clip((loc - 0.30) / 0.35, 0.0, 1.0)
        cents[o0:o1] = vib_depth * env * np.sin(2.0 * np.pi * vib_rate * loc)
    drift = rng.standard_normal(f0.size)
    w = max(1, int(0.5 * fps))
    drift = np.convolve(drift, np.ones(w) / w, mode="same")
    cents += 4.0 * drift / (np.std(drift) or 1.0)
    return traj, f0 * 2.0 ** (cents / 1200.0), slots


def _sung_dynamics(fr, slots, evenness=0.8, sustain=0.85):
    """Turn speech dynamics into *sung* dynamics.

    Two things differ between speaking and singing, and both are fixed here:

    * **Across notes** -- speech stresses some syllables and throws others away,
      so unstressed ones drop out of the melody.  A singer gives every syllable
      full voice, so each note is levelled toward the phrase median.
    * **Within a note** -- a spoken vowel decays as the speaker moves on, and
      because WORLD's spectral envelope carries loudness, holding that vowel
      holds its decay too and the note fades out.  A singer sustains at constant
      volume, so the voiced part of each note is levelled against its own median.

    A musical arc (a little louder toward higher notes and the middle of the
    phrase) is then applied on top, so the result is shaped, not flat.
    """
    power = np.sqrt(fr.sp.sum(axis=1) + 1e-12)
    voiced = fr.f0 > 0
    levels = [float(np.median(power[a:b][voiced[a:b]]) if voiced[a:b].any()
                    else np.mean(power[a:b])) or 1e-6 for a, b, _ in slots]
    ref = float(np.median([lv for lv in levels if lv > 1e-6]) or 1.0)
    mids = [m for _, _, m in slots]
    lo, hi = min(mids), max(mids)

    gain = np.ones(fr.n)
    for j, (a, b, m) in enumerate(slots):
        note_ref = levels[j]
        # Hold the sustained vowel at a steady level (voiced frames only --
        # consonants are meant to be quieter and noisier).
        seg = np.ones(b - a)
        v = voiced[a:b]
        if v.any():
            seg[v] = np.clip((note_ref / power[a:b][v]) ** sustain, 0.35, 2.8)
        even = np.clip((ref / note_ref) ** evenness, 0.5, 2.2)
        pn = (m - lo) / (hi - lo) if hi > lo else 0.5
        arc = np.sin(np.pi * (j + 0.5) / len(slots))
        gain[a:b] = seg * even * (0.86 + 0.08 * pn + 0.08 * arc)
    k = max(1, int(0.03 * 1000.0 / fr.frame_period))
    gain = np.convolve(np.pad(gain, (k, k), mode="edge"),
                       np.ones(k) / k, mode="same")[k:-k]
    # Envelope is a power spectrum, so amplitude gain g -> g**2 on sp.
    return world.Frames(fr.f0, fr.sp * (gain ** 2)[:, None], fr.ap, fr.sr,
                        fr.frame_period)


# ---------------------------------------------------------------------------
# phrase / song rendering
# ---------------------------------------------------------------------------

def _render_phrase(phrase, sr, voice_name, beat, glide_ms, vib_depth, vib_rate,
                   formant_shift, breath, seed):
    text = " ".join(word for word, _ in phrase)
    notes = [nt for _, wnotes in phrase for nt in wnotes]
    utt = voice.speak(text, name=voice_name)
    fr = world.analyze(utt.audio, utt.sr)
    spans = _syllables(utt, len(notes))

    traj, f0, slots = _build_trajectory(fr, spans, notes, beat, utt.sr,
                                        glide_ms, vib_depth, vib_rate, seed)
    warped = world.resample_frames(fr, traj)
    # Voiced frames sing the melody; unvoiced frames stay unvoiced (consonants).
    sung = world.Frames(np.where(warped.f0 > 0, f0, 0.0), warped.sp, warped.ap,
                        warped.sr, warped.frame_period)
    sung = world.shift_formants(sung, formant_shift)
    sung = world.breathiness(sung, breath)
    sung = _sung_dynamics(sung, slots)

    y = world.synthesize(sung)
    if utt.sr != sr:
        y = util.resample(y, utt.sr, sr)
    n = y.size
    at = min(int(0.015 * sr), n)
    rel = min(int(0.12 * sr), n)
    y[:at] *= np.linspace(0.0, 1.0, at)
    y[-rel:] *= np.linspace(1.0, 0.0, rel) ** 1.3
    return y


def _phrases(score, beat):
    """Group a score into (start_seconds, [(word, notes), ...]) phrases."""
    out, t, cur, cur_t = [], 0.0, [], 0.0
    for item in score:
        if item[0] == "rest":
            if cur:
                out.append((cur_t, cur))
                cur = []
            t += item[1] * beat
        else:
            if not cur:
                cur_t = t
            cur.append(item)
            t += sum(b for _, b in item[1]) * beat
    if cur:
        out.append((cur_t, cur))
    return out, t


def sing(score, sr=DEFAULT_SR, bpm=100, voice_name=voice.DEFAULT_VOICE,
         glide_ms=45.0, vib_depth=22.0, vib_rate=5.5, formant_shift=1.0,
         breath=0.0, seed=5):
    """Render a word-based score to a continuous, phrased sung mono line."""
    beat = 60.0 / bpm
    phrases, total = _phrases(score, beat)
    out = np.zeros(int(total * sr) + sr, dtype=np.float32)
    # Phrases are levelled to a common *loudness*.  Peak-normalising each phrase
    # instead (as this used to) makes a line with one sharp transient quiet and a
    # smooth line loud, which is heard as the volume jumping between lines.
    bufs = []
    for k, (start, phrase) in enumerate(phrases):
        buf = _render_phrase(phrase, sr, voice_name, beat, glide_ms, vib_depth,
                             vib_rate, formant_shift, breath, seed + k)
        bufs.append((start, buf))
    levels = [_loudness(b) for _, b in bufs]
    ref = float(np.median([lv for lv in levels if lv > 1e-6]) or 1.0)
    for (start, buf), lv in zip(bufs, levels):
        if lv > 1e-6:
            buf = buf * np.clip(ref / lv, 0.6, 1.7)
        p = int(start * sr)
        e = min(out.size, p + buf.size)
        out[p:e] += buf[:e - p]
    out = out[:int(total * sr) + int(0.3 * sr)]
    return util.normalize_peak(out, 0.9)


def _loudness(x):
    """RMS of the audible part -- silence must not drag the measurement down."""
    a = np.abs(x)
    live = x[a > 0.05 * (float(a.max()) or 1.0)]
    return float(np.sqrt(np.mean(live ** 2))) if live.size else 0.0


# ---------------------------------------------------------------------------
# mastering
# ---------------------------------------------------------------------------

def _voice_timbre(x, sr):
    """Put back the chest/body that singing above the speaking range loses.

    The notes sit well above the voice's natural speaking pitch, which shifts
    energy up the spectrum and reads as thin or "tinny".  A low shelf restores
    the body, a gentle dip tames the 4 kHz edge, and only a little air is added
    on top -- the neural voice already has a natural spectrum, so this stays
    light.
    """
    return biquad_fft([highpass(80.0, sr, 0.7),
                       low_shelf(350.0, sr, 3.5),      # body / chest
                       peaking(900.0, sr, 1.0, 1.0),   # warmth in the low mids
                       peaking(4000.0, sr, 1.6, -2.0),  # take off the edge
                       high_shelf(9000.0, sr, 1.0)], x)


def _deess(x, sr, amount=0.5):
    hi = biquad_fft(highpass(6500.0, sr, 0.7), x)
    lo = np.asarray(x, dtype=np.float64) - hi
    env = one_pole_lp_fft(np.abs(hi), sr, 55.0)
    nz = env[env > 1e-5]
    thr = 1.8 * float(np.median(nz)) if nz.size else 1.0
    gain = one_pole_lp_fft(np.clip(thr / (env + 1e-6), 1.0 - amount, 1.0), sr, 80.0)
    return (lo + hi * gain).astype(np.float32)


def render_song(score, sr=DEFAULT_SR, bpm=100, voice_name=voice.DEFAULT_VOICE,
                reverb_mix=0.16, width=1.15, **kw):
    """Full render: sing -> timbre -> de-ess -> stereo -> reverb -> master."""
    dry = sing(score, sr, bpm, voice_name, **kw)
    dry = _voice_timbre(dry, sr)
    dry = _deess(dry, sr)
    dry = util.normalize_peak(dry, 0.92)
    stereo = effects.stereoize(dry, sr, haas_ms=8.0, width=width)
    stereo = effects.reverb(stereo, sr, mix=reverb_mix, size=0.7, damp=0.5,
                            width=1.15)
    stereo = util.normalize_percentile(stereo, target=0.85)
    return util.soft_limit(stereo, 0.98)
