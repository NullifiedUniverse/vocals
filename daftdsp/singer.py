"""
Aria -- a lifelike singing voice.

Every stage is modelled on what human singers measurably do, rather than on what
is convenient to compute:

===========================  ==================================================
human behaviour              how it is implemented here
===========================  ==================================================
the **vowel** lands on the   notes are planned so each vowel *onset* sits on the
beat; consonants come        beat, and its consonant occupies the time just
before it                    before it (borrowed from the previous note)
a sustained vowel holds a    the steadiest frames of the vowel are found and
steady vocal-tract shape     held, instead of freezing a transitional moment
subglottal pressure is       loudness is flattened across the sustain *after*
steady, so loudness is       synthesis, where it can actually be measured
vibrato: ~6 Hz, ±40-60       f0 vibrato at those values, delayed ~250 ms and
cents, entering after the    ramped in, with the small amplitude tremolo that
note has settled             accompanies it in a real voice
legato: pitch moves in       note-to-note pitch glides over ~70 ms, continuous
~50-120 ms, never jumping
phonation is never perfectly small F0 jitter and amplitude shimmer at measured
periodic                     human levels (~0.3 % / ~3 %)
phrases arch and are         phrase-level dynamic arch, and a release only at
released at the end          the end of a phrase
===========================  ==================================================

The source voice is a neural TTS (see :mod:`daftdsp.voice`) and pitch/time are
manipulated with the WORLD vocoder (:mod:`daftdsp.world`), so the timbre stays
that of a real recorded human.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import util, voice, world
from .biquad import biquad_fft, high_shelf, highpass, low_shelf, peaking
from .util import midi_to_freq, note_to_midi

DEFAULT_SR = 22050

# --- human singing constants (see module docstring for the behaviour each one
# --- encodes).  Times in seconds, pitch in cents.
VIBRATO_HZ = 6.0            # trained singers sit at 5.5-7.5 Hz
VIBRATO_CENTS = 58.0        # peak deviation; ±30-60 is the normal range
VIBRATO_ONSET = 0.25        # vibrato enters after the note has settled
VIBRATO_RAMP = 0.25
TREMOLO = 0.05              # amplitude modulation that rides with vibrato
JITTER_CENTS = 5.0          # cycle-to-cycle pitch irregularity
SHIMMER = 0.03              # cycle-to-cycle amplitude irregularity
GLIDE = 0.07                # legato note-to-note pitch move
ATTACK = 0.035              # onset of a sung note
RELEASE = 0.18              # only at the end of a phrase
MAX_ONSET = 0.16            # a consonant cluster never eats more than this
MAX_CODA = 0.12


@dataclass
class NotePlan:
    """One note: where it comes from in the source, and where it goes in time."""
    src_on: tuple[float, float]     # consonant span, source frames
    src_vowel: tuple[float, float]  # vowel span, source frames
    src_coda: tuple[float, float]   # coda span, source frames
    out_start: int                  # first output frame (consonant onset)
    out_vowel: int                  # frame where the vowel starts (= the beat)
    out_end: int                    # one past the last output frame
    midi: int


# ---------------------------------------------------------------------------
# planning: syllables -> notes, with the vowel on the beat
# ---------------------------------------------------------------------------

def _syllable_spans(utt, n_notes):
    """``n_notes`` spans of ``(start, vowel_start, vowel_end, end)`` in samples.

    Vowel positions come from the neural voice's phoneme alignment, so they are
    known rather than inferred, and everything is clamped to the region that
    actually contains speech (the surrounding silence belongs to no syllable).
    """
    groups = utt.vowel_groups()
    lo, hi = utt.speech_span()
    if not groups:
        step = (hi - lo) / max(1, n_notes)
        return [(int(lo + k * step),) * 2 + (int(lo + (k + 1) * step),) * 2
                for k in range(n_notes)]

    if len(groups) > n_notes:                    # drop the weakest nuclei
        groups = sorted(sorted(groups, key=lambda g: g[0] - g[1])[:n_notes])
    while len(groups) < n_notes:                 # split the longest
        i = int(np.argmax([b - a for a, b in groups]))
        a, b = groups[i]
        mid = (a + b) // 2
        groups[i:i + 1] = [(a, mid), (mid, b)]
        groups = sorted(groups)

    spans = []
    for k, (v0, v1) in enumerate(groups):
        start = lo if k == 0 else (groups[k - 1][1] + v0) // 2
        end = hi if k == len(groups) - 1 else (v1 + groups[k + 1][0]) // 2
        v0 = int(min(max(v0, lo), hi))
        v1 = int(min(max(v1, v0 + 1), hi))
        spans.append((int(min(start, v0)), v0, v1, int(max(v1, min(end, hi)))))
    return spans


def _plan_notes(fr, spans, notes, beat, fps):
    """Lay the syllables out in time with each **vowel onset on its beat**.

    This is the single biggest difference between singing and stretched speech:
    a singer places the vowel on the beat and fits the consonant into the time
    just before it, so the rhythm is carried by the vowels.
    """
    beats = np.cumsum([0.0] + [b for _, b in notes]) * beat
    onsets = []
    for (s0, v0, _, _) in spans:
        onsets.append(min((v0 - s0) / fr.sr, MAX_ONSET))

    plans = []
    for k, ((s0, v0, v1, s1), (note, nb)) in enumerate(zip(spans, notes)):
        vowel_at = beats[k]
        # The consonant is borrowed from the time before the beat (but never
        # from before the phrase, and never more than half the previous note).
        room = vowel_at if k == 0 else (beats[k] - beats[k - 1]) * 0.5
        on_dur = min(onsets[k], max(0.0, room))
        # The vowel runs until the next note's consonant has to start.
        next_at = beats[k + 1] if k + 1 < len(notes) else beats[-1]
        next_on = min(onsets[k + 1], (beats[k + 1] - beats[k]) * 0.5) \
            if k + 1 < len(notes) else 0.0
        vowel_end = max(vowel_at + 0.05, next_at - next_on)

        plans.append(NotePlan(
            src_on=(fr.samples_to_frame(s0), fr.samples_to_frame(v0)),
            src_vowel=(fr.samples_to_frame(v0), fr.samples_to_frame(v1)),
            src_coda=(fr.samples_to_frame(v1), fr.samples_to_frame(s1)),
            out_start=int(round((vowel_at - on_dur) * fps)),
            out_vowel=int(round(vowel_at * fps)),
            out_end=int(round(vowel_end * fps)),
            midi=note_to_midi(note)))
    return plans, beats[-1]


def _steadiest(fr, a, b):
    """The most stable stretch of a vowel -- what a singer actually holds.

    Spectral flux is lowest where the vocal tract has stopped moving, so that is
    the part of the vowel to sustain; holding a transitional frame instead is
    what makes a held note sound synthetic.
    """
    a, b = int(max(0, a)), int(min(fr.n, b))
    if b - a < 3:
        return float(a + b) * 0.5
    sp = np.log(fr.sp[a:b] + 1e-12)
    flux = np.r_[0.0, np.mean(np.abs(np.diff(sp, axis=0)), axis=1)]
    k = min(5, max(1, (b - a) // 3))
    smooth = np.convolve(flux, np.ones(k) / k, mode="same")
    best = a + int(np.argmin(smooth))
    # Only a voiced frame can be sustained -- holding an unvoiced one gives a
    # note with no tone at all (measured: a completely silent note).  Keep the
    # steadiest choice and simply move to the nearest voiced frame if it is not
    # voiced; re-searching under a mask instead picks a transitional frame.
    if fr.f0[min(best, fr.n - 1)] > 0:
        return float(best)
    vi = np.where(fr.f0[a:b] > 0)[0]
    if vi.size:
        return float(a + vi[np.argmin(np.abs(vi - (best - a)))])
    vi = np.where(fr.f0 > 0)[0]
    return float(vi[np.argmin(np.abs(vi - best))]) if vi.size else float(best)


# ---------------------------------------------------------------------------
# trajectory + melody
# ---------------------------------------------------------------------------

def _trajectory(fr, plans, fps):
    """Source-frame index for every output frame (continuous, corner-free)."""
    n_out = plans[-1].out_end
    traj = np.zeros(n_out)
    prev_end = plans[0].src_on[0]
    voiced = fr.f0 > 0
    for p in plans:
        hold = _steadiest(fr, *p.src_vowel)
        v0, v1 = p.src_vowel
        # Keep the whole sung part of the note inside the voiced run that
        # contains the held frame: if a ramp runs off into unvoiced frames the
        # note loses its tone (measured as notes that came out silent).
        h = int(np.clip(hold, 0, fr.n - 1))
        if voiced[h]:
            lo = h
            while lo > 0 and voiced[lo - 1]:
                lo -= 1
            hi = h
            while hi + 1 < fr.n and voiced[hi + 1]:
                hi += 1
            v0 = max(v0, lo)
            v1 = min(max(v1, v0 + 1), hi + 1)
        # consonant: play it at its natural speed, arriving at the vowel on time
        a, b = p.out_start, p.out_vowel
        if b > a:
            traj[max(0, a):b] = np.linspace(prev_end, v0, b - max(0, a))
        # vowel: reach the steady posture quickly, hold it, then move on
        c = p.out_end
        m = c - b
        if m > 0:
            reach = min(int(0.06 * fps), max(1, m // 3))
            traj[b:b + reach] = np.linspace(v0, hold, reach)
            tail = min(int(0.05 * fps), max(0, (m - reach) // 3))
            body = m - reach - tail
            if body > 0:
                traj[b + reach:b + reach + body] = hold
            if tail > 0:
                traj[c - tail:c] = np.linspace(hold, min(v1, hold + (v1 - hold)),
                                               tail)
        prev_end = v1
    # Round the corners: a step in read-*rate* is audible as a tick.
    k = 5
    traj = np.convolve(np.pad(traj, (k, k), mode="edge"),
                       np.ones(k) / k, mode="same")[k:-k]
    return np.clip(traj, 0, fr.n - 1)


def _melody(plans, fps, seed):
    """f0 per output frame: legato glides, vibrato that enters late, jitter."""
    n_out = plans[-1].out_end
    f0 = np.zeros(n_out)
    prev = 0.0
    for p in plans:
        hz = midi_to_freq(p.midi)
        a, b = max(0, p.out_start), p.out_end
        f0[a:b] = hz
        if prev > 0:
            g = min(int(GLIDE * fps), max(1, (b - a) // 2))
            f0[a:a + g] = prev * (hz / prev) ** np.linspace(0.0, 1.0, g)
        prev = hz

    rng = np.random.default_rng(seed)
    cents = np.zeros(n_out)
    for p in plans:
        a, b = max(0, p.out_vowel), p.out_end
        if b <= a:
            continue
        t = np.arange(b - a) / fps
        env = np.clip((t - VIBRATO_ONSET) / VIBRATO_RAMP, 0.0, 1.0)
        cents[a:b] = VIBRATO_CENTS * env * np.sin(2.0 * np.pi * VIBRATO_HZ * t)
    jitter = rng.standard_normal(n_out)
    w = max(1, int(0.08 * fps))
    jitter = np.convolve(jitter, np.ones(w) / w, mode="same")
    cents += JITTER_CENTS * jitter / (np.std(jitter) or 1.0)
    return f0 * 2.0 ** (cents / 1200.0)


# ---------------------------------------------------------------------------
# amplitude: what a singer's breath support actually produces
# ---------------------------------------------------------------------------

def _amplitude(y, sr, plans, fps, target=None, phrase_arch=True):
    """Shape loudness in the time domain, after synthesis.

    Loudness has to be corrected *here*: WORLD's output level depends on the f0
    it is given as well as on the spectral envelope, so flattening the envelope
    alone does not produce a steady note (measured: a sustained note still decayed
    from 0.18 to 0.004).

    Every sung note is levelled to **one common target** rather than to its own
    average -- this is the standard fix in singing synthesis, where the voiced
    sections of every unit are gain-matched to a single global RMS so that no
    syllable is louder than another.  Levelling each note to its own mean instead
    (as this used to) preserves exactly the speech-loudness differences that make
    a sung line sound uneven; measured note-to-note spread was 2.6x.
    """
    n = y.size
    env = np.abs(y).astype(np.float64)
    k = max(1, int(0.03 * sr))
    env = np.convolve(env, np.ones(k) / k, mode="same")

    gain = np.ones(n)
    mids = [p.midi for p in plans]
    lo, hi = min(mids), max(mids)
    for j, p in enumerate(plans):
        a = int(max(0, p.out_vowel) / fps * sr)
        b = int(min(p.out_end / fps * sr, n))
        if b - a < 8:
            continue
        seg = env[a:b]
        live = seg[seg > 0.15 * (seg.max() or 1.0)]
        ref = float(np.median(live)) if live.size else float(seg.mean())
        if ref <= 1e-6:
            continue
        # Steady breath support within the note, and the same level as every
        # other note (the global target), not merely self-consistent.
        aim = target if target else ref
        gain[a:b] = np.clip(aim / np.maximum(seg, 0.05 * ref), 0.25, 4.0)
        if phrase_arch:                        # phrases arch; high notes carry
            pn = (p.midi - lo) / (hi - lo) if hi > lo else 0.5
            arc = np.sin(np.pi * (j + 0.5) / len(plans))
            gain[a:b] *= 0.86 + 0.09 * pn + 0.09 * arc

    k = max(1, int(0.04 * sr))
    gain = np.convolve(np.pad(gain, (k, k), mode="edge"),
                       np.ones(k) / k, mode="same")[k:-k]
    y = y * gain

    # Tremolo rides with the vibrato, and phonation is never perfectly even.
    t = np.arange(n) / sr
    rng = np.random.default_rng(7)
    sh = rng.standard_normal(n)
    w = max(1, int(0.05 * sr))
    sh = np.convolve(sh, np.ones(w) / w, mode="same")
    y = y * (1.0 + TREMOLO * np.sin(2.0 * np.pi * VIBRATO_HZ * t + 1.1)
             + SHIMMER * sh / (np.std(sh) or 1.0))

    # Sung attack, and a release only where the phrase ends.
    at = min(int(ATTACK * sr), n)
    rel = min(int(RELEASE * sr), n)
    y[:at] *= np.linspace(0.0, 1.0, at) ** 0.6
    y[-rel:] *= np.linspace(1.0, 0.0, rel) ** 1.5
    return y


# ---------------------------------------------------------------------------
# phrase / song
# ---------------------------------------------------------------------------

def _render_phrase(phrase, sr, voice_name, beat, formant_shift, breath, seed,
                   target=None):
    text = " ".join(word for word, _ in phrase)
    notes = [nt for _, wnotes in phrase for nt in wnotes]
    # Close the between-word pauses: a sung phrase is one connected line.
    utt = voice.speak(text, name=voice_name).legato()
    fr = world.analyze(utt.audio, utt.sr)
    fps = 1000.0 / fr.frame_period

    spans = _syllable_spans(utt, len(notes))
    plans, _ = _plan_notes(fr, spans, notes, beat, fps)
    traj = _trajectory(fr, plans, fps)
    f0 = _melody(plans, fps, seed)

    warped = world.resample_frames(fr, traj)
    sung = world.Frames(np.where(warped.f0 > 0, f0[:warped.n], 0.0),
                        warped.sp, warped.ap, warped.sr, warped.frame_period)
    sung = world.shift_formants(sung, formant_shift)
    sung = world.breathiness(sung, breath)

    y = world.synthesize(sung).astype(np.float64)
    y = _amplitude(y, utt.sr, plans, fps, target=target)
    if utt.sr != sr:
        y = util.resample(y.astype(np.float32), utt.sr, sr)
    # The consonant of the first note starts before its beat.
    lead = max(0.0, (plans[0].out_vowel - max(0, plans[0].out_start)) / fps)
    return np.asarray(y, dtype=np.float32), lead


def _phrases(score, beat):
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


def _loudness(x):
    """RMS of the audible part -- silence must not drag the measurement down."""
    a = np.abs(x)
    live = x[a > 0.05 * (float(a.max()) or 1.0)]
    return float(np.sqrt(np.mean(live ** 2))) if live.size else 0.0


def sing(score, sr=DEFAULT_SR, bpm=100, voice_name=voice.DEFAULT_VOICE,
         formant_shift=1.0, breath=0.0, seed=5):
    """Render a word-based score to a sung mono line."""
    beat = 60.0 / bpm
    phrases, total = _phrases(score, beat)
    out = np.zeros(int(total * sr) + sr, dtype=np.float32)

    rendered = []
    for k, (start, phrase) in enumerate(phrases):
        buf, lead = _render_phrase(phrase, sr, voice_name, beat, formant_shift,
                                   breath, seed + k)
        rendered.append((start, lead, buf))
    # One loudness target for the whole song, applied to every note, so no
    # syllable or line is louder than another.
    levels = [_loudness(b) for _, _, b in rendered]
    ref = float(np.median([lv for lv in levels if lv > 1e-6]) or 1.0)
    rendered = [(start, lead,
                 _render_phrase(phrase, sr, voice_name, beat, formant_shift,
                                breath, seed + k, target=ref)[0])
                for k, ((start, lead, _), (_, phrase))
                in enumerate(zip(rendered, phrases))]
    levels = [_loudness(b) for _, _, b in rendered]
    for (start, lead, buf), lv in zip(rendered, levels):
        if lv > 1e-6:
            buf = buf * np.clip(ref / lv, 0.7, 1.5)
        p = max(0, int((start - lead) * sr))
        e = min(out.size, p + buf.size)
        out[p:e] += buf[:e - p]
    return util.normalize_peak(out[:int(total * sr) + int(0.4 * sr)], 0.9)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def _voice_timbre(x, sr):
    """Put back the chest resonance that singing above the speaking range loses.

    Nothing else is applied: chorus, stereo widening and heavy reverb all make a
    solo voice sound processed rather than present, so they are gone.
    """
    return biquad_fft([highpass(80.0, sr, 0.7),
                       low_shelf(350.0, sr, 3.0),
                       peaking(900.0, sr, 1.0, 1.0),
                       peaking(4000.0, sr, 1.6, -2.0),
                       high_shelf(9000.0, sr, 1.0)], x)


def render_song(score, sr=DEFAULT_SR, bpm=100, voice_name=voice.DEFAULT_VOICE,
                room=0.08, **kw):
    """Render a song: sing -> restore body -> a touch of room.

    ``room`` is a small early-reflection blend so the voice is not clinically
    dry; set it to 0 for the raw voice.  Output is mono -- a single singer is a
    point source, and widening one only makes it sound artificial.
    """
    y = sing(score, sr, bpm, voice_name, **kw)
    y = _voice_timbre(y, sr)
    if room > 0:
        y = _room(y, sr, room)
    y = util.normalize_percentile(y, target=0.82)
    return util.soft_limit(y, 0.97)


def _room(x, sr, mix):
    """A few early reflections -- the sound of a room, not an effect."""
    out = np.asarray(x, dtype=np.float64).copy()
    for delay_ms, g in ((17.0, 0.5), (23.0, 0.4), (31.0, 0.3), (43.0, 0.22)):
        d = int(delay_ms / 1000.0 * sr)
        if d < out.size:
            out[d:] += mix * g * x[:-d]
    return util.normalize_peak(out.astype(np.float32), 0.95)
