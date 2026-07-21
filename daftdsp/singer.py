"""
Aria -- a from-scratch singing voice.

Rather than chopping espeak into phoneme fragments, Aria speaks whole **words**
naturally (correct pronunciation + coarticulation), then warps them onto the
melody with epoch-based Time-Domain PSOLA:

  * each word is split into its syllables at the vowel nuclei,
  * every syllable's *vowel* is stretched onto its note by slowly ping-ponging
    the read pointer through the steady middle of the vowel -- so a held note is
    the real vowel, sustained and gently moving, never a frozen grain and never
    decaying,
  * pitch is set by overlap-add grain spacing (formants preserved), with legato
    glides between notes.

Because every grain is Hann-windowed and overlap-added at pitch spacing, the
result is click-free -- no loops, no wander, no injected noise.
"""
from __future__ import annotations

import numpy as np

from . import effects, harmonic, pitch, tts, util
from .biquad import (biquad_fft, high_shelf, highpass, low_shelf, lowpass,
                     one_pole_lp_fft, peaking)
from .util import midi_to_freq, note_to_midi


def _hann(n):
    n = max(2, int(n))
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def _marks(sr, f0s, voiced, n):
    """Analysis pitch marks at local-period spacing across the signal."""
    out = []
    i = 0.0
    while i < n - 1:
        ii = int(i)
        if voiced[ii] and f0s[ii] > 0:
            p = float(np.clip(sr / f0s[ii], 40.0, 900.0))
        else:
            p = sr / 150.0
        out.append(ii)
        i += p
    if len(out) < 2:
        out = [0, max(1, n - 1)]
    return np.array(out, dtype=np.int64)


def _voiced_energy(x, sr, voiced):
    return one_pole_lp_fft(np.abs(x) * (voiced.astype(np.float64)), sr, 24.0)


def _split_syllables(x, sr, track, nsyl):
    """Split a word into ``nsyl`` (start, end) syllable spans at vowel nuclei."""
    n = x.size
    f0s, voiced = track.to_per_sample(n)
    if nsyl <= 1:
        return [(0, n)], (f0s, voiced)
    e = _voiced_energy(x, sr, voiced)
    emax = float(e.max()) or 1.0
    mind = int(0.07 * sr)
    peaks = []
    for i in range(1, e.size - 1):
        if e[i] > e[i - 1] and e[i] >= e[i + 1] and e[i] > 0.22 * emax:
            if not peaks or i - peaks[-1] >= mind:
                peaks.append(i)
            elif e[i] > e[peaks[-1]]:
                peaks[-1] = i
    if len(peaks) > nsyl:
        peaks = sorted(sorted(peaks, key=lambda p: -e[p])[:nsyl])
    if len(peaks) < nsyl:                                   # fallback: even split
        vi = np.where(voiced)[0]
        lo, hi = (vi[0], vi[-1]) if vi.size >= 2 else (0, n)
        spans = [(int(lo + (hi - lo) * k / nsyl),
                  int(lo + (hi - lo) * (k + 1) / nsyl)) for k in range(nsyl)]
        return spans, (f0s, voiced)
    bounds = [0]
    for a, b in zip(peaks[:-1], peaks[1:]):
        bounds.append(a + int(np.argmin(e[a:b])))
    bounds.append(n)
    return [(bounds[k], bounds[k + 1]) for k in range(nsyl)], (f0s, voiced)


def _vowel_region(x, sr, span, voiced):
    s0, s1 = span
    seg = np.abs(x[s0:s1]) * voiced[s0:s1]
    if seg.size < 8:
        return s0, s1
    e = one_pole_lp_fft(seg, sr, 45.0)
    thr = 0.4 * (float(e.max()) or 1.0)
    idx = np.where(e > thr)[0]
    if idx.size == 0:
        return s0, s1
    return s0 + int(idx[0]), s0 + int(idx[-1]) + 1


def _render_syllable(x, sr, marks, f0s, voiced, span, note_dur, tgt_f0,
                     prev_f0, glide_ms, coda):
    """PSOLA-warp one syllable's audio onto a note of ``note_dur`` at ``tgt_f0``."""
    s0, s1 = span
    v0, v1 = _vowel_region(x, sr, span, voiced)
    out_len = max(int(note_dur * sr), int(0.10 * sr))

    onset_out = min(max(0, v0 - s0), int(0.13 * sr))
    coda_out = min(max(0, s1 - v1), int(0.12 * sr)) if coda else 0
    sustain_out = max(1, out_len - onset_out - coda_out)

    # input-time trajectory: onset 1:1, then slow ping-pong through the steady
    # middle of the vowel (sustain), then coda 1:1.
    ti = np.empty(out_len)
    if onset_out > 0:
        ti[:onset_out] = np.linspace(s0, v0, onset_out)
    vs0 = v0 + 0.18 * (v1 - v0)
    vs1 = v1 - 0.18 * (v1 - v0)
    if vs1 <= vs0 + 1:
        vs0, vs1 = v0, max(v0 + 1, v1)
    st = np.arange(sustain_out)
    tri = 0.5 - 0.5 * np.cos(2.0 * np.pi * 0.8 * st / sr)   # slow scan 0..1..0
    ti[onset_out:onset_out + sustain_out] = vs0 + (vs1 - vs0) * tri
    if coda_out > 0:
        ti[onset_out + sustain_out:] = np.linspace(v1, s1, out_len - onset_out - sustain_out)

    tgt = np.full(out_len, tgt_f0)
    if prev_f0 > 0 and glide_ms > 0:
        g = min(int(glide_ms / 1000.0 * sr), out_len // 2)
        if g > 1:
            tgt[:g] = prev_f0 * (tgt_f0 / prev_f0) ** np.linspace(0.0, 1.0, g)

    pad = 1024
    out = np.zeros(out_len + pad)
    nrm = np.zeros(out_len + pad)
    po = 0.0
    while po < out_len:
        poi = int(po)
        tii = float(ti[min(poi, out_len - 1)])
        si = int(np.clip(tii, 0, x.size - 1))
        ps = float(np.clip(sr / f0s[si], 40.0, 900.0)) if (voiced[si] and f0s[si] > 0) else sr / 150.0
        half = max(2, int(round(ps)))
        k = int(np.searchsorted(marks, tii))
        if k >= marks.size:
            k = marks.size - 1
        elif k > 0 and abs(marks[k - 1] - tii) < abs(marks[k] - tii):
            k -= 1
        a = int(marks[k])
        g0, g1 = a - half, a + half
        lo, hi = max(0, g0), min(x.size, g1)
        grain = np.zeros(2 * half)
        grain[lo - g0:hi - g0] = x[lo:hi]
        w = _hann(2 * half)
        grain *= w
        d0 = poi - half
        dlo, dhi = max(0, d0), min(out.size, d0 + 2 * half)
        if dhi > dlo:
            out[dlo:dhi] += grain[dlo - d0:dhi - d0]
            nrm[dlo:dhi] += w[dlo - d0:dhi - d0]
        po += float(np.clip(sr / tgt[min(poi, out_len - 1)], 40.0, 900.0))

    m = nrm > 1e-6
    out[m] /= nrm[m]
    note = out[:out_len]

    # Gentle AGC over the sustain so held notes stay level (not decaying/pumping).
    env = one_pole_lp_fft(np.abs(note), sr, 26.0)
    emax = float(env.max()) or 1.0
    ref = np.median(env[env > 0.35 * emax]) if np.any(env > 0.35 * emax) else emax
    gain = np.clip(ref / (env + 1e-4), 0.6, 1.8)
    gain = np.where(env > 0.18 * emax, gain, 1.0)
    note = note * one_pole_lp_fft(gain, sr, 12.0)
    return note.astype(np.float32)


def _render_syllable_hnm(ana, sr, span, vowel, note_dur, tgt_f0, prev_f0,
                         glide_ms, hop, rng):
    """Render one note by harmonic+noise resynthesis (harmonic.py): hold the
    vowel's envelope frames while synthesizing clean sinusoids at the target
    pitch (with glide + vibrato baked straight into the synthesis frequency)."""
    s0, s1 = span
    v0, v1 = vowel
    nf = ana["env"].shape[0]
    out_len = max(int(note_dur * sr), int(0.10 * sr))
    onset_out = min(max(0, v0 - s0), int(0.13 * sr))
    coda_out = min(max(0, s1 - v1), int(0.12 * sr))
    sustain_out = max(1, out_len - onset_out - coda_out)

    fof = np.empty(out_len)
    if onset_out > 0:
        fof[:onset_out] = np.linspace(s0 / hop, v0 / hop, onset_out)
    fv0, fv1 = v0 / hop, v1 / hop
    a = fv0 + 0.2 * (fv1 - fv0)
    b = fv1 - 0.2 * (fv1 - fv0)
    if b <= a:
        a, b = fv0, max(fv0 + 0.5, fv1)
    st = np.arange(sustain_out)
    tri = 0.5 - 0.5 * np.cos(2.0 * np.pi * 0.7 * st / sr)
    fof[onset_out:onset_out + sustain_out] = a + (b - a) * tri
    if coda_out > 0:
        fof[onset_out + sustain_out:] = np.linspace(fv1, s1 / hop,
                                                    out_len - onset_out - sustain_out)
    fof = np.clip(fof, 0, nf - 1)

    t = np.arange(out_len)
    f0 = np.full(out_len, float(tgt_f0))
    if prev_f0 > 0 and glide_ms > 0:
        g = min(int(glide_ms / 1000.0 * sr), out_len // 2)
        if g > 1:
            f0[:g] = prev_f0 * (tgt_f0 / prev_f0) ** np.linspace(0.0, 1.0, g)
    # Vibrato (swells in) + flutter, straight into the synthesis frequency.
    venv = np.clip((t / sr - 0.28) / 0.3, 0.0, 1.0)
    cents = 33.0 * venv * np.sin(2.0 * np.pi * 5.7 * t / sr)
    fl = rng.standard_normal(out_len)
    w = max(1, int(sr * 0.13))
    fl = np.convolve(fl, np.ones(w) / w, mode="same")
    cents += 5.0 * (fl / (np.std(fl) or 1.0))
    f0 = f0 * 2.0 ** (cents / 1200.0)
    return harmonic.resynth(ana, fof, f0, sr)


def _read_cubic(x, read):
    n = x.size
    i = np.floor(read).astype(np.int64)
    f = read - i

    def tap(o):
        return x[np.clip(i + o, 0, n - 1)]

    p0, p1, p2, p3 = tap(-1), tap(0), tap(1), tap(2)
    return p1 + 0.5 * f * (p2 - p0 + f * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
                          + f * (3.0 * (p1 - p2) + p3 - p0)))


def _expression(x, sr, rng, vib_rate=5.7, vib_depth=32.0, vib_delay=0.28,
                flutter_cents=5.0):
    """Vibrato that swells in + tiny random pitch flutter, applied as exact
    pitch modulation via a resampling read-index (click-free)."""
    n = x.size
    t = np.arange(n)
    cents = np.zeros(n)
    if vib_depth > 0:
        env = np.clip((t / sr - vib_delay) / 0.3, 0.0, 1.0)
        cents += vib_depth * env * np.sin(2.0 * np.pi * vib_rate * t / sr)
    if flutter_cents > 0:
        noise = rng.standard_normal(n)
        w = max(1, int(sr * 0.13))
        noise = np.convolve(noise, np.ones(w) / w, mode="same")
        s = np.std(noise)
        cents += flutter_cents * (noise / s if s > 1e-9 else noise)
    ratio = 2.0 ** (cents / 1200.0) - 1.0
    ratio -= float(np.mean(ratio))
    read = np.cumsum(1.0 + ratio)
    read = np.clip(read - read[0], 0.0, n - 1)
    y = _read_cubic(x.astype(np.float64), read)
    # Real vibrato modulates loudness a little too (tremolo coupled to pitch).
    if vib_depth > 0:
        env = np.clip((t / sr - vib_delay) / 0.3, 0.0, 1.0)
        y = y * (1.0 + 0.05 * env * np.sin(2.0 * np.pi * vib_rate * t / sr + 0.6))
    return y.astype(np.float32)


def _deess(x, sr, amount=0.6):
    """Tame sibilants ('s'/'sh') so the air boost doesn't get harsh."""
    hi = biquad_fft(highpass(6500.0, sr, 0.7), x)
    lo = np.asarray(x, dtype=np.float64) - hi
    env = one_pole_lp_fft(np.abs(hi), sr, 55.0)
    nz = env[env > 1e-5]
    thr = 1.6 * float(np.median(nz)) if nz.size else 1.0
    gain = one_pole_lp_fft(np.clip(thr / (env + 1e-6), 1.0 - amount, 1.0), sr, 80.0)
    return (lo + hi * gain).astype(np.float32)


def _voice_timbre(x, sr):
    """Shape espeak's thin/buzzy tone toward a natural sung voice (frequencies
    from espeak's measured spectrum): low-mid warmth, a 3 kHz singer's-formant
    ring to fill the dip, tame the ~4 kHz buzz, and gentle air on top."""
    chain = [
        highpass(95.0, sr, 0.7),
        low_shelf(330.0, sr, 4.0),           # warmth / body
        peaking(2900.0, sr, 1.4, 3.0),       # singer's-formant ring (fills dip)
        peaking(4200.0, sr, 2.4, -3.5),      # tame the harsh buzz
        high_shelf(8500.0, sr, 2.5),         # air
    ]
    return biquad_fft(chain, x)


def sing(score, sr=44100, bpm=100, voice="en+f4", base_pitch=64, wpm=150,
         glide_ms=55.0, crossfade_ms=22.0, seed=5, engine="psola"):
    """Render a word-based score to a continuous sung mono line.

    Score items: ``("rest", beats)`` or ``(word_text, [(note, beats), ...])``
    where the note list has one entry per syllable of the word.  ``engine`` picks
    the synthesiser: ``"psola"`` (warp espeak's waveform) or ``"hnm"`` (harmonic +
    noise resynthesis -- cleaner, more synthetic-diva tone)."""
    beat = 60.0 / bpm
    xf = int(crossfade_ms / 1000.0 * sr)
    rng = np.random.default_rng(seed)

    # Render every note into a record dict (audio + timing + musical context).
    items = []
    t = 0.0
    prev_f0 = 0.0
    phrase_id = 0
    hum = np.random.default_rng(seed + 1)
    for it in score:
        if it[0] == "rest":
            t += it[1] * beat
            if prev_f0 != 0.0:
                phrase_id += 1
            prev_f0 = 0.0
            continue
        word, notes = it
        raw = tts.text_to_vocal(word, sr, voice=voice, pitch=base_pitch, wpm=wpm)
        raw = util.normalize_peak(raw, 0.9)
        track = pitch.track_pitch(raw, sr, max_f0=1000.0)
        f0s, voiced = track.to_per_sample(raw.size)
        marks = _marks(sr, f0s, voiced, raw.size)
        spans, _ = _split_syllables(raw, sr, track, len(notes))
        ana = harmonic.analyze(raw, sr) if engine == "hnm" else None
        for k, ((note, beats), span) in enumerate(zip(notes, spans)):
            midi = note_to_midi(note)
            f0 = midi_to_freq(midi)
            dur = beats * beat
            if engine == "hnm":
                vowel = _vowel_region(raw, sr, span, voiced)
                a = _render_syllable_hnm(ana, sr, span, vowel, dur + xf / sr, f0,
                                         prev_f0, glide_ms, ana["hop"],
                                         np.random.default_rng(int(midi * 97 + k)))
            else:
                # Keep the consonant tail on every syllable (clearer articulation).
                a = _render_syllable(raw, sr, marks, f0s, voiced, span,
                                     dur + xf / sr, f0, prev_f0, glide_ms, coda=True)
                a = _expression(a, sr, np.random.default_rng(int(midi * 97 + a.size)))
            new_phrase = prev_f0 == 0.0
            jitter = 0 if new_phrase else int(hum.normal(0.0, 0.004) * sr)
            items.append({"a": a, "pos": max(0, int(t * sr) + jitter),
                          "midi": midi, "phrase": phrase_id, "start": new_phrase})
            t += dur
            prev_f0 = f0

    # Musical dynamics: build a loudness per note (higher notes and the phrase
    # peak sing louder), grouped by phrase, and mark phrase ends.
    for i, item in enumerate(items):
        item["end"] = (i + 1 == len(items)) or (items[i + 1]["phrase"] != item["phrase"])
    by_phrase = {}
    for i, item in enumerate(items):
        by_phrase.setdefault(item["phrase"], []).append(i)
    gains = [1.0] * len(items)
    for idxs in by_phrase.values():
        mids = [items[i]["midi"] for i in idxs]
        pmin, pmax = min(mids), max(mids)
        for j, i in enumerate(idxs):
            pn = (items[i]["midi"] - pmin) / (pmax - pmin) if pmax > pmin else 0.5
            arc = np.sin(np.pi * (j + 0.5) / len(idxs))
            gains[i] = 0.70 + 0.16 * pn + 0.16 * arc

    out = np.zeros(int(t * sr) + sr)
    ramp = np.linspace(0.0, 1.0, xf)
    for i, item in enumerate(items):
        a = item["a"].copy() * gains[i]
        if not item["start"]:
            a[:xf] *= np.sin(0.5 * np.pi * ramp)
        if not item["end"]:
            a[-xf:] *= np.cos(0.5 * np.pi * ramp)
        else:
            rel = min(int(0.16 * sr), a.size)
            a[-rel:] *= np.linspace(1.0, 0.0, rel) ** 1.3
        pos = item["pos"]
        e = min(out.size, pos + a.size)
        out[pos:e] += a[:e - pos]
    return out[:int(t * sr) + int(0.3 * sr)].astype(np.float32)


def render_song(score, sr=44100, bpm=100, voice="en+f4", base_pitch=64,
                reverb_mix=0.19, width=1.2, engine="psola"):
    """Full render: sing -> timbre-shape -> stereo -> reverb."""
    dry = sing(score, sr, bpm, voice, base_pitch, engine=engine)
    dry = _voice_timbre(dry, sr)
    dry = _deess(dry, sr)
    dry = util.normalize_peak(dry, 0.92)
    stereo = effects.stereoize(dry, sr, haas_ms=9.0, width=width)
    stereo = effects.reverb(stereo, sr, mix=reverb_mix, size=0.7, damp=0.48,
                            width=1.2)
    stereo = util.normalize_percentile(stereo, target=0.85)
    stereo = util.soft_limit(stereo, 0.98)
    return stereo

