"""
Harmonic + noise resynthesis (a from-scratch WORLD/STRAIGHT-style vocoder).

The idea that lifts quality past espeak's buzzy tone: separate *what vowel* is
being said (the spectral envelope = formants) from *the source* (pitch + glottal
buzz), then rebuild the voice from clean, phase-dispersed **sinusoids** at the
target melodic pitch, shaped by that envelope, plus a **noise** band for
breath/consonants.

  * ``analyze`` -- per-frame smooth spectral envelope via cepstral liftering,
    plus voicing and energy.
  * ``resynth`` -- continuous overlap-free block synthesis: harmonics with
    carried phase (no clicks), amplitudes read from the (time-warped) envelope,
    so a held note is a sustained, non-decaying, non-buzzy vowel.

Only numpy FFTs are used -- no DSP libraries.
"""
from __future__ import annotations

import numpy as np

from . import pitch


def _hann(n):
    n = max(2, int(n))
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def analyze(x, sr, frame_ms=32.0, hop_ms=5.0):
    """Return per-frame linear spectral envelopes + voicing + energy.

    envelope[f] is a smooth magnitude spectrum (harmonics removed by cepstral
    liftering) -- the vocal-tract / formant shape at frame f."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    W = max(64, int(frame_ms / 1000.0 * sr))
    H = max(1, int(hop_ms / 1000.0 * sr))
    nfft = 1 << int(np.ceil(np.log2(W)))
    win = _hann(W)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)

    track = pitch.track_pitch(x, sr, max_f0=1000.0)
    f0s, voiced = track.to_per_sample(n)
    med_f0 = float(np.median(f0s[f0s > 0])) if np.any(f0s > 0) else 200.0
    # Cepstral lifter cutoff: keep quefrencies below the pitch peak (sr/f0) so
    # the harmonic ripple is removed but the formants survive.
    lifter = int(np.clip(0.42 * sr / max(90.0, med_f0), 30, 260))

    starts = list(range(0, max(1, n - W), H))
    env = np.empty((len(starts), freqs.size))
    vflag = np.empty(len(starts))
    energy = np.empty(len(starts))
    for i, s in enumerate(starts):
        seg = x[s:s + W] * win
        energy[i] = np.sqrt(np.mean(seg * seg) + 1e-12)
        mag = np.abs(np.fft.rfft(seg, nfft)) + 1e-7
        logmag = np.log(mag)
        cep = np.fft.irfft(logmag, nfft)
        cep[lifter:nfft - lifter + 1] = 0.0          # low-quefrency lifter
        env[i] = np.exp(np.fft.rfft(cep, nfft).real)
        c = s + W // 2
        vflag[i] = 1.0 if (c < n and voiced[c]) else 0.0
    return {"env": env, "freqs": freqs, "hop": H, "voiced": vflag,
            "energy": energy, "n": n, "sr": sr}


def _harmonic(ana, frame_of_out, f0_out, sr, block=128, seed=1):
    """Voiced part: phase-dispersed sinusoids at the target pitch, amplitudes
    read from the (time-warped) envelope.  Phase is carried across blocks so the
    tone is continuous and click-free."""
    env = ana["env"]
    freqs = ana["freqs"]
    voiced = ana["voiced"]
    n_out = frame_of_out.size
    nyq = sr * 0.5
    fmin = float(np.min(f0_out[f0_out > 0], initial=120.0))
    max_harm = max(20, int(min(150, (nyq * 0.95) / max(60.0, fmin))))
    k = np.arange(1, max_harm + 1)
    phase0 = np.random.default_rng(seed).uniform(0, 2 * np.pi, max_harm)

    out = np.zeros(n_out + block)
    phase = phase0.copy()
    n_frames = env.shape[0]
    amp_prev = None
    for b0 in range(0, n_out, block):
        b1 = min(n_out, b0 + block)
        m = b1 - b0
        c = min((b0 + b1) // 2, n_out - 1)
        fi = int(np.clip(frame_of_out[c], 0, n_frames - 1))
        f0 = float(f0_out[c]) or 120.0
        vc = voiced[fi]
        hf = k * f0
        live = hf < nyq * 0.98
        # Natural glottal spectral tilt (~-8 dB/oct above 1.8 kHz) so the many
        # high harmonics don't accumulate into brightness/buzz.
        tilt = (1800.0 / np.maximum(hf, 1800.0)) ** 1.3
        amp = np.zeros(max_harm)
        amp[live] = np.interp(hf[live], freqs, env[fi]) * vc * tilt[live]
        if amp_prev is None:
            amp_prev = amp
        tt = np.arange(m)
        ph = phase[:, None] + (2.0 * np.pi * hf[:, None] / sr) * tt[None, :]
        ramp = np.linspace(0.0, 1.0, m)
        amp_t = amp_prev[:, None] * (1 - ramp)[None, :] + amp[:, None] * ramp[None, :]
        out[b0:b1] = (np.cos(ph) * amp_t).sum(axis=0)
        phase = (phase + 2.0 * np.pi * hf / sr * m) % (2.0 * np.pi)
        amp_prev = amp
    return out[:n_out]


def _shaped_noise(ana, frame_of_out, sr, voiced_breath=0.05, seed=2):
    """Noise part: white noise coloured by the (time-warped) envelope -- full for
    unvoiced frames (real fricatives/consonants), a whisper for voiced frames
    (breath).  Overlap-add STFT so it's smooth."""
    env = ana["env"]
    voiced = ana["voiced"]
    n = frame_of_out.size
    W = (env.shape[1] - 1) * 2
    H = W // 4
    win = _hann(W)
    rng = np.random.default_rng(seed)
    out = np.zeros(n + W)
    norm = np.zeros(n + W)
    n_frames = env.shape[0]
    for s in range(0, n, H):
        c = min(s + W // 2, n - 1)
        fi = int(np.clip(frame_of_out[c], 0, n_frames - 1))
        vc = voiced[fi]
        gain = env[fi] * ((1.0 - vc) + vc * voiced_breath)
        nz = rng.standard_normal(W) * win
        shaped = np.fft.irfft(np.fft.rfft(nz, W) * gain, W)
        out[s:s + W] += shaped * win
        norm[s:s + W] += win * win
    m = norm > 1e-9
    out[m] /= norm[m]
    return out[:n]


def resynth(ana, frame_of_out, f0_out, sr, breath=0.04, noise_level=0.08,
            block=128):
    """Harmonic + envelope-shaped noise resynthesis.  Returns mono float32 the
    length of ``frame_of_out``.

    The noise carries its own voiced/unvoiced balance from the envelope; mixing
    it in at a low ``noise_level`` keeps voiced breath a whisper while unvoiced
    consonants come through.  ``breath`` sets how much air rides on voiced notes."""
    harm = _harmonic(ana, frame_of_out, f0_out, sr, block=block)
    noise = _shaped_noise(ana, frame_of_out, sr, voiced_breath=breath)
    harm = harm / (float(np.max(np.abs(harm))) or 1.0)
    noise = noise / (float(np.max(np.abs(noise))) or 1.0)
    y = harm + noise_level * noise
    peak = float(np.max(np.abs(y))) or 1.0
    return (y / peak * 0.9).astype(np.float32)
