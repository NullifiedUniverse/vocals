"""
Channel vocoder (default 32 bands, 100 Hz - 10 kHz).

Both the modulator (voice) and carrier (synth) are split into the same log-spaced
band-pass filter bank.  Per band the voice energy is measured (full-wave rectify +
low-pass), and that envelope gates the matching carrier band; the gated carrier
bands are summed -- the classic robot-choir sound.

Two implementations of the *linear* band-pass / envelope filters are provided and
give equivalent results:

  * ``method="fft"``  (default) evaluates each biquad's exact frequency response
    H(e^jw) -- derived by hand from the coefficients -- and applies it via one
    FFT/IFFT.  No per-sample Python loop, so it is fast enough for a live server.
  * ``method="iir"``  runs the literal time-domain sample-by-sample recursion
    through :class:`~daftdsp.biquad.BandBank` (clear but slower).

``formant_shift`` scales the carrier band centres relative to the modulator bands,
moving the imposed spectral envelope independently of pitch.  A noise ``sibilance``
path restores unvoiced consonants a tonal carrier cannot reproduce.
"""
from __future__ import annotations

import numpy as np

from .biquad import BandBank, bandpass, biquad_fft, highpass, one_pole_lp_fft


def band_frequencies(n_bands=32, f_lo=100.0, f_hi=10000.0):
    """Log-spaced band centre frequencies."""
    return np.logspace(np.log10(f_lo), np.log10(f_hi), int(n_bands))


def _biquad_response(b0, b1, b2, a1, a2, omega):
    """Exact biquad frequency response for coefficient *arrays* (one per band).

    Returns an ``(n_bands, n_freq)`` complex matrix H(e^jw)."""
    z1 = np.exp(-1j * omega)[None, :]
    z2 = z1 * z1
    num = b0[:, None] + b1[:, None] * z1 + b2[:, None] * z2
    den = 1.0 + a1[:, None] * z1 + a2[:, None] * z2
    return num / den


def _bank_coeffs(freqs, sr, q):
    n = freqs.size
    b0 = np.empty(n); b1 = np.empty(n); b2 = np.empty(n)
    a1 = np.empty(n); a2 = np.empty(n)
    for k, f in enumerate(freqs):
        b0[k], b1[k], b2[k], a1[k], a2[k] = bandpass(f, sr, q)
    return b0, b1, b2, a1, a2


def _envelope_follow(rect, sr, atk_ms, rel_ms):
    """Per-band attack/release envelope follower over a ``(N, bands)`` array
    (used by the time-domain path and the sibilance detector)."""
    n = rect.shape[0]
    nb = rect.shape[1]
    aa = np.exp(-1.0 / (max(0.05, atk_ms) / 1000.0 * sr))
    ar = np.exp(-1.0 / (max(1.0, rel_ms) / 1000.0 * sr))
    env = np.zeros(nb)
    out = np.empty_like(rect)
    for i in range(n):
        x = rect[i]
        coeff = np.where(x > env, aa, ar)
        env = coeff * env + (1.0 - coeff) * x
        out[i] = env
    return out


def vocode(modulator, carrier, sr, *, n_bands=32, f_lo=100.0, f_hi=10000.0,
           band_q=7.0, attack_ms=3.0, release_ms=18.0, formant_shift=1.0,
           sibilance=0.35, level=1.0, method="fft"):
    """Run the channel vocoder and return a mono signal the length of ``carrier``."""
    n = min(len(modulator), len(carrier))
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    modulator = np.asarray(modulator[:n], dtype=np.float64)
    carrier = np.asarray(carrier[:n], dtype=np.float64)

    freqs = band_frequencies(n_bands, f_lo, f_hi)
    car_freqs = np.clip(freqs * formant_shift, 20.0, sr * 0.45)

    if method == "iir":
        out = _vocode_iir(modulator, carrier, freqs, car_freqs, sr,
                          band_q, attack_ms, release_ms)
    else:
        out = _vocode_fft(modulator, carrier, freqs, car_freqs, sr,
                          band_q, release_ms)

    # Unvoiced-consonant path: high-band voice energy modulates white noise so
    # sibilants survive the tonal carrier.
    if sibilance > 0.0:
        hp = biquad_fft(highpass(3500.0, sr, 0.7), modulator)
        s_env = one_pole_lp_fft(np.abs(hp), sr, 60.0)
        noise = np.random.default_rng(7).standard_normal(n)
        out = out[:n] + sibilance * 3.0 * s_env * noise

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1e-9:
        out = out / peak
    return (out * level).astype(np.float32)


def _vocode_fft(modulator, carrier, freqs, car_freqs, sr, band_q, release_ms):
    """Frequency-domain vocoder: apply each band's exact biquad response via FFT,
    then a one-pole low-pass on the rectified voice bands for the envelope."""
    n = modulator.size
    # Guard band so circular wrap doesn't leak, rounded up to a power of two so
    # the FFT stays fast (numpy is slow on awkward composite lengths).
    nfft = 1 << int(np.ceil(np.log2(n + 4096)))
    omega = 2.0 * np.pi * np.fft.rfftfreq(nfft)

    mb = _bank_coeffs(freqs, sr, band_q)
    cb = _bank_coeffs(car_freqs, sr, band_q)
    h_mod = _biquad_response(*mb, omega)          # (bands, nfreq)
    h_car = _biquad_response(*cb, omega)

    m_fft = np.fft.rfft(modulator, nfft)[None, :]
    c_fft = np.fft.rfft(carrier, nfft)[None, :]
    mod_bands = np.fft.irfft(m_fft * h_mod, nfft, axis=1)[:, :n]   # (bands, n)
    car_bands = np.fft.irfft(c_fft * h_car, nfft, axis=1)[:, :n]

    # Envelope = |voice band| low-passed.  One-pole LP applied in the FFT domain.
    fc = 1000.0 / max(1.0, release_ms)            # ms -> approx cutoff Hz
    a = np.exp(-2.0 * np.pi * fc / sr)
    z1 = np.exp(-1j * omega)
    h_lp = (1.0 - a) / (1.0 - a * z1)
    rect = np.abs(mod_bands)
    r_fft = np.fft.rfft(rect, nfft, axis=1)
    env = np.fft.irfft(r_fft * h_lp[None, :], nfft, axis=1)[:, :n]
    env = np.maximum(env, 0.0)

    return np.sum(car_bands * env, axis=0)


def _vocode_iir(modulator, carrier, freqs, car_freqs, sr, band_q, atk, rel):
    """Reference time-domain vocoder using the sample-by-sample biquad bank."""
    mod_bank = BandBank(freqs, sr, q=band_q)
    car_bank = BandBank(car_freqs, sr, q=band_q)
    env = _envelope_follow(np.abs(mod_bank.filter_bank(modulator)), sr, atk, rel)
    c_bands = car_bank.filter_bank(carrier)
    return np.sum(c_bands * env, axis=1)
