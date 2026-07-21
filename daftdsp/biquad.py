"""
From-scratch biquad (second-order IIR) filters.

Coefficients follow the well-known Audio-EQ ("RBJ cookbook") bilinear-transform
formulas -- these are just algebra, not a library call.  Processing uses the
Direct-Form-II-Transposed recursion, hand-written so nothing here depends on
scipy / any DSP package.

Three processors are provided:
  * :func:`df2t`        -- one biquad over a mono or multi-channel signal.
  * :func:`df2t_tv`     -- one *first-order* all-pass with per-sample (time
                           varying) coefficient, for the LFO-swept phaser.
  * :class:`BandBank`   -- N parallel band-pass biquads sharing one input,
                           vectorised across bands (the vocoder hot path).
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Coefficient calculators  (return (b0, b1, b2, a1, a2), normalised by a0)
# ---------------------------------------------------------------------------

def _wc(freq, sr):
    return TWO_PI * min(max(freq, 1.0), sr * 0.49) / sr


def lowpass(freq, sr, q=0.707):
    w0 = _wc(freq, sr)
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    b0 = (1.0 - cw) / 2.0
    b1 = 1.0 - cw
    b2 = (1.0 - cw) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cw
    a2 = 1.0 - alpha
    return _norm(b0, b1, b2, a0, a1, a2)


def highpass(freq, sr, q=0.707):
    w0 = _wc(freq, sr)
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    b0 = (1.0 + cw) / 2.0
    b1 = -(1.0 + cw)
    b2 = (1.0 + cw) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cw
    a2 = 1.0 - alpha
    return _norm(b0, b1, b2, a0, a1, a2)


def bandpass(freq, sr, q=4.0):
    """Constant 0 dB peak gain band-pass."""
    w0 = _wc(freq, sr)
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * cw
    a2 = 1.0 - alpha
    return _norm(b0, b1, b2, a0, a1, a2)


def peaking(freq, sr, q, gain_db):
    """Resonant peak / notch (used for formants)."""
    w0 = _wc(freq, sr)
    cw, sw = np.cos(w0), np.sin(w0)
    A = 10.0 ** (gain_db / 40.0)
    alpha = sw / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cw
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cw
    a2 = 1.0 - alpha / A
    return _norm(b0, b1, b2, a0, a1, a2)


def high_shelf(freq, sr, gain_db, slope=0.9):
    w0 = _wc(freq, sr)
    cw, sw = np.cos(w0), np.sin(w0)
    A = 10.0 ** (gain_db / 40.0)
    alpha = sw / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    tsa = 2.0 * np.sqrt(A) * alpha
    b0 = A * ((A + 1.0) + (A - 1.0) * cw + tsa)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cw)
    b2 = A * ((A + 1.0) + (A - 1.0) * cw - tsa)
    a0 = (A + 1.0) - (A - 1.0) * cw + tsa
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cw)
    a2 = (A + 1.0) - (A - 1.0) * cw - tsa
    return _norm(b0, b1, b2, a0, a1, a2)


def low_shelf(freq, sr, gain_db, slope=0.9):
    w0 = _wc(freq, sr)
    cw, sw = np.cos(w0), np.sin(w0)
    A = 10.0 ** (gain_db / 40.0)
    alpha = sw / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    tsa = 2.0 * np.sqrt(A) * alpha
    b0 = A * ((A + 1.0) - (A - 1.0) * cw + tsa)
    b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cw)
    b2 = A * ((A + 1.0) - (A - 1.0) * cw - tsa)
    a0 = (A + 1.0) + (A - 1.0) * cw + tsa
    a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cw)
    a2 = (A + 1.0) + (A - 1.0) * cw - tsa
    return _norm(b0, b1, b2, a0, a1, a2)


def _norm(b0, b1, b2, a0, a1, a2):
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------

def df2t(coeffs, x):
    """Direct-Form-II-Transposed biquad over a mono ``(N,)`` or ``(N, C)`` signal."""
    b0, b1, b2, a1, a2 = coeffs
    x = np.asarray(x, dtype=np.float64)
    mono = x.ndim == 1
    if mono:
        x = x[:, None]
    n, c = x.shape
    y = np.empty_like(x)
    s1 = np.zeros(c)
    s2 = np.zeros(c)
    for i in range(n):
        xi = x[i]
        yi = b0 * xi + s1
        s1 = b1 * xi - a1 * yi + s2
        s2 = b2 * xi - a2 * yi
        y[i] = yi
    return y[:, 0].astype(np.float32) if mono else y.astype(np.float32)


def df2t_chain(coeff_list, x):
    """Apply a list of biquads in series (time-domain recursion)."""
    for c in coeff_list:
        x = df2t(c, x)
    return x


def _biquad_H(coeffs, omega):
    """Exact transfer function H(e^jw) of one biquad, sampled at ``omega``."""
    b0, b1, b2, a1, a2 = coeffs
    z1 = np.exp(-1j * omega)
    z2 = z1 * z1
    return (b0 + b1 * z1 + b2 * z2) / (1.0 + a1 * z1 + a2 * z2)


def biquad_fft(coeffs, x, pad=2048):
    """Apply one biquad via its exact transfer function and an FFT.

    Mathematically identical to :func:`df2t` for an LTI biquad, but with no
    per-sample Python loop -- the fast path for longer signals.  Accepts mono
    ``(N,)`` or multi-channel ``(N, C)``.  ``coeffs`` may also be a *list* of
    biquads, whose responses are multiplied so a whole chain costs one FFT.
    """
    x = np.asarray(x, dtype=np.float64)
    mono = x.ndim == 1
    if mono:
        x = x[:, None]
    n = x.shape[0]
    nfft = 1 << int(np.ceil(np.log2(n + pad)))
    omega = 2.0 * np.pi * np.fft.rfftfreq(nfft)
    if isinstance(coeffs, list):
        h = np.ones(omega.size, dtype=np.complex128)
        for c in coeffs:
            h = h * _biquad_H(c, omega)
    else:
        h = _biquad_H(coeffs, omega)
    xf = np.fft.rfft(x, nfft, axis=0)
    y = np.fft.irfft(xf * h[:, None], nfft, axis=0)[:n]
    return y[:, 0].astype(np.float32) if mono else y.astype(np.float32)


def one_pole_lp_fft(x, sr, cutoff_hz, pad=2048, axis=0):
    """One-pole low-pass (magnitude) applied via FFT along ``axis``.  Used for
    envelope smoothing without a per-sample loop."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[axis]
    nfft = 1 << int(np.ceil(np.log2(n + pad)))
    omega = 2.0 * np.pi * np.fft.rfftfreq(nfft)
    a = np.exp(-2.0 * np.pi * max(1.0, cutoff_hz) / sr)
    h = (1.0 - a) / (1.0 - a * np.exp(-1j * omega))
    xf = np.fft.rfft(x, nfft, axis=axis)
    shape = [1] * x.ndim
    shape[axis] = h.size
    y = np.fft.irfft(xf * h.reshape(shape), nfft, axis=axis)
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, n)
    return y[tuple(sl)]


def first_order_allpass_tv(x, g):
    """First-order all-pass with a *time-varying* coefficient ``g`` (per sample).

    Transfer function  H(z) = (-g + z^-1) / (1 - g z^-1).  ``g`` is an array the
    same length as ``x``; this powers the LFO-swept phaser.
    """
    x = np.asarray(x, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    y = np.empty_like(x)
    xm1 = 0.0
    ym1 = 0.0
    for i in range(x.size):
        xi = x[i]
        yi = -g[i] * xi + xm1 + g[i] * ym1
        y[i] = yi
        xm1 = xi
        ym1 = yi
    return y.astype(np.float32)


class BandBank:
    """A bank of parallel 2nd-order band-pass filters sharing a single input.

    All bands are advanced together inside one time loop (vectorised across the
    band axis), which is dramatically faster than filtering the signal N separate
    times.  Also exposes an integrated one-pole envelope follower so the vocoder
    can extract per-band energy in the same pass.
    """

    def __init__(self, freqs, sr, q=6.0):
        freqs = np.asarray(freqs, dtype=np.float64)
        b0 = np.empty(freqs.size)
        b1 = np.empty(freqs.size)
        b2 = np.empty(freqs.size)
        a1 = np.empty(freqs.size)
        a2 = np.empty(freqs.size)
        for k, f in enumerate(freqs):
            b0[k], b1[k], b2[k], a1[k], a2[k] = bandpass(f, sr, q)
        self.b0, self.b1, self.b2, self.a1, self.a2 = b0, b1, b2, a1, a2
        self.n_bands = freqs.size

    def filter_bank(self, x):
        """Return ``(N, n_bands)`` -- ``x`` passed through every band-pass."""
        x = np.asarray(x, dtype=np.float64)
        n = x.size
        out = np.empty((n, self.n_bands))
        s1 = np.zeros(self.n_bands)
        s2 = np.zeros(self.n_bands)
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        for i in range(n):
            xi = x[i]
            yi = b0 * xi + s1
            s1 = b1 * xi - a1 * yi + s2
            s2 = b2 * xi - a2 * yi
            out[i] = yi
        return out

    def envelope_bank(self, x, sr, atk_ms, rel_ms):
        """Band-pass, full-wave rectify and attack/release smooth in a single
        pass -- the vocoder's per-band voice-energy envelope ``(N, n_bands)``."""
        x = np.asarray(x, dtype=np.float64)
        n = x.size
        out = np.empty((n, self.n_bands))
        s1 = np.zeros(self.n_bands)
        s2 = np.zeros(self.n_bands)
        env = np.zeros(self.n_bands)
        aa = np.exp(-1.0 / (max(0.05, atk_ms) / 1000.0 * sr))
        ar = np.exp(-1.0 / (max(1.0, rel_ms) / 1000.0 * sr))
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        for i in range(n):
            xi = x[i]
            yi = b0 * xi + s1
            s1 = b1 * xi - a1 * yi + s2
            s2 = b2 * xi - a2 * yi
            r = np.abs(yi)
            coeff = np.where(r > env, aa, ar)
            env = coeff * env + (1.0 - coeff) * r
            out[i] = env
        return out
