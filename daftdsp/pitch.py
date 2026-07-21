"""
Monophonic pitch tracking via the YIN algorithm (de Cheveigne & Kawahara, 2002),
implemented from scratch, plus musical scale quantisation.

Pipeline per frame:
  1. squared-difference function d(tau)
  2. cumulative mean normalised difference (CMND)
  3. absolute-threshold pitch pick with a local-minimum refine
  4. parabolic interpolation for sub-sample period accuracy
The CMND value at the chosen lag doubles as an aperiodicity / voicing score.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .util import quantize_freq_to_scale


@dataclass
class PitchTrack:
    hop: int
    frame_centers: np.ndarray   # sample index of each frame centre
    f0: np.ndarray              # Hz per frame (0 = unvoiced)
    voiced: np.ndarray          # bool per frame
    sr: int

    def to_per_sample(self, n: int):
        """Interpolate the frame-rate f0/voicing to one value per sample."""
        centers = self.frame_centers
        idx = np.arange(n)
        f0_s = np.interp(idx, centers, self.f0)
        v_s = np.interp(idx, centers, self.voiced.astype(np.float64)) > 0.5
        return f0_s, v_s


def _yin_frame(frame, sr, tau_min, tau_max, threshold):
    n = frame.size - tau_max
    if n <= 0:
        return 0.0, 1.0
    x0 = frame[:n]
    diff = np.empty(tau_max + 1, dtype=np.float64)
    diff[0] = 0.0
    for tau in range(1, tau_max + 1):
        d = x0 - frame[tau:tau + n]
        diff[tau] = np.dot(d, d)

    # Cumulative mean normalised difference.
    cmnd = np.empty(tau_max + 1, dtype=np.float64)
    cmnd[0] = 1.0
    running = 0.0
    for tau in range(1, tau_max + 1):
        running += diff[tau]
        cmnd[tau] = diff[tau] * tau / running if running > 0 else 1.0

    # Absolute threshold: first dip below threshold, refined to its local min.
    tau_est = -1
    tau = tau_min
    while tau < tau_max:
        if cmnd[tau] < threshold:
            while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            tau_est = tau
            break
        tau += 1
    if tau_est == -1:
        tau_est = tau_min + int(np.argmin(cmnd[tau_min:tau_max]))
    aper = float(cmnd[tau_est])

    # Parabolic interpolation around the chosen lag.
    if 1 <= tau_est < tau_max:
        a, b, c = cmnd[tau_est - 1], cmnd[tau_est], cmnd[tau_est + 1]
        denom = 2.0 * (a - 2.0 * b + c)
        shift = (a - c) / denom if denom != 0 else 0.0
        tau_ref = tau_est + np.clip(shift, -1.0, 1.0)
    else:
        tau_ref = float(tau_est)

    f0 = sr / tau_ref if tau_ref > 0 else 0.0
    return f0, aper


def track_pitch(x, sr, *, min_f0=70.0, max_f0=500.0, hop=256, frame=2048,
                threshold=0.15, voiced_energy=1e-4):
    """Run YIN across ``x`` and return a :class:`PitchTrack`."""
    x = np.asarray(x, dtype=np.float64)
    tau_min = max(2, int(sr / max_f0))
    tau_max = min(frame - 1, int(sr / min_f0))
    win = tau_max + tau_min + 1
    if win < frame:
        win = frame

    centers, f0s, voiced = [], [], []
    pos = 0
    while pos + win <= x.size:
        seg = x[pos:pos + win]
        energy = float(np.mean(seg * seg))
        if energy < voiced_energy:
            f0, aper = 0.0, 1.0
        else:
            f0, aper = _yin_frame(seg, sr, tau_min, tau_max, threshold)
        is_voiced = (aper < threshold * 1.6) and (min_f0 <= f0 <= max_f0) \
            and (energy >= voiced_energy)
        centers.append(pos + win // 2)
        f0s.append(f0 if is_voiced else 0.0)
        voiced.append(is_voiced)
        pos += hop

    if not centers:
        centers = [0, max(1, x.size - 1)]
        f0s = [0.0, 0.0]
        voiced = [False, False]

    f0s = _median_smooth(np.array(f0s), 5)
    return PitchTrack(hop, np.array(centers), f0s, np.array(voiced, dtype=bool), sr)


def _median_smooth(a, k):
    """Median filter that ignores zeros (unvoiced) when smoothing f0."""
    if a.size == 0:
        return a
    out = a.copy()
    half = k // 2
    for i in range(a.size):
        lo, hi = max(0, i - half), min(a.size, i + half + 1)
        window = a[lo:hi]
        nz = window[window > 0]
        if a[i] > 0 and nz.size:
            out[i] = np.median(nz)
    return out


def quantize_track(f0, key_root, scale):
    """Snap every voiced f0 in a track to ``scale`` (rooted at ``key_root``)."""
    out = np.zeros_like(f0)
    for i, f in enumerate(f0):
        out[i] = quantize_freq_to_scale(f, key_root, scale) if f > 0 else 0.0
    return out
