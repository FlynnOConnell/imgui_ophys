"""
d' (the sensitivity index) for one ROI: how far a single spike's fluorescence stands above the baseline noise, in
units of that noise. The form is from the jGCaMP8 paper (Zhang et al. 2023, Methods), which takes it from signal
detection theory (McNicol 2004):

    d' = (mean(F_top) - mean(F_bottom)) / sqrt(0.5 * (var(F_top) + var(F_bottom)))

F_top is a short window of frames at the response peak, F_bottom a window of the same length just before the
spike. The windows of every isolated spike are pooled, so d' is one number per cell. An ideal observer with the
threshold halfway between the two means calls a single spike correctly with probability Phi(d' / 2): d' = 1 gives
69%, d' = 2 84%, d' = 3 93%, d' = 4 98%.

The same paper's in vivo form, d' = mean(A) / sqrt(0.5 * var(A)) with A the isolated single-spike amplitudes
(top minus bottom), agrees in expectation: var(A) is var(top) + var(bottom) when the two windows are independent.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from imgui_ophys import sim

QUIET_DECAYS = 4.0  # an event is isolated when no spike fell within this many decay time constants before it


@dataclass
class Score:
    """d' for one trace, with the windows it came from: top and bottom are (events, frames)."""

    d: float
    top: np.ndarray
    bottom: np.ndarray
    spikes: np.ndarray  # s, the isolated spikes used
    centers: np.ndarray  # frame of each spike
    peak: int  # frames from a spike to its response peak

    @property
    def n(self) -> int:
        return int(self.spikes.size)

    @property
    def amplitudes(self) -> np.ndarray:
        """A per event: mean of its top window minus mean of its bottom window."""
        if self.n == 0:
            return np.zeros(0)
        return self.top.mean(axis=1) - self.bottom.mean(axis=1)

    @property
    def d_amplitudes(self) -> float:
        """The in vivo form: mean(A) / sqrt(0.5 var(A)). NaN with fewer than two events."""
        a = self.amplitudes
        if a.size < 2 or a.var() == 0:
            return math.nan
        return float(a.mean() / math.sqrt(0.5 * a.var()))


def dprime(top: np.ndarray, bottom: np.ndarray) -> float:
    """(mean top - mean bottom) / pooled SD. NaN when a pool is empty or both are constant."""
    top, bottom = np.ravel(top), np.ravel(bottom)
    if top.size == 0 or bottom.size == 0:
        return math.nan
    pooled = math.sqrt(0.5 * (top.var() + bottom.var()))
    if pooled == 0:
        return math.nan
    return float((top.mean() - bottom.mean()) / pooled)


def accuracy(d: float) -> float:
    """Phi(d / 2): the fraction of single spikes an ideal threshold between the two means gets right."""
    if math.isnan(d):
        return math.nan
    return 0.5 * (1 + math.erf(d / 2 / math.sqrt(2)))


def peak_frames(indicator: sim.Indicator, fs: float) -> int:
    """Frames from a spike to the peak of the indicator's single-spike response."""
    dt = 1e-3
    k = sim.calcium_kernel(indicator.rise, indicator.decay, dt)
    return int(round(float(np.argmax(k)) * dt * fs))


def quiet_before(indicator: sim.Indicator) -> float:
    """s a spike must have had to itself beforehand, so the last response has decayed out of its baseline window."""
    return max(0.5, QUIET_DECAYS * indicator.decay)


def isolated(spikes: np.ndarray, before: float, after: float) -> np.ndarray:
    """The spikes with no other spike within before s earlier or after s later."""
    if spikes.size < 2:
        return spikes
    gaps = np.diff(spikes)
    alone = np.ones(spikes.size, bool)
    alone[1:] &= gaps >= before
    alone[:-1] &= gaps >= after
    return spikes[alone]


def score(f: np.ndarray, spikes: np.ndarray, fs: float, peak: int, frames: int, before: float) -> Score:
    """
    d' of f (any per-frame fluorescence: photons or dF/F, the ratio does not care) from the spikes.

    Per isolated spike at frame c: bottom is the frames frames before c, top the frames frames from c + peak on.
    A spike counts as isolated with no spike within before s earlier and none until its top window has passed.
    Events whose windows run off the trace are dropped.
    """
    used, tops, bottoms, centers = [], [], [], []
    for s in isolated(spikes, before, (peak + frames) / fs):
        c = int(round(s * fs))
        if c - frames < 0 or c + peak + frames > f.size:
            continue
        bottoms.append(f[c - frames : c])
        tops.append(f[c + peak : c + peak + frames])
        used.append(s)
        centers.append(c)
    top = np.array(tops) if tops else np.zeros((0, frames))
    bottom = np.array(bottoms) if bottoms else np.zeros((0, frames))
    return Score(dprime(top, bottom), top, bottom, np.array(used), np.array(centers, dtype=int), peak)


def score_params(p: sim.TraceParams, frames: int, trace: Optional[sim.Trace] = None) -> Score:
    """Simulate p (or take trace) and score it on the recorded photons with the true spike times."""
    tr = sim.make_trace(p) if trace is None else trace
    ind = sim.BY_NAME[p.indicator]
    return score(tr.recorded, tr.spikes, p.fs, peak_frames(ind, p.fs), frames, quiet_before(ind))


def noise_rows(f: np.ndarray, spikes: np.ndarray, fs: float, peak: int, frames: int, before: float, targets: tuple, seed: int = 7) -> list:
    """
    (d', trace) per target: one draw of white noise added to f, scaled by bisection until d' lands on the target;
    f itself comes last. A target above f's own d' cannot be reached.

    Raises
    ------
    ValueError
        If a target exceeds f's d'.
    """
    d_clean = score(f, spikes, fs, peak, frames, before).d
    z = np.random.default_rng(seed).standard_normal(f.size)
    rows = []
    for target in targets:
        if not target < d_clean:
            raise ValueError(f"d' = {target} is not below the clean trace's {d_clean:.2f}")
        lo, hi = 0.0, float(np.std(f))
        while score(f + hi * z, spikes, fs, peak, frames, before).d > target:
            hi *= 2.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if score(f + mid * z, spikes, fs, peak, frames, before).d > target:
                lo = mid
            else:
                hi = mid
        noisy = f + 0.5 * (lo + hi) * z
        rows.append((score(noisy, spikes, fs, peak, frames, before).d, noisy))
    rows.append((d_clean, f))
    return rows
