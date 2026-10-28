"""
The signal-quality page's numbers: a noise-free recording from `sim`, counted at any brightness, and what each metric
reads from it.

    nu = median |dF/F[t+1] - dF/F[t]| / sqrt(fs)       (dF/F in %; Rupprecht et al. 2021, eq. 1)
    d' = (filter output at spikes - its median elsewhere) / its SD elsewhere,   filter = the average spike
    r1 = sum x[t+1] x[t] / sum x[t]^2,  x one pixel of a movie, mean removed

Shot noise alone at phi0 photons / s gives dF/F an SD of 100 sqrt(fs / phi0) % per sample, so nu = 0.954 x 100 /
sqrt(phi0) whatever fs is (0.954 = median |a - b| / SD for two independent Gaussian samples). For a known spike
shape h in white noise of SD s per sample, d' = ||h|| / s: amplitude, duration and noise in one number. White
noise on top of light L gives r1 = r1(L) x var(L) / var(x): near the share of variance that is not shot noise when L
changes slower than a frame, and 0 +/- 1 / sqrt(T) over T frames when nothing does.
"""

from dataclasses import dataclass

import numpy as np

from imgui_ophys import sim

MEDIAN_ABS_DIFF = 0.6744897501960817 * np.sqrt(2)  # median |a - b| / SD, a and b independent Gaussian samples
DURATION = {"calcium": 120.0, "voltage": 10.0}  # s: enough isolated spikes for a stable d'
RATE = {"calcium": 0.5, "voltage": 6.0}  # spikes / s, as on the Traces page
PHOTONS = (1e3, 1e8)  # photons / s from the cell at rest: the brightness slider's range
SWEEP = np.logspace(3, 8, 26)
NU_LOW, NU_HIGH = 1.0, 8.0  # "very low" and "high" noise levels, Rupprecht et al. 2021
D_RELIABLE = 5.0  # threshold 3.7 SD: about 0.1 false spikes / s at 1 kHz, 90% of spikes found
MOTION = ((0.3, 0.2), (3.0, 0.5), (10.0, 0.3))  # (Hz, share of the amplitude): drift, breathing, heartbeat
SIDE = 20  # the movie is SIDE x SIDE pixels
# (x, y, radius px, brightness, fires): brightness relative to the cell the photon rate is for; the third is silent
CELLS = ((5, 6, 2.2, 1.0, True), (13, 5, 2.2, 1.0, True), (8, 14, 2.2, 3.0, False), (15, 14, 2.0, 0.4, True))
BACKGROUND = 0.002  # out-of-focus light per pixel, relative to a cell's total
MOVIE_FRAMES = 2000  # at most, so the movie can be recounted while a slider moves


@dataclass
class Clean:
    """A noise-free recording at fs, and which spikes are far enough apart to average into a template."""

    indicator: sim.Indicator
    fs: float
    t: np.ndarray  # sample times, s
    dff: np.ndarray  # true dF/F, %
    spikes: np.ndarray  # spike times, s
    isolated: np.ndarray  # sample index of each spike with no other within the template's span
    before: int  # template samples before the spike's sample
    after: int  # template samples after it


@dataclass
class Recording:
    """One brightness: the recorded trace and what the metrics read from it."""

    phi0: float  # photons / s from the cell at rest
    dff: np.ndarray  # recorded dF/F, %
    truth: np.ndarray  # true dF/F through the same low-pass, %
    template: np.ndarray  # mean isolated spike in truth, baseline removed, samples -before..after, %
    z: np.ndarray  # matched-filter output in SDs of its spike-free part
    free: np.ndarray  # bool: samples whose filter output no spike reaches
    spike_z: np.ndarray  # z at the isolated spikes
    nu: float
    dprime: float


@dataclass
class Field:
    """A noise-free movie of CELLS: each cell's footprint and dF/F."""

    fs: float
    t: np.ndarray  # frame times, s
    footprints: np.ndarray  # (pixels, cells), each summing to its cell's brightness
    dff: np.ndarray  # (cells, frames), true dF/F as a fraction


def make_clean(indicator: str, fs: float, rate: float, seed: int = 1) -> Clean:
    ind = sim.BY_NAME[indicator]
    tr = sim.make_trace(sim.TraceParams(indicator=indicator, fs=fs, duration=DURATION[ind.kind], rate=rate, seed=seed))
    # the template spans 1.5 decay constants of a calcium transient (95% of its energy), or a spike and its AHP
    pre, post = (0.1, 1.5 * ind.decay) if ind.kind == "calcium" else (0.002, 0.015)
    before, after = max(1, int(np.ceil(pre * fs))), max(1, int(np.ceil(post * fs)))
    frames = (tr.spikes * fs).astype(int)
    gaps = np.diff(tr.spikes)
    alone = np.ones(frames.size, bool)
    alone[1:] &= gaps > post
    alone[:-1] &= gaps > post
    inside = (frames >= before) & (frames + after < tr.dff.size)
    return Clean(ind, fs, tr.t, 100 * tr.dff, tr.spikes, frames[alone & inside], before, after)


def nu(dff: np.ndarray, fs: float) -> float:
    """Rupprecht et al. 2021, eq. 1, with dF/F in %: %·Hz^-1/2."""
    return float(np.median(np.abs(np.diff(dff))) / np.sqrt(fs))


def nu_shot(phi0: float) -> float:
    """nu of shot noise alone at phi0 photons / s."""
    return float(MEDIAN_ABS_DIFF * 100 / np.sqrt(phi0))


def dprime_white(template: np.ndarray, fs: float, phi0: float) -> float:
    """d' a matched filter reaches in white shot noise: ||h|| / SD per sample."""
    return float(np.linalg.norm(template) / (100 * np.sqrt(fs / phi0)))


def motion(t: np.ndarray, amplitude: float) -> np.ndarray:
    """Fractional change in collected light from movement and blood flow, at most amplitude % of F."""
    return amplitude / 100 * sum(share * np.sin(2 * np.pi * hz * t) for hz, share in MOTION)


def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """Zero-phase low-pass along the last axis: an order-`order` Butterworth's magnitude applied in the frequency domain."""
    n = x.shape[-1]
    f = np.fft.rfftfreq(n, 1 / fs)
    return np.fft.irfft(np.fft.rfft(x) / np.sqrt(1 + (f / cutoff) ** (2 * order)), n=n)


def matched_filter(x: np.ndarray, h: np.ndarray, before: int) -> np.ndarray:
    """y[i] = sum_k x[i - before + k] h[k]: h laid on x with its spike sample at i."""
    padded = np.pad(x, (before, h.size - 1 - before))
    return np.lib.stride_tricks.sliding_window_view(padded, h.size) @ h


def spike_free(spikes: np.ndarray, fs: float, n: int, before: int, after: int) -> np.ndarray:
    """Samples whose filter output (x[i - before .. i + after]) overlaps no spike (x[s - 1 .. s + after])."""
    frames = (spikes * fs).astype(int)
    edges = np.zeros(n + 1)
    np.add.at(edges, np.clip(frames - 1 - after, 0, n), 1)
    np.add.at(edges, np.clip(frames + after + before + 1, 0, n), -1)
    free = np.cumsum(edges)[:n] == 0
    free[:before] = False
    free[n - after :] = False
    return free


def record(clean: Clean, phi0: float, artifact: float = 0.0, cutoff: float = 0.0, seed: int = 0) -> Recording:
    """
    Count photons at phi0 / s: Poisson(phi0 / fs x (1 + dF/F) x (1 + motion)), back to dF/F with F0 known, then an
    optional low-pass (cutoff Hz; 0 for none). artifact is the motion's amplitude, % of F.
    """
    rng = np.random.default_rng(seed)
    per_sample = phi0 / clean.fs
    light = per_sample * (1 + clean.dff / 100) * (1 + motion(clean.t, artifact))
    dff = 100 * (rng.poisson(light) / per_sample - 1)
    truth = clean.dff
    if cutoff > 0:
        dff, truth = lowpass(dff, clean.fs, cutoff), lowpass(truth, clean.fs, cutoff)
    lags = np.arange(-clean.before, clean.after + 1)
    template = np.zeros(lags.size)
    if clean.isolated.size:
        template = truth[clean.isolated[:, None] + lags].mean(axis=0)  # (lags,)
        template = template - template[: clean.before].mean()
    y = matched_filter(dff, template, clean.before)
    free = spike_free(clean.spikes, clean.fs, dff.size, clean.before, clean.after)
    reference = y[free] if free.any() else y  # spikes everywhere: z-score against everything
    center = np.median(reference)
    spread = 1.4826 * np.median(np.abs(reference - center)) or 1.0
    z = (y - center) / spread
    spike_z = z[clean.isolated]
    dprime = float(spike_z.mean()) if spike_z.size else 0.0
    return Recording(phi0, dff, truth, template, z, free, spike_z, nu(dff, clean.fs), dprime)


def sweep(clean: Clean, phis: np.ndarray, artifact: float = 0.0, cutoff: float = 0.0, repeats: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """(nu, d') at each photon rate in phis, each the mean of `repeats` recordings."""
    recordings = [[record(clean, phi, artifact, cutoff, seed=repeats * i + j) for j in range(repeats)] for i, phi in enumerate(phis)]
    nus = np.array([[r.nu for r in row] for row in recordings])  # (phis, repeats)
    dprimes = np.array([[r.dprime for r in row] for row in recordings])
    return nus.mean(axis=1), dprimes.mean(axis=1)


def survival(values: np.ndarray, points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """(x, fraction of values >= x), points spaced evenly in log fraction so the tail shows."""
    ordered = np.sort(values)
    n = ordered.size
    k = np.unique(np.round(np.geomspace(1, n, points)).astype(int))
    return ordered[n - k], k / n


def spectrum(x: np.ndarray, fs: float, seconds: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """
    One-sided power spectral density, %^2 / Hz (Welch: Hann windows `seconds` long, half overlapping), without
    0 Hz. White noise of SD s per sample gives 2 s^2 / fs.
    """
    n = min(x.size, int(round(seconds * fs)))
    window = np.hanning(n)
    segments = np.lib.stride_tricks.sliding_window_view(x, n)[:: max(1, n // 2)]
    segments = segments - segments.mean(axis=1, keepdims=True)
    power = np.mean(np.abs(np.fft.rfft(segments * window, axis=1)) ** 2, axis=0) * 2 / (fs * np.sum(window**2))
    if n % 2 == 0:
        power[-1] /= 2  # Nyquist has no mirror image
    return np.fft.rfftfreq(n, 1 / fs)[1:], power[1:]


def shot_psd(phi0: float) -> float:
    """Shot noise's flat level in `spectrum`'s units: 2 (100 sqrt(fs / phi0))^2 / fs."""
    return 2e4 / phi0


def slow_band(fs: float) -> tuple[float, float]:
    """Where motion, heartbeat and subthreshold voltage live, Hz."""
    return 1.0, min(30.0, fs / 4)


def slow_excess(noise: np.ndarray, fs: float, phi0: float) -> float:
    """Mean power of noise in the slow band over shot noise's."""
    f, p = spectrum(noise, fs)
    lo, hi = slow_band(fs)
    return float(np.mean(p[(f >= lo) & (f <= hi)]) / shot_psd(phi0))


def make_field(indicator: str, fs: float, rate: float, seed: int = 1) -> Field:
    """CELLS as Gaussian footprints, each with its own spikes at rate (none for a silent one), at most MOVIE_FRAMES long."""
    duration = min(DURATION[sim.BY_NAME[indicator].kind], MOVIE_FRAMES / fs)
    yy, xx = np.mgrid[0:SIDE, 0:SIDE]
    footprints, dff = [], []
    for k, (cx, cy, radius, brightness, fires) in enumerate(CELLS):
        a = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius**2)).ravel()
        footprints.append(brightness * a / a.sum())
        p = sim.TraceParams(indicator=indicator, fs=fs, duration=duration, rate=rate if fires else 0.0, seed=1000 * seed + k)
        dff.append(sim.make_trace(p).dff)
    dff = np.array(dff)
    return Field(fs, np.arange(dff.shape[1]) / fs, np.array(footprints).T, dff)


def count_field(field: Field, phi0: float, artifact: float = 0.0, cutoff: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    (recorded, expected) photons per frame, each (frames, pixels): Poisson(phi0 / fs x light x (1 + motion)), light
    the footprints x (1 + dF/F) plus the background, then the same optional low-pass as `record`.
    """
    light = (1 + field.dff).T @ field.footprints.T + BACKGROUND  # (frames, pixels)
    expected = phi0 / field.fs * light * (1 + motion(field.t, artifact))[:, None]
    recorded = np.random.default_rng(seed).poisson(expected).astype(float)
    if cutoff > 0:
        recorded, expected = lowpass(recorded.T, field.fs, cutoff).T, lowpass(expected.T, field.fs, cutoff).T
    return recorded, expected


def lag1(x: np.ndarray) -> np.ndarray:
    """Per-pixel lag-1 autocorrelation of x (T, n)."""
    x = x - x.mean(0)
    return (x[1:] * x[:-1]).sum(0) / np.maximum((x * x).sum(0), 1e-12)


def share_not_shot(recorded: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Per pixel, the share of recorded's variance (T, n) that expected carries: all but the shot noise."""
    return expected.var(0) / np.maximum(recorded.var(0), 1e-12)
