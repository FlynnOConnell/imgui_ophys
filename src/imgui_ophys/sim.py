"""
Simulated recordings the apps share: spikes, an indicator's response, and what a camera counts from one ROI.

Everything is simulated on a fine time grid first (1 kHz for calcium, 20 kHz for voltage) and then averaged into
camera frames, as an exposure integrates light: a 1 ms action potential seen at 500 Hz is mostly averaged away.

A frame's photons from an ROI of `pixels` pixels at `brightness` photons / pixel / frame:

    expected = brightness * pixels * (1 + dF/F) * bleach   +   neuropil
    counted  = Poisson(expected) + Normal(0, read_noise * sqrt(pixels))

The rail asks for an SNR rather than a brightness: the single-spike peak in dF/F over the noise SD of dF/F per
frame. `brightness_for_snr` is the closed form of that ratio, so the slider and the simulation agree.

Indicator numbers are single-step or single-AP values from the papers, good for shapes and orders of magnitude only.
The voltage list is gevisim's (github.com/FlynnOConnell/gevisim), one YAML per indicator with its provenance.
"""

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class Indicator:
    name: str
    kind: str  # "calcium" or "voltage"
    amp: float  # calcium: peak dF/F per AP; voltage: dF/F per 100 mV (negative: dimmer when depolarized)
    rise: float  # s, time constant (calcium: of the kernel; voltage: of the low-pass)
    decay: float  # s, time constant
    fs: float  # a typical frame rate, Hz
    duration: float  # a typical recording length for this app, s
    note: str  # what was measured, and where: the time constants above are fit to it


# Calcium kernels are fit so their time to peak (or half-rise) and half-decay match the paper's 1 AP numbers.
# Voltage: amp is the dF/F of a -70 to +30 mV step, rise the activation and decay the deactivation time constant.
INDICATORS = (
    Indicator("GCaMP6f", "calcium", 0.19, 0.017, 0.18, 30.0, 30.0,
              "1 AP in vivo (mouse V1): dF/F 19%, time to peak 45 ms, half-decay 142 ms. Chen et al. 2013, Suppl. Table 3"),
    Indicator("GCaMP6s", "calcium", 0.23, 0.072, 0.67, 30.0, 30.0,
              "1 AP in vivo (mouse V1): dF/F 23%, time to peak 179 ms, half-decay 550 ms. Chen et al. 2013, Suppl. Table 3"),
    Indicator("jGCaMP8m", "calcium", 0.75, 0.014, 0.17, 30.0, 30.0,
              "1 AP in culture: dF/F 0.75, half-rise 7.4 ms, half-decay 134 ms. Zhang et al. 2023, Ext. Data Table 3"),
    Indicator("ASAP6.1", "voltage", 1.64, 0.00399, 0.00489, 1000.0, 10.0,
              "HEK293A, 1P, room temperature: dF/F +164% for -70 to +30 mV; activation 3.99 ms, deactivation 4.89 ms. Lee et al. 2024, Suppl. Tables 2-3"),
    Indicator("ASAP6.2", "voltage", 0.92, 0.00374, 0.00616, 1000.0, 10.0,
              "HEK293A, 1P, room temperature: dF/F +92% for -70 to +30 mV; activation 3.74 ms, deactivation 6.16 ms. Lee et al. 2024, Suppl. Tables 2-3"),
    Indicator("ASAP6.3", "voltage", 2.22, 0.00393, 0.00441, 1000.0, 10.0,
              "HEK293A, 1P, room temperature: dF/F +222% for -70 to +30 mV; activation 3.93 ms, deactivation 4.41 ms. Lee et al. 2024, Suppl. Tables 2-3"),
    Indicator("ASAP7y", "voltage", -0.80, 0.0030, 0.0037, 1000.0, 10.0,
              "HEK293, 1P, 25 C: dF/F -80% for -70 to +30 mV; activation 3.0 ms (0.9 ms at 37 C), deactivation not quantified (3.7 ms assumed). Hao et al. 2026, Fig. 1b, ED Fig. 1b"),
    Indicator("JEDI-2P", "voltage", -0.418, 0.00055, 0.0012, 1000.0, 10.0,
              "HEK293-Kir2.1, 2P, 23 C: dF/F -41.8% for -70 to +30 mV; activation 0.55 ms (fast component), deactivation 1.20 ms at 33 C. Land et al. 2026, Table 1, Suppl. Fig. 3"),
    Indicator("JEDI3hyp", "voltage", -0.515, 0.00048, 0.00211, 1000.0, 10.0,
              "HEK293-Kir2.1, 2P, 23 C: dF/F -51.5% for -70 to +30 mV; activation 0.48 ms (fast component), deactivation 2.11 ms at 33 C. Land et al. 2026, Table 1, Suppl. Fig. 3"),
    Indicator("JEDI3sub", "voltage", -0.635, 0.00063, 0.00223, 1000.0, 10.0,
              "HEK293-Kir2.1, 2P, 23 C: dF/F -63.5% for -70 to +30 mV; activation 0.63 ms (fast component), deactivation 2.23 ms at 33 C. Land et al. 2026, Table 1, Suppl. Fig. 3"),
)
BY_NAME = {i.name: i for i in INDICATORS}
KINDS = ("calcium", "voltage")

FINE_HZ = {"calcium": 1000.0, "voltage": 20000.0}
V_REST, V_PEAK = -65.0, 30.0  # mV
RATE = {"calcium": 0.5, "voltage": 6.0}  # spikes / s the simulated cell fires
BRIGHTNESS = {"calcium": 20.0, "voltage": 3.0}  # photons / pixel / frame at rest: the cell the pages open on


@dataclass
class TraceParams:
    indicator: str = "GCaMP6f"
    fs: float = 30.0  # camera frame rate, Hz
    duration: float = 30.0  # s
    rate: float = 0.5  # mean spikes / s
    brightness: float = 20.0  # photons / pixel / frame at rest
    pixels: int = 50  # pixels in the ROI
    read_noise: float = 1.5  # electrons RMS / pixel / frame
    neuropil: float = 0.3  # neuropil light / ROI light
    bleach: float = 0.2  # fraction of light lost by the end
    seed: int = 1
    # which parts of the recording are on
    shot: bool = True
    read: bool = True
    background: bool = True
    bleaching: bool = True


@dataclass
class Trace:
    """One ROI's recording, frame by frame, with each part kept apart so it can be shown or left out."""

    t: np.ndarray  # frame times, s
    spikes: np.ndarray  # spike times, s
    fine_t: np.ndarray  # the fine grid, s
    fine_signal: np.ndarray  # calcium: dF/F; voltage: membrane potential, mV
    dff: np.ndarray  # the indicator's true dF/F per frame
    parts: dict = field(default_factory=dict)  # name -> photons per frame: signal, bleaching, neuropil, shot, read
    on: dict = field(default_factory=dict)  # name -> whether the part is in `recorded`

    @property
    def recorded(self) -> np.ndarray:
        """Photons counted per frame: the signal plus every part that is on."""
        out = self.parts["signal"].copy()
        for name, values in self.parts.items():
            if name != "signal" and self.on.get(name, True):
                out = out + values
        return out

    @property
    def baseline(self) -> float:
        """F0: the cell's photons per frame at rest (the signal is that times 1 + dF/F)."""
        return float(np.median(self.parts["signal"] / (1 + self.dff)))


def spike_times(rate: float, duration: float, rng: np.random.Generator, refractory: float = 0.005) -> np.ndarray:
    """Poisson spike times with a refractory period, s."""
    n = rng.poisson(rate * duration)
    times = np.sort(rng.uniform(0, duration, n))
    keep = np.ones(n, bool)
    last = -np.inf
    for i, t in enumerate(times):
        if t - last < refractory:
            keep[i] = False
        else:
            last = t
    return times[keep]


def calcium_kernel(rise: float, decay: float, dt: float) -> np.ndarray:
    """Difference of exponentials, peak 1, long enough to decay to 1%."""
    t = np.arange(0, decay * 6, dt)
    k = np.exp(-t / decay) - np.exp(-t / rise)
    return k / k.max()


def ar1(x: np.ndarray, a: float, start: float = 0.0) -> np.ndarray:
    """y[i] = a y[i-1] + x[i] with y[-1] = start, as one convolution: the kernel a**k is cut where it falls under 1e-9."""
    k = a ** np.arange(int(np.ceil(np.log(1e-9) / np.log(a))) + 1) if 0 < a < 1 else np.ones(1)
    head = np.full(k.size - 1, start * (1 - a))  # a run of `start` before the first sample, so y[-1] = start
    full = np.concatenate([head, x])
    n = full.size + k.size - 1
    y = np.fft.irfft(np.fft.rfft(full, n) * np.fft.rfft(k, n), n)
    return y[k.size - 1 : k.size - 1 + x.size]


def membrane(times: np.ndarray, fine_t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Membrane potential in mV: resting level, slow subthreshold wander, 1 ms spikes with an afterhyperpolarization."""
    dt = fine_t[1] - fine_t[0]
    v = np.full(fine_t.size, V_REST)
    # Ornstein-Uhlenbeck subthreshold fluctuations, tau 20 ms, SD 3 mV
    tau, sd = 0.02, 3.0
    a = np.exp(-dt / tau)
    noise = rng.standard_normal(fine_t.size) * sd * np.sqrt(1 - a * a)
    v += ar1(noise, a)
    w = np.arange(0, 0.015, dt)
    shape = (V_PEAK - V_REST) * np.exp(-((w - 0.0005) ** 2) / (2 * 0.00025**2)) - 6.0 * (w > 0.001) * np.exp(-(w - 0.001) / 0.006)
    for s in times:
        i = int(s / dt)
        n = min(shape.size, fine_t.size - i)
        v[i : i + n] += shape[:n]
    return v


def lowpass(x: np.ndarray, rise: float, dt: float) -> np.ndarray:
    """First-order low-pass with time constant rise: an indicator that cannot follow faster than that."""
    a = np.exp(-dt / rise)
    return ar1((1 - a) * x, a, start=float(x[0]))


def bin_frames(x: np.ndarray, per_frame: int) -> np.ndarray:
    """Average consecutive samples into frames, as an exposure integrates light."""
    n = x.size // per_frame
    return x[: n * per_frame].reshape(n, per_frame).mean(axis=1)


def make_trace(p: TraceParams) -> Trace:
    ind = BY_NAME[p.indicator]
    rng = np.random.default_rng(p.seed)
    fine = FINE_HZ[ind.kind]
    per_frame = max(1, int(round(fine / p.fs)))
    fine_hz = p.fs * per_frame
    dt = 1.0 / fine_hz
    fine_t = np.arange(int(p.duration * fine_hz)) * dt
    times = spike_times(p.rate, p.duration, rng)

    if ind.kind == "calcium":
        train = np.zeros(fine_t.size)
        np.add.at(train, np.minimum((times / dt).astype(int), fine_t.size - 1), 1.0)
        fine_signal = ind.amp * np.convolve(train, calcium_kernel(ind.rise, ind.decay, dt))[: fine_t.size]
        fine_dff = fine_signal
    else:
        fine_signal = membrane(times, fine_t, rng)
        fine_dff = lowpass(ind.amp * (fine_signal - V_REST) / 100.0, ind.rise, dt)

    dff = bin_frames(fine_dff, per_frame)
    t = np.arange(dff.size) / p.fs
    rest = p.brightness * p.pixels
    signal = rest * (1 + dff)
    bleach = 1 - p.bleach * (1 - np.exp(-t / max(p.duration / 3, 1e-9))) / (1 - np.exp(-3))
    # neuropil: other cells' activity blurred together, slow and smaller
    # run 3 s before the recording starts, so the slow filter is already at its steady level
    warm = int(3 * p.fs)
    kernel = np.exp(-np.arange(warm) / p.fs)
    other = np.convolve(rng.poisson(p.rate * 20 / p.fs, dff.size + warm).astype(float), kernel)[warm : warm + dff.size]
    other /= max(float(np.mean(other)), 1e-9)
    neuropil = p.neuropil * rest * (1 + 0.2 * (other - 1))
    expected = (signal + neuropil) * bleach
    shot = rng.poisson(expected) - expected
    read = rng.standard_normal(dff.size) * p.read_noise * np.sqrt(p.pixels)
    parts = {
        "signal": signal,
        "bleaching": (signal + neuropil) * (bleach - 1),
        "neuropil": neuropil,
        "shot": shot,
        "read": read,
    }
    on = {"bleaching": p.bleaching, "neuropil": p.background, "shot": p.shot, "read": p.read}
    return Trace(t, times, fine_t, fine_signal, dff, parts, on)


def spike_response(ind: Indicator, fs: float) -> tuple[int, float]:
    """
    One spike's response as the camera sees it: (frames from the spike to the peak frame, |dF/F| at that frame).

    The spike sits at the start of a frame; a calcium kernel or a 1 ms action potential through the indicator's
    low-pass is averaged into frames and the largest frame taken.
    """
    fine = FINE_HZ[ind.kind]
    per_frame = max(1, int(round(fine / fs)))
    dt = 1.0 / (fs * per_frame)
    if ind.kind == "calcium":
        fine_dff = ind.amp * calcium_kernel(ind.rise, ind.decay, dt)
    else:
        t = np.arange(int(0.05 / dt)) * dt
        v = V_REST + (V_PEAK - V_REST) * np.exp(-((t - 0.0005) ** 2) / (2 * 0.00025**2))
        fine_dff = lowpass(ind.amp * (v - V_REST) / 100.0, ind.rise, dt)
    frames = np.abs(bin_frames(fine_dff, per_frame))
    at = int(np.argmax(frames))
    return at, float(frames[at])


def spike_peak(ind: Indicator, fs: float) -> float:
    """|dF/F| at the peak frame of one spike's response: the SNR's numerator."""
    return spike_response(ind, fs)[1]


def noise_sd(brightness: float, pixels: int, read_noise: float, neuropil: float) -> float:
    """SD of dF/F per frame from shot and read noise at rest, in units of the cell's resting light."""
    rest = brightness * pixels
    return float(np.sqrt(rest * (1 + neuropil) + read_noise**2 * pixels) / rest)


def brightness_for_snr(peak: float, snr: float, pixels: int, read_noise: float, neuropil: float) -> float:
    """The brightness at which peak / noise_sd equals snr: the positive root of the quadratic noise_sd gives."""
    a = (peak / snr) ** 2 * pixels**2
    b = pixels * (1 + neuropil)
    c = read_noise**2 * pixels
    return float((b + np.sqrt(b * b + 4 * a * c)) / (2 * a))


def snr_of(peak: float, brightness: float, pixels: int, read_noise: float, neuropil: float) -> float:
    return peak / noise_sd(brightness, pixels, read_noise, neuropil)


@dataclass
class Recording:
    """The rail's choice, shared by every page that shows a trace: the indicator and how clearly one spike shows."""

    indicator: str = "GCaMP6f"
    snr: float = 0.0  # 0: the kind's opening brightness

    @property
    def kind(self) -> str:
        return BY_NAME[self.indicator].kind

    def default_snr(self) -> float:
        ind = BY_NAME[self.indicator]
        p = TraceParams()
        return snr_of(spike_peak(ind, ind.fs), BRIGHTNESS[ind.kind], p.pixels, p.read_noise, p.neuropil)

    def set_kind(self, kind: str) -> None:
        """The first indicator of that kind, at its opening SNR."""
        if kind != self.kind:
            self.indicator = next(i.name for i in INDICATORS if i.kind == kind)
            self.snr = 0.0

    def params(self, **overrides) -> TraceParams:
        """TraceParams for the simulated cell: the indicator's frame rate and length, the brightness this SNR needs."""
        ind = BY_NAME[self.indicator]
        p = TraceParams(indicator=ind.name, fs=ind.fs, duration=ind.duration, rate=RATE[ind.kind])
        p = replace(p, **overrides)
        snr = self.snr if self.snr > 0 else self.default_snr()
        p.brightness = brightness_for_snr(spike_peak(ind, p.fs), snr, p.pixels, p.read_noise, p.neuropil)
        return p


def upright(dff: np.ndarray, ind: Indicator) -> np.ndarray:
    """dF/F with depolarization up: unchanged for calcium and brightening indicators, negated for dimming ones."""
    return -dff if ind.amp < 0 else dff


BASELINES = ("whole recording", "running window")


def baseline(f: np.ndarray, method: str, window: int, q: float = 20.0) -> np.ndarray:
    """F0 per frame: the q-th percentile of f over the whole recording, or over a running window of frames."""
    if method != "running window" or window >= f.size:
        return np.full(f.size, float(np.percentile(f, q)))
    window = max(window, 3)
    stride = max(1, window // 8)
    centers = np.arange(0, f.size, stride)
    values = [np.percentile(f[max(0, c - window // 2) : c + window // 2 + 1], q) for c in centers]
    return np.interp(np.arange(f.size), centers, values)


def to_dff(f: np.ndarray, f0) -> np.ndarray:
    return (f - f0) / f0


# ----------------------------------------------------------------------
# denoising
# ----------------------------------------------------------------------

DENOISERS = ("none", "moving average", "gaussian", "exponential")


def denoise(x: np.ndarray, method: str, width: float) -> np.ndarray:
    """Smooth x; width is in frames (the box length, the Gaussian SD, or the exponential time constant)."""
    if method == "moving average":
        n = max(1, int(round(width)))
        k = np.ones(n) / n
    elif method == "gaussian":
        s = max(width, 1e-3)
        r = np.arange(-int(4 * s) - 1, int(4 * s) + 2)
        k = np.exp(-(r**2) / (2 * s * s))
        k /= k.sum()
    elif method == "exponential":
        # causal: what an online filter can do; it delays the peak
        a = np.exp(-1 / max(width, 1e-3))
        y = np.empty_like(x)
        acc = x[0]
        for i, v in enumerate(x):
            acc = a * acc + (1 - a) * v
            y[i] = acc
        return y
    else:
        return x.copy()
    pad = len(k) // 2
    xp = np.pad(x, (pad, len(k) - 1 - pad), mode="edge")
    return np.convolve(xp, k, mode="valid")


def event_average(x: np.ndarray, spikes: np.ndarray, fs: float, before: float, after: float, isolation: float) -> tuple[np.ndarray, np.ndarray]:
    """(lag s, mean of x around each spike with no other spike within isolation s); empty when none qualify."""
    lag = np.arange(-int(before * fs), int(after * fs) + 1)
    rows = []
    for i, s in enumerate(spikes):
        others = np.delete(spikes, i)
        if others.size and np.min(np.abs(others - s)) < isolation:
            continue
        c = int(round(s * fs))
        if c + lag[0] < 0 or c + lag[-1] >= x.size:
            continue
        rows.append(x[c + lag])
    if not rows:
        return lag / fs, np.zeros(lag.size)
    return lag / fs, np.mean(rows, axis=0)
