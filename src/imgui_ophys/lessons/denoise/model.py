"""
vnoiser's Morlet wavelet denoiser on one trace, every stage kept for the page: F to z the way
`vnoiser.preprocess.domain_zscore` does it, `Denoiser.upstream` (the lab's settings) on z, the kept bands with
their event windows and masks, the lab's saved trace (`vnoiser.pf.final_trace`) and `detect_peaks` on it.

vnoiser's sample counts (baseline sigmas, the first samples replaced, the wavelet scales, the peak spacing) were set
at 1075 Hz; here they scale with the frame rate so they keep their length in seconds and the rows their frequency.
"""

from typing import Optional

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

try:
    import scipy.ndimage
    import vnoiser.denoiser
    import vnoiser.events
    import vnoiser.pf
    import vnoiser.preprocess
except ImportError:  # the denoise extra is optional; the page says how to install it
    vnoiser = None

REFERENCE_FS = 1075.2688  # Hz, the frame rate vnoiser's sample counts were set at
N_SCALES = 100
SCALE_RANGE = (1.0, 1000.0)  # samples at REFERENCE_FS
DEFAULT_FS = 1000.0  # Hz, for a file that does not say
TOLERANCE_S = 0.01  # a spike has a peak when one lies within this
IMAGE_COLUMNS = 800  # the scalogram image's width in samples (block maxima), so ImPlot draws it at frame rate


def available() -> bool:
    return vnoiser is not None


@dataclass
class Band:
    """One band the denoiser kept: a cluster of CWT rows averaged, with its event windows and mask."""

    lo_hz: float
    hi_hz: float
    trace: np.ndarray  # the reduced band, real part
    windows: np.ndarray  # (n_events, 2) start and end samples
    mask: np.ndarray  # 1 on events, the floor elsewhere, Gaussian-blurred
    masked: np.ndarray  # trace / sqrt(mean scale) x mask, real: what the band adds to the sum

    @property
    def span(self) -> str:
        return f"{fmt_hz(self.lo_hz)} to {fmt_hz(self.hi_hz)} Hz"


@dataclass
class Run:
    """Every stage's output for one trace."""

    fs: float
    f: np.ndarray
    f_baseline: np.ndarray  # Gaussian-smoothed F
    dfof: np.ndarray  # (F - f_baseline) / f_baseline, the first samples replaced
    dfof_baseline: np.ndarray  # the slower Gaussian baseline under the z-score
    z: np.ndarray
    freqs: np.ndarray  # Hz per CWT row, decreasing
    amp: np.ndarray  # |coeff|, float32, (n_scales, n_samples)
    image: np.ndarray  # |coeff| / sqrt(scale) in IMAGE_COLUMNS blocks of time, block maxima, float32
    bands: list  # the kept Bands, slow to fast
    rescaled: np.ndarray  # the masked bands added up, real
    baseline: np.ndarray  # the 1 Hz FIR low-pass of z
    denoised: np.ndarray  # rescaled + baseline
    final: np.ndarray  # the trace the lab saves: rescaled + baseline[0]
    peaks: np.ndarray  # sample indices on `final`
    spikes: Optional[np.ndarray] = None  # ground truth spike times, s, when simulated
    hits: Optional[tuple] = None  # (spikes with a peak within TOLERANCE_S, spikes, peaks with no spike)

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.f.size) / self.fs

    @property
    def duration(self) -> float:
        return self.f.size / self.fs

    @property
    def peak_times(self) -> np.ndarray:
        return self.peaks / self.fs


def fmt_hz(f: float) -> str:
    return f"{f:.0f}" if f >= 10 else f"{f:.1f}"


def rate_factor(fs: float) -> float:
    """fs over the frame rate vnoiser's sample counts were set at."""
    return fs / REFERENCE_FS


def dfof_config(fs: float, negative: bool = True):
    """vnoiser's DfofConfig with its sample counts scaled to fs, so the baselines keep their length in seconds."""
    k = rate_factor(fs)
    cfg = vnoiser.preprocess.DfofConfig()
    return replace(cfg, sigma_dfof=cfg.sigma_dfof * k, sigma_baseline=cfg.sigma_baseline * k,
                   n_startup=int(round(cfg.n_startup * k)), negative=negative)


def zscore(f: np.ndarray, fs: float, negative: bool = True) -> tuple:
    """
    domain_zscore on one trace, keeping each step.

    Parameters
    ----------
    f : np.ndarray
        Fluorescence per frame.
    fs : float
        Frame rate, Hz.
    negative : bool
        Flip the sign, so a dimming indicator's depolarization points up.

    Returns
    -------
    tuple
        (f_baseline, dfof, dfof_baseline, z); dfof has its first n_startup samples replaced by the mean of the rest.
    """
    cfg = dfof_config(fs, negative)
    f = np.asarray(f, dtype=float)
    f_baseline = scipy.ndimage.gaussian_filter1d(f, sigma=cfg.sigma_dfof)
    dfof = (f - f_baseline) / f_baseline
    if 0 < cfg.n_startup < dfof.size:
        dfof[: cfg.n_startup] = dfof[cfg.n_startup :].mean()
    dfof_baseline = scipy.ndimage.gaussian_filter1d(dfof, sigma=cfg.sigma_baseline)
    z = (dfof - dfof_baseline) / dfof.std()
    return f_baseline, dfof, dfof_baseline, -z if negative else z


def scales_at(fs: float) -> np.ndarray:
    """The wavelet scales in samples at fs: 1 to 1000 at REFERENCE_FS, scaled with the rate, never under 1."""
    k = rate_factor(fs)
    lo = max(1.0, SCALE_RANGE[0] * k)
    return np.logspace(np.log10(lo), np.log10(max(lo, SCALE_RANGE[1] * k)), N_SCALES)


def spike_config(fs: float):
    """detect_peaks' defaults at fs: the band-pass top under Nyquist, the peak spacing scaled with the rate."""
    cfg = vnoiser.events.SpikeDetectConfig()
    return replace(cfg, bp=(float(cfg.bp[0]), min(float(cfg.bp[1]), 0.95 * fs / 2)),
                   distance_samples=max(1, int(round(cfg.distance_samples * rate_factor(fs)))))


def peak_hits(spikes_s: np.ndarray, peaks_s: np.ndarray, tolerance_s: float = TOLERANCE_S) -> tuple:
    """(spikes with a peak within tolerance_s, spikes, peaks with no spike within tolerance_s)."""
    spikes_s, peaks_s = np.asarray(spikes_s, dtype=float), np.asarray(peaks_s, dtype=float)
    if spikes_s.size == 0 or peaks_s.size == 0:
        return 0, int(spikes_s.size), int(peaks_s.size)
    gap = np.abs(spikes_s[:, None] - peaks_s[None, :])  # (n_spikes, n_peaks)
    return int((gap.min(axis=1) <= tolerance_s).sum()), int(spikes_s.size), int((gap.min(axis=0) > tolerance_s).sum())


def block_max(a: np.ndarray, columns: int) -> np.ndarray:
    """a's last axis cut into at most `columns` blocks, the maximum of each."""
    n = a.shape[-1]
    per = -(-n // columns)
    pad = per * -(-n // per) - n
    padded = np.concatenate([a, np.repeat(a[..., -1:], pad, axis=-1)], axis=-1)
    return padded.reshape(*a.shape[:-1], -1, per).max(axis=-1)


def band_masks(denoiser) -> list:
    """
    The kept bands of a Denoiser that has run, slow to fast, with their masks after the blur.

    AdaptiveThreshold returns the mask before its Gaussian blur (`clu_label`) and the masked band after it
    (`all_scaled`); the blur is redone here with its sigma.
    """
    reduced, thres, cfg = denoiser.reduced_, denoiser.threshold_result_, denoiser.cfg_thres
    sigmas = cfg.soft_sigma if cfg.thres_type == "soft" else cfg.hard_sigma
    bands = []
    for row, (lo, hi) in enumerate(reduced["reduced_freqs"]):
        onsets = list(reduced["event_onsets"].values())[row]
        label = thres["clu_label"][row]
        mask = scipy.ndimage.gaussian_filter1d(label, sigma=sigmas[0] if hi <= 120 else sigmas[1])
        bands.append(Band(lo_hz=float(lo), hi_hz=float(hi), trace=np.real(reduced["reduced_coeff"][row]),
                          windows=np.asarray(onsets["crossings"], dtype=int), mask=mask,
                          masked=np.real(thres["all_scaled"][row])))
    bands.sort(key=lambda band: band.lo_hz)
    return bands


def run(f: np.ndarray, fs: float, spikes: Optional[np.ndarray] = None, negative: bool = True) -> Run:
    """
    Every stage on one fluorescence trace.

    Parameters
    ----------
    f : np.ndarray
        Fluorescence per frame.
    fs : float
        Frame rate, Hz.
    spikes : np.ndarray, optional
        Ground truth spike times in s; sets `Run.hits`.
    negative : bool
        Flip the sign in the z-score (a dimming indicator).

    Raises
    ------
    ImportError
        Without the denoise extra (vnoiser, scipy, scikit-learn, PyWavelets).
    """
    if vnoiser is None:
        raise ImportError("the denoise extra is not installed: uv pip install -e '.[denoise]'")
    f = np.asarray(f, dtype=float).ravel()
    f_baseline, dfof, dfof_baseline, z = zscore(f, fs, negative)
    scales = scales_at(fs)
    denoiser = vnoiser.denoiser.Denoiser.upstream(fs, freq_scales=scales)
    denoised, _ = denoiser.run(z)
    amp = np.abs(denoiser.coeff_).astype(np.float32)
    final = vnoiser.pf.final_trace(denoiser.rescaled_signal_, denoiser.lp_dfof_)
    peaks = vnoiser.events.detect_peaks(final, float(np.floor(fs)), spike_config(fs))
    out = Run(
        fs=fs, f=f, f_baseline=f_baseline, dfof=dfof, dfof_baseline=dfof_baseline, z=z,
        freqs=np.asarray(denoiser.freqs_, dtype=float), amp=amp,
        image=block_max(amp / np.sqrt(scales, dtype=np.float32)[:, None], IMAGE_COLUMNS),
        bands=band_masks(denoiser), rescaled=np.real(denoiser.rescaled_signal_), baseline=denoiser.lp_dfof_,
        denoised=np.real(denoised), final=final, peaks=peaks,
    )
    if spikes is not None:
        out.spikes = np.asarray(spikes, dtype=float)
        out.hits = peak_hits(out.spikes, out.peak_times)
    return out


def read_trace(path: str) -> tuple:
    """
    One fluorescence trace from a file: (f, fs or None).

    .npz as the line-scan export writes it: every `roi_<i>` array (frames, width) averaged over its pixels into one
    F, `fs` from the file. .npy, .csv, .txt, .tsv: one trace per column, the first taken, the rate unknown.
    """
    file = Path(path).expanduser()
    if file.suffix == ".npz":
        d = np.load(file)
        names = sorted((k for k in d.files if k.startswith("roi_")), key=lambda k: int(k[4:]))
        if not names:
            raise ValueError(f"{file.name} holds no roi_<i> arrays")
        pixels = np.concatenate([np.asarray(d[k], dtype=float).T for k in names])  # (pixels, frames)
        return pixels.mean(axis=0), float(d["fs"])
    if file.suffix == ".npy":
        data = np.asarray(np.load(file), dtype=float)
    else:
        rows = []
        for line in file.read_text().splitlines():
            parts = line.replace(",", " ").replace(";", " ").split()
            if parts and all(is_number(v) for v in parts):
                rows.append([float(v) for v in parts])
        if not rows:
            raise ValueError(f"no numeric rows in {file.name}")
        width = min(len(r) for r in rows)
        data = np.array([r[:width] for r in rows])
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f"{file.name} holds a {data.ndim}-d array; need one trace per column")
    if data.shape[0] < data.shape[1]:
        data = data.T
    return data[:, 0].copy(), None


def is_number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True
