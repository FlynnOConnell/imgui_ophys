"""
The two inputs of the Denoising page, each a `Signal` the pipeline runs on: a gevisim cell (one JEDI3sub soma under a
two-photon point focus, its shot noise scaled by one knob) or a recorded MUnit exported from a .mesc.

Both are pixels over time, averaged to F and turned into z the way `vnoiser.preprocess.domain_zscore` does it, with
every step kept for the page.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.ndimage

SOURCES = (
    ("gevisim", "Simulated cell (gevisim)", "One JEDI3sub soma under a two-photon point focus, 20 s at 1075 Hz, "
                                            "4 spikes/s. The noise knob scales its shot noise; 1 is the photon budget "
                                            "the simulation gives, 0 is the noiseless cell."),
    ("munit", "Recorded MUnit", "A .mesc unit exported by scripts/export_mesc_linescan.py (roi_<i> pixel arrays and "
                                "fs), or the lab's per-ROI export (F, fs). Found under ~/data."),
)

REFERENCE_FS = 1075.2688  # Hz, the frame rate vnoiser's sample counts were set at
SIM_DURATION_S = 20.0
SIM_FS_VM = 10_000.0  # Hz, the membrane potential grid; a 0.8 ms action potential needs it
SIM_RATE_HZ = 4.0  # spikes per second
SIM_SEED = 3  # the cell, its spikes and its footprint; the noise draw has its own seed
SIM_ROI_FRACTION = 0.2  # a pixel is in the cell when its footprint reaches this share of the brightest pixel's


@dataclass
class SignalParams:
    """Everything that picks and shapes the input trace."""

    source: str = "gevisim"
    noise: float = 1.0  # gevisim: shot noise SD relative to the simulated photon budget; 0 is the noiseless cell
    seed: int = 7  # gevisim: the noise draw
    duration_s: float = SIM_DURATION_S  # the window shown, from start_s
    start_s: float = 0.0  # munit: where the window starts
    scan_path: str = ""  # munit: the exported .npz
    rois: tuple = (0,)  # munit: the lines averaged into one domain


@dataclass
class Pixels:
    """
    The steps from pixels to the z-scored trace, each kept for plotting, for the recorded pixels and, when simulated,
    the same pixels without noise (`clean_*`).

    Parameters
    ----------
    pixels : np.ndarray
        (pixels, samples), every block's pixels stacked, raw counts or photons.
    clean_pixels : np.ndarray, optional
        The same without shot noise; None for a recording.
    edges : list
        (first, last + 1) pixel of each block in pixels.
    means : np.ndarray
        (blocks, samples), each block's mean.
    f : np.ndarray
        The mean over every pixel.
    f_baseline : np.ndarray
        Gaussian-smoothed f.
    dfof_raw : np.ndarray
        (f - f_baseline) / f_baseline.
    dfof : np.ndarray
        dfof_raw with the first n_startup samples replaced by the mean of the rest.
    dfof_baseline : np.ndarray
        Gaussian-smoothed dfof.
    n_startup : int
        Samples replaced.
    clean_f, clean_dfof : np.ndarray, optional
        f and dfof of clean_pixels.
    names : list
        Each block's name in the legend.
    sigma_s : tuple
        The two Gaussian baselines' SDs in seconds: F's, then dF/F's.
    """

    pixels: np.ndarray
    clean_pixels: Optional[np.ndarray]
    edges: list
    means: np.ndarray
    f: np.ndarray
    f_baseline: np.ndarray
    dfof_raw: np.ndarray
    dfof: np.ndarray
    dfof_baseline: np.ndarray
    n_startup: int
    clean_f: Optional[np.ndarray] = None
    clean_dfof: Optional[np.ndarray] = None
    names: list = field(default_factory=list)
    sigma_s: tuple = (0.0, 0.0)


@dataclass
class Signal:
    """
    One trace in the units the denoiser expects (z-scored dF/F, depolarization positive).

    Parameters
    ----------
    name : str
        The plots' name for it.
    fs : float
        Sampling rate in Hz.
    x : np.ndarray
        The trace the denoiser runs on.
    truth : np.ndarray, optional
        The same trace without noise, when simulated.
    note : str
        One line about where it came from.
    events_s : dict
        Ground truth event times in s by kind ("AP"), when simulated.
    pixels : Pixels
        The steps from pixels to x.
    """

    name: str
    fs: float
    x: np.ndarray
    truth: Optional[np.ndarray] = None
    note: str = ""
    events_s: dict = field(default_factory=dict)
    pixels: Optional[Pixels] = None

    @property
    def n(self) -> int:
        return int(self.x.size)

    @property
    def duration(self) -> float:
        return self.n / self.fs

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n) / self.fs


def make_signal(p: SignalParams) -> Signal:
    """The trace p describes. Raises ValueError with a message for the window when it cannot be made."""
    if p.source == "gevisim":
        return simulated_cell(p)
    if p.source == "munit":
        return munit_scan(p)
    raise ValueError(f"Unknown source {p.source!r}.")


def dfof_config(fs: float):
    """vnoiser's DfofConfig with its sample counts scaled from 1075 Hz to fs, as mbo_utilities' pipeline does, so
    the baselines keep their length in seconds."""
    from vnoiser.preprocess import DfofConfig

    cfg = DfofConfig()
    k = fs / REFERENCE_FS
    if abs(k - 1) < 1e-6:
        return cfg
    return DfofConfig(sigma_dfof=cfg.sigma_dfof * k, sigma_baseline=cfg.sigma_baseline * k,
                      n_startup=int(round(cfg.n_startup * k)), negative=cfg.negative)


def baseline_seconds(fs: float) -> tuple:
    """The SDs of the two Gaussian baselines in seconds: the one F is divided by, and the slower one taken off dF/F."""
    cfg = dfof_config(fs)
    return cfg.sigma_dfof / fs, cfg.sigma_baseline / fs


def preprocess_steps(f: np.ndarray, fs: float = REFERENCE_FS):
    """vnoiser.preprocess.domain_zscore on one mean trace, keeping every step: (f_baseline, dfof_raw, dfof,
    dfof_baseline, n_startup)."""
    cfg = dfof_config(fs)
    f_baseline = scipy.ndimage.gaussian_filter1d(f, sigma=cfg.sigma_dfof)
    dfof_raw = (f - f_baseline) / f_baseline
    dfof = dfof_raw.copy()
    n = int(cfg.n_startup) if dfof.size > cfg.n_startup else 0
    if n:
        dfof[:n] = dfof[n:].mean()
    return f_baseline, dfof_raw, dfof, scipy.ndimage.gaussian_filter1d(dfof, sigma=cfg.sigma_baseline), n


def pixels_to_signal(name: str, fs: float, blocks: list, names: list, clean: Optional[list], note: str,
                     events_s: Optional[dict] = None) -> Signal:
    """Stack the (pixels, samples) blocks, average them to F and turn F into z as the lab's pipeline does.
    Depolarization is negative on these indicators, so z is flipped."""
    edges, start = [], 0
    for block in blocks:
        edges.append((start, start + block.shape[0]))
        start += block.shape[0]
    pixels = np.concatenate(blocks).astype(np.float64)
    f = pixels.mean(axis=0)
    f_baseline, dfof_raw, dfof, dfof_baseline, n_startup = preprocess_steps(f, fs)
    sd = dfof.std()
    x = -(dfof - dfof_baseline) / sd
    px = Pixels(
        pixels=pixels.astype(np.float32), clean_pixels=None, edges=edges,
        means=np.stack([b.mean(axis=0) for b in blocks]), f=f, f_baseline=f_baseline, dfof_raw=dfof_raw,
        dfof=dfof, dfof_baseline=dfof_baseline, n_startup=n_startup, names=names, sigma_s=baseline_seconds(fs),
    )
    truth = None
    if clean is not None:
        clean_pixels = np.concatenate(clean).astype(np.float64)
        px.clean_pixels = clean_pixels.astype(np.float32)
        px.clean_f = clean_pixels.mean(axis=0)
        _, _, px.clean_dfof, clean_base, _ = preprocess_steps(px.clean_f, fs)
        truth = -(px.clean_dfof - clean_base) / sd
    return Signal(name=name, fs=fs, x=x, truth=truth, note=note, events_s=events_s or {}, pixels=px)


# ----------------------------------------------------------------------
# gevisim
# ----------------------------------------------------------------------

_CELL: dict = {}  # the simulated cell, once per process: (expected photons (pixels, frames), spike times s, fs)


def gevisim_cell() -> tuple:
    """Expected photons per frame in the soma's pixels, the spike times and the frame rate of the fixed cell."""
    if "cell" not in _CELL:
        from gevisim import kinetics
        from gevisim.distribution import SomaTargeted
        from gevisim.morphology import Spheres
        from gevisim.optics import Grid, NonImaging, Optics, PointFocus
        from gevisim.simulation import Simulation
        from gevisim.voltage import ap_train, poisson_spikes

        sim = Simulation(
            Spheres(extent_um=(40.0, 40.0, 30.0), n_cells=1), SomaTargeted(),
            Optics(PointFocus(wavelength_nm=940, na=0.8), NonImaging(na=0.8)),
            kinetics.IndicatorModel(kinetics.load("JEDI3sub"), temperature_C=37),
            grid=Grid((24, 24), 0.8), seed=SIM_SEED,
        )
        spikes = poisson_spikes(1, SIM_RATE_HZ, SIM_DURATION_S, np.random.default_rng(SIM_SEED))
        vm = ap_train(spikes, SIM_DURATION_S, SIM_FS_VM)
        result = sim.run(vm, SIM_FS_VM, REFERENCE_FS, shot_noise=False)
        image = sim.footprints.image()
        soma = image > SIM_ROI_FRACTION * image.max()
        _CELL["cell"] = (result.expected[:, soma].T.copy(), np.asarray(spikes[0], dtype=float), REFERENCE_FS)
    return _CELL["cell"]


def simulated_cell(p: SignalParams) -> Signal:
    """The gevisim cell's first duration_s seconds with shot noise at p.noise times the budget's: Poisson draws of
    expected / noise^2, times noise^2, keep the mean and scale the SD by noise."""
    expected, spikes, fs = gevisim_cell()
    n = int(round(min(p.duration_s, SIM_DURATION_S) * fs))
    if n < 64:
        raise ValueError("Need at least 64 samples; raise the duration.")
    clean = expected[:, :n].astype(np.float64)
    if p.noise > 0:
        k = p.noise**2
        noisy = np.random.default_rng(p.seed).poisson(clean / k) * k
    else:
        noisy = clean.copy()
    note = f"gevisim JEDI3sub soma, {clean.shape[0]} pixels, {n / fs:.0f} s at {fs:.0f} Hz, noise x {p.noise:g}"
    return pixels_to_signal("Simulated cell", fs, [noisy], ["soma"], [clean], note,
                            events_s={"AP": spikes[spikes < n / fs]})


# ----------------------------------------------------------------------
# a recorded MUnit
# ----------------------------------------------------------------------

_SCANS: dict = {}  # path -> (the ROIs' pixel arrays, info), so changing lines or the window does not reread it


def open_scan(path: str):
    """The exported .npz at path as {roi: (frames, width) pixels} and its info: fs, unit, frames, each ROI's width.

    Two layouts are read: the pixel export (`roi_<i>` uint16 arrays, `fs`, `unit`) and the lab's per-ROI export
    (`F` as (channels, rois, frames), `fs`), whose channel 0 trace stands in as a one-pixel ROI."""
    file = Path(path).expanduser()
    if not file.is_file():
        raise ValueError(f"No file at {file}.")
    key = str(file.resolve())
    if key not in _SCANS:
        d = np.load(file)
        names = sorted((k for k in d.files if k.startswith("roi_")), key=lambda k: int(k[4:]))
        if names:
            rois = {int(k[4:]): d[k] for k in names}
        elif "F" in d.files and d["F"].ndim == 3:
            rois = {i: d["F"][0, i][:, None] for i in range(d["F"].shape[1])}
        else:
            raise ValueError(f"{file.name} holds neither roi_<i> arrays nor F; export it with scripts/export_mesc_linescan.py.")
        first = rois[min(rois)]
        info = {
            "fs": float(d["fs"]), "unit": str(d["unit"]) if "unit" in d.files else file.stem,
            "frames": int(first.shape[0]), "widths": {i: int(a.shape[1]) for i, a in rois.items()},
        }
        _SCANS[key] = (rois, info)
    return _SCANS[key]


def scan_info(path: str) -> dict:
    return open_scan(path)[1]


def find_scans(root: str = "~/data") -> list:
    """Exported units under root, up to five folders deep: <mesc folder>/linescans/*.npz and MSession_*_MUnit_*.npz."""
    base = Path(root).expanduser()
    if not base.is_dir():
        return []
    found = set()
    for depth in range(1, 6):
        for name in ("linescans/*.npz", "MSession_*MUnit_*.npz"):
            found.update(base.glob("/".join(["*"] * (depth - 1) + [name])))
    return sorted(str(f) for f in found)


def munit_scan(p: SignalParams) -> Signal:
    """A window of an exported unit: the chosen lines' pixels, averaged and turned into z."""
    if not p.scan_path:
        raise ValueError("Choose an exported MUnit (.npz).")
    data, info = open_scan(p.scan_path)
    fs = info["fs"]
    rois = [int(r) for r in p.rois if int(r) in data]
    if not rois:
        raise ValueError("Pick at least one line.")
    n_all = info["frames"]
    i0 = int(np.clip(round(p.start_s * fs), 0, max(n_all - 64, 0)))
    i1 = int(min(n_all, i0 + round(p.duration_s * fs)))
    if i1 - i0 < 64:
        raise ValueError("The window holds fewer than 64 frames.")
    blocks = [data[roi][i0:i1].T.astype(np.float64) for roi in rois]  # (width, frames) each
    note = f"{info['unit']}, lines {', '.join(str(r) for r in rois)}, {i0 / fs:.1f} to {i1 / fs:.1f} s at {fs:.1f} Hz"
    return pixels_to_signal(Path(p.scan_path).stem, fs, blocks, [f"line {roi}" for roi in rois], None, note)
