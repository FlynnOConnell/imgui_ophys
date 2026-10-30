"""
The Denoising (vnoiser) stage: one trace from F to z, its Morlet scalogram, the bands the denoiser keeps with their
event masks, and the masked sum against z with the peaks the lab's detector finds on it. Runs on the rail's
Recording: the simulated cell, or a file when one is open.
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui, implot

from imgui_ophys import diagram as D
from imgui_ophys import plotting as P
from imgui_ophys import sim
from imgui_ophys.lesson import Lesson
from imgui_ophys.lessons.denoise import model as M
from imgui_ophys.widgets import KEY, push_size, u32

ImVec2, ImVec4 = imgui.ImVec2, imgui.ImVec4
TRUTH = ImVec4(0.35, 0.85, 0.45, 1.0)  # ground truth spikes
BAR = ImVec4(0.52, 0.52, 0.55, 1.0)  # scale bars sit back a little from the traces
LAB = ImVec4(0.92, 0.92, 0.94, 1.0)  # the trace the lab saves
TIME_AXIS = implot.AxisFlags_.no_decorations  # pans and zooms, draws nothing: a time bar stands in
VALUE_AXIS = implot.AxisFlags_.no_decorations | implot.AxisFlags_.auto_fit
INSTALL = "Install the denoise extra: uv pip install -e '.[denoise]'"


def nice(x: float) -> float:
    """x rounded down to 1, 2 or 5 times a power of ten."""
    e = 10 ** math.floor(math.log10(x))
    return e * max(m for m in (1.0, 2.0, 5.0) if m * e <= x)


def plural(noun: str, n: int) -> str:
    return noun if n == 1 else noun + "s"


def seconds_label(seconds: float) -> str:
    return f"{seconds * 1000:g} ms" if seconds < 1 else f"{seconds:g} s"


def time_bar(dl, fraction: float = 0.2) -> None:
    """A bar for a round fraction of the visible time, at the bottom right of the current plot, its length under it."""
    pos, size = implot.get_plot_pos(), implot.get_plot_size()
    limits = implot.get_plot_limits().x
    seconds = nice(fraction * (limits.max - limits.min))
    right = pos.x + size.x - D.em(0.6)
    left = right - seconds / (limits.max - limits.min) * size.x
    y = pos.y + size.y - D.em(1.2)
    dl.add_line(ImVec2(left, y), ImVec2(right, y), u32(BAR), 3.0 * P.SCALE)
    D.text(dl, 0.5 * (left + right), y + D.em(0.1), seconds_label(seconds), BAR, 0.85, center=True)


def value_bar(dl, height: float, label: str) -> None:
    """A vertical bar of `height` plot units at the right of the current plot, from its top, labelled to its left."""
    pos, size = implot.get_plot_pos(), implot.get_plot_size()
    limits = implot.get_plot_limits()
    x = pos.x + size.x - D.em(0.6)
    top = pos.y + D.em(0.4)
    bottom = top + height / (limits.y.max - limits.y.min) * size.y
    dl.add_line(ImVec2(x, top), ImVec2(x, bottom), u32(BAR), 3.0 * P.SCALE)
    D.text(dl, x - D.text_width(label, 0.85) - D.em(0.3), 0.5 * (top + bottom) - D.em(0.5), label, BAR, 0.85)


def spike_ticks(dl, spikes: np.ndarray, color: ImVec4 = TRUTH) -> None:
    """A tick per spike along the bottom of the current plot."""
    pos, size = implot.get_plot_pos(), implot.get_plot_size()
    limits = implot.get_plot_limits().x
    y1 = pos.y + size.y - D.em(0.15)
    for s in spikes:
        if limits.min <= s <= limits.max:
            x = implot.plot_to_pixels(float(s), 0.0).x
            dl.add_line(ImVec2(x, y1 - D.em(0.5)), ImVec2(x, y1), u32(color), 1.5 * P.SCALE)


def dots(color: ImVec4) -> implot.Spec:
    sp = P.spec(color, 0.0)
    sp.marker = implot.Marker_.circle
    sp.marker_size = 3.2 * P.SCALE
    sp.marker_fill_color = ImVec4(color.x, color.y, color.z, 0.9)
    sp.marker_line_color = ImVec4(color.x, color.y, color.z, 0.0)
    return sp


def begin_trace(name: str, size: ImVec2, duration: float, legend: bool = True) -> bool:
    """A plot over 0..duration s with no axis decorations; x pans and zooms, y fits the data."""
    flags = P.TRACE_FLAGS | (0 if legend else implot.Flags_.no_legend)
    if not P.begin_plot(name, size, flags):
        return False
    implot.setup_axes("", "", TIME_AXIS, VALUE_AXIS)
    implot.setup_axis_limits(implot.ImAxis_.x1, 0.0, duration, imgui.Cond_.once)
    return True


def window_fill(windows: np.ndarray, n: int, height: float) -> np.ndarray:
    """`height` inside each (start, end) window, 0 elsewhere: a step to shade events with."""
    fill = np.zeros(n)
    for start, end in windows:
        fill[start:end] = height
    return fill


class DenoiseLesson(Lesson):
    key = "denoise"
    title = "Denoising (vnoiser)"
    summary = "the masked Morlet bands of z added back up, plus z's slow part"
    uses_recording = True
    sections = (
        (fa.ICON_FA_WAVE_SQUARE, "From F to z"),
        (fa.ICON_FA_BARS_STAGGERED, "Scalogram"),
        (fa.ICON_FA_LAYER_GROUP, "Bands and masks"),
        (fa.ICON_FA_BULLSEYE, "Denoised and peaks"),
    )
    references = (
        ("Land MA, et al. (2026). JEDI-3 voltage indicators for two-photon imaging of subthreshold and spiking activity. Nat Methods 23:986-997.",
         "https://doi.org/10.1038/s41592-026-03043-8"),
        ("vnoiser: the spatial JEDI pipeline's Morlet wavelet denoiser, as a package.",
         "https://github.com/FlynnOConnell/vnoiser"),
        ("Torrence C, Compo GP (1998). A practical guide to wavelet analysis. Bull. Am. Meteorol. Soc. 79:61-78.",
         "https://doi.org/10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2"),
    )

    def __init__(self):
        self.run: Optional[M.Run] = None
        self.source = ""  # the provenance line under the traces
        self.notes: list = []  # one-line remarks beside it: a calcium indicator, an assumed frame rate
        self.event_fills: list = []  # per kept band, a step at 1 inside its event windows
        self.image_max = 1.0

    def set_recording(self, recording: sim.Recording) -> None:
        self.recording = recording
        self.run, self.notes = None, []
        if not M.available():
            return
        if recording.path:
            f, fs = M.read_trace(recording.path)
            if fs is None:
                fs = M.DEFAULT_FS
                self.notes.append(f"the file gives no frame rate: {fs:g} Hz assumed")
            spikes, negative = None, True
            self.source = f"{Path(recording.path).name}, {fs:g} Hz, {f.size / fs:.1f} s"
        else:
            p = recording.params()
            trace = sim.make_trace(p)
            f, fs, spikes = trace.recorded, p.fs, trace.spikes
            negative = sim.BY_NAME[p.indicator].amp < 0
            if recording.kind == "calcium":
                self.notes.append("vnoiser is a voltage pipeline; it runs on this calcium trace all the same")
            self.source = f"{p.indicator}, simulated, {fs:g} Hz, {p.duration:g} s"
        self.run = M.run(f=f, fs=fs, spikes=spikes, negative=negative)
        self.event_fills = [window_fill(band.windows, f.size, 1.0) for band in self.run.bands]
        self.image_max = float(np.percentile(self.run.image, 99.5)) or 1.0

    def page(self) -> None:
        if not M.available():
            imgui.text_colored(P.WARN, INSTALL)
            return
        if self.run is not None:
            super().page()

    # ------------------------------------------------------------------
    # page
    # ------------------------------------------------------------------

    def section_0(self) -> None:
        D.heading(*self.sections[0])
        em, w = D.em(), D.guide_width()
        r = self.run
        t, fs = r.t, r.fs
        cfg = M.dfof_config(fs)
        unit = "photons / frame" if r.spikes is not None else "counts"
        sign = "-" if sim.BY_NAME[self.recording.indicator].amp < 0 or r.spikes is None else ""
        rows = (
            ("##f", (f"F ({'simulated' if r.spikes is not None else 'file'}), {unit}", r.f, P.RAW, 1.0),
             (f"Gaussian baseline, sigma {cfg.sigma_dfof / fs:.1f} s", r.f_baseline, KEY, 1.8), unit),
            ("##dfof", (f"dF/F, first {cfg.n_startup / fs:.1f} s replaced", r.dfof, P.RAW, 1.0),
             (f"slower baseline, sigma {cfg.sigma_baseline / fs:.1f} s", r.dfof_baseline, KEY, 1.8), "dF/F"),
            ("##z", (f"z = {sign}(dF/F - baseline) / SD", r.z, P.RAW, 1.0), None, ""),
        )
        for name, line, baseline, bar_unit in rows:
            if not begin_trace(name, ImVec2(w, 7 * em), r.duration):
                continue
            values = line[1] if baseline is None else np.concatenate([line[1], baseline[1]])
            lo, hi = float(values.min()), float(values.max())
            span = (hi - lo) or 1.0
            implot.setup_axis_limits(implot.ImAxis_.x1, 0.0, r.duration, imgui.Cond_.always)
            implot.setup_axis_limits(implot.ImAxis_.y1, lo - (0.35 if name == "##z" else 0.05) * span, hi + 0.4 * span, imgui.Cond_.always)
            implot.plot_line(line[0], t, np.ascontiguousarray(line[1]), P.spec(line[2], line[3]))
            if baseline is not None:
                implot.plot_line(baseline[0], t, np.ascontiguousarray(baseline[1]), P.spec(baseline[2], baseline[3]))
            if name == "##z" and r.spikes is not None:
                implot.plot_dummy("Ground truth spikes (simulated)", P.spec(TRUTH, 2.0))
            dl = implot.get_plot_draw_list()
            implot.push_plot_clip_rect()
            if bar_unit:
                height = nice(0.3 * span)
                label = f"{100 * height:g}% dF/F" if bar_unit == "dF/F" else f"{height:g} {bar_unit}"
                value_bar(dl, height, label)
            if name == "##z":
                if r.spikes is not None:
                    spike_ticks(dl, r.spikes)
                time_bar(dl)
            implot.pop_plot_clip_rect()
            implot.end_plot()
        imgui.text_disabled("  /  ".join([self.source, *self.notes]))

    def section_1(self) -> None:
        D.heading(*self.sections[1])
        em, w = D.em(), D.guide_width()
        r = self.run
        n_rows = r.freqs.size
        shown = range(0, n_rows, 10)
        implot.push_colormap(implot.Colormap_.viridis)
        if P.begin_plot("##scalogram", ImVec2(w, 16 * em), P.TRACE_FLAGS | implot.Flags_.no_legend):
            implot.setup_axes("", "", TIME_AXIS, implot.AxisFlags_.lock | implot.AxisFlags_.no_grid_lines | implot.AxisFlags_.no_tick_marks)
            implot.setup_axis_limits(implot.ImAxis_.x1, 0.0, r.duration, imgui.Cond_.once)
            implot.setup_axis_limits(implot.ImAxis_.y1, 0.0, float(n_rows), imgui.Cond_.always)
            implot.setup_axis_ticks(implot.ImAxis_.y1, [n_rows - i - 0.5 for i in shown], [f"{M.fmt_hz(r.freqs[i])} Hz" for i in shown])
            implot.plot_heatmap("##amp", np.ascontiguousarray(r.image, dtype=np.float64), 0.0, self.image_max, "",
                                implot.Point(0.0, 0.0), implot.Point(r.duration, float(n_rows)))
            edges = []
            for band in r.bands:
                edges.append(n_rows - int(np.argmin(np.abs(r.freqs - band.hi_hz))))
                edges.append(n_rows - int(np.argmin(np.abs(r.freqs - band.lo_hz))) - 1)
            if edges:
                implot.plot_inf_lines("##edges", np.array(edges, dtype=np.float64),
                                      P.spec(ImVec4(P.ACCENT.x, P.ACCENT.y, P.ACCENT.z, 0.7), 1.0, flags=implot.InfLinesFlags_.horizontal))
            dl = implot.get_plot_draw_list()
            implot.push_plot_clip_rect()
            time_bar(dl)
            implot.pop_plot_clip_rect()
            implot.end_plot()
        implot.pop_colormap()
        imgui.text_disabled(f"|CWT| / sqrt(scale), cmor0.5-1.0, {n_rows} scales; the lines are the edges of the {len(r.bands)} kept bands")

    def section_2(self) -> None:
        D.heading(*self.sections[2])
        em, w = D.em(), D.guide_width()
        r = self.run
        t = r.t
        for label, color in (("band: the mean of its CWT rows", P.RAW), ("mask", P.ACCENT), ("band x mask", KEY)):
            imgui.text_colored(color, label)
            imgui.same_line(0, 1.5 * em)
        imgui.new_line()
        if not r.bands:
            imgui.text_colored(P.WARN, "no band had an event, so nothing is kept")
            return
        n = len(r.bands)
        flags = implot.SubplotFlags_.link_all_x | implot.SubplotFlags_.no_title | implot.SubplotFlags_.no_legend
        if not implot.begin_subplots("##bands", n, 1, ImVec2(w, n * 3.6 * em), flags):
            return
        for k, band in enumerate(r.bands):
            if not begin_trace(f"##band{k}", ImVec2(-1, 0), r.duration, legend=False):
                continue
            implot.setup_axis(implot.ImAxis_.y2, None, implot.AxisFlags_.no_decorations | implot.AxisFlags_.lock)
            implot.setup_axis_limits(implot.ImAxis_.y2, -0.02, 1.35, imgui.Cond_.always)
            implot.set_axes(implot.ImAxis_.x1, implot.ImAxis_.y2)
            implot.plot_shaded("##events", t, np.ascontiguousarray(1.35 * self.event_fills[k]), 0.0,
                               spec=P.spec(ImVec4(P.ACCENT.x, P.ACCENT.y, P.ACCENT.z, 1.0), 0.0, fill_alpha=0.2))
            implot.plot_line("##mask", t, np.ascontiguousarray(band.mask), P.spec(P.ACCENT, 1.0))
            implot.set_axes(implot.ImAxis_.x1, implot.ImAxis_.y1)
            implot.plot_line("##band", t, np.ascontiguousarray(band.trace), P.spec(P.RAW, 1.0))
            implot.plot_line("##masked", t, np.ascontiguousarray(band.trace * band.mask), P.spec(KEY, 1.2))
            dl = implot.get_plot_draw_list()
            pos = implot.get_plot_pos()
            implot.push_plot_clip_rect()
            D.text(dl, pos.x + 0.5 * em, pos.y + 0.15 * em, f"{band.span}, {band.windows.shape[0]} events", P.DIM, 0.8)
            if k == n - 1:
                time_bar(dl)
            implot.pop_plot_clip_rect()
            implot.end_plot()
        implot.end_subplots()

    def section_3(self) -> None:
        D.heading(*self.sections[3])
        em, w = D.em(), D.guide_width()
        r = self.run
        t = r.t
        if begin_trace("##denoised", ImVec2(w, 14 * em), r.duration):
            implot.plot_line("z", t, np.ascontiguousarray(r.z), P.spec(P.RAW, 1.0))
            implot.plot_line("denoised", t, np.ascontiguousarray(r.denoised), P.spec(P.ACCENT, 1.5))
            implot.plot_line("the lab's saved trace", t, np.ascontiguousarray(r.final), P.spec(LAB, 1.0))
            if r.peaks.size:
                implot.plot_scatter("peaks", np.ascontiguousarray(t[r.peaks]), np.ascontiguousarray(r.final[r.peaks]), dots(KEY))
            if r.spikes is not None:
                implot.plot_dummy("Ground truth spikes (simulated)", P.spec(TRUTH, 2.0))
            dl = implot.get_plot_draw_list()
            implot.push_plot_clip_rect()
            if r.spikes is not None:
                spike_ticks(dl, r.spikes)
            time_bar(dl)
            implot.pop_plot_clip_rect()
            implot.end_plot()
        imgui.text_disabled("denoised = bands x masks + the 1 Hz baseline of z; the saved trace = bands x masks + baseline[0]; peaks: detect_peaks on the saved trace")
        imgui.dummy(ImVec2(0, 0.4 * em))
        push_size(2.2)
        imgui.text_colored(KEY, f"{r.peaks.size} {plural('peak', r.peaks.size)}")
        imgui.pop_font()
        if r.hits is not None:
            hit, n_spikes, stray = r.hits
            imgui.same_line(0, 1.5 * em)
            imgui.text_disabled(f"{hit} of {n_spikes} spikes have a peak within {1000 * M.TOLERANCE_S:g} ms, "
                                f"{stray} {plural('peak', stray)} with none")
