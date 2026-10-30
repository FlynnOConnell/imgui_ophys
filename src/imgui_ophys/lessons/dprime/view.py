"""
The Signal quality (d') stage, built the way the story tells it: one trace, one spike +/- 0.8 s, the frames before
the spike and at its peak clustered, the number; then example traces at d' = 0.5 to 4. Scored on the Traces
simulator with the spike times known, and re-scored as the light and the bleaching change.
"""

import math
from dataclasses import replace

import numpy as np
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui, implot

from imgui_ophys import diagram as D
from imgui_ophys import plotting as P
from imgui_ophys import sim
from imgui_ophys.lesson import Lesson
from imgui_ophys.lessons.dprime import model as M
from imgui_ophys.widgets import KEY, push_size, section, u32

ImVec2, ImVec4 = imgui.ImVec2, imgui.ImVec4
TRACE = ImVec4(0.70, 0.70, 0.72, 1.0)
BAR = ImVec4(0.52, 0.52, 0.55, 1.0)  # scale bars sit back a little from the traces
TRUTH = ImVec4(0.35, 0.85, 0.45, 1.0)  # ground truth spikes
BEFORE = P.ACCENT  # the frames just before a spike
PEAK = KEY  # the frames at the response peak
WINDOW = {"calcium": 6, "voltage": 1}  # frames in each window: the jGCaMP8 paper's six; a 1 ms spike fills one frame
ZOOM = 24  # frames either side of the followed spike
F0_WINDOW = 5.0  # s, the running 20th-percentile baseline
MORPH = 1.2  # s a parameter change takes to play out: the trace morphs into the new one and d' counts along
TARGETS = (0.5, 1.0, 2.0, 3.0, 4.0)  # the examples: white noise added to a clean cell until d' lands here
# the cells the story shows, as (brightness over the kind's opening brightness, bleach, F0); the plot ranges are
# fixed over them so a change is seen as motion
REFERENCE = ((1.0, 0.2, "running window"), (0.2, 0.2, "running window"), (5.0, 0.2, "running window"), (5.0, 0.35, "whole recording"))
PRESETS = (
    ("Add noise", 0.2, 0.2, "running window"),
    ("Remove noise", 5.0, 0.2, "running window"),
    ("Add photobleaching", 5.0, 0.35, "whole recording"),
    ("Reset", 1.0, 0.2, "running window"),
)
DURATION = {"calcium": 40.0, "voltage": 10.0}  # s


def to_dff(f: np.ndarray, fs: float, f0: str, ind: sim.Indicator) -> np.ndarray:
    """dF/F against the chosen F0, with depolarization up whatever the indicator's polarity."""
    return sim.upright(sim.to_dff(f, sim.baseline(f, f0, int(F0_WINDOW * fs))), ind)


def nice(x: float) -> float:
    """x rounded down to 1, 2 or 5 times a power of ten."""
    e = 10 ** math.floor(math.log10(x))
    return e * max(m for m in (1.0, 2.0, 5.0) if m * e <= x)


def px(x: float, y: float) -> ImVec2:
    return implot.plot_to_pixels(implot.Point(float(x), float(y)))


def begin_hidden(name: str, size: ImVec2, x0: float, x1: float, y0: float, y1: float, legend: bool = False) -> bool:
    """A plot with no axes, locked to these limits."""
    flags = P.TRACE_FLAGS | (0 if legend else implot.Flags_.no_legend)
    if not P.begin_plot(name, size, flags):
        return False
    implot.setup_axes("", "", P.HIDDEN_AXIS, P.HIDDEN_AXIS)
    implot.setup_axis_limits(implot.ImAxis_.x1, x0, x1, imgui.Cond_.always)
    implot.setup_axis_limits(implot.ImAxis_.y1, y0, y1, imgui.Cond_.always)
    return True


def dots(color: ImVec4) -> implot.Spec:
    sp = P.spec(color, 0.0)
    sp.marker = implot.Marker_.circle
    sp.marker_size = 3.2 * P.SCALE
    sp.marker_fill_color = ImVec4(color.x, color.y, color.z, 0.9)
    sp.marker_line_color = ImVec4(color.x, color.y, color.z, 0.0)
    return sp


def seconds_label(seconds: float) -> str:
    return f"{seconds:g} s" if seconds >= 0.1 else f"{1000 * seconds:g} ms"


def time_bar(dl, seconds: float, x_max: float, y: float) -> None:
    """A horizontal bar ending at x_max with its length written under it."""
    a, b = px(x_max - seconds, y), px(x_max, y)
    dl.add_line(a, b, u32(BAR), 3.0 * P.SCALE)
    D.text(dl, 0.5 * (a.x + b.x), b.y + D.em(0.1), seconds_label(seconds), BAR, 0.85, center=True)


def dff_bar(dl, dff: float, x_max: float, y_hi: float, unit: str = "dF/F") -> None:
    """A vertical bar at x_max down from y_hi with its height written to its left."""
    a, b = px(x_max, y_hi - dff), px(x_max, y_hi)
    dl.add_line(a, b, u32(BAR), 3.0 * P.SCALE)
    label = f"{100 * dff:g}% {unit}"
    D.text(dl, a.x - D.text_width(label, 0.85) - D.em(0.3), 0.5 * (a.y + b.y) - D.em(0.5), label, BAR, 0.85)


def vertical(dl, x: float, y0: float, y1: float, color: ImVec4, thickness: float, cap: float = 0.0) -> None:
    a, b = px(x, y0), px(x, y1)
    dl.add_line(a, b, u32(color), thickness * P.SCALE)
    for p in (a, b) if cap else ():
        dl.add_line(ImVec2(p.x - cap, p.y), ImVec2(p.x + cap, p.y), u32(color), thickness * P.SCALE)


class DprimeLesson(Lesson):
    key = "dprime"
    title = "Signal quality (d')"
    summary = "a measure of how well spikes stand out from noise"
    uses_recording = True
    sections = (
        (fa.ICON_FA_WAVE_SQUARE, "Trace"),
        (fa.ICON_FA_MAGNIFYING_GLASS, "Spike +/- 0.8 s"),
        (fa.ICON_FA_CIRCLE_DOT, "Clusters"),
        (fa.ICON_FA_TABLE_LIST, "Examples"),
    )
    references = (
        ("Zhang Y, Rozsa M, Liang Y, et al. (2023). Fast and sensitive GCaMP calcium indicators for imaging neural populations. Nature 615:884-891. Methods, 'Screening in neuronal cell culture' and 'Imaging in mouse cortex'.",
         "https://doi.org/10.1038/s41586-023-05828-9"),
        ("McNicol D (2004). A Primer of Signal Detection Theory. Psychology Press.",
         "https://doi.org/10.4324/9781410611949"),
        ("Chen T-W, Wardill TJ, Sun Y, et al. (2013). Ultrasensitive fluorescent proteins for imaging neuronal activity. Nature 499:295-300.",
         "https://doi.org/10.1038/nature12354"),
    )

    def __init__(self):
        self.params = self.cell(self.recording)
        self.frames = WINDOW[self.recording.kind]
        self.f0 = "running window"
        self.chosen_time = math.nan  # s, the spike the zoom follows
        self.dff = None
        self.morph = MORPH  # s into the current morph
        self.reseed()

    @staticmethod
    def cell(recording: sim.Recording) -> sim.TraceParams:
        return recording.params(duration=DURATION[recording.kind], seed=3)

    def set_recording(self, recording: sim.Recording) -> None:
        """A new indicator reseeds (new spike train, ranges and examples); a new SNR morphs the trace."""
        self.recording = recording
        was = self.params
        self.params = replace(self.cell(recording), bleach=was.bleach)
        self.frames = WINDOW[recording.kind]
        if self.params.indicator != was.indicator:
            self.reseed()
        else:
            self.make()

    @property
    def indicator(self) -> sim.Indicator:
        return sim.BY_NAME[self.params.indicator]

    @property
    def unit(self) -> str:
        return "-dF/F" if self.indicator.amp < 0 else "dF/F"

    def dff_of(self, times_brightness: float, bleach: float, f0: str) -> np.ndarray:
        p = replace(self.params, brightness=times_brightness * sim.BRIGHTNESS[self.indicator.kind], bleach=bleach)
        return to_dff(sim.make_trace(p).recorded, p.fs, f0, self.indicator)

    def rescore(self, dff: np.ndarray, spikes: np.ndarray) -> M.Score:
        return M.score(dff, spikes, self.params.fs, self.peak, self.frames, self.quiet)

    def reseed(self) -> None:
        """Everything that depends on the spike train: the plot ranges, the followed spike and the example rows."""
        p = self.params
        self.peak = M.peak_frames(self.indicator, p.fs)
        self.quiet = M.quiet_before(self.indicator)
        spikes = sim.make_trace(p).spikes
        traces = [self.dff_of(*cell) for cell in REFERENCE]
        lo = float(min(np.percentile(d, 0.05) for d in traces))
        hi = float(max(np.percentile(d, 99.95) for d in traces))
        pad = 0.05 * (hi - lo)
        self.y_range = (lo - pad, hi + pad)
        pools = np.concatenate([np.r_[s.top.ravel(), s.bottom.ravel()] for s in (self.rescore(d, spikes) for d in traces)])
        if pools.size:
            lo, hi = float(np.percentile(pools, 0.5)), float(np.percentile(pools, 99.5))
            pad = 0.12 * (hi - lo)
            self.z_range = (lo - pad, hi + pad)
        else:
            self.z_range = self.y_range
        base = self.rescore(traces[0], spikes)
        amplitudes = base.amplitudes
        self.chosen_time = float(base.spikes[np.argmin(np.abs(amplitudes - np.median(amplitudes)))]) if base.n else math.nan
        clean = replace(p, brightness=5000.0, bleach=0.0, neuropil=0.0, read_noise=0.0)
        dff_clean = to_dff(sim.make_trace(clean).recorded, p.fs, "running window", self.indicator)
        half = ZOOM / p.fs
        self.sections = (*self.sections[:1], (self.sections[1][0], f"Spike +/- {seconds_label(half)}"), *self.sections[2:])
        reachable = tuple(d for d in TARGETS if d < 0.98 * self.rescore(dff_clean, spikes).d)
        self.examples = M.noise_rows(dff_clean, spikes, p.fs, self.peak, self.frames, self.quiet, reachable)
        self.dff = None
        self.make()

    def make(self) -> None:
        """Re-simulate and start the morph from what is on screen to the new trace; a new spike train snaps."""
        p = self.params
        self.trace = sim.make_trace(p)
        self.dff_to = to_dff(self.trace.recorded, p.fs, self.f0, self.indicator)
        target = self.rescore(self.dff_to, self.trace.spikes)
        self.d_to = target.d
        self.chosen = int(np.argmin(np.abs(target.spikes - self.chosen_time))) if target.n else None
        if self.dff is None:
            self.dff_from, self.d_from = self.dff_to, self.d_to
            self.settle()
            return
        self.dff_from, self.d_from = self.dff, self.score.d
        self.morph = 0.0
        self.show(0.0)

    def show(self, u: float) -> None:
        """Put the trace u of the way (0 to 1) from the old to the new on screen, with d' counting linearly between the two."""
        b = u * u * (3 - 2 * u)
        self.dff = (1 - b) * self.dff_from + b * self.dff_to
        self.score = self.rescore(self.dff, self.trace.spikes)
        self.score.d = (1 - b) * self.d_from + b * self.d_to

    def settle(self) -> None:
        """Finish the morph now."""
        self.morph = MORPH
        self.show(1.0)

    def update(self, dt: float) -> None:
        if self.morph < MORPH:
            self.morph = min(MORPH, self.morph + dt)
            self.show(self.morph / MORPH)

    def animating(self) -> bool:
        return self.morph < MORPH

    def zoom_bounds(self) -> tuple[int, int, int]:
        """(first frame, last frame + 1, spike frame) of the window around the followed spike."""
        center = int(self.score.centers[self.chosen])
        half = ZOOM
        return max(0, center - half), min(self.dff.size, center + self.peak + half), center

    # ------------------------------------------------------------------
    # rail
    # ------------------------------------------------------------------

    def controls(self) -> None:
        """The story's cells: a fifth of the light, five times the light, and five times the light with photobleaching."""
        section(f"{fa.ICON_FA_WAND_MAGIC_SPARKLES}  Cells")
        rec, ind, p = self.recording, self.indicator, self.params
        for name, times_brightness, bleach, f0 in PRESETS:
            if imgui.button(name):
                brightness = times_brightness * sim.BRIGHTNESS[ind.kind]
                rec.snr = sim.snr_of(sim.spike_peak(ind, p.fs), brightness, p.pixels, p.read_noise, p.neuropil)
                p.bleach, self.f0 = bleach, f0
                if self.touch:
                    self.touch()
                else:
                    self.set_recording(rec)
        imgui.push_style_color(imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled))
        imgui.text_wrapped(f"{self.frames} frame{'s' if self.frames > 1 else ''} per window; a spike is scored with none in the {seconds_label(self.quiet)} before it")
        imgui.pop_style_color()

    # ------------------------------------------------------------------
    # page
    # ------------------------------------------------------------------

    def section_0(self) -> None:
        D.heading(*self.sections[0])
        em, w = D.em(), D.guide_width()
        t, dff, s = self.trace.t, self.dff, self.score
        lo, hi = self.y_range
        span = hi - lo
        if begin_hidden("##dprime_trace", ImVec2(w, 11 * em), float(t[0]), float(t[-1]), lo - 0.5 * span, hi + 0.05 * span, legend=True):
            implot.plot_dummy("Ground truth spikes (simulated)", P.spec(TRUTH, 2.0))
            implot.plot_line("##dff", t, np.ascontiguousarray(dff), P.spec(TRACE, 1.3))
            dl = implot.get_plot_draw_list()
            implot.push_plot_clip_rect()
            for sp in self.trace.spikes:
                vertical(dl, sp, lo - 0.1 * span, lo - 0.03 * span, TRUTH, 2.0)
            if self.chosen is not None:
                a, b, _ = self.zoom_bounds()
                dl.add_rect(px(t[a], hi), px(t[b - 1], lo), u32(P.DIM), 0, 1.5 * P.SCALE)
            time_bar(dl, nice(float(t[-1]) / 8), float(t[-1]), lo - 0.24 * span)
            dff_bar(dl, nice(0.3 * span), float(t[-1]), hi, self.unit)
            implot.pop_plot_clip_rect()
            implot.end_plot()

    def section_1(self) -> None:
        D.heading(*self.sections[1])
        em, w = D.em(), D.guide_width()
        if self.chosen is None:
            imgui.text_colored(P.WARN, "no isolated spike to score")
            return
        t, dff, fs = self.trace.t, self.dff, self.params.fs
        a, b, center = self.zoom_bounds()
        lo, hi = self.z_range
        span = hi - lo
        before = np.arange(center - self.frames, center)
        peak = np.arange(center + self.peak, center + self.peak + self.frames)
        if begin_hidden("##dprime_zoom", ImVec2(min(w * 0.6, 34 * em), 11 * em), float(t[a]), float(t[b - 1]), lo - 0.32 * span, hi, legend=True):
            implot.plot_line("##zoom", t[a:b], np.ascontiguousarray(dff[a:b]), P.spec(TRACE, 1.6))
            frames = f"{self.frames} frames" if self.frames > 1 else "the frame"
            implot.plot_scatter(f"{frames} before the spike", t[before], np.ascontiguousarray(np.clip(dff[before], lo, hi)), dots(BEFORE))
            implot.plot_scatter(f"{frames} at the peak", t[peak], np.ascontiguousarray(np.clip(dff[peak], lo, hi)), dots(PEAK))
            dl = implot.get_plot_draw_list()
            implot.push_plot_clip_rect()
            time_bar(dl, nice(float(t[b - 1] - t[a]) / 3), float(t[b - 1]), lo - 0.1 * span)
            implot.pop_plot_clip_rect()
            implot.end_plot()

    def section_2(self) -> None:
        D.heading(*self.sections[2])
        em = D.em()
        s = self.score
        if self.chosen is None:
            imgui.text_colored(P.WARN, "no isolated spike to score")
            return
        lo, hi = self.z_range
        top, bottom = s.top.ravel(), s.bottom.ravel()
        pooled = math.sqrt(0.5 * (top.var() + bottom.var()))
        x_before, x_peak = 0.5, 1.5
        jitter = np.random.default_rng(0).uniform(-0.28, 0.28, top.size)
        if begin_hidden("##dprime_clusters", ImVec2(16 * em, 12 * em), 0.0, 2.0, lo, hi):
            implot.plot_scatter("##before", x_before + jitter, np.clip(bottom, lo, hi), dots(BEFORE))
            implot.plot_scatter("##peak", x_peak + jitter, np.clip(top, lo, hi), dots(PEAK))
            for x, values, color in ((x_before, bottom, BEFORE), (x_peak, top, PEAK)):
                implot.plot_line(f"##mean{x}", np.array([x - 0.42, x + 0.42]), np.full(2, float(values.mean())), P.spec(color, 3.0))
            dl = implot.get_plot_draw_list()
            implot.push_plot_clip_rect()
            vertical(dl, x_before - 0.46, bottom.mean() - pooled, bottom.mean() + pooled, BEFORE, 2.0, cap=0.3 * em)
            vertical(dl, x_peak + 0.46, top.mean() - pooled, top.mean() + pooled, PEAK, 2.0, cap=0.3 * em)
            a, b = px(1.0, bottom.mean()), px(1.0, top.mean())
            D.arrow(dl, a.x, a.y, b.x, b.y, D.TEXT, 2.0 * P.SCALE)
            D.arrow(dl, b.x, b.y, a.x, a.y, D.TEXT, 2.0 * P.SCALE)
            implot.pop_plot_clip_rect()
            implot.end_plot()
        imgui.same_line(0, 2 * em)
        imgui.begin_group()
        imgui.dummy(ImVec2(0, 3.5 * em))
        push_size(2.2)
        imgui.text_colored(PEAK, f"d' = {s.d:.1f}")
        imgui.pop_font()
        imgui.text_disabled(f"{s.n} spikes x {self.frames} frame{'s' if self.frames > 1 else ''}")
        imgui.end_group()

    def section_3(self) -> None:
        D.heading(*self.sections[3])
        em, w = D.em(), D.guide_width()
        t = self.trace.t
        values = np.concatenate([dff for _, dff in self.examples])
        lo, hi = float(np.percentile(values, 0.5)), float(np.percentile(values, 99.5))
        row_h = 2.6 * em
        if not imgui.begin_table("##dprime_examples", 3, imgui.TableFlags_.none, ImVec2(w, 0)):
            return
        imgui.table_setup_column("trace", imgui.TableColumnFlags_.width_stretch)
        imgui.table_setup_column("d'", imgui.TableColumnFlags_.width_fixed, 5 * em)
        imgui.table_setup_column("% correct", imgui.TableColumnFlags_.width_fixed, 6 * em)
        imgui.table_next_row()
        for k, name in enumerate(("", "d'", "% correct")):
            imgui.table_set_column_index(k)
            imgui.text_disabled(name)
        for k, (d, dff) in enumerate(self.examples):
            imgui.table_next_row()
            imgui.table_set_column_index(0)
            if begin_hidden(f"##example{k}", ImVec2(-1, row_h), float(t[0]), float(t[-1]), lo, hi):
                implot.plot_line("##dff", t, np.ascontiguousarray(dff), P.spec(TRACE, 1.2))
                implot.end_plot()
            for column, text, color in ((1, f"d' = {d:.1f}", PEAK), (2, f"{100 * M.accuracy(d):.0f}%", D.TEXT)):
                imgui.table_set_column_index(column)
                imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 0.5 * (row_h - em))
                imgui.text_colored(color, text)
        imgui.end_table()
