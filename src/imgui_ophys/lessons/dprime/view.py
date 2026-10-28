"""
The Signal quality (d') stage: what the score says, how it is computed from a trace, what moves it, and what it
cannot see. Scored on the Traces simulator with the spike times known.
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
from imgui_ophys.widgets import KEY, help_mark, section, table

ImVec2, ImVec4 = imgui.ImVec2, imgui.ImVec4
TRUTH = ImVec4(0.35, 0.85, 0.45, 1.0)  # the true spikes and the true dF/F
TOP = KEY  # peak windows
BOTTOM = ImVec4(0.62, 0.62, 0.70, 1.0)  # baseline windows
DETECTED = P.HIGHLIGHT  # what a threshold picker called a spike
SWEEP_DELAY = 0.3  # s after the last parameter change before the curves re-simulate
SWEEP_DURATION = 60.0  # s per re-simulated recording in "What moves it"
F0_WINDOW = 5.0  # s, the running 20th-percentile baseline for dF/F, so bleaching does not drift the pools
ACCURACY_ROWS = (1.0, 2.0, 3.0, 4.0)
# one line per curve: what the setting does to the numerator or the denominator
WHY = {
    "brightness": "signal grows with photons N, shot noise with sqrt(N): d' goes up like sqrt(N)",
    "pixels": "more pixels averaged is more photons: sqrt(pixels), until the ROI leaves the cell",
    "neuropil": "light that is not the cell adds shot noise and no signal",
    "read_noise": "only matters once the photons per frame are few",
    "fs": "faster frames split the same light: fewer photons each, and the window spans less of the decay",
}


def to_dff(f: np.ndarray, fs: float) -> np.ndarray:
    return sim.to_dff(f, sim.baseline(f, "running window", int(F0_WINDOW * fs)))


class DprimeLesson(Lesson):
    key = "dprime"
    title = "Signal quality (d')"
    summary = "how far one spike stands above the noise"
    sections = (
        (fa.ICON_FA_CHART_SIMPLE, "What d' says"),
        (fa.ICON_FA_CALCULATOR, "How it is computed"),
        (fa.ICON_FA_SLIDERS, "What moves it"),
        (fa.ICON_FA_EYE_SLASH, "What it misses"),
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
        self.params = sim.TraceParams(duration=60.0)
        self.frames = 6  # window length
        self.threshold = 3.0  # robust SDs, for the event picker
        self.sweeps: dict[str, np.ndarray] = {}
        self.indicators: list[tuple[str, float]] = []
        self.sweep_wait = 0.0  # s left before the curves re-simulate; 0 when they are current
        self.sweep_params = None
        self.make()
        self.resweep()

    @property
    def indicator(self) -> sim.Indicator:
        return sim.BY_NAME[self.params.indicator]

    def make(self) -> None:
        p = self.params
        self.trace = sim.make_trace(p)
        self.peak = M.peak_frames(self.indicator, p.fs)
        self.quiet = M.quiet_before(self.indicator)
        self.dff = to_dff(self.trace.recorded, p.fs)
        self.score = M.score(self.dff, self.trace.spikes, p.fs, self.peak, self.frames, self.quiet)
        self.redetect()
        self.sweep_wait = SWEEP_DELAY

    def redetect(self) -> None:
        p = self.params
        gap = max(1, self.peak + self.frames)
        self.detected = M.detected_spike_times(self.dff, p.fs, self.peak, self.threshold, gap)
        self.score_detected = M.score(self.dff, self.detected, p.fs, self.peak, self.frames, self.quiet)
        silent = sim.make_trace(replace(p, rate=0.0))
        dff = to_dff(silent.recorded, p.fs)
        self.detected_silent = M.detected_spike_times(dff, p.fs, self.peak, self.threshold, gap)
        self.score_silent = M.score(dff, self.detected_silent, p.fs, self.peak, self.frames, self.quiet)

    def resweep(self) -> None:
        p = replace(self.params, duration=max(self.params.duration, SWEEP_DURATION))
        self.sweeps = {attr: M.sweep(p, attr, values, self.frames) for attr, _, values in M.SWEEPS}
        self.indicators = M.by_indicator(p, self.frames)
        self.sweep_wait = 0.0

    def update(self, dt: float) -> None:
        if self.sweep_wait > 0:
            self.sweep_wait = max(0.0, self.sweep_wait - dt)
            if self.sweep_wait == 0:
                self.resweep()

    def animating(self) -> bool:
        return self.sweep_wait > 0

    # ------------------------------------------------------------------
    # rail
    # ------------------------------------------------------------------

    def controls(self) -> None:
        p = self.params
        section(f"{fa.ICON_FA_DNA}  Indicator")
        names = [i.name for i in M.CALCIUM]
        changed, idx = imgui.combo("indicator", names.index(p.indicator), names)
        dirty = changed
        if changed:
            p.indicator = names[idx]
        help_mark(self.indicator.note)
        for label, attr, lo, hi, fmt in (
            ("frame rate", "fs", 5.0, 200.0, "%.0f Hz"),
            ("duration", "duration", 10.0, 300.0, "%.0f s"),
            ("spike rate", "rate", 0.05, 5.0, "%.2f /s"),
        ):
            c, v = imgui.slider_float(label, getattr(p, attr), lo, hi, fmt, imgui.SliderFlags_.logarithmic)
            if c:
                setattr(p, attr, v)
                dirty = True

        section(f"{fa.ICON_FA_CAMERA}  Light")
        for label, attr, lo, hi, fmt, tip in (
            ("brightness", "brightness", 0.2, 200.0, "%.1f ph/px", "Photons per pixel per frame at rest: laser power, expression, depth."),
            ("ROI pixels", "pixels", 1, 500, "%d", "Pixels averaged into the trace."),
            ("read noise", "read_noise", 0.0, 10.0, "%.1f e-", "Electrons RMS per pixel per frame."),
            ("neuropil", "neuropil", 0.0, 1.5, "%.2f", "Neuropil light / ROI light."),
            ("bleaching", "bleach", 0.0, 0.8, "%.2f", "Fraction of light lost by the end."),
        ):
            if attr == "pixels":
                c, v = imgui.slider_int(label, p.pixels, lo, hi, fmt, imgui.SliderFlags_.logarithmic)
            else:
                c, v = imgui.slider_float(label, getattr(p, attr), lo, hi, fmt, imgui.SliderFlags_.logarithmic if attr == "brightness" else 0)
            if c:
                setattr(p, attr, v)
                dirty = True
            help_mark(tip)
        c, p.seed = imgui.input_int("seed", p.seed)
        dirty |= c

        section(f"{fa.ICON_FA_CALCULATOR}  d'")
        c, self.frames = imgui.slider_int("window", self.frames, 1, 20, "%d frames")
        dirty |= c
        help_mark("Frames in each window. The paper used 6 at 200 Hz (30 ms). The top window starts at the response peak, so a long window at a slow frame rate averages in the decay.")
        imgui.text_disabled(f"isolated: no spike {self.quiet:.2f} s before")
        help_mark(f"{M.QUIET_DECAYS:.0f} decay time constants of {self.indicator.name}, so the previous response is out of the baseline window.")
        if dirty:
            self.make()

        section(f"{fa.ICON_FA_CROSSHAIRS}  Event picker")
        c, self.threshold = imgui.slider_float("threshold", self.threshold, 1.0, 6.0, "%.1f SD")
        if c:
            self.redetect()
        help_mark("Local maxima of dF/F above the median by this many robust SDs. What you are left with when the spike times are not known.")

    # ------------------------------------------------------------------
    # 0: what it says
    # ------------------------------------------------------------------

    def section_0(self) -> None:
        D.heading(fa.ICON_FA_CHART_SIMPLE, "What d' says",
                  "d' is the height of a one-spike bump divided by the SD of the noise. Frames before a spike and frames at its peak each make a pile; d' is the distance between the piles in units of their width.")
        em = D.em()
        s = self.score
        w = D.guide_width()
        plot_w = min(w * 0.62, 40 * em)
        if P.begin_plot("##dprime_piles", ImVec2(plot_w, 14 * em)):
            implot.setup_axes("dF/F", "frames", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            if s.n:
                bottom, top = s.bottom.ravel(), s.top.ravel()
                lo, hi = float(min(bottom.min(), top.min())), float(max(bottom.max(), top.max()))
                edges = np.linspace(lo, hi, 25)
                width = float(edges[1] - edges[0])
                centers = (edges[:-1] + edges[1:]) / 2
                for name, values, color in (("before the spike", bottom, BOTTOM), ("at the peak", top, TOP)):
                    counts, _ = np.histogram(values, edges)
                    implot.plot_bars(name, centers, counts.astype(np.float64), width * 0.9, P.spec(color, 1.0, 0.45))
                pooled = math.sqrt(0.5 * (bottom.var() + top.var()))
                xs = np.linspace(lo, hi, 200)
                scale = bottom.size * width
                for values, color in ((bottom, BOTTOM), (top, TOP)):
                    curve = scale / (pooled * math.sqrt(2 * math.pi)) * np.exp(-((xs - values.mean()) ** 2) / (2 * pooled**2))
                    implot.plot_line(f"##fit{id(values)}", xs, curve, P.spec(color, 2.0))
                mid = (bottom.mean() + top.mean()) / 2
                implot.plot_inf_lines("ideal threshold", np.array([mid]), P.spec(DETECTED, 1.5))
            implot.end_plot()
        imgui.same_line(0, 1.5 * em)
        imgui.begin_group()
        if s.n:
            imgui.text_colored(KEY, f"d' = {s.d:.2f}")
            imgui.text_wrapped(f"The piles are {s.d:.1f} noise SDs apart. A threshold halfway between them calls a single spike correctly {100 * M.accuracy(s.d):.0f}% of the time, and that is the best any threshold can do.")
        else:
            imgui.text_colored(P.WARN, "no isolated spike to score")
        imgui.spacing()
        imgui.text_disabled("ideal accuracy = Phi(d' / 2)")
        rows = [(f"d' = {d:.0f}", f"{100 * M.accuracy(d):.0f}% of single spikes called right") for d in ACCURACY_ROWS]
        table("accuracy", rows)
        imgui.end_group()

    # ------------------------------------------------------------------
    # 1: how it is computed
    # ------------------------------------------------------------------

    def section_1(self) -> None:
        D.heading(fa.ICON_FA_CALCULATOR, "How it is computed",
                  "For every isolated spike: a window of frames just before it (bottom) and a window from the response peak on (top). Pool the windows across spikes, then one formula.")
        em, dl = D.em(), imgui.get_window_draw_list()
        p, s, tr = self.params, self.score, self.trace
        w = D.guide_width()
        n_frames = self.frames
        if P.begin_plot("##dprime_windows", ImVec2(w, 13 * em)):
            implot.setup_axes("s", "dF/F", 0, implot.AxisFlags_.auto_fit)
            if s.n:
                c = int(s.centers[0])
                t0, t1 = tr.t[max(0, c - 4 * n_frames)], tr.t[min(tr.t.size - 1, c + self.peak + 4 * n_frames)]
                implot.setup_axis_limits(implot.ImAxis_.x1, float(t0), float(t1), imgui.Cond_.once)
            implot.plot_line("recorded dF/F", tr.t, np.ascontiguousarray(self.dff), P.spec(P.RAW, 1.0))
            implot.plot_line("true dF/F", tr.t, np.ascontiguousarray(tr.dff), P.spec(TRUTH, 1.5))
            if s.spikes.size:
                implot.plot_inf_lines("isolated spikes", np.ascontiguousarray(s.spikes), P.spec(ImVec4(TRUTH.x, TRUTH.y, TRUTH.z, 0.5), 1.0))
            pdl = implot.get_plot_draw_list()
            implot.push_plot_clip_rect()
            lim = implot.get_plot_limits()
            for c in s.centers:
                for start, color in ((c - n_frames, BOTTOM), (c + self.peak, TOP)):
                    a = implot.plot_to_pixels(implot.Point(float(tr.t[start]) - 0.5 / p.fs, lim.y.max))
                    b = implot.plot_to_pixels(implot.Point(float(tr.t[start + n_frames - 1]) + 0.5 / p.fs, lim.y.min))
                    pdl.add_rect_filled(a, b, D.u32(color, 0.18))
                    pdl.add_rect(a, b, D.u32(color, 0.7), 0, 1.0)
            implot.pop_plot_clip_rect()
            implot.end_plot()
        imgui.text_disabled("scroll to zoom, drag to pan, double-click to see every spike")

        x, y = D.canvas(w, 2.8 * em)
        bx = x
        for label, color in (("d'", KEY), ("=", P.DIM), ("mean top - mean bottom", D.TEXT), ("/", P.DIM), ("sqrt( (var top + var bottom) / 2 )", D.TEXT)):
            if label in ("=", "/"):
                D.text(dl, bx + 0.4 * em, y + 0.5 * em, label, color, 1.2)
                bx += 1.6 * em
                continue
            bw = D.text_width(label) + 1.4 * em
            D.box(dl, bx, y, bw, 2.0 * em, label, color)
            bx += bw
        if s.n:
            top, bottom = s.top.ravel(), s.bottom.ravel()
            x, y = D.canvas(w, 1.5 * em)
            D.segments(dl, x, y, [
                ("= (", P.DIM), (f"{top.mean():.3f}", TOP), (" - ", P.DIM), (f"{bottom.mean():.3f}", BOTTOM), (") / sqrt((", P.DIM),
                (f"{top.var():.4f}", TOP), (" + ", P.DIM), (f"{bottom.var():.4f}", BOTTOM), (") / 2)  =  ", P.DIM), (f"{s.d:.2f}", KEY),
                (f"      {s.n} isolated spikes x {n_frames} frames in each pool", P.DIM),
            ], 0.9)
        table("windows", (
            ("top", f"{n_frames} frames from the peak on. {self.indicator.name} peaks {self.peak} frame(s) after the spike at {p.fs:.0f} Hz; the window spans {1000 * n_frames / p.fs:.0f} ms of a {1000 * self.indicator.decay:.0f} ms decay."),
            ("bottom", f"the {n_frames} frames before the spike, with no spike in the {self.quiet:.2f} s before that"),
            ("units", "any: photons or dF/F give the same d', the ratio cancels the scale"),
            ("in vivo form", f"the same paper scores mouse cortex as mean(A) / sqrt(var(A) / 2), A the single-spike amplitudes. With A a {n_frames}-frame mean that reads {s.d_amplitudes:.1f} here: a different number, not comparable." if s.n >= 2 else "the same paper scores mouse cortex as mean(A) / sqrt(var(A) / 2), A the single-spike amplitudes; needs two or more events"),
        ))

    # ------------------------------------------------------------------
    # 2: what moves it
    # ------------------------------------------------------------------

    def section_2(self) -> None:
        D.heading(fa.ICON_FA_SLIDERS, "What moves it",
                  f"Each curve re-simulates this cell for {SWEEP_DURATION:.0f} s with one setting changed and scores it with the true spikes. The orange line is the current setting.")
        em = D.em()
        w = D.guide_width()
        across = 3 if w > 48 * em else 2
        gap = 1.2 * em
        pw = (w - (across - 1) * gap) / across
        ph = 10 * em
        if self.sweep_wait > 0:
            imgui.text_disabled("re-simulating...")
        p = self.params
        panels = [(attr, label, values, self.sweeps.get(attr)) for attr, label, values in M.SWEEPS]
        top = max([float(np.nanmax(d)) for d in self.sweeps.values() if d.size] + [d for _, d in self.indicators] + [1.0])
        for i, (attr, label, values, d) in enumerate(panels):
            if i % across:
                imgui.same_line(0, gap)
            imgui.begin_group()
            if P.begin_plot(f"##sweep_{attr}", ImVec2(pw, ph), implot.Flags_.no_legend):
                implot.setup_axes(label, "d'", 0, 0)
                if attr in ("brightness", "pixels", "fs"):
                    implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10)
                implot.setup_axis_limits(implot.ImAxis_.x1, float(values[0]) * 0.9, float(values[-1]) * 1.1, imgui.Cond_.always)
                implot.setup_axis_limits(implot.ImAxis_.y1, 0.0, top * 1.1, imgui.Cond_.always)
                if d is not None:
                    sp = P.spec(P.ACCENT, 2.0)
                    sp.marker = implot.Marker_.circle
                    sp.marker_size = 3.0 * P.SCALE
                    implot.plot_line("d'", values, np.ascontiguousarray(d), sp)
                implot.plot_inf_lines("now", np.array([float(getattr(p, attr))]), P.spec(DETECTED, 1.5))
                implot.end_plot()
            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + pw)
            imgui.text_disabled(WHY[attr])
            imgui.pop_text_wrap_pos()
            imgui.end_group()
        if len(panels) % across:
            imgui.same_line(0, gap)
        imgui.begin_group()
        if P.begin_plot("##sweep_indicator", ImVec2(pw, ph), implot.Flags_.no_legend):
            names = [n for n, _ in self.indicators]
            implot.setup_axes("indicator", "d'", implot.AxisFlags_.no_grid_lines, 0)
            implot.setup_axis_limits(implot.ImAxis_.y1, 0.0, top * 1.1, imgui.Cond_.always)
            if names:
                implot.setup_axis_ticks(implot.ImAxis_.x1, 0.0, float(len(names) - 1), len(names), names)
                implot.setup_axis_limits(implot.ImAxis_.x1, -0.6, len(names) - 0.4, imgui.Cond_.always)
                ds = np.array([d for _, d in self.indicators], dtype=np.float64)
                xs = np.arange(len(names), dtype=np.float64)
                implot.plot_bars("d'", xs, ds, 0.6, P.spec(P.ACCENT, 1.0, 0.6))
                k = names.index(p.indicator)
                implot.plot_bars("now", np.array([float(k)]), ds[k : k + 1], 0.6, P.spec(DETECTED, 1.0, 0.8))
            implot.end_plot()
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + pw)
        imgui.text_disabled("dF/F per spike, the numerator: " + ", ".join(f"{i.name} {i.amp:.2f}" for i in M.CALCIUM))
        imgui.pop_text_wrap_pos()
        imgui.end_group()

    # ------------------------------------------------------------------
    # 3: what it misses
    # ------------------------------------------------------------------

    def section_3(self) -> None:
        D.heading(fa.ICON_FA_EYE_SLASH, "What it misses",
                  "d' needs to know where the spikes are. Without ground truth you score the events a picker chose, so the score partly measures the picker. And it never sees motion, drift or crosstalk.")
        em = D.em()
        p, tr = self.params, self.trace
        w = D.guide_width()
        if P.begin_plot("##dprime_picker", ImVec2(w, 10 * em)):
            implot.setup_axes("s", "dF/F", 0, implot.AxisFlags_.auto_fit)
            implot.plot_line("recorded dF/F", tr.t, np.ascontiguousarray(self.dff), P.spec(P.RAW, 1.0))
            if tr.spikes.size:
                implot.plot_inf_lines("true spikes", np.ascontiguousarray(tr.spikes), P.spec(ImVec4(TRUTH.x, TRUTH.y, TRUTH.z, 0.6), 1.0))
            if self.detected.size:
                picked = np.ascontiguousarray(self.detected)
                sp = P.spec(DETECTED, 0.0)
                sp.marker = implot.Marker_.down
                sp.marker_size = 4.0 * P.SCALE
                sp.marker_fill_color = DETECTED
                sp.marker_line_color = DETECTED
                implot.plot_scatter(f"picked at {self.threshold:.1f} SD", picked, np.full(picked.size, float(self.dff.max()) * 1.05), sp)
            implot.end_plot()
        s_true, s_pick, s_silent = self.score, self.score_detected, self.score_silent
        rows = [
            ("true spikes", self.verdict(s_true, f"{s_true.n} isolated of {tr.spikes.size}")),
            ("picked events", self.verdict(s_pick, f"{self.detected.size} picked, {s_pick.n} isolated. The score now measures the picker as much as the cell: noise it picks pulls d' down, small events it drops push d' up")),
            ("no spikes at all", self.verdict(s_silent, f"the same picker on a cell that never fires, {self.detected_silent.size} picked. A silent cell and a dim cell both read low")),
        ]
        table("picker", rows, w)
        imgui.spacing()
        half = (w - 1.5 * em) / 2
        imgui.begin_group()
        imgui.text_colored(KEY, "it reads")
        table("reads", (
            ("dF/F per spike", "indicator, expression, calcium buffering"),
            ("photons", "laser power, depth, dwell time, pixels per cell"),
            ("neuropil", "out-of-focus light in the baseline"),
            ("frame rate", "photons per frame, and what the window spans"),
        ), half)
        imgui.end_group()
        imgui.same_line(0, 1.5 * em)
        imgui.begin_group()
        imgui.text_colored(P.WARN, "it does not read")
        table("misses", (
            ("motion, z-drift", "only if it lands in a window, and then it looks like noise"),
            ("bleaching, crosstalk", f"drift widens the pools unless dF/F has a running baseline (here {F0_WINDOW:.0f} s); stripes, crosstalk not at all"),
            ("silent cells", "no events, no score: same as a dim cell"),
            ("which spikes are real", "one number per cell, not a confidence per event"),
        ), half)
        imgui.end_group()
        imgui.spacing()
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + w)
        imgui.text_disabled("At the scope: report the fraction of cells with d' above 3 as 'single spikes resolvable', with a motion or drift number next to it, so a low d' is not read as a broken microscope when the cells are quiet or dim.")
        imgui.pop_text_wrap_pos()

    @staticmethod
    def verdict(s: M.Score, note: str) -> str:
        if s.n == 0 or math.isnan(s.d):
            return f"no score  ({note})"
        few = "  (too few events to mean much)" if s.n < 4 else ""
        return f"d' = {s.d:.2f}, ideal accuracy {100 * M.accuracy(s.d):.0f}%{few}  ({note})"
