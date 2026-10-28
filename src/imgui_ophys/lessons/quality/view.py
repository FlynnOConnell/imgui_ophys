"""
The Signal quality stage: how much a recording says about spikes, against the photons collected. Subsections: both
metrics swept over brightness for every indicator, then what each one reads: nu from neighbouring samples, d' from a
matched filter, where nu looks in the noise spectrum, and the lag-1 autocorrelation of each pixel of a small movie.
"""

from functools import lru_cache

import numpy as np
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui, implot

from imgui_ophys import diagram as D
from imgui_ophys import plotting as P
from imgui_ophys import sim
from imgui_ophys.lesson import Lesson
from imgui_ophys.lessons.quality import model as M
from imgui_ophys.lessons.traces.view import TRUTH
from imgui_ophys.widgets import KEY, help_mark, section, table

ImVec2 = imgui.ImVec2
COLORS = {
    "GCaMP6f": P.component_color(0),
    "GCaMP6s": P.component_color(2),
    "jGCaMP8m": P.component_color(1),
    "ASAP3": P.component_color(5),
    "JEDI-2P": P.component_color(4),
}
D_FLOOR = 0.1  # d' at or below 0 has no place on a log axis
LOG = implot.Scale_.log10
ALWAYS = imgui.Cond_.always
HORIZONTAL = implot.InfLinesFlags_.horizontal
NO_LEGEND = P.TRACE_FLAGS | implot.Flags_.no_legend


@lru_cache(maxsize=16)
def cached_clean(indicator: str, fs: float, rate: float, seed: int) -> M.Clean:
    return M.make_clean(indicator, fs, rate, seed)


@lru_cache(maxsize=16)
def cached_field(indicator: str, fs: float, rate: float, seed: int) -> M.Field:
    return M.make_field(indicator, fs, rate, seed)


def dots(color: imgui.ImVec4, size: float = 2.5, weight: float = 1.0) -> implot.Spec:
    """A line with a dot at every sample."""
    return implot.Spec(line_color=color, fill_color=color, line_weight=weight * P.SCALE, marker=implot.Marker_.circle,
                       marker_size=size * P.SCALE)


def note(text: str) -> None:
    """Dim text wrapped at the guide width."""
    imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + D.guide_width())
    imgui.text_colored(P.DIM, text)
    imgui.pop_text_wrap_pos()


def log_limits(axis: int, values: np.ndarray, below: float = 0.7, above: float = 1.4) -> None:
    """Fit a log axis to the positive values, with a margin."""
    v = values[values > 0]
    implot.setup_axis_limits(axis, float(v.min()) * below, float(v.max()) * above, ALWAYS)


class QualityLesson(Lesson):
    key = "quality"
    title = "Signal quality"
    summary = "noise level nu and spike detectability d' against the photons collected"
    sections = (
        (fa.ICON_FA_SUN, "Brightness"),
        (fa.ICON_FA_STAIRS, "Noise level nu"),
        (fa.ICON_FA_CROSSHAIRS, "Spike detectability d'"),
        (fa.ICON_FA_WAVE_SQUARE, "Noise spectrum"),
        (fa.ICON_FA_TABLE_CELLS, "Lag-1 autocorrelation"),
    )
    references = (
        ("Rupprecht P, Carta S, Hoffmann A, et al. (2021). A database and deep learning toolbox for noise-optimized, generalized spike inference from calcium imaging. Nat Neurosci 24:1324-1337. (Methods: computation of noise levels.)",
         "https://doi.org/10.1038/s41593-021-00895-5"),
        ("Rupprecht P (2021). Large-scale calcium imaging & noise levels. Blog post.",
         "https://gcamp6f.com/2021/10/04/large-scale-calcium-imaging-noise-levels/"),
        ("Wilt BA, Fitzgerald JE, Schnitzer MJ (2013). Photon shot noise limits on optical detection of neuronal spikes and estimation of spike timing. Biophys J 104:51-62.",
         "https://doi.org/10.1016/j.bpj.2012.07.058"),
        ("Villette V, Chavarha M, Dimov IK, et al. (2019). Ultrafast two-photon imaging of a high-gain voltage indicator in awake behaving mice. Cell 179:1590-1608.",
         "https://doi.org/10.1016/j.cell.2019.11.004"),
        ("Welch P (1967). The use of fast Fourier transform for the estimation of power spectra. IEEE Trans Audio Electroacoust 15:70-73.",
         "https://doi.org/10.1109/TAU.1967.1161901"),
    )

    def __init__(self):
        self.indicator = "GCaMP6f"
        self.fs = 30.0
        self.rate = M.RATE["calcium"]
        self.seed = 1
        self.phi0 = 1e4  # photons / s from the cell at rest
        self.artifact = 0.0  # motion, % of F
        self.filtered = False
        self.cutoff = 100.0  # Hz
        self.sweeps: dict = {}  # indicator -> (nu, d') over M.SWEEP; empty until Brightness is drawn
        self.pick = M.CELLS[0][:2]  # (x, y) pixel of the movie
        self.cmaps = None  # P.Colormaps, made on the first page() (it needs an ImPlot context)
        self.make()

    @property
    def ind(self) -> sim.Indicator:
        return sim.BY_NAME[self.indicator]

    @property
    def lowpass(self) -> float:
        return self.cutoff if self.filtered else 0.0

    def make(self) -> None:
        self.clean = cached_clean(self.indicator, self.fs, self.rate, self.seed)
        self.field = cached_field(self.indicator, self.fs, self.rate, self.seed)
        self.sweeps = {}
        self.recount()

    def recount(self) -> None:
        """Everything at the current brightness: the recording, its spectra and the readouts."""
        c = self.clean
        self.rec = rec = M.record(c, self.phi0, self.artifact, self.lowpass, self.seed)
        plain = rec if not (self.artifact or self.lowpass) else M.record(c, self.phi0, seed=self.seed)
        self.nu_plain = plain.nu
        noise = rec.dff - rec.truth
        f, recorded = M.spectrum(rec.dff, c.fs)
        _, truth = M.spectrum(rec.truth, c.fs)
        _, residual = M.spectrum(noise, c.fs)
        self.spectra = (f, recorded, truth, residual)
        self.excess = M.slow_excess(noise, c.fs, self.phi0)
        self.steps = M.survival(np.abs(np.diff(rec.dff)))
        self.truth_steps = M.survival(np.abs(np.diff(rec.truth)))
        self.movie = recorded, expected = M.count_field(self.field, self.phi0, self.artifact, self.lowpass, self.seed)
        self.r1 = M.lag1(recorded)
        self.r1_light = M.lag1(expected)
        self.share = M.share_not_shot(recorded, expected)
        self.mean_img = recorded.mean(axis=0).reshape(M.SIDE, M.SIDE)

    def resweep(self) -> None:
        self.sweeps = {}
        for ind in sim.INDICATORS:
            if ind.name == self.indicator:
                c = self.clean
            else:
                c = cached_clean(ind.name, ind.fs, M.RATE[ind.kind], 1)
            self.sweeps[ind.name] = M.sweep(c, M.SWEEP, self.artifact, self.lowpass)

    # ------------------------------------------------------------------
    # rail
    # ------------------------------------------------------------------

    def controls(self) -> None:
        log = imgui.SliderFlags_.logarithmic
        section(f"{fa.ICON_FA_DNA}  Recording")
        names = [i.name for i in sim.INDICATORS]
        changed, idx = imgui.combo("indicator", names.index(self.indicator), names)
        if changed:
            new = sim.INDICATORS[idx]
            self.indicator, self.fs, self.rate = new.name, new.fs, M.RATE[new.kind]
            self.make()
        help_mark(self.ind.note)
        changed, self.phi0 = imgui.slider_float("photons / s", self.phi0, *M.PHOTONS, "%.2g", log)
        if changed:
            self.recount()
        help_mark("Photons collected from the cell per second at rest: summed over its pixels, or over the AOD dwells. "
                  "Per sample: this / sample rate.")
        _, self.fs = imgui.slider_float("sample rate", self.fs, 5.0, 2000.0, "%.0f Hz", log)
        remake = imgui.is_item_deactivated_after_edit()
        _, self.rate = imgui.slider_float("spike rate", self.rate, 0.05, 20.0, "%.2f /s", log)
        remake |= imgui.is_item_deactivated_after_edit()
        changed, self.seed = imgui.input_int("seed", self.seed)
        if remake or changed:
            self.make()

        section(f"{fa.ICON_FA_TRIANGLE_EXCLAMATION}  Beyond shot noise")
        _, self.artifact = imgui.slider_float("motion", self.artifact, 0.0, 10.0, "%.1f %% of F")
        redo = imgui.is_item_deactivated_after_edit()
        help_mark("Light that changes with movement and blood flow: drift 0.3 Hz, breathing 3 Hz, heartbeat 10 Hz.")
        changed, self.filtered = imgui.checkbox("low-pass", self.filtered)
        redo |= changed
        _, self.cutoff = imgui.slider_float("cutoff", self.cutoff, 2.0, 1000.0, "%.0f Hz", log)
        redo |= imgui.is_item_deactivated_after_edit() and self.filtered
        help_mark("Zero-phase order-4 Butterworth on the recorded dF/F, as an amplifier or analysis filter would.")
        if redo:
            self.recount()
            self.sweeps = {}

        section(f"{fa.ICON_FA_GAUGE}  Metrics")
        table("quality_metrics", self.readout())

    def readout(self) -> list[tuple[str, str]]:
        rec, c = self.rec, self.clean
        rows = [
            ("photons / sample", f"{self.phi0 / c.fs:.3g}"),
            ("nu", f"{rec.nu:.3g} %/sqrt(Hz)"),
            ("shot noise alone", f"{M.nu_shot(self.phi0):.3g}"),
            ("d'", f"{rec.dprime:.3g}"),
            ("d', white noise", f"{M.dprime_white(rec.template, c.fs, self.phi0):.3g}"),
            ("slow noise", f"{self.excess:.2g} x shot"),
        ]
        if self.ind.kind == "voltage":
            # amp is dF/F per 100 mV, so |amp| x 100 % / 100 mV = |amp| % per mV
            rows.append(("nu in mV", f"{rec.nu / abs(self.ind.amp):.3g} mV/sqrt(Hz)"))
        return rows

    # ------------------------------------------------------------------
    # page
    # ------------------------------------------------------------------

    def window(self, pre: tuple[float, float], length: tuple[float, float]) -> tuple[int, int]:
        """Samples [i0, i1) from (calcium, voltage) pre s before the first isolated spike, length s long."""
        c = self.clean
        k = 0 if self.ind.kind == "calcium" else 1
        start = c.isolated[0] / c.fs - pre[k] if c.isolated.size else 0.0
        i0 = max(0, int(start * c.fs))
        return i0, min(c.t.size, i0 + max(4, int(length[k] * c.fs)))

    def time_unit(self) -> tuple[float, str]:
        return (1.0, "s") if self.ind.kind == "calcium" else (1000.0, "ms")

    def section_0(self) -> None:
        D.heading(fa.ICON_FA_SUN, "Brightness",
                  "Each point is the recording counted at that many photons per second. nu follows the photons alone, so it is the same for every indicator; d' also depends on how large and how long a spike's signal is.")
        if not self.sweeps:
            self.resweep()
        em, dl = D.em(), imgui.get_window_draw_list()
        w = D.guide_width()
        x, y = D.canvas(w, 1.4 * em)
        for name, color in [*COLORS.items(), ("shot noise alone", P.DIM), ("now", P.HIGHLIGHT)]:
            x += D.legend_chip(dl, x, y, name, color)

        gap = em
        pw, ph = (w - 2 * gap) / 3, 15 * em
        shot = np.array([M.nu_shot(p) for p in M.SWEEP])
        nus = np.concatenate([nu for nu, _ in self.sweeps.values()] + [shot])
        ds = np.maximum(np.concatenate([d for _, d in self.sweeps.values()]), D_FLOOR)
        dprime = max(self.rec.dprime, D_FLOOR)
        rec = self.rec

        if P.begin_plot("##quality_nu", ImVec2(pw, ph), NO_LEGEND):
            implot.setup_axes("photons / s", "nu, %/sqrt(Hz)", 0, 0)
            implot.setup_axis_scale(implot.ImAxis_.x1, LOG)
            implot.setup_axis_scale(implot.ImAxis_.y1, LOG)
            implot.setup_axis_limits(implot.ImAxis_.x1, *M.PHOTONS, ALWAYS)
            log_limits(implot.ImAxis_.y1, np.r_[nus, M.NU_HIGH], above=2.0)
            implot.plot_inf_lines("##nu_marks", np.array([M.NU_LOW, M.NU_HIGH]), P.spec(P.DIM, 1.0, flags=HORIZONTAL))
            implot.plot_text("very low", M.PHOTONS[1], M.NU_LOW, ImVec2(-2.4 * em, -0.6 * em))
            implot.plot_text("high", M.PHOTONS[1], M.NU_HIGH, ImVec2(-1.6 * em, -0.6 * em))
            implot.plot_line("##shot", M.SWEEP, shot, P.spec(P.DIM, 3.0))
            self.plot_sweeps("photons", "nu")
            implot.plot_inf_lines("##phi", np.array([self.phi0]), P.spec(P.HIGHLIGHT, 1.0))
            implot.plot_scatter("##now", np.array([self.phi0]), np.array([rec.nu]), dots(P.HIGHLIGHT, 5.0))
            implot.end_plot()
        imgui.same_line(0, gap)
        if P.begin_plot("##quality_d", ImVec2(pw, ph), NO_LEGEND):
            implot.setup_axes("photons / s", "d'", 0, 0)
            implot.setup_axis_scale(implot.ImAxis_.x1, LOG)
            implot.setup_axis_scale(implot.ImAxis_.y1, LOG)
            implot.setup_axis_limits(implot.ImAxis_.x1, *M.PHOTONS, ALWAYS)
            log_limits(implot.ImAxis_.y1, np.r_[ds, M.D_RELIABLE])
            self.reliable_line()
            self.plot_sweeps("photons", "d")
            implot.plot_inf_lines("##phi", np.array([self.phi0]), P.spec(P.HIGHLIGHT, 1.0))
            implot.plot_scatter("##now", np.array([self.phi0]), np.array([dprime]), dots(P.HIGHLIGHT, 5.0))
            implot.end_plot()
        imgui.same_line(0, gap)
        if P.begin_plot("##quality_nu_d", ImVec2(pw, ph), NO_LEGEND):
            implot.setup_axes("nu, %/sqrt(Hz)", "d'", 0, 0)
            implot.setup_axis_scale(implot.ImAxis_.x1, LOG)
            implot.setup_axis_scale(implot.ImAxis_.y1, LOG)
            log_limits(implot.ImAxis_.x1, np.r_[nus, M.NU_HIGH])
            log_limits(implot.ImAxis_.y1, np.r_[ds, M.D_RELIABLE])
            implot.plot_inf_lines("##nu_marks", np.array([M.NU_LOW, M.NU_HIGH]), P.spec(P.DIM, 1.0))
            self.reliable_line()
            self.plot_sweeps("nu", "d")
            implot.plot_scatter("##now", np.array([rec.nu]), np.array([dprime]), dots(P.HIGHLIGHT, 5.0))
            implot.end_plot()

        note("nu follows shot noise (gray) until the signal itself moves between samples. At equal nu, calcium spikes are ~10x more "
             "detectable than voltage spikes: their signal lasts ~100x longer. d' levels off where what is left is the cell's own variability "
             "(other spikes, subthreshold voltage), not photons.")
        note(f"Marks: nu {M.NU_LOW:.0f} very low and {M.NU_HIGH:.0f} high (Rupprecht et al. 2021). d' {M.D_RELIABLE:.0f}: a threshold at 3.7 SD "
             "finds 90% of spikes with ~0.1 false ones / s at 1 kHz.")

    def plot_sweeps(self, x: str, y: str) -> None:
        """One line per indicator, x and y each "photons", "nu" or "d"."""
        for name, (nu, d) in self.sweeps.items():
            series = {"photons": M.SWEEP, "nu": nu, "d": np.maximum(d, D_FLOOR)}
            implot.plot_line(f"##{name}", series[x], series[y], P.spec(COLORS[name], 3.0 if name == self.indicator else 1.3))

    def reliable_line(self) -> None:
        implot.plot_inf_lines("##reliable", np.array([M.D_RELIABLE]), P.spec(P.DIM, 1.0, flags=HORIZONTAL))

    def section_1(self) -> None:
        D.heading(fa.ICON_FA_STAIRS, "Noise level nu",
                  "nu = median |dF/F[t+1] - dF/F[t]| / sqrt(fs), dF/F in %. Neighbouring samples share the signal, so their difference is mostly noise; the median ignores the few large steps a spike makes; / sqrt(fs) turns noise per sample into noise per second, so frame rates compare.")
        em = D.em()
        c, rec = self.clean, self.rec
        scale, unit = self.time_unit()
        i0, i1 = self.window((0.5, 0.03), (3.0, 0.1))
        t = c.t[i0:i1] * scale
        median_step = float(np.median(np.abs(np.diff(rec.dff))))
        w = D.guide_width()
        lw = 0.6 * w
        imgui.begin_group()
        if P.begin_plot("##quality_trace", ImVec2(lw, 9 * em), P.TRACE_FLAGS):
            implot.setup_axes("", "dF/F, %", implot.AxisFlags_.no_tick_labels, implot.AxisFlags_.auto_fit)
            implot.setup_axis_limits(implot.ImAxis_.x1, float(t[0]), float(t[-1]), ALWAYS)
            implot.plot_line("recorded", t, np.ascontiguousarray(rec.dff[i0:i1]), dots(P.RAW))
            implot.plot_line("truth", t, np.ascontiguousarray(rec.truth[i0:i1]), dots(TRUTH, 2.0, 1.5))
            implot.end_plot()
        if P.begin_plot("##quality_steps", ImVec2(lw, 9 * em), P.TRACE_FLAGS):
            implot.setup_axes(unit, "step, %", 0, implot.AxisFlags_.auto_fit)
            implot.setup_axis_limits(implot.ImAxis_.x1, float(t[0]), float(t[-1]), ALWAYS)
            mid = (t[:-1] + t[1:]) / 2
            implot.plot_stems("steps, recorded", mid, np.diff(rec.dff[i0:i1]), 0, P.spec(P.RAW, 1.5))
            implot.plot_stems("steps, truth", mid, np.diff(rec.truth[i0:i1]), 0, P.spec(TRUTH, 1.5))
            implot.plot_inf_lines("+/- median |step|", np.array([-median_step, median_step]), P.spec(KEY, 1.5, flags=HORIZONTAL))
            implot.end_plot()
        imgui.end_group()
        imgui.same_line(0, em)
        if P.begin_plot("##quality_survival", ImVec2(-1, 18 * em + imgui.get_style().item_spacing.y), P.TRACE_FLAGS):
            implot.setup_axes("|step|, %", "fraction of steps this big or bigger", 0, 0)
            implot.setup_axis_scale(implot.ImAxis_.x1, LOG)
            implot.setup_axis_scale(implot.ImAxis_.y1, LOG)
            xs, ys = self.steps
            xt, yt = self.truth_steps
            implot.setup_axis_limits(implot.ImAxis_.x1, median_step / 30, float(max(xs[0], xt[0], median_step)) * 1.5, ALWAYS)
            implot.setup_axis_limits(implot.ImAxis_.y1, float(ys[0]), 1.0, ALWAYS)
            implot.plot_line("recorded", xs[xs > 0], ys[xs > 0], P.spec(P.RAW, 2.0))
            implot.plot_line("truth alone", xt[xt > 0], yt[xt > 0], P.spec(TRUTH, 2.0))
            implot.plot_inf_lines("##half", np.array([0.5]), P.spec(P.DIM, 1.0, flags=HORIZONTAL))
            implot.plot_inf_lines("median", np.array([median_step]), P.spec(KEY, 1.5))
            implot.end_plot()

        dl = imgui.get_window_draw_list()
        x, y = D.canvas(w, 1.4 * em)
        shot = M.nu_shot(self.phi0)
        D.segments(dl, x, y, [("median |step| ", P.DIM), (f"{median_step:.3g} %", KEY), ("  /  sqrt(", P.DIM), (f"{c.fs:.0f} Hz", KEY), (")  =  nu ", P.DIM),
                              (f"{rec.nu:.3g}", KEY), (f"      shot noise alone {shot:.3g}  (x{rec.nu / shot:.2f})", P.DIM)], 0.85)
        note("Right: the steps the signal makes by itself (green) move the median only once they are about as common as noise-sized steps.")

    def section_2(self) -> None:
        D.heading(fa.ICON_FA_CROSSHAIRS, "Spike detectability d'",
                  "Slide the average spike h along the trace (a matched filter) and z-score the output on stretches no spike reaches. d' is the mean output at spikes: amplitude, duration and noise in one number, the same at any sample rate.")
        em = D.em()
        c, rec = self.clean, self.rec
        scale, unit = self.time_unit()
        w = D.guide_width()
        gap = em
        tw, zw = 0.22 * w, 0.48 * w
        hw = w - tw - zw - 2 * gap
        ph = 13 * em
        lags = np.arange(-c.before, c.after + 1) / c.fs * scale
        if P.begin_plot("##quality_template", ImVec2(tw, ph), NO_LEGEND):
            implot.setup_axes(f"{unit} from spike", "h, dF/F %", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            implot.plot_inf_lines("##zero", np.array([0.0]), P.spec(P.DIM, 1.0, flags=HORIZONTAL))
            implot.plot_line("h", lags, rec.template, dots(TRUTH, 3.0, 2.0))
            implot.end_plot()
        imgui.same_line(0, gap)
        i0, i1 = self.window((2.0, 0.05), (30.0, 0.6))
        t = c.t[i0:i1] * scale
        spikes = c.spikes[(c.spikes >= c.t[i0]) & (c.spikes < c.t[i1 - 1])] * scale
        if P.begin_plot("##quality_z", ImVec2(zw, ph), P.TRACE_FLAGS):
            implot.setup_axes(unit, "filter output, SD", 0, implot.AxisFlags_.auto_fit)
            implot.setup_axis_limits(implot.ImAxis_.x1, float(t[0]), float(t[-1]), ALWAYS)
            if spikes.size:
                implot.plot_inf_lines("spikes", np.ascontiguousarray(spikes), P.spec(P.HIGHLIGHT, 1.0))
            implot.plot_line("output", t, np.ascontiguousarray(rec.z[i0:i1]), P.spec(P.ACCENT, 1.2))
            implot.plot_inf_lines("d'", np.array([rec.dprime]), P.spec(KEY, 1.5, flags=HORIZONTAL))
            implot.end_plot()
        imgui.same_line(0, gap)
        free = rec.z[rec.free]
        if P.begin_plot("##quality_hist", ImVec2(hw, ph), P.TRACE_FLAGS):
            implot.setup_axes("filter output, SD", "density", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            values = np.r_[free, rec.spike_z]
            edges = np.linspace(float(values.min()), float(values.max()) + 1e-9, 41)
            width = edges[1] - edges[0]
            centers = (edges[:-1] + edges[1:]) / 2
            for label, v, color in (("no spike", free, P.ACCENT), ("at spikes", rec.spike_z, P.HIGHLIGHT)):
                if v.size:
                    density = np.histogram(v, edges)[0] / (v.size * width)
                    implot.plot_bars(label, centers, density, width, P.spec(color, 1.0, fill_alpha=0.55))
            implot.plot_inf_lines("##d", np.array([0.0, rec.dprime]), P.spec(KEY, 1.5))
            implot.end_plot()

        dl = imgui.get_window_draw_list()
        x, y = D.canvas(w, 1.4 * em)
        norm = float(np.linalg.norm(rec.template))
        sd = 100 * np.sqrt(c.fs / self.phi0)
        D.segments(dl, x, y, [("d' = ", P.DIM), (f"{rec.dprime:.3g}", KEY), (f"  ({c.isolated.size} isolated spikes)      white shot noise alone: ||h|| / SD = ", P.DIM),
                              (f"{norm:.3g} % / {sd:.3g} %", KEY), (" = ", P.DIM), (f"{M.dprime_white(rec.template, c.fs, self.phi0):.3g}", KEY)], 0.85)
        note("For an exponential transient of size A and time constant tau, ||h|| / SD is about A sqrt(tau / 2) / nu: the same nu with a 100x "
             "shorter tau gives a 10x lower d'.")

    def section_3(self) -> None:
        D.heading(fa.ICON_FA_WAVE_SQUARE, "Noise spectrum",
                  "nu weighs the spectrum by the step filter's gain, 4 sin^2(pi f / fs) (right axis): nothing at 0 Hz, most at the Nyquist frequency. Slow noise (motion, heartbeat, subthreshold voltage) sits where nu barely looks; a low-pass removes exactly what nu reads.")
        em = D.em()
        c, rec = self.clean, self.rec
        f, recorded, truth, residual = self.spectra
        shot = np.sqrt(M.shot_psd(self.phi0))
        level = np.sqrt(2) * rec.nu / M.MEDIAN_ABS_DIFF  # the flat spectrum that would give this nu
        lo, hi = M.slow_band(c.fs)
        w = D.guide_width()
        if P.begin_plot("##quality_spectrum", ImVec2(w, 17 * em), P.TRACE_FLAGS):
            implot.setup_axes("Hz", "ASD, %/sqrt(Hz)", 0, 0)
            implot.setup_axis(implot.ImAxis_.y2, "step filter gain", implot.AxisFlags_.aux_default)
            implot.setup_axis_scale(implot.ImAxis_.x1, LOG)
            implot.setup_axis_scale(implot.ImAxis_.y1, LOG)
            implot.setup_axis_limits(implot.ImAxis_.x1, float(f[0]), float(f[-1]), ALWAYS)
            low = min(float(np.sqrt(residual).min()), shot, level)
            high = max(float(np.sqrt(recorded).max()), shot, level)
            implot.setup_axis_limits(implot.ImAxis_.y1, 0.3 * low, 3 * high, ALWAYS)
            implot.setup_axis_limits(implot.ImAxis_.y2, 0.0, 4.4, ALWAYS)
            band = np.array([lo, hi])
            implot.plot_shaded("slow band", band, np.full(2, 0.3 * low), np.full(2, 3 * high), P.spec(P.DIM, 1.0, fill_alpha=0.25))
            implot.plot_line("recorded", f, np.sqrt(recorded), P.spec(P.RAW, 1.5))
            implot.plot_line("truth", f, np.sqrt(truth), P.spec(TRUTH, 1.5))
            implot.plot_line("noise (recorded - truth)", f, np.sqrt(residual), P.spec(P.ACCENT, 1.5))
            implot.plot_inf_lines("shot noise alone", np.array([shot]), P.spec(P.DIM, 2.0, flags=HORIZONTAL))
            implot.plot_inf_lines("flat level nu implies", np.array([level]), P.spec(KEY, 1.5, flags=HORIZONTAL))
            implot.set_axes(implot.ImAxis_.x1, implot.ImAxis_.y2)
            implot.plot_line("step filter gain", f, 4 * np.sin(np.pi * f / c.fs) ** 2, P.spec(P.WHITE, 1.0))
            implot.end_plot()
        dl = imgui.get_window_draw_list()
        x, y = D.canvas(w, 1.4 * em)
        D.segments(dl, x, y, [(f"slow band {lo:.0f}-{hi:.0f} Hz: noise power ", P.DIM), (f"{self.excess:.2g}x", KEY), (" shot noise's.      nu ", P.DIM),
                              (f"{rec.nu:.3g}", KEY), (", without motion and filter ", P.DIM), (f"{self.nu_plain:.3g}", KEY)], 0.85)
        note("The noise line is known only in simulation. On data, record a voltage-insensitive reference: a background spot, or a second fluorophore.")

    def section_4(self) -> None:
        D.heading(fa.ICON_FA_TABLE_CELLS, "Lag-1 autocorrelation",
                  "r1 = sum x[t+1] x[t] / sum x[t]^2 for each pixel of a movie, mean removed: how much of a pixel's fluctuation is still there one frame later. Shot noise forgets at once, so a pixel of noise alone reads 0 +/- 1 / sqrt(frames); light that changes slower than a frame keeps it. r1 is about the share of the pixel's variance that is not shot noise.")
        if self.cmaps is None:
            self.cmaps = P.Colormaps()
        em = D.em()
        recorded, expected = self.movie
        frames = recorded.shape[0]
        null = 1 / np.sqrt(frames)
        p = self.pick[1] * M.SIDE + self.pick[0]
        w = D.guide_width()
        gap = em
        side = 0.27 * w
        x0 = imgui.get_cursor_pos_x()
        self.picture("mean photons / frame", self.mean_img, side, float(self.mean_img.max()), self.cmaps.gray)
        imgui.same_line(0, gap)
        hi = max(float(self.r1.max()), 5 * null)
        self.picture("r1", self.r1.reshape(M.SIDE, M.SIDE), side, hi, self.cmaps.sequential[0])
        imgui.same_line(0, 0.3 * em)
        implot.colormap_scale("##lag_scale", 0.0, hi, ImVec2(0, side), "%.2f", 0, self.cmaps.sequential[0])
        imgui.same_line(0, gap)
        top = 1.1 * max(float(self.share.max()), 0.05)
        if P.begin_plot("##lag_share", ImVec2(w - (imgui.get_cursor_pos_x() - x0), side), P.TRACE_FLAGS):
            implot.setup_axes("share of variance not shot noise", "r1", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            ends = np.array([0.0, top])
            implot.plot_shaded("noise, 3 SD", ends, np.full(2, -3 * null), np.full(2, 3 * null), P.spec(P.DIM, 1.0, fill_alpha=0.2))
            implot.plot_inf_lines("noise, 3 SD", np.array([-3 * null, 3 * null]), P.spec(P.DIM, 1.0, flags=HORIZONTAL))
            implot.plot_line("r1 = share", ends, ends, P.spec(KEY, 1.5))
            implot.plot_scatter("pixels", self.share, self.r1, dots(P.ACCENT, 2.0))
            implot.plot_scatter("##picked", self.share[p : p + 1], self.r1[p : p + 1], dots(P.HIGHLIGHT, 5.0))
            implot.end_plot()

        scale, unit = self.time_unit()
        i1 = min(frames, max(4, int((10.0 if self.ind.kind == "calcium" else 0.2) * self.field.fs)))
        t = self.field.t[:i1] * scale
        x = recorded[:, p] - recorded[:, p].mean()
        if P.begin_plot("##lag_trace", ImVec2(w - side - gap, side), P.TRACE_FLAGS):
            implot.setup_axes(unit, "photons / frame", 0, implot.AxisFlags_.auto_fit)
            implot.setup_axis_limits(implot.ImAxis_.x1, float(t[0]), float(t[-1]), ALWAYS)
            implot.plot_line("recorded", t, np.ascontiguousarray(recorded[:i1, p]), dots(P.RAW))
            implot.plot_line("light, no shot noise", t, np.ascontiguousarray(expected[:i1, p]), P.spec(TRUTH, 2.0))
            implot.end_plot()
        imgui.same_line(0, gap)
        r = float(self.r1[p])
        lim = 1.05 * float(np.abs(x).max()) or 1.0
        if P.begin_plot("##lag_pairs", ImVec2(side, side), P.TRACE_FLAGS | implot.Flags_.equal):
            implot.setup_axes("x[t], mean removed", "x[t+1]", 0, 0)
            implot.setup_axes_limits(-lim, lim, -lim, lim, ALWAYS)
            implot.plot_scatter("frame pairs", x[:-1], x[1:], dots(imgui.ImVec4(P.RAW.x, P.RAW.y, P.RAW.z, 0.4), 1.5))
            ends = np.array([-lim, lim])
            implot.plot_line("slope 1", ends, ends, P.spec(P.DIM, 1.0))
            implot.plot_line("slope r1", ends, r * ends, P.spec(KEY, 2.0))
            implot.end_plot()

        dl = imgui.get_window_draw_list()
        x, y = D.canvas(w, 1.4 * em)
        light, share = float(self.r1_light[p]), float(self.share[p])
        D.segments(dl, x, y, [("pixel ", P.DIM), (f"{self.pick[0]}, {self.pick[1]}", P.HIGHLIGHT), ("   r1 ", P.DIM), (f"{r:.3f}", KEY),
                              ("  =  r1 of its light ", P.DIM), (f"{light:.3f}", KEY), ("  x  share ", P.DIM), (f"{share:.3f}", KEY),
                              (f"  ({light * share:.3f})      photons / frame ", P.DIM), (f"{self.mean_img.flat[p]:.3g}", KEY)], 0.85)
        x, y = D.canvas(w, 1.4 * em)
        D.segments(dl, x, y, [(f"noise alone: 0 +/- {null:.3f} over {frames} frames      pixels past 3 SD ", P.DIM),
                              (f"{int((self.r1 > 3 * null).sum())} / {self.r1.size}", KEY)], 0.85)
        note("The cell that never spikes (lower left) is the brightest in the mean image, and with a calcium indicator reads about 0: r1 maps "
             "change, not brightness (a voltage indicator still shows its subthreshold voltage). With motion it lights up along with everything "
             "else whose light moves; a low-pass below fs / 2 correlates the noise itself and lights up every pixel. nu reads the same "
             "neighbouring pairs: var(x[t+1] - x[t]) = 2 var(x) (1 - r1).")
        note("A cell's light is spread over its pixels: at 1e4 photons / s GCaMP6f's cells stay inside the noise band here, though the cell's "
             "summed trace above finds its spikes. GCaMP6s's longer transients, or 10x the photons, bring them out.")

    def picture(self, label: str, img: np.ndarray, side: float, hi: float, cmap: int) -> None:
        """A movie-sized image from 0 to hi that picks a pixel on click, its label centered under it."""
        imgui.begin_group()
        P.image_plot(f"##{label}", img, side, 0.0, hi, cmap, on_click=self.select, mark=self.pick)
        if imgui.is_item_hovered():
            imgui.set_tooltip("click or drag: pick a pixel")
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (side - imgui.calc_text_size(label).x) / 2)
        imgui.text_colored(P.DIM, label)
        imgui.end_group()

    def select(self, x: int, y: int) -> None:
        self.pick = (x, y)
