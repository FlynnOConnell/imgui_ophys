"""
The Traces stage: one ROI's recording from spikes to photons, for calcium or voltage indicators. Subsections: indicator
response, dF/F and its baseline, neuropil, bleaching and noise (each part switchable), and temporal smoothing, with
what it does to an event's shape.
"""

import numpy as np
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui, implot

from imgui_ophys import diagram as D
from imgui_ophys import plotting as P
from imgui_ophys import sim
from imgui_ophys.lesson import Lesson
from imgui_ophys.widgets import KEY, help_mark, section

ImVec2, ImVec4 = imgui.ImVec2, imgui.ImVec4
TRUTH = ImVec4(0.35, 0.85, 0.45, 1.0)
# the parts of a recording, in equation order, with their colors and what each is
PARTS = (
    ("signal", TRUTH, "the cell: resting light x (1 + dF/F)"),
    ("bleaching", ImVec4(0.95, 0.55, 0.15, 1.0), "the indicator fading over the session"),
    ("neuropil", ImVec4(0.58, 0.40, 0.74, 1.0), "out-of-focus light from everything around the cell"),
    ("shot", ImVec4(0.35, 0.65, 0.95, 1.0), "photon counting: Poisson, SD = sqrt(photons)"),
    ("read", ImVec4(0.75, 0.75, 0.78, 1.0), "the camera's electronics, the same every frame"),
)
DOT_FRAMES = 8
PRESETS = {"calcium": dict(rate=0.5, brightness=20.0), "voltage": dict(rate=6.0, brightness=3.0)}


class TracesLesson(Lesson):
    key = "traces"
    title = "Traces"
    summary = "from spikes to fluorescence to dF/F"
    sections = (
        (fa.ICON_FA_BOLT, "Indicator response"),
        (fa.ICON_FA_PERCENT, "dF/F"),
        (fa.ICON_FA_LAYER_GROUP, "Neuropil, bleaching and noise"),
        (fa.ICON_FA_FILTER, "Temporal smoothing"),
    )
    references = (
        ("Chen T-W, Wardill TJ, Sun Y, et al. (2013). Ultrasensitive fluorescent proteins for imaging neuronal activity. Nature 499:295-300.",
         "https://doi.org/10.1038/nature12354"),
        ("Zhang Y, Rozsa M, Liang Y, et al. (2023). Fast and sensitive GCaMP calcium indicators for imaging neural populations. Nature 615:884-891.",
         "https://doi.org/10.1038/s41586-023-05828-9"),
        ("Villette V, Chavarha M, Dimov IK, et al. (2019). Ultrafast two-photon imaging of a high-gain voltage indicator in awake behaving mice. Cell 179:1590-1608.",
         "https://doi.org/10.1016/j.cell.2019.11.004"),
        ("Liu Z, Lu X, Villette V, et al. (2022). Sustained deep-tissue voltage recording using a fast indicator evolved for two-photon microscopy. Cell 185:3408-3425.",
         "https://doi.org/10.1016/j.cell.2022.07.013"),
        ("Giovannucci A, Friedrich J, Gunn P, et al. (2019). CaImAn an open source tool for scalable calcium imaging data analysis. eLife 8:e38173.",
         "https://doi.org/10.7554/eLife.38173"),
        ("suite2p documentation: Signal extraction (neuropil) and spike deconvolution (baseline).",
         "https://suite2p.readthedocs.io/en/latest/roiextraction/"),
    )

    def __init__(self):
        self.params = sim.TraceParams()
        self.method = "gaussian"
        self.width = 1.0  # frames
        self.baseline = "whole recording"
        self.window = 5.0  # s, for the running baseline
        self.dot_clock = 0.0
        self.make()

    def make(self) -> None:
        self.trace = sim.make_trace(self.params)
        self.refilter()

    def refilter(self) -> None:
        tr = self.trace
        f = tr.recorded
        self.f0 = sim.baseline(f, self.baseline, int(self.window * self.params.fs))
        self.dff = sim.to_dff(f, self.f0)
        self.clean = self.trace.dff
        self.smooth = sim.denoise(self.dff, self.method, self.width)
        fs = self.params.fs
        ind = sim.BY_NAME[self.params.indicator]
        span = (0.2, 2.0) if ind.kind == "calcium" else (0.01, 0.03)
        iso = span[1] if ind.kind == "calcium" else 0.03
        self.lag, self.ev_truth = sim.event_average(self.clean, tr.spikes, fs, span[0], span[1], iso)
        _, self.ev_raw = sim.event_average(self.dff, tr.spikes, fs, span[0], span[1], iso)
        _, self.ev_smooth = sim.event_average(self.smooth, tr.spikes, fs, span[0], span[1], iso)

    @property
    def indicator(self) -> sim.Indicator:
        return sim.BY_NAME[self.params.indicator]

    def update(self, dt: float) -> None:
        self.dot_clock += dt

    def animating(self) -> bool:
        return True  # the photon dots re-roll

    # ------------------------------------------------------------------
    # rail
    # ------------------------------------------------------------------

    def controls(self) -> None:
        p = self.params
        section(f"{fa.ICON_FA_DNA}  Indicator")
        names = [i.name for i in sim.INDICATORS]
        changed, idx = imgui.combo("indicator", names.index(p.indicator), names)
        if changed:
            new = sim.INDICATORS[idx]
            if new.kind != self.indicator.kind:
                p.fs, p.duration = new.fs, new.duration
                p.rate, p.brightness = PRESETS[new.kind]["rate"], PRESETS[new.kind]["brightness"]
            p.indicator = new.name
            self.make()
        help_mark(self.indicator.note)
        dirty = False
        for label, attr, lo, hi, fmt in (
            ("frame rate", "fs", 5.0, 2000.0, "%.0f Hz"),
            ("duration", "duration", 1.0, 120.0, "%.0f s"),
            ("spike rate", "rate", 0.05, 20.0, "%.2f /s"),
        ):
            c, v = imgui.slider_float(label, getattr(p, attr), lo, hi, fmt, imgui.SliderFlags_.logarithmic)
            if c:
                setattr(p, attr, v)
                dirty = True

        section(f"{fa.ICON_FA_CAMERA}  Light")
        for label, attr, lo, hi, fmt, tip in (
            ("brightness", "brightness", 0.2, 200.0, "%.1f ph/px", "Photons per pixel per frame at rest."),
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
        if dirty:
            self.make()

        section(f"{fa.ICON_FA_PERCENT}  dF/F")
        changed, idx = imgui.combo("F0", sim.BASELINES.index(self.baseline), sim.BASELINES)
        if changed:
            self.baseline = sim.BASELINES[idx]
            self.refilter()
        help_mark("F0: the 20th percentile of F, over the whole recording or a running window.")
        c, self.window = imgui.slider_float("window", self.window, 0.2, 60.0, "%.1f s", imgui.SliderFlags_.logarithmic)
        if c:
            self.refilter()

        section(f"{fa.ICON_FA_FILTER}  Smoothing")
        changed, idx = imgui.combo("method", sim.DENOISERS.index(self.method), sim.DENOISERS)
        if changed:
            self.method = sim.DENOISERS[idx]
            self.refilter()
        c, self.width = imgui.slider_float("width", self.width, 0.2, 30.0, "%.1f frames", imgui.SliderFlags_.logarithmic)
        if c:
            self.refilter()
        help_mark("Box length, Gaussian SD, or exponential time constant, in frames.")

    def zoom(self) -> tuple[float, float]:
        """A window around the first spike with room on both sides: (t0, t1) in s."""
        ind, tr = self.indicator, self.trace
        span = (0.5, 3.0) if ind.kind == "calcium" else (0.01, 0.04)
        dur = tr.t[-1] if tr.t.size else self.params.duration
        for s in tr.spikes:
            if s - span[0] >= 0 and s + span[1] <= dur:
                return s - span[0], s + span[1]
        return 0.0, min(dur, span[0] + span[1])

    def section_0(self) -> None:
        ind, tr, p = self.indicator, self.trace, self.params
        t0, t1 = self.zoom()
        width_txt = f"{(t1 - t0) * 1000:.0f} ms" if t1 - t0 < 1 else f"{t1 - t0:.1f} s"
        D.heading(fa.ICON_FA_BOLT, "Indicator response", f"One spike through the chain ({ind.name}, {width_txt} around it). The camera averages light over each frame.")
        em, dl = D.em(), imgui.get_window_draw_list()
        w = D.guide_width()
        n = 5
        gap = 1.6 * em
        across = n if w > 50 * em else 3
        cw = (w - (across - 1) * gap) / across
        ch = 7.5 * em
        rows = -(-n // across)
        x, y = D.canvas(w, rows * (ch + gap))
        fine = (tr.fine_t >= t0) & (tr.fine_t <= t1)
        frames = (tr.t >= t0) & (tr.t <= t1)
        spikes = tr.spikes[(tr.spikes >= t0) & (tr.spikes <= t1)] - t0
        fine_dff = None
        if ind.kind == "calcium":
            fine_dff = tr.fine_signal[fine]
            response, response_label = fine_dff, "calcium -> dF/F"
        else:
            response, response_label = tr.fine_signal[fine], "membrane, mV"
        rec = self.dff[frames]
        cards = (
            ("spike", "ticks"),
            (response_label, "line"),
            ("camera frames", "bins"),
            ("photons / frame", "dots"),
            ("recorded dF/F", "rec"),
        )
        for i, (label, kind) in enumerate(cards):
            r, c = divmod(i, across)
            bx, by = x + c * (cw + gap), y + r * (ch + gap)
            D.box(dl, bx, by, cw, ch, "", P.DIM)
            D.text(dl, bx + 0.5 * em, by + 0.3 * em, f"{i + 1}  {label}", KEY, 0.85)
            px, py, pw, ph = bx + 0.4 * em, by + 1.6 * em, cw - 0.8 * em, ch - 2.0 * em
            if kind == "ticks":
                D.ticks(dl, px, py, pw, ph, spikes, t1 - t0)
            elif kind == "line":
                color = TRUTH if ind.kind == "calcium" else P.WARN
                D.sparkline(dl, px, py, pw, ph, response, color)
            elif kind == "bins":
                truth = tr.dff[frames]
                fd = sim.lowpass(ind.amp * (tr.fine_signal[fine] - sim.V_REST) / 100.0, ind.rise, tr.fine_t[1] - tr.fine_t[0]) if ind.kind == "voltage" else fine_dff
                lo, hi = float(min(fd.min(), truth.min())), float(max(fd.max(), truth.max()))
                D.sparkline(dl, px, py, pw, ph, fd, P.DIM, lo, hi)
                self.draw_bars(dl, px, py, pw, ph, truth, lo, hi, TRUTH)
            elif kind == "dots":
                counts = tr.recorded[frames]
                self.draw_photon_columns(dl, px, py, pw, ph, counts[: max(8, min(counts.size, int(pw / D.em(0.45))))])
            else:
                D.sparkline(dl, px, py, pw, ph, rec, P.RAW)
                D.sparkline(dl, px, py, pw, ph, tr.dff[frames], TRUTH, float(rec.min()), float(rec.max()), frame=False, thickness=1.0)
            if c:
                D.arrow(dl, bx - gap + 0.2 * em, by + ch / 2, bx - 0.2 * em, by + ch / 2)
        x, y = D.canvas(w, 1.4 * em)
        D.segments(dl, x, y, [(f"{p.fs:.0f} Hz", KEY), (" frames, ", P.DIM), (f"{1000 / p.fs:.1f} ms", KEY),
                              (" each.   ", P.DIM), (f"{ind.name}", KEY), (f"  tau rise {ind.rise * 1000:.1f} ms, tau decay {ind.decay * 1000:.0f} ms", P.DIM)], 0.85)
        if D.mouse_in(x, y, w, 1.4 * em):
            imgui.set_tooltip(ind.note)

    def draw_bars(self, dl, x, y, w, h, values, lo, hi, color) -> None:
        """One bar per frame from 0 to its value: what the camera keeps of the line behind."""
        n = values.size
        if n == 0:
            return
        pad = D.em(0.25)
        span = (hi - lo) or 1.0
        bw = (w - 2 * pad) / n
        zero = y + h - pad - (np.clip(0.0, lo, hi) - lo) / span * (h - 2 * pad)
        for i, v in enumerate(values):
            top = y + h - pad - (v - lo) / span * (h - 2 * pad)
            a, b = ImVec2(x + pad + i * bw, min(top, zero)), ImVec2(x + pad + (i + 1) * bw - 1, max(top, zero))
            dl.add_rect_filled(a, b, D.u32(color, 0.35))
            dl.add_rect(a, b, D.u32(color, 0.9))

    def draw_photon_columns(self, dl, x, y, w, h, counts) -> None:
        """One column per frame filled with dots from the bottom, a dot per so many photons: taller is brighter."""
        n = counts.size
        if n == 0:
            return
        pad = D.em(0.25)
        dl.add_rect_filled(ImVec2(x, y), ImVec2(x + w, y + h), D.u32(D.PLOT), 3.0)
        cw = (w - 2 * pad) / n
        across = max(1, min(3, int(cw / D.em(0.3))))
        r = max(1.0, min(cw / across * 0.35, D.em(0.12)))
        rows = max(1, int((h - 2 * pad) / (2.3 * r)))
        per_dot = max(1.0, float(counts.max()) / (rows * across))
        for i, count in enumerate(counts):
            k = int(max(count, 0) / per_dot)
            for d in range(min(k, rows * across)):
                row, col = divmod(d, across)
                cx = x + pad + i * cw + (col + 0.5) * cw / across
                cy = y + h - pad - (row + 0.5) * (h - 2 * pad) / rows
                dl.add_circle_filled(ImVec2(cx, cy), r, D.u32(KEY, 0.85))
        label = f"1 dot = {per_dot:.0f} photons"
        D.text(dl, x + w - D.text_width(label, 0.65) - pad, y - D.em(1.15), label, P.DIM, 0.65)

    def section_1(self) -> None:
        D.heading(fa.ICON_FA_PERCENT, "dF/F",
                  "Divide the change by the resting level: dF/F = (F - F0) / F0. F0 is not measured, it is estimated, here as a low percentile of F.")
        em, dl = D.em(), imgui.get_window_draw_list()
        tr = self.trace
        f = tr.recorded
        w = D.guide_width()
        x, y = D.canvas(w, 2.6 * em)
        bx = x
        for i, (label, color) in enumerate((("dF/F", TRUTH), ("=", P.DIM), ("(F - F0)", D.TEXT), ("/", P.DIM), ("F0", KEY))):
            if label in ("=", "/"):
                D.text(dl, bx + 0.4 * em, y + 0.5 * em, label, color, 1.2)
                bx += 1.6 * em
                continue
            bw = D.text_width(label) + 1.4 * em
            D.box(dl, bx, y, bw, 2.0 * em, label, color)
            bx += bw
        D.text(dl, bx + 1.5 * em, y + 0.5 * em, f"F0 = 20th percentile, {self.baseline}" + (f" ({self.window:.1f} s)" if self.baseline != "whole recording" else ""), P.DIM, 0.8)
        x, y = D.canvas(w, 13.5 * em)
        D.text(dl, x, y, "F: photons per frame, and F0", P.DIM, 0.75)
        f0 = np.broadcast_to(self.f0, f.shape)
        lo, hi = float(min(f.min(), f0.min())), float(f.max())
        D.sparkline(dl, x, y + 1.2 * em, w, 5 * em, f, P.RAW, lo, hi)
        D.sparkline(dl, x, y + 1.2 * em, w, 5 * em, f0, KEY, lo, hi, frame=False, thickness=2.0)
        D.text(dl, x, y + 6.8 * em, "dF/F from that F0 (gray) and the true dF/F (green)", P.DIM, 0.75)
        lo, hi = float(min(self.dff.min(), self.clean.min())), float(max(self.dff.max(), self.clean.max()))
        D.sparkline(dl, x, y + 8.0 * em, w, 5 * em, self.dff, P.RAW, lo, hi)
        D.sparkline(dl, x, y + 8.0 * em, w, 5 * em, self.clean, TRUTH, lo, hi, frame=False, thickness=1.5)
        parts = self.trace.on
        notes = []
        if parts.get("neuropil"):
            notes.append("neuropil light raises F0, so every dF/F comes out smaller")
        if parts.get("bleaching") and self.baseline == "whole recording":
            notes.append("bleaching drifts F under a fixed F0: try the running window")
        if notes:
            imgui.text_disabled("  /  ".join(notes))

    def section_2(self) -> None:
        D.heading(fa.ICON_FA_LAYER_GROUP, "Neuropil, bleaching and noise", "The recorded F is the cell's light plus everything else. Click a part to take it out or put it back.")
        em, dl = D.em(), imgui.get_window_draw_list()
        tr = self.trace
        w = D.guide_width()
        n = len(PARTS)
        across = n + 1 if w > 56 * em else 3
        gap = 1.6 * em
        cw = (w - (across - 1) * gap) / across
        ch = 5.4 * em
        rows = -(-(n + 1) // across)
        x, y = D.canvas(w, rows * (ch + gap))
        lo = float(min(tr.recorded.min(), tr.parts["signal"].min()))
        hi = float(max(tr.recorded.max(), tr.parts["signal"].max()))
        for i, (name, color, tip) in enumerate((*PARTS, ("recorded", P.RAW, "the photons counted per frame"))):
            r, c = divmod(i, across)
            bx, by = x + c * (cw + gap), y + r * (ch + gap)
            on = name in ("signal", "recorded") or tr.on.get(name, True)
            hovered = D.mouse_in(bx, by, cw, ch)
            D.box(dl, bx, by, cw, ch, "", color if on else P.DIM, 0.10 if on else 0.0, hovered and name not in ("signal", "recorded"))
            D.text(dl, bx + 0.5 * em, by + 0.3 * em, name, color if on else P.DIM, 0.85)
            values = tr.recorded if name == "recorded" else tr.parts[name]
            if name in ("signal", "recorded"):
                D.sparkline(dl, bx + 0.4 * em, by + 1.6 * em, cw - 0.8 * em, ch - 2.0 * em, values, color if on else P.DIM, lo, hi)
            elif name in ("shot", "read"):
                m = float(np.abs(values).max()) or 1.0
                D.sparkline(dl, bx + 0.4 * em, by + 1.6 * em, cw - 0.8 * em, ch - 2.0 * em, values, color if on else P.DIM, -m, m)
            else:
                D.sparkline(dl, bx + 0.4 * em, by + 1.6 * em, cw - 0.8 * em, ch - 2.0 * em, values, color if on else P.DIM)
                if not on:
                    dl.add_line(ImVec2(bx + 0.4 * em, by + ch - 0.4 * em), ImVec2(bx + cw - 0.4 * em, by + 1.6 * em), D.u32(P.ERROR, 0.8), 2.0)
            if hovered:
                imgui.set_tooltip(tip)
                if name not in ("signal", "recorded"):
                    imgui.set_mouse_cursor(imgui.MouseCursor_.hand)
                    if imgui.is_mouse_clicked(0):
                        self.toggle(name)
            if c:
                symbol = "=" if name == "recorded" else "+"
                D.text(dl, bx - gap / 2, by + ch / 2 - 0.7 * em, symbol, P.DIM, 1.4, center=True)

    def toggle(self, name: str) -> None:
        attr = {"bleaching": "bleaching", "neuropil": "background", "shot": "shot", "read": "read"}[name]
        setattr(self.params, attr, not getattr(self.params, attr))
        self.trace.on[name] = getattr(self.params, attr)
        self.refilter()

    def section_3(self) -> None:
        D.heading(fa.ICON_FA_FILTER, "Temporal smoothing", "Smoothing averages neighbouring frames: less noise, but events get lower and wider. Fast voltage spikes suffer most.")
        em = D.em()
        tr = self.trace
        h = 14 * em
        if P.begin_plot("##traces_main", ImVec2(D.guide_width(), h)):
            implot.setup_axes("s", "dF/F", 0, implot.AxisFlags_.auto_fit)
            implot.plot_line("recorded", tr.t, np.ascontiguousarray(self.dff), P.spec(P.RAW, 1.0))
            implot.plot_line("truth", tr.t, np.ascontiguousarray(self.clean), P.spec(TRUTH, 1.5))
            if self.method != "none":
                implot.plot_line(f"{self.method}", tr.t, np.ascontiguousarray(self.smooth), P.spec(P.ACCENT, 2.0))
            if tr.spikes.size:
                implot.plot_inf_lines("spikes", np.ascontiguousarray(tr.spikes), P.spec(ImVec4(P.HIGHLIGHT.x, P.HIGHLIGHT.y, P.HIGHLIGHT.z, 0.35), 1.0))
            implot.end_plot()
        imgui.text_disabled("scroll to zoom, drag to pan, double-click to fit")

        w = D.guide_width()
        imgui.dummy(ImVec2(0, 0.6 * em))
        side = min(w * 0.5, 26 * em)
        if P.begin_plot("##traces_event", ImVec2(side, 12 * em)):
            implot.setup_axes("s from spike", "dF/F", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            lag = np.ascontiguousarray(self.lag)
            implot.plot_line("truth", lag, np.ascontiguousarray(self.ev_truth), P.spec(TRUTH, 2.0))
            implot.plot_line("recorded", lag, np.ascontiguousarray(self.ev_raw), P.spec(P.RAW, 1.0))
            if self.method != "none":
                implot.plot_line(self.method, lag, np.ascontiguousarray(self.ev_smooth), P.spec(P.ACCENT, 2.0))
            implot.end_plot()
        imgui.same_line(0, 1.5 * em)
        imgui.begin_group()
        imgui.text_colored(KEY, "average isolated event")
        for label, value in self.metrics():
            imgui.text_disabled(label)
            imgui.same_line(12 * em)
            imgui.text(value)
        imgui.end_group()

    def metrics(self) -> list[tuple[str, str]]:
        def peak(x: np.ndarray) -> float:
            return float(x[np.argmax(np.abs(x))]) if x.size else 0.0

        def noise(x: np.ndarray) -> float:
            return float(np.std(x - self.clean))

        truth_peak = peak(self.ev_truth)
        out = [
            ("noise SD, recorded", f"{noise(self.dff):.3f}"),
            ("noise SD, smoothed", f"{noise(self.smooth):.3f}"),
        ]
        if truth_peak:
            raw_p, sm_p = peak(self.ev_raw), peak(self.ev_smooth)
            out += [
                ("peak kept, recorded", f"{100 * raw_p / truth_peak:.0f}%"),
                ("peak kept, smoothed", f"{100 * sm_p / truth_peak:.0f}%"),
                ("SNR, recorded", f"{abs(truth_peak) / max(noise(self.dff), 1e-9):.1f}"),
                ("SNR, smoothed", f"{abs(sm_p) / max(noise(self.smooth), 1e-9):.1f}"),
            ]
            d = (np.argmax(np.abs(self.ev_smooth)) - np.argmax(np.abs(self.ev_truth))) / self.params.fs
            out.append(("peak delay", f"{1000 * d:+.0f} ms"))
        else:
            out.append(("events", "no isolated spike"))
        return out
