"""
The Traces stage: one ROI's recording from spikes to photons, for calcium or voltage indicators. Subsections: indicator
response, dF/F and its baseline, neuropil, bleaching and noise (each part switchable on the page), and temporal
smoothing at two widths, which lowers and widens the average event. The cell comes from the rail's Recording.
"""

import numpy as np
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui, implot

from imgui_ophys import diagram as D
from imgui_ophys import plotting as P
from imgui_ophys import sim
from imgui_ophys.lesson import Lesson
from imgui_ophys.widgets import KEY

ImVec2, ImVec4 = imgui.ImVec2, imgui.ImVec4
TRUTH = ImVec4(0.35, 0.85, 0.45, 1.0)
PARTS = (
    ("signal", TRUTH, "the cell: resting light x (1 + dF/F)"),
    ("bleaching", ImVec4(0.95, 0.55, 0.15, 1.0), "the indicator fading over the session"),
    ("neuropil", ImVec4(0.58, 0.40, 0.74, 1.0), "out-of-focus light from everything around the cell"),
    ("shot", ImVec4(0.35, 0.65, 0.95, 1.0), "photon counting: Poisson, SD = sqrt(photons)"),
    ("read", ImVec4(0.75, 0.75, 0.78, 1.0), "the camera's electronics, the same every frame"),
)
DOT_FRAMES = 8
WIDTHS = (1.0, 3.0)  # Gaussian SDs in frames the smoothing section shows
SMOOTH = (P.ACCENT, ImVec4(0.91, 0.48, 0.64, 1.0))  # one color per width
F0_WINDOW = 5.0  # s, the running 20th-percentile baseline


class TracesLesson(Lesson):
    key = "traces"
    title = "Traces"
    summary = "from spikes to fluorescence to dF/F"
    uses_recording = True
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
        self.params = self.recording.params()
        self.baseline = "running window"
        self.dot_clock = 0.0
        self.make()

    def set_recording(self, recording: sim.Recording) -> None:
        self.recording = recording
        on = self.params
        self.params = recording.params(shot=on.shot, read=on.read, background=on.background, bleaching=on.bleaching)
        self.make()

    def make(self) -> None:
        self.trace = sim.make_trace(self.params)
        self.refilter()

    def refilter(self) -> None:
        tr = self.trace
        f = tr.recorded
        self.f0 = sim.baseline(f, self.baseline, int(F0_WINDOW * self.params.fs))
        self.dff = sim.to_dff(f, self.f0)
        self.clean = self.trace.dff
        self.smooth = {w: sim.denoise(self.dff, "gaussian", w) for w in WIDTHS}
        fs = self.params.fs
        ind = sim.BY_NAME[self.params.indicator]
        span = (0.2, 2.0) if ind.kind == "calcium" else (0.01, 0.03)
        iso = span[1] if ind.kind == "calcium" else 0.03
        self.lag, self.ev_truth = sim.event_average(self.clean, tr.spikes, fs, span[0], span[1], iso)
        _, self.ev_raw = sim.event_average(self.dff, tr.spikes, fs, span[0], span[1], iso)
        self.ev_smooth = {w: sim.event_average(self.smooth[w], tr.spikes, fs, span[0], span[1], iso)[1] for w in WIDTHS}

    @property
    def indicator(self) -> sim.Indicator:
        return sim.BY_NAME[self.params.indicator]

    def update(self, dt: float) -> None:
        self.dot_clock += dt

    def animating(self) -> bool:
        return True  # the photon dots re-roll

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
        D.text(dl, bx + 1.5 * em, y + 0.5 * em, f"F0 = 20th percentile of F over a running {F0_WINDOW:g} s window", P.DIM, 0.8)
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
        D.heading(fa.ICON_FA_FILTER, "Temporal smoothing", f"A Gaussian over neighbouring frames, SD {WIDTHS[0]:g} and {WIDTHS[1]:g} frames: less noise, but events get lower and wider. Fast voltage spikes suffer most.")
        em = D.em()
        tr = self.trace
        h = 14 * em
        if P.begin_plot("##traces_main", ImVec2(D.guide_width(), h)):
            implot.setup_axes("s", "dF/F", 0, implot.AxisFlags_.auto_fit)
            implot.plot_line("recorded", tr.t, np.ascontiguousarray(self.dff), P.spec(P.RAW, 1.0))
            implot.plot_line("truth", tr.t, np.ascontiguousarray(self.clean), P.spec(TRUTH, 1.5))
            for w, color in zip(WIDTHS, SMOOTH):
                implot.plot_line(f"SD {w:g} frames", tr.t, np.ascontiguousarray(self.smooth[w]), P.spec(color, 2.0))
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
            for width, color in zip(WIDTHS, SMOOTH):
                implot.plot_line(f"SD {width:g} frames", lag, np.ascontiguousarray(self.ev_smooth[width]), P.spec(color, 2.0))
            implot.end_plot()
        imgui.same_line(0, 1.5 * em)
        imgui.begin_group()
        imgui.text_colored(KEY, "average isolated event")
        for label, value in self.metrics():
            imgui.text_disabled(label)
            imgui.same_line(13 * em)
            imgui.text(value)
        imgui.end_group()

    def metrics(self) -> list[tuple[str, str]]:
        """Noise SD, the share of the event's peak kept, and the SNR, for the recorded trace and each smoothing."""
        if not self.ev_truth.size or not np.any(self.ev_truth):
            return [("events", "no isolated spike")]
        at = int(np.argmax(np.abs(self.ev_truth)))  # the frame of the true event's peak; the others are read there
        truth_peak = float(self.ev_truth[at])
        rows = [("recorded", self.dff, self.ev_raw)] + [(f"SD {w:g} frames", self.smooth[w], self.ev_smooth[w]) for w in WIDTHS]
        out = []
        for name, trace, event in rows:
            sd = float(np.std(trace - self.clean))
            out.append((f"noise SD, {name}", f"{sd:.3f}"))
            out.append((f"peak kept, {name}", f"{100 * event[at] / truth_peak:.0f}%"))
            out.append((f"SNR, {name}", f"{abs(event[at]) / max(sd, 1e-9):.1f}"))
        return out
