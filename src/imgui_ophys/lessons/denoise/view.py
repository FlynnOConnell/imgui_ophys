"""
The Denoising stage: vnoiser's Morlet wavelet denoiser on one trace, one section per stage, every parameter in the
rail, and the stages after a changed parameter recomputed on a worker thread while the page keeps drawing the last
run. The trace is a gevisim cell with one noise knob or a recorded MUnit; both go from pixels to z as the lab's
pipeline does. Every time plot shares one window (`t0`, `t1`), so zooming one zooms all.
"""

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui, implot
from imgui_bundle import portable_file_dialogs as pfd

from imgui_ophys import diagram as D
from imgui_ophys import plotting as P
from imgui_ophys.lesson import Lesson
from imgui_ophys.lessons.denoise import plots as X
from imgui_ophys.widgets import CARD, CARD_HOVER, KEY, push_size, section

try:
    from imgui_ophys.lessons.denoise import pipeline as M
    from imgui_ophys.lessons.denoise import signals as S
except ImportError:  # the denoise extra is optional; the page says how to install it
    M = S = None

ImVec2, ImVec4 = imgui.ImVec2, imgui.ImVec4
INSTALL = "Install the denoise extra: uv pip install -e '.[denoise]'"
LIT = ImVec4(P.ACCENT.x, P.ACCENT.y, P.ACCENT.z, 0.22)
STAGE_NAMES = {"signal": "trace", "cwt": "transform", "cluster": "grouping", "reduce": "bands", "mask": "masks",
               "final": "sum", "peaks": "peaks"}
HEAT_MODES = ("amplitude", "amplitude / median", "standardized", "real part")
VIEWS = (("all", 0.0), ("4 s", 4.0), ("1 s", 1.0), ("200 ms", 0.2))
SECONDS_COPIED = 3.0  # how long the rail says the code was copied
IMAGE_COLUMNS = 1000  # heatmaps are pooled to about this many columns of the view
IMAGE_ROWS = 40  # and a pixel image to this many rows per block


class DenoiseLesson(Lesson):
    key = "denoise"
    title = "Denoising (vnoiser)"
    summary = "the masked Morlet bands of z added back up, plus z's slow part"
    sections = (
        (fa.ICON_FA_GRIP_LINES, "Pixels to z"),
        (fa.ICON_FA_CHART_AREA, "Scalogram"),
        (fa.ICON_FA_LAYER_GROUP, "Bands"),
        (fa.ICON_FA_FILTER, "Masks"),
        (fa.ICON_FA_WAVE_SQUARE, "Denoised"),
        (fa.ICON_FA_BOLT, "Peaks"),
    )
    references = (
        ("Land MA, et al. (2026). JEDI-3 voltage indicators for two-photon imaging of subthreshold and spiking activity. Nat Methods 23:986-997.",
         "https://doi.org/10.1038/s41592-026-03043-8"),
        ("vnoiser: the spatial JEDI pipeline's Morlet wavelet denoiser, as a package.",
         "https://github.com/FlynnOConnell/vnoiser"),
        ("gevisim: voltage imaging simulator with published GEVI kinetics and photophysics.",
         "https://github.com/FlynnOConnell/gevisim"),
        ("Torrence C, Compo GP (1998). A practical guide to wavelet analysis. Bull. Am. Meteorol. Soc. 79:61-78.",
         "https://doi.org/10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2"),
    )

    def __init__(self):
        self.settings = M.Settings() if M else None
        self.run = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future: Optional[Future] = None
        self.submitted = None
        self.stage_running = ""
        self.t0 = implot.BoxedValue(0.0)
        self.t1 = implot.BoxedValue(S.SIM_DURATION_S if S else 1.0)
        self.row = 40
        self.heat_mode = 1
        self.heat_cache = (None, None)
        self.pixel_cache = (None, None)
        self.colormaps: Optional[dict] = None
        self.band_id: Optional[int] = None
        self.picker = None
        self.scans: Optional[list] = None  # exported units found under ~/data
        self.time_copied = -math.inf

    # ------------------------------------------------------------------
    # the worker
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        if M is not None:
            self.poll()

    def animating(self) -> bool:
        return self.future is not None

    def poll(self) -> None:
        """Land a finished run, then start the next if the settings moved on."""
        if self.future is not None and self.future.done():
            run = self.future.result()
            new_signal = self.run is None or run.signal is not self.run.signal
            self.run, self.future, self.stage_running = run, None, ""
            if new_signal and run.signal is not None:
                self.t0.value, self.t1.value = 0.0, run.signal.duration
            self.check_selection()
        if self.future is None and self.settings.first_change(self.submitted) is not None:
            start = self.settings.first_change(self.run.settings if self.run else None) or "final"
            self.submitted = self.settings.copy()
            self.future = self.executor.submit(M.compute, self.submitted, start, self.run, self.on_stage)
        if self.picker is not None and self.picker.ready():
            chosen = self.picker.result()
            self.picker = None
            if chosen:
                self.settings.signal.scan_path = chosen[0]

    def on_stage(self, stage: str) -> None:
        self.stage_running = stage

    def wait(self):
        """Block until the settings' run is in (for scripts and tests)."""
        while self.future is not None or self.settings.first_change(self.submitted) is not None:
            if self.future is not None:
                self.future.result()
            self.poll()
        return self.run

    def check_selection(self) -> None:
        run = self.run
        if run is None or not run.bands:
            return
        if run.coeff is not None:
            self.row = int(np.clip(self.row, 0, run.coeff.shape[0] - 1))
        if run.band(self.band_id) is None:
            kept = run.kept or run.bands
            self.band_id = kept[len(kept) // 2].id

    @property
    def ready(self) -> bool:
        return self.run is not None and self.run.signal is not None

    def has(self, stage: str) -> bool:
        """Whether the last run got through stage."""
        return self.run is not None and stage in self.run.timing

    def band_order(self) -> dict:
        """Band id -> its color index (bands in frequency order)."""
        return {band.id: i for i, band in enumerate(self.run.bands)} if self.run else {}

    def state(self) -> tuple:
        """(color, text): what the pipeline is doing or how it ended."""
        run = self.run
        if run is not None and run.error:
            return P.ERROR, f"failed at the {STAGE_NAMES.get(run.failed, run.failed)}: {run.error}"
        if self.future is not None:
            return P.WARN, f"running: {STAGE_NAMES.get(self.stage_running, 'starting')}"
        if self.has("peaks"):
            return P.DIM, f"{len(run.kept)} of {len(run.bands)} bands kept, {run.peaks.peaks.size} peaks"
        if self.has("final"):
            return P.DIM, f"{len(run.kept)} of {len(run.bands)} bands kept"
        return P.DIM, "starting"

    # ------------------------------------------------------------------
    # the rail
    # ------------------------------------------------------------------

    def controls(self) -> None:
        if M is None:
            imgui.text_colored(P.WARN, INSTALL)
            return
        self.controls_source()
        imgui.dummy(ImVec2(0, 0.4 * D.em()))
        for title, draw in (
            ("Transform", self.controls_scalogram), ("Grouping", self.controls_bands), ("Masks", self.controls_masks),
            ("Sum", self.controls_final), ("Peaks", self.controls_peaks),
        ):
            if imgui.collapsing_header(title):
                draw()
        imgui.dummy(ImVec2(0, 0.4 * D.em()))
        self.controls_presets()

    def controls_source(self) -> None:
        sp = self.settings.signal
        section(f"{fa.ICON_FA_GRIP_LINES}  Source")
        for key, label, hint in S.SOURCES:
            if imgui.radio_button(label, sp.source == key):
                sp.source = key
            X.tooltip(hint)
        if sp.source == "munit":
            self.controls_munit()
            return
        if X.begin_form("gevisim"):
            X.row("noise", "Shot noise SD relative to the simulated photon budget; 0 is the noiseless cell")
            _, sp.noise = imgui.slider_float("##noise", sp.noise, 0.0, 5.0, "%.2f x")
            imgui.end_table()
        if imgui.button(f"{fa.ICON_FA_DICE}  New noise"):
            sp.seed += 1
        X.tooltip("Another draw of the shot noise on the same cell")

    def controls_munit(self) -> None:
        sp = self.settings.signal
        if self.scans is None:
            self.scans = S.find_scans()
            if not sp.scan_path and self.scans:
                sp.scan_path = self.scans[0]
        choices = list(self.scans)
        if sp.scan_path and sp.scan_path not in choices:
            choices.append(sp.scan_path)
        names = [scan_label(f) for f in choices]
        imgui.set_next_item_width(-1)
        changed, i = imgui.combo("##scan", choices.index(sp.scan_path) if sp.scan_path in choices else -1, names)
        if changed:
            sp.scan_path = choices[i]
        if imgui.button(f"{fa.ICON_FA_FOLDER_OPEN}  Browse...") and self.picker is None:
            self.picker = pfd.open_file("Open an exported MUnit", sp.scan_path or str(Path.home()), ["MUnit exports", "*.npz"])
        if not sp.scan_path:
            imgui.text_colored(P.WARN, "No exported unit under ~/data; see scripts/export_mesc_linescan.py")
            return
        try:
            info = S.scan_info(sp.scan_path)
        except ValueError as e:
            imgui.push_text_wrap_pos(0)
            imgui.text_colored(P.ERROR, str(e))
            imgui.pop_text_wrap_pos()
            return
        fs, total = info["fs"], info["frames"] / info["fs"]
        imgui.text_colored(P.DIM, f"{info['unit']}, {fs:.0f} Hz, {total:.0f} s")
        imgui.text_disabled("lines in the domain")
        chosen = set(sp.rois)
        right = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
        for k, (roi, width) in enumerate(sorted(info["widths"].items())):
            label = f"{roi} ({width} px)" if width > 1 else str(roi)
            if k:
                imgui.same_line()
                need = imgui.get_frame_height() + imgui.get_style().item_inner_spacing.x + imgui.calc_text_size(label).x
                if imgui.get_cursor_pos_x() + need > right:
                    imgui.new_line()
            changed, on = imgui.checkbox(f"{label}##roi{roi}", roi in chosen)
            if changed:
                chosen = chosen | {roi} if on else chosen - {roi}
        if chosen:
            sp.rois = tuple(sorted(chosen))
        sp.duration_s = float(min(sp.duration_s, total))
        if X.begin_form("window"):
            X.row("length")
            _, sp.duration_s = imgui.slider_float("##length", sp.duration_s, 1.0, total, "%.1f s", imgui.SliderFlags_.logarithmic)
            X.row("start")
            _, sp.start_s = imgui.slider_float("##start", sp.start_s, 0.0, max(total - sp.duration_s, 0.0), "%.1f s")
            imgui.end_table()

    def controls_scalogram(self) -> None:
        s = self.settings
        if not X.begin_form("scalogram"):
            return
        X.row("color", "amplitude: |row|. amplitude / median: each row over its own median, so every row's usual level "
                       "looks the same. standardized: each complex row minus its mean, over its SD. real part: the "
                       "filtered trace itself.")
        _, self.heat_mode = imgui.combo("##heat", self.heat_mode, HEAT_MODES)
        n = self.run.coeff.shape[0] if self.has("cwt") else int(s.n_scales)
        X.row("row", "Or drag the white line on the scalogram")
        freq = f"  ({self.run.freqs[self.row]:.3g} Hz)" if self.has("cwt") else ""
        _, self.row = imgui.slider_int("##row", self.row, 0, n - 1, f"%d{freq}")
        self.row = int(np.clip(self.row, 0, n - 1))
        X.row("at this fs", "The scales are set in samples at 1075 Hz. Ticked, they are multiplied by fs / 1075 so the "
                            "rows keep their frequencies in Hz on a scan at another rate, as mbo_utilities' at_fs does "
                            "for the lab's pipeline. Unticked, Denoiser's 1 to 1000 samples at any rate.")
        _, s.scale_to_fs = imgui.checkbox("##scale_to_fs", s.scale_to_fs)
        imgui.end_table()
        if self.ready:
            fs = self.run.signal.fs
            lo, hi = s.scale_bounds(fs)
            imgui.text_disabled(f"{int(s.n_scales)} scales, {lo:.3g} to {hi:.3g} samples: {s.c * fs / hi:.2g} to {s.c * fs / lo:.0f} Hz")

    def controls_bands(self) -> None:
        c = self.settings.clust
        n_rows = int(self.settings.n_scales)
        if X.begin_form("bands"):
            X.row("standardize", "Each row minus its mean, over its SD, before the PCA")
            _, c.standardize = imgui.checkbox("##standardize", c.standardize)
            X.row("PCA", "Principal components computed from the rows")
            _, c.n_components = imgui.slider_int("##pca", c.n_components, 2, min(60, n_rows), "%d components")
            X.row("clustered on", "The components the clustering uses, counted from the first")
            _, c.n_comp_clu = imgui.slider_int("##pcs", c.n_comp_clu, 1, c.n_components, "first %d")
            methods = ["ward", "average", "complete", "single"]
            X.row("linkage")
            changed, i = imgui.combo("##linkage", methods.index(c.method_linkage) if c.method_linkage in methods else 0, methods)
            if changed:
                c.method_linkage = methods[i]
            X.row("bands", "The most bands the tree is cut into")
            _, c.n_subclusters = imgui.slider_int("##n_bands", c.n_subclusters, 2, min(30, n_rows - 1))
            imgui.end_table()
        c.n_comp_clu = min(c.n_comp_clu, c.n_components)
        if not self.has("reduce"):
            return
        order = self.band_order()
        if imgui.begin_table("##band_table", 4, imgui.TableFlags_.row_bg | imgui.TableFlags_.sizing_fixed_fit):
            for col in ("", "Hz", "rows", "events"):
                imgui.table_setup_column(col)
            imgui.table_headers_row()
            for band in self.run.bands:
                imgui.table_next_row()
                imgui.table_next_column()
                X.swatch(f"c{band.id}", X.band_color(order[band.id]))
                imgui.table_next_column()
                imgui.text(band.span)
                imgui.table_next_column()
                imgui.text(str(int(band.rows.sum())))
                imgui.table_next_column()
                if band.kept:
                    imgui.text(str(len(band.windows)))
                else:
                    imgui.text_colored(P.WARN, "0")
            imgui.end_table()

    def controls_masks(self) -> None:
        r, t = self.settings.reducer, self.settings.thres
        if self.has("reduce"):
            imgui.text_disabled("band on the page")
            order = self.band_order()
            for band in reversed(self.run.bands):
                X.swatch(f"b{band.id}", X.band_color(order[band.id]))
                label = band.span if band.kept else f"{band.span}, dropped"
                if imgui.selectable(f"{label}##{band.id}", self.band_id == band.id)[0]:
                    self.band_id = band.id
        imgui.text_disabled("events")
        if X.begin_form("events"):
            X.row("under 30 Hz", "Where an event opens, in SDs of the band")
            _, r.slow_upthres = imgui.slider_float("##slow", r.slow_upthres, 0.5, 6.0, "%.2f SD")
            X.row("30 Hz and up")
            _, r.fast_upthres = imgui.slider_float("##fast", r.fast_upthres, 0.5, 6.0, "%.2f SD")
            X.row("smoothing", "savgol windows in samples, for bands at or below 75 Hz, slowest band first")
            changed, windows = imgui.input_int3("##smoothing", list(r.smooth_windows))
            if changed:
                r.smooth_windows = tuple(max(3, int(w)) for w in windows)
            X.row("complex", "Bands above 75 Hz keep their complex mean: its real part is what's added back")
            _, r.complex_bands = imgui.checkbox("above 75 Hz##complex", r.complex_bands)
            imgui.end_table()
        imgui.text_disabled("between events")
        if X.begin_form("mask"):
            kinds = ["soft", "hard"]
            X.row("type", "soft: a floor per band speed. hard: one floor for every band")
            changed, i = imgui.combo("##type", kinds.index(t.thres_type), kinds)
            if changed:
                t.thres_type = kinds[i]
            if t.thres_type == "soft":
                X.row("floor", "Under 5 Hz, under 30, under 80, 80 and up")
                changed, levels = imgui.slider_float4("##levels", list(t.soft_levels), 0.0, 1.0, "%.2f")
                if changed:
                    t.soft_levels = tuple(levels)
                X.row("blur", "Gaussian SD of the mask in samples: up to 120 Hz, above")
                changed, sigmas = imgui.input_float2("##blur", list(t.soft_sigma), "%.1f")
                if changed:
                    t.soft_sigma = tuple(max(0.1, v) for v in sigmas)
            else:
                X.row("floor")
                _, t.hard_floor = imgui.slider_float("##floor", t.hard_floor, 0.0, 1.0, "%.3f")
                X.row("blur", "Gaussian SD of the mask in samples: up to 120 Hz, above")
                changed, sigmas = imgui.input_float2("##blur", list(t.hard_sigma), "%.1f")
                if changed:
                    t.hard_sigma = tuple(max(0.1, v) for v in sigmas)
            imgui.end_table()

    def controls_final(self) -> None:
        s = self.settings
        if X.begin_form("final"):
            X.row("slow part", "The input below the cutoff, through a zero-phase FIR low-pass, added to the bands")
            _, s.add_baseline = imgui.checkbox("##add", s.add_baseline)
            X.row("below")
            _, s.lp_cutoff = imgui.slider_float("##cutoff", s.lp_cutoff, 0.05, 20.0, "%.3g Hz", imgui.SliderFlags_.logarithmic)
            imgui.end_table()
        if not self.has("mask"):
            return
        imgui.text_disabled("bands in the sum")
        order = self.band_order()
        excluded = set(s.excluded)
        for band in reversed(self.run.bands):
            if not band.kept:
                imgui.text_disabled(f"{band.span}, dropped")
                continue
            imgui.push_style_color(imgui.Col_.check_mark, X.band_color(order[band.id]))
            changed, on = imgui.checkbox(f"{band.span}##inc{band.id}", band.id not in excluded)
            imgui.pop_style_color()
            if changed:
                excluded = excluded - {band.id} if on else excluded | {band.id}
        s.excluded = tuple(sorted(excluded))

    def controls_peaks(self) -> None:
        c = self.settings.spike
        if not X.begin_form("peaks"):
            return
        X.row("band-pass", "A zero-phase Butterworth band-pass (order 5) of the trace; the top is capped under Nyquist")
        lo, hi = c.bp
        _, lo = imgui.slider_float("##bp_lo", float(lo), 0.2, 50.0, "from %.3g Hz", imgui.SliderFlags_.logarithmic)
        X.row("")
        _, hi = imgui.slider_float("##bp_hi", float(hi), 20.0, 2000.0, "to %.3g Hz", imgui.SliderFlags_.logarithmic)
        c.bp = (float(lo), float(max(hi, 1.5 * lo)))
        X.row("band-passed", "A local maximum of the band-passed trace must reach its mean plus this many SD")
        _, c.thres_bp_sd = imgui.slider_float("##bp_sd", c.thres_bp_sd, 0.5, 10.0, "over %.1f SD")
        X.row("trace", "And the trace itself its mean plus this many SD")
        _, c.thres_amp_sd = imgui.slider_float("##amp_sd", c.thres_amp_sd, 0.5, 10.0, "over %.1f SD")
        X.row("lasting", "The stretch of the trace over the on/off level around the peak must last this long")
        _, c.duration_thres_ms = imgui.slider_float("##dur", c.duration_thres_ms, 0.0, 30.0, "%.1f ms")
        X.row("on/off", "That level: the trace's median plus this many SD")
        _, c.onoff_sd = imgui.slider_float("##onoff", c.onoff_sd, 0.0, 3.0, "%.2f SD")
        X.row("spacing", "Peaks closer than this keep the higher one; in samples at 1075 Hz, scaled with the rate")
        _, c.distance_samples = imgui.slider_int("##dist", int(c.distance_samples), 1, 30, "%d samples")
        imgui.end_table()

    def controls_presets(self) -> None:
        """vnoiser's defaults, the lab's settings, and the code for the current ones."""
        width = imgui.get_content_region_avail().x
        half = (width - imgui.get_style().item_spacing.x) / 2
        if imgui.button(f"{fa.ICON_FA_ARROW_ROTATE_LEFT}  Defaults", ImVec2(half, 0)):
            self.settings.use_defaults()
        X.tooltip("Denoiser(): vnoiser's defaults")
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_FLASK}  Lab settings", ImVec2(half, 0)):
            self.settings.use_upstream()
        X.tooltip("Denoiser.upstream: the lab's original settings. Bands above 75 Hz stay complex, and the floor above "
                  "80 Hz is 0.01 instead of 0.1.")
        copied = time.monotonic() - self.time_copied < SECONDS_COPIED
        label = f"{fa.ICON_FA_CHECK}  Copied" if copied else f"{fa.ICON_FA_COPY}  Copy code"
        if imgui.button(label, ImVec2(width, 0)) and self.ready:
            imgui.set_clipboard_text(self.settings.python(self.run.signal.fs))
            self.time_copied = time.monotonic()
        X.tooltip("The Python that runs vnoiser with these settings")

    # ------------------------------------------------------------------
    # the page
    # ------------------------------------------------------------------

    def page(self) -> None:
        if M is None:
            imgui.text_colored(P.WARN, INSTALL)
            return
        if self.colormaps is None:
            self.colormaps = X.colormaps()
        if not self.ready:
            self.draw_waiting()
            return
        self.draw_views()
        super().page()

    def draw_waiting(self) -> None:
        """The stages, the done ones checked, until the first run is in."""
        em = D.em()
        imgui.dummy(ImVec2(0, 2 * em))
        run = self.run
        if run is not None and run.error:
            imgui.text_colored(P.ERROR, run.error)
            return
        current = M.STAGES.index(self.stage_running) if self.stage_running in M.STAGES else -1
        for i, stage in enumerate(M.STAGES):
            if i < current:
                color, icon = X.TRUTH, fa.ICON_FA_CIRCLE_CHECK
            elif i == current:
                color, icon = P.WARN, fa.ICON_FA_SPINNER
            else:
                color, icon = P.DIM, fa.ICON_FA_CIRCLE
            imgui.text_colored(color, f"{icon}  {STAGE_NAMES[stage]}")

    def draw_views(self) -> None:
        """The pipeline's state on the left; on the right, chips that set how much time the plots show."""
        color, state = self.state()
        imgui.text_colored(color, state)
        duration = self.run.signal.duration
        span_now = self.t1.value - self.t0.value
        texts = [text for text, _ in VIEWS]
        width = sum(imgui.calc_text_size(t).x + 2 * imgui.get_style().frame_padding.x for t in texts)
        width += imgui.get_style().item_spacing.x * (len(texts) - 1)
        imgui.same_line()
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, D.guide_width() - width - (imgui.get_cursor_pos_x() - imgui.get_cursor_start_pos().x)))
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 5.0)
        for i, (text, span) in enumerate(VIEWS):
            span = min(span or duration, duration)
            lit = abs(span_now - span) < 0.01 * span
            if i:
                imgui.same_line()
            imgui.push_style_color(imgui.Col_.button, LIT if lit else CARD)
            imgui.push_style_color(imgui.Col_.button_hovered, LIT if lit else CARD_HOVER)
            imgui.push_style_color(imgui.Col_.button_active, LIT)
            imgui.push_style_color(imgui.Col_.text, P.ACCENT if lit else P.DIM)
            if imgui.button(f"{text}##view"):
                center = 0.5 * (self.t0.value + self.t1.value) if span < duration else duration / 2
                lo = float(np.clip(center - span / 2, 0, duration - span))
                self.t0.value, self.t1.value = lo, lo + span
            imgui.pop_style_color(4)
        imgui.pop_style_var()

    def number(self, big: str, provenance: str) -> None:
        """One number per section, large in KEY under its plots, with where it came from beside it."""
        if not big:
            return
        em = D.em()
        imgui.dummy(ImVec2(0, 0.3 * em))
        push_size(2.0)
        imgui.text_colored(KEY, big)
        imgui.pop_font()
        if provenance:
            imgui.same_line(0, 1.2 * em)
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 0.85 * em)
            imgui.text_disabled(provenance)

    # ------------------------------------------------------------------
    # plot frames
    # ------------------------------------------------------------------

    def begin_time_plot(self, title: str, height: float, flags: int = 0, y_flags: Optional[int] = None,
                        legend: bool = True) -> bool:
        """A plot against time on the shared window, no axis decorations; y fits the data unless y_flags say
        otherwise. Close it with end_time_plot."""
        flags |= P.TRACE_FLAGS | (0 if legend else implot.Flags_.no_legend)
        if not P.begin_plot(title, ImVec2(D.guide_width(), height), flags):
            return False
        implot.setup_axes("", "", X.TIME_AXIS, X.VALUE_AXIS if y_flags is None else y_flags)
        implot.setup_axis_links(implot.ImAxis_.x1, self.t0, self.t1)
        return True

    def end_time_plot(self, time_bar: bool = False, spikes: Optional[np.ndarray] = None, value_bar: Optional[tuple] = None) -> None:
        """End the plot after its scale bars: a time bar on the last plot of a stack, spike ticks, a value bar as
        (height in plot units, label)."""
        dl = implot.get_plot_draw_list()
        implot.push_plot_clip_rect()
        if value_bar is not None:
            X.value_bar(dl, *value_bar)
        if spikes is not None:
            X.spike_ticks(dl, spikes)
        if time_bar:
            X.time_bar(dl)
        implot.pop_plot_clip_rect()
        implot.end_plot()

    def line(self, label: str, y: np.ndarray, color=None, weight: float = 1.0, flags: int = 0) -> None:
        X.line(label, y, self.run.signal.fs, self.t0.value, self.t1.value, color, weight, flags)

    def begin_spectrum(self, title: str, size: ImVec2) -> bool:
        if not P.begin_plot(title, size, P.TRACE_FLAGS):
            return False
        implot.setup_axes("frequency (Hz)", "power / Hz", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
        implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10)
        implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10)
        return True

    def plot_spectrum_lines(self, psd: dict) -> None:
        implot.plot_line("input", psd["f"], psd["raw"], P.spec(X.INPUT))
        if "truth" in psd:
            implot.plot_line("noiseless (simulated)", psd["f"], psd["truth"], P.spec(X.TRUTH, 1.5))
        if "denoised" in psd:
            implot.plot_line("denoised", psd["f"], psd["denoised"] + 1e-30, P.spec(X.DENOISED, 1.8))

    # ------------------------------------------------------------------
    # Pixels to z
    # ------------------------------------------------------------------

    def pixel_image(self) -> tuple:
        """(rows x columns, t_start, t_end, lo, hi): the pixels over the noiseless pixels, one blank row between,
        pooled to about IMAGE_COLUMNS columns of the view and IMAGE_ROWS rows per block."""
        run = self.run
        fs, px = run.signal.fs, run.signal.pixels
        i0, i1 = X.window_indices(run.signal.n, fs, self.t0.value, self.t1.value)
        key = (id(run), i0, i1)
        if self.pixel_cache[0] == key:
            return self.pixel_cache[1]
        noisy, whole = X.pool_columns(px.pixels[:, i0:i1].astype(np.float64), IMAGE_COLUMNS)
        noisy = X.pool_rows(noisy, IMAGE_ROWS)
        lo, hi = X.fit_scale(noisy, 1.0, 99.5)
        if px.clean_pixels is None:
            m = noisy
        else:
            clean, _ = X.pool_columns(px.clean_pixels[:, i0:i1].astype(np.float64), IMAGE_COLUMNS)
            gap = np.full((1, noisy.shape[1]), lo)
            m = np.vstack([noisy, gap, X.pool_rows(clean, IMAGE_ROWS)])  # heatmap row 0 is drawn at the top
        value = (np.ascontiguousarray(m), i0 / fs, (i0 + whole) / fs, lo, hi)
        self.pixel_cache = (key, value)
        return value

    def section_0(self) -> None:
        D.heading(*self.sections[0])
        sig = self.run.signal
        px = sig.pixels
        em = D.em()
        clean = px.clean_pixels is not None
        name = "noisy" if clean else "recorded"
        m, ta, tb, lo, hi = self.pixel_image()
        n_rows = m.shape[0]
        block = (n_rows - 1) // 2 if clean else n_rows
        implot.push_colormap(self.colormaps["scan"])
        if self.begin_time_plot("##pixels", 12 * em, implot.Flags_.no_legend, implot.AxisFlags_.lock | implot.AxisFlags_.no_grid_lines):
            implot.setup_axis_limits(implot.ImAxis_.y1, 0, n_rows, implot.Cond_.always)
            ticks = [n_rows - block / 2, block / 2] if clean else [block / 2]
            implot.setup_axis_ticks(implot.ImAxis_.y1, ticks, ["noisy", "noiseless"] if clean else [name], False)
            implot.plot_heatmap("##px", m, lo, hi, "", implot.Point(ta, 0), implot.Point(tb, n_rows))
            if len(px.edges) > 1:
                scale = block / px.pixels.shape[0]
                for k, (a, b) in enumerate(px.edges):
                    for top in (n_rows, block) if clean else (n_rows,):
                        y0, y1 = top - b * scale, top - a * scale
                        implot.plot_shaded(f"##line{k}{top}", np.array([ta, ta + 0.006 * (tb - ta)]), np.full(2, y0),
                                           np.full(2, y1), P.spec(X.band_color(k), fill_alpha=1.0))
            self.end_time_plot()
        implot.pop_colormap()
        unit = "photons" if clean else "counts"
        if self.begin_time_plot("##F", 7 * em):
            if len(px.edges) > 1:
                for k, mean in enumerate(px.means):
                    self.line(px.names[k], mean, X.band_color(k, 0.6))
            self.line(f"F, all pixels ({unit})", px.f, X.TEXT, 1.5)
            if clean:
                self.line("noiseless (simulated)", px.clean_f, X.TRUTH, 1.5)
            self.line(f"Gaussian baseline, sigma {px.sigma_s[0]:.1f} s", px.f_baseline, X.BASELINE, 2)
            span = float(np.ptp(px.f)) or 1.0
            height = X.nice(0.3 * span)
            self.end_time_plot(value_bar=(height, f"{height:g} {unit}"))
        if self.begin_time_plot("##dfof", 7 * em):
            self.line(f"dF/F, first {px.n_startup / sig.fs:.1f} s replaced", px.dfof, X.INPUT)
            if clean:
                self.line("noiseless (simulated)", px.clean_dfof, X.TRUTH, 1.5)
            self.line(f"slower baseline, sigma {px.sigma_s[1]:.1f} s", px.dfof_baseline, X.BASELINE, 2)
            height = X.nice(0.3 * (float(np.ptp(px.dfof)) or 1.0))
            self.end_time_plot(value_bar=(height, f"{100 * height:g}% dF/F"))
        if self.begin_time_plot("##z", 8 * em):
            self.line("z = -(dF/F - baseline) / SD", sig.x, X.INPUT)
            if sig.truth is not None:
                self.line("noiseless (simulated)", sig.truth, X.TRUTH, 1.5)
            spikes = sig.events_s.get("AP")
            if spikes is not None:
                implot.plot_dummy("action potentials (simulated)", P.spec(X.TRUTH, 2.0))
            self.end_time_plot(time_bar=True, spikes=spikes, value_bar=(2.0, "2 z"))
        imgui.text_disabled(sig.note)
        if sig.truth is not None:
            n0 = px.n_startup
            r = float(np.corrcoef(sig.x[n0:], sig.truth[n0:])[0, 1])
            self.number(f"r = {r:.2f}", "z against the noiseless z (simulated)")

    # ------------------------------------------------------------------
    # Scalogram
    # ------------------------------------------------------------------

    def heat_matrix(self) -> tuple:
        """(values rows x columns, t_start, t_end, lo, hi) for the current view and color mode, cached."""
        run = self.run
        fs = run.signal.fs
        i0, i1 = X.window_indices(run.signal.n, fs, self.t0.value, self.t1.value)
        key = (id(run), self.heat_mode, i0, i1)
        if self.heat_cache[0] == key:
            return self.heat_cache[1]
        if self.heat_mode == 3:
            m = run.coeff.real[:, i0:i1] / np.sqrt(run.scales)[:, None]
        elif self.heat_mode == 2:
            if not hasattr(run, "_row_mean"):
                run._row_mean = run.coeff.mean(axis=1, keepdims=True)
                run._row_sd = run.coeff.std(axis=1, keepdims=True)
            m = np.abs((run.coeff[:, i0:i1] - run._row_mean) / run._row_sd)
        else:
            m = run.amp[:, i0:i1].astype(np.float64)
            if self.heat_mode == 1:
                if not hasattr(run, "_row_median"):
                    run._row_median = np.median(run.amp, axis=1, keepdims=True) + 1e-12
                m = m / run._row_median
        if self.heat_mode == 3:
            width = max(1, int(math.ceil(m.shape[1] / IMAGE_COLUMNS)))
            whole = max(m.shape[1] // width * width, 1)
            if width > 1:
                blocks = m[:, :whole].reshape(m.shape[0], -1, width)
                pick = np.abs(blocks).argmax(axis=2)[..., None]
                m = np.take_along_axis(blocks, pick, axis=2)[..., 0]
            hi = float(np.percentile(np.abs(m), 99.5)) or 1e-9
            lo = -hi
        else:
            m, whole = X.pool_columns(m, IMAGE_COLUMNS, "max")
            m = np.log10(m + 1e-12)
            lo, hi = X.fit_scale(m, 2.0, 99.7)
        value = (np.ascontiguousarray(m, dtype=np.float64), i0 / fs, (i0 + whole) / fs, lo, hi)
        self.heat_cache = (key, value)
        return value

    def section_1(self) -> None:
        D.heading(*self.sections[1])
        if not self.has("cwt"):
            imgui.text_disabled("The transform has not run.")
            return
        run = self.run
        sig, freqs = run.signal, run.freqs
        m, ta, tb, lo, hi = self.heat_matrix()
        log_f = np.log10(freqs)
        step = (log_f[0] - log_f[-1]) / max(len(freqs) - 1, 1)
        top, bottom = log_f[0] + step / 2, log_f[-1] - step / 2
        em = D.em()
        h = 16 * em
        implot.push_colormap(self.colormaps["diverging" if self.heat_mode == 3 else "sequential"])
        if self.begin_time_plot("##scalogram", h, implot.Flags_.no_legend, implot.AxisFlags_.lock | implot.AxisFlags_.no_grid_lines):
            implot.setup_axis_limits(implot.ImAxis_.y1, bottom, top, implot.Cond_.always)
            X.hz_ticks(freqs.min(), freqs.max())
            implot.plot_heatmap("##W", m, lo, hi, "", implot.Point(ta, bottom), implot.Point(tb, top))
            if run.bands and run.bands[0].rows.size == len(freqs):
                for below, above in zip(run.bands, run.bands[1:]):
                    X.hline("##edge", math.log10(math.sqrt(below.hi_hz * above.lo_hz)), X.faded(X.TEXT, 0.4))
            if sig.fs / 2 < freqs.max():
                X.hline("##nyquist", math.log10(sig.fs / 2), P.WARN)
            changed, y, *_ = implot.drag_line_y(1, float(log_f[self.row]), X.TEXT, 1.5)
            if changed:
                self.row = int(np.argmin(np.abs(log_f - y)))
            if implot.is_plot_hovered():
                mouse = implot.get_plot_mouse_pos()
                k = int(np.argmin(np.abs(log_f - mouse.y)))
                imgui.set_tooltip(f"row {k}, {freqs[k]:.3g} Hz, at {mouse.x:.3f} s")
            self.end_time_plot()
        implot.pop_colormap()
        imgui.text_disabled(f"{HEAT_MODES[self.heat_mode]} of the complex Morlet transform ({M.morlet.name(run.settings.b, run.settings.c)}), "
                            f"{len(freqs)} rows in Hz; the lines are the band edges")
        k = self.row
        root = math.sqrt(run.scales[k])
        if self.begin_time_plot("##row", 8 * em):
            self.line("z", sig.x, X.INPUT)
            self.line(f"row {k}, {freqs[k]:.3g} Hz: z filtered there", run.coeff[k].real / root, P.ACCENT, 1.5)
            self.line("its envelope", run.amp[k], X.LEVEL_UP, 1.0)
            self.end_time_plot(time_bar=True)

    # ------------------------------------------------------------------
    # Bands
    # ------------------------------------------------------------------

    def section_2(self) -> None:
        D.heading(*self.sections[2])
        if not self.has("cluster"):
            imgui.text_disabled("The grouping has not run.")
            return
        run = self.run
        order = self.band_order()
        freqs = run.freqs
        em, w = D.em(), D.guide_width()
        ratio = math.sqrt(freqs[0] / freqs[1]) if len(freqs) > 1 else 1.0
        if P.begin_plot(f"{freqs.size} rows in {len(run.bands)} bands; dropped bands faded", ImVec2(w, 9 * em), P.TRACE_FLAGS | implot.Flags_.no_legend):
            implot.setup_axes("frequency (Hz)", "", implot.AxisFlags_.auto_fit, implot.AxisFlags_.no_tick_labels | implot.AxisFlags_.lock)
            implot.setup_axis_limits(implot.ImAxis_.y1, 0, 1.05, implot.Cond_.always)
            implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10)
            implot.setup_axis(implot.ImAxis_.y2, "gain of the kept bands summed", implot.AxisFlags_.auto_fit | implot.AxisFlags_.opposite)
            for band in run.bands:
                f = freqs[band.rows]
                xs = np.array([f.min() / ratio, f.max() * ratio])
                c = X.band_color(order[band.id])
                implot.plot_shaded(f"##{band.id}", xs, np.ones(2), 0.0, P.spec(c, fill_alpha=0.45 if band.kept else 0.12))
                implot.plot_scatter(f"##rows{band.id}", f, np.full(f.size, 0.5), X.marker(c, 3.5, X.faded(X.TEXT, 0.8)))
            X.vlines("##nyquist", [run.signal.fs / 2], P.WARN)
            if self.has("reduce"):
                fx = np.logspace(np.log10(freqs.min()), np.log10(min(freqs.max(), run.signal.fs / 2)), 500)
                implot.set_axes(implot.ImAxis_.x1, implot.ImAxis_.y2)
                implot.plot_line("##gain", fx, M.band_sum_gain(run, fx), P.spec(X.TEXT, 2))
            implot.end_plot()
        spectrum = run.psd if self.has("final") else None
        edge = None
        if spectrum and self.begin_spectrum("##spectrum", ImVec2(w, 12 * em)):
            f = spectrum["f"]
            top = float(max(spectrum["raw"].max(), spectrum.get("truth", spectrum["raw"]).max())) * 2
            low = float(min(spectrum["raw"].min(), spectrum["denoised"].min() + 1e-30))
            for band in run.bands:
                fb = freqs[band.rows]
                xs = np.array([fb.min() / ratio, min(fb.max() * ratio, f.max())])
                implot.plot_shaded(f"##bg{band.id}", xs, np.full(2, top), low,
                                   P.spec(X.band_color(order[band.id]), fill_alpha=0.18 if band.kept else 0.05))
            self.plot_spectrum_lines(spectrum)
            edge = M.noise_floor_hz(spectrum, run.signal.fs)
            if edge is not None:
                X.vlines(f"noise floor from {edge:.0f} Hz", [edge], P.DIM, 1.5)
            implot.end_plot()
        if self.has("reduce") and P.begin_plot("kept share of the recording per band; events on each bar", ImVec2(w, 8 * em), P.TRACE_FLAGS | implot.Flags_.no_legend):
            n = run.signal.n
            bands = run.bands
            implot.setup_axes("", "kept (%)", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            implot.setup_axis_ticks(implot.ImAxis_.x1, [float(i) for i in range(len(bands))], [b.span for b in bands], False)
            for i, band in enumerate(bands):
                if band.label is not None:
                    kept = float(np.mean(band.label >= 1.0)) * 100
                else:
                    kept = float(np.sum(band.windows[:, 1] - band.windows[:, 0])) / n * 100 if band.kept else 0.0
                implot.plot_bars(f"##k{band.id}", np.array([float(i)]), np.array([kept]), 0.7,
                                 P.spec(X.band_color(order[band.id]), fill_alpha=1.0 if band.kept else 0.3))
                implot.plot_text(f"{len(band.windows) if band.kept else 0}", float(i), kept, ImVec2(0, -0.7 * em))
            implot.end_plot()
        if run.bands:
            above = sum(band.lo_hz > edge for band in run.kept) if edge is not None else None
            self.number(f"{len(run.kept)} of {len(run.bands)} bands kept",
                        f"{above} above the {edge:.0f} Hz noise floor" if above is not None else "")

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------

    def section_3(self) -> None:
        D.heading(*self.sections[3])
        if not self.has("reduce"):
            imgui.text_disabled("The bands have not run.")
            return
        run = self.run
        band = run.band(self.band_id)
        if band is None:
            return
        fs = run.signal.fs
        color = X.band_color(self.band_order()[band.id])
        em = D.em()
        h = 6 * em
        if self.begin_time_plot("##rows", h):
            for k in np.flatnonzero(band.rows):
                self.line(f"{int(band.rows.sum())} rows", run.coeff[k].real, X.faded(X.TEXT, 0.35))
            self.line(f"{band.span}, their mean", band.average.real, color, 2)
            self.end_time_plot()
        if self.begin_time_plot("##levels", h):
            if band.kept and band.extra:
                wide = np.clip(band.windows + np.array([-band.extra, band.extra]), 0, run.signal.n)
                X.shade_intervals(wide, fs, self.t0.value, self.t1.value, X.faded(color, 0.10))
            X.shade_intervals(band.windows, fs, self.t0.value, self.t1.value, X.faded(color, 0.22))
            if band.smooth_window:
                self.line("mean", band.average.real, X.faded(X.TEXT, 0.35))
            self.line("smoothed" if band.smooth_window else "mean", band.smoothed.real, color, 1.5)
            X.hline(f"opens {band.up:.2f}", band.up, X.LEVEL_UP, 1.5)
            X.hline(f"closes {band.down:.2f}", band.down, X.LEVEL_DOWN, 1.0)
            self.end_time_plot(time_bar=not band.kept)
        if not band.kept:
            imgui.text_colored(P.WARN, f"{fa.ICON_FA_CIRCLE_INFO}  never reaches {band.up:.2f}: left out of the sum")
        elif band.mask is not None:
            if self.begin_time_plot("##mask", h, y_flags=implot.AxisFlags_.no_decorations | implot.AxisFlags_.lock):
                implot.setup_axis_limits(implot.ImAxis_.y1, -0.05, 1.08, implot.Cond_.always)
                self.line("before the blur", band.label, X.faded(X.TEXT, 0.5))
                self.line("mask", band.mask, color, 2)
                self.end_time_plot()
            if self.begin_time_plot("##masked", h):
                self.line("band", band.scaled.real, X.faded(X.TEXT, 0.35))
                self.line("band x mask", band.masked.real, color, 1.5)
                self.end_time_plot(time_bar=True)
        self.draw_mask_image()
        if band.kept:
            self.number(f"{len(band.windows)} events", f"{band.span}, the mean of {int(band.rows.sum())} rows")

    def draw_mask_image(self) -> None:
        run = self.run
        kept = [band for band in run.bands if band.kept and band.mask is not None][::-1]
        if not kept:
            return
        fs = run.signal.fs
        i0, i1 = X.window_indices(run.signal.n, fs, self.t0.value, self.t1.value)
        m, whole = X.pool_columns(np.stack([band.mask[i0:i1] for band in kept]), IMAGE_COLUMNS)
        nb = len(kept)
        em = D.em()
        imgui.text_disabled("every kept band's mask")
        implot.push_colormap(self.colormaps["sequential"])
        if self.begin_time_plot("##masks", (nb + 3) * em, implot.Flags_.no_legend, implot.AxisFlags_.lock | implot.AxisFlags_.no_grid_lines):
            implot.setup_axis_limits(implot.ImAxis_.y1, 0, nb, implot.Cond_.always)
            implot.setup_axis_ticks(implot.ImAxis_.y1, [nb - k - 0.5 for k in range(nb)], [b.span for b in kept], False)
            implot.plot_heatmap("##all_masks", np.ascontiguousarray(m, dtype=np.float64), 0.0, 1.0, "",
                                implot.Point(i0 / fs, 0), implot.Point((i0 + whole) / fs, nb))
            self.end_time_plot()
        implot.pop_colormap()

    # ------------------------------------------------------------------
    # Denoised
    # ------------------------------------------------------------------

    def section_4(self) -> None:
        D.heading(*self.sections[4])
        if not self.has("final"):
            imgui.text_disabled("The sum has not run.")
            return
        run = self.run
        sig = run.signal
        order = self.band_order()
        excluded = set(run.settings.excluded)
        kept = [band for band in run.bands if band.kept and band.masked is not None]
        em, w = D.em(), D.guide_width()
        spread = max((np.percentile(np.abs(b.masked.real), 99.5) for b in kept), default=1.0) * 2.2 or 1.0
        if kept and self.begin_time_plot("##stack", max(8.0, 1.6 * len(kept)) * em, implot.Flags_.no_legend,
                                         implot.AxisFlags_.no_decorations | implot.AxisFlags_.lock):
            implot.setup_axis_limits(implot.ImAxis_.y1, -0.7 * spread, (len(kept) - 0.2) * spread, implot.Cond_.always)
            for k, band in enumerate(kept):
                alpha = 0.25 if band.id in excluded else 1.0
                self.line(band.span, band.masked.real + k * spread, X.band_color(order[band.id], alpha), 1.2)
                implot.plot_text(band.span, self.t0.value, k * spread, ImVec2(3 * em, -0.6 * em))
            self.end_time_plot()
        imgui.text_disabled("each kept band times its mask, slowest at the bottom; bands left out of the sum faded")
        if self.begin_time_plot("##sum", 9 * em):
            self.line("z", sig.x, X.INPUT)
            if sig.truth is not None:
                self.line("noiseless (simulated)", sig.truth, X.TRUTH, 1.5)
            if run.settings.add_baseline:
                self.line("slow part", run.baseline, X.BASELINE, 1.0)
            self.line("denoised", run.denoised, X.DENOISED, 1.8)
            self.end_time_plot(value_bar=(2.0, "2 z"))
        if self.begin_time_plot("##removed", 6 * em):
            self.line("z - denoised", sig.x - run.denoised, X.REMOVED)
            self.end_time_plot(time_bar=True)
        if run.psd and self.begin_spectrum("##final_spectrum", ImVec2(w, 11 * em)):
            self.plot_spectrum_lines(run.psd)
            implot.end_plot()
        numbers = M.metrics(run)
        if "corr denoised vs ground truth" in numbers:
            self.number(f"r = {numbers['corr denoised vs ground truth']:.2f}",
                        f"denoised against the noiseless z (simulated); z itself: {numbers['corr raw vs ground truth']:.2f}")

    # ------------------------------------------------------------------
    # Peaks
    # ------------------------------------------------------------------

    def section_5(self) -> None:
        D.heading(*self.sections[5])
        if not self.has("peaks"):
            imgui.text_disabled("The peaks have not run.")
            return
        run = self.run
        sig = run.signal
        p = run.peaks
        fs = sig.fs
        t_peaks = p.peaks / fs
        em = D.em()
        spikes = sig.events_s.get("AP")
        if self.begin_time_plot("##lab_trace", 11 * em):
            if spikes is not None:
                implot.plot_dummy("action potentials (simulated)", P.spec(X.TRUTH, 2.0))
            if sig.truth is not None:
                self.line("noiseless (simulated)", sig.truth, X.TRUTH, 1.5)
            self.line("saved trace", p.trace, X.DENOISED, 1.8)
            X.hline(f"over {p.level_amp:.2f}", p.level_amp, X.LEVEL_UP, 1.5)
            X.hline(f"on/off {p.level_onoff:.2f}", p.level_onoff, X.LEVEL_DOWN, 1.0)
            if p.dropped.size:
                implot.plot_scatter("too short", p.dropped / fs, p.trace[p.dropped], X.marker(X.faded(X.TEXT, 0.6), 4.0))
            if p.peaks.size:
                implot.plot_scatter(f"{p.peaks.size} peaks", t_peaks, p.trace[p.peaks], X.marker(X.LEVEL_UP, 5.0, X.TEXT))
            self.end_time_plot(spikes=spikes, value_bar=(2.0, "2 z"))
        if self.begin_time_plot("##bandpassed", 7 * em):
            lo, hi = run.settings.spike_at(fs).bp
            self.line(f"band-passed {lo:g} to {hi:.3g} Hz", p.bandpassed, X.REMOVED, 1.2)
            X.hline(f"over {p.level_bp:.2f}", p.level_bp, X.LEVEL_UP, 1.5)
            if p.peaks.size:
                implot.plot_scatter("peaks", t_peaks, p.bandpassed[p.peaks], X.marker(X.LEVEL_UP, 4.0, X.TEXT))
            self.end_time_plot(time_bar=True)
        imgui.text_disabled("the saved trace is bands x masks + baseline[0] (vnoiser.pf.final_trace); detect_peaks runs on it at floor(fs)")
        hits = M.peak_hits(run)
        if hits is None:
            self.number(f"{p.peaks.size} peaks", f"detect_peaks at {p.fs:g} Hz")
        else:
            hit, n_ap, stray = hits
            self.number(f"{p.peaks.size} peaks", f"{hit} of {n_ap} simulated action potentials within 10 ms; {stray} with none")


def scan_label(path: str) -> str:
    """<recording folder> / <unit> for the combo."""
    f = Path(path)
    folder = f.parent.parent if f.parent.name == "linescans" else f.parent
    return f"{folder.name} / {f.stem}"
