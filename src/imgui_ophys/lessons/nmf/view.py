"""
The NMF lesson: watch NMF unmix a small synthetic imaging movie, against PCA and the ground truth used to build it.
The Demixing (NMF) stage. The first subsections (guide.py) walk a 6 x 6 movie to footprints and traces, every value
drawn; the last, NMF vs PCA, fits a 24 x 24 movie live: the frame written as a sum of its parts, a pickable pixel, every
footprint x trace.
"""

from typing import Optional

import numpy as np
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui

from imgui_ophys import diagram as D
from imgui_ophys import plotting as P
from imgui_ophys.lesson import Lesson
from imgui_ophys.lessons.nmf.model import N_TRUE, NMF, MovieParams, make_movie, match_to_truth, pca, rel_error
from imgui_ophys.lessons.nmf.guide import NmfGuide
from imgui_ophys.widgets import KEY, help_mark, operator, section

METHODS = (("nmf", "NMF"), ("pca", "PCA"), ("truth", "truth"))
VIEWS = ("Recorded", "Average over time", "Sum of parts", "Residual")
MAX_K = len(P.COMPONENT_COLORS)
SHOW_WEIGHT = 0.03  # parts weighing less than this at the selected pixel stay out of its plot


class NmfLesson(Lesson):
    key = "nmf"
    title = "Demixing (NMF)"
    summary = "Y = A C: spatial footprints times temporal traces, all non-negative"
    sections = (
        (fa.ICON_FA_TABLE_CELLS, "The movie as a matrix"),
        (fa.ICON_FA_LAYER_GROUP, "Footprints and traces"),
        (fa.ICON_FA_ROTATE, "HALS updates"),
        (fa.ICON_FA_FLASK, "NMF vs PCA"),
    )
    references = (
        ("Lee DD, Seung HS (1999). Learning the parts of objects by non-negative matrix factorization. Nature 401:788-791.",
         "https://doi.org/10.1038/44565"),
        ("Cichocki A, Phan A-H (2009). Fast local algorithms for large scale nonnegative matrix and tensor factorizations. IEICE Trans Fundamentals E92-A:708-721.",
         "https://doi.org/10.1587/transfun.E92.A.708"),
        ("Pnevmatikakis EA, Soudry D, Gao Y, et al. (2016). Simultaneous denoising, deconvolution, and demixing of calcium imaging data. Neuron 89:285-299.",
         "https://doi.org/10.1016/j.neuron.2015.11.037"),
    )

    def __init__(self, params: Optional[MovieParams] = None):
        self.params = params or MovieParams()
        self.k = N_TRUE
        self.max_iters = 150
        self.iters_per_frame = 3
        self.nmf_seed = 0
        self.mode = "none"
        self.nmf: Optional[NMF] = None
        self.A: Optional[np.ndarray] = None
        self.C: Optional[np.ndarray] = None
        self.offset: np.ndarray | float = 0.0  # the mean frame for PCA, else 0
        self.frame = 0
        self.playing = True
        self.fps = 15.0
        self.play_accum = 0.0
        self.sel = (10, 10)  # (x, y) of the selected pixel
        self.view = 0
        self.cmaps: Optional[P.Colormaps] = None
        self.guide = NmfGuide()
        self.make_movie()

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def make_movie(self) -> None:
        self.movie = make_movie(self.params)
        self.vmax = float(np.percentile(self.movie.Y, 99.5))
        self.mean_img = self.movie.image(self.movie.Y.mean(axis=1))
        self.t = np.arange(self.movie.frames, dtype=np.float64)
        self.frame = min(self.frame, self.movie.frames - 1)
        self.run(self.mode)

    def run(self, mode: str) -> None:
        self.mode, self.nmf, self.offset = mode, None, 0.0
        Y = self.movie.Y
        if mode == "nmf":
            self.nmf = NMF(Y, self.k, seed=self.nmf_seed)
            self.A, self.C = self.nmf.A, self.nmf.C
        elif mode == "pca":
            self.A, self.C, self.offset = pca(Y, self.k)
        elif mode == "truth":
            self.A, self.C = self.movie.A, self.movie.C
        else:
            self.A = self.C = None

    @property
    def fitting(self) -> bool:
        return self.nmf is not None and self.nmf.iter < self.max_iters

    @property
    def sel_index(self) -> int:
        x, y = self.sel
        return y * self.movie.side + x

    @property
    def mean_frame(self) -> np.ndarray:
        """The (pixels,) offset added to the parts: PCA's mean frame, else zeros."""
        if isinstance(self.offset, np.ndarray):
            return self.offset[:, 0]
        return np.zeros(self.movie.Y.shape[0])

    def update(self, dt: float) -> None:
        self.guide.update(dt)
        if self.fitting:
            for _ in range(min(self.iters_per_frame, self.max_iters - self.nmf.iter)):
                self.nmf.step()
            if not self.fitting:
                self.nmf.A, self.nmf.C = match_to_truth(self.nmf.A, self.nmf.C, self.movie.A)
            self.A, self.C = self.nmf.A, self.nmf.C
        if self.playing:
            self.play_accum += dt
            step = 1.0 / self.fps
            while self.play_accum >= step:
                self.play_accum -= step
                self.frame = (self.frame + 1) % self.movie.frames

    def animating(self) -> bool:
        return self.guide.animating() or self.playing or self.fitting

    def error(self) -> Optional[float]:
        return None if self.A is None else rel_error(self.movie.Y, self.A, self.C, self.offset)

    # ------------------------------------------------------------------
    # rail
    # ------------------------------------------------------------------

    def controls(self) -> None:
        section(f"{fa.ICON_FA_WAND_MAGIC_SPARKLES}  NMF vs PCA")
        for i, (mode, label) in enumerate(METHODS):
            if i:
                imgui.same_line()
            if imgui.radio_button(label, self.mode == mode):
                self.run(mode)
        changed, k = imgui.slider_int("parts (K)", self.k, 1, MAX_K)
        if changed:
            self.k = k
            if self.mode in ("nmf", "pca"):
                self.run(self.mode)
        help_mark("The movie has 4: three cells and the background.")
        section(f"{fa.ICON_FA_FILM}  Movie")
        if imgui.button(fa.ICON_FA_PAUSE if self.playing else fa.ICON_FA_PLAY):
            self.playing = not self.playing
        imgui.same_line()
        _, self.view = imgui.combo("##movie", self.view, VIEWS)

    # ------------------------------------------------------------------
    # page
    # ------------------------------------------------------------------

    def section_0(self) -> None:
        self.guide.frames_to_matrix()

    def section_1(self) -> None:
        self.guide.why_it_factors()

    def section_2(self) -> None:
        self.guide.hals()

    def section_3(self) -> None:
        D.heading(fa.ICON_FA_FLASK, "NMF vs PCA", "A 24 x 24 movie of three cells on a bleaching background. Pick a method in the left panel.")
        self.lab()

    def lab(self) -> None:
        if self.cmaps is None:
            self.cmaps = P.Colormaps()
        em = imgui.get_font_size()
        self.draw_status()
        imgui.dummy(imgui.ImVec2(0, 0.6 * em))
        self.draw_equation()
        imgui.dummy(imgui.ImVec2(0, 0.6 * em))
        imgui.separator()

        side = min(16 * em, 0.34 * imgui.get_content_region_avail().x)
        top = side + 1.5 * imgui.get_frame_height_with_spacing()
        if imgui.begin_child("movie", imgui.ImVec2(side + em, top)):
            self.draw_movie(side)
        imgui.end_child()
        imgui.same_line()
        if imgui.begin_child("pixel", imgui.ImVec2(0, top)):
            self.draw_pixel(top - imgui.get_frame_height_with_spacing())
        imgui.end_child()

        imgui.separator()
        self.draw_parts(5.5 * em)

    def draw_status(self) -> None:
        """The fit's progress and how much it leaves unexplained."""
        em = imgui.get_font_size()
        imgui.begin_group()
        width = min(16 * em, imgui.get_content_region_avail().x)
        if self.nmf is not None:
            imgui.progress_bar(self.nmf.iter / self.max_iters, imgui.ImVec2(width, 0), f"sweep {self.nmf.iter}")
        else:
            imgui.dummy(imgui.ImVec2(width, imgui.get_frame_height()))
        err = self.error()
        if err is not None:
            imgui.progress_bar(err, imgui.ImVec2(width, 0), f"{100 * err:.0f}% unexplained")
            help_mark("||Y - A C|| / ||Y||. The ground truth leaves only the noise.")
        else:
            imgui.text_disabled(f"{fa.ICON_FA_ARROW_LEFT}  pick a method")
        imgui.end_group()

    def draw_equation(self) -> None:
        """This frame as a sum: recorded = (mean +) part 1 + ... + part K + residual, each a picture."""
        em = imgui.get_font_size()
        Y, f = self.movie.Y, self.frame
        terms = [("frame", self.movie.image(Y[:, f]), 0.0, self.vmax, self.cmaps.gray, P.RAW)]
        if self.A is not None:
            if self.mode == "pca":
                terms.append(("mean", self.movie.image(self.mean_frame), 0.0, self.vmax, self.cmaps.gray, P.RAW))
            for k in range(self.A.shape[1]):
                img = self.movie.image(self.A[:, k] * self.C[k, f])
                # each part against its own peak, so it lights up when its cell fires
                peak = float(np.abs(self.A[:, k]).max() * np.abs(self.C[k]).max()) or 1.0
                i = k % len(P.COMPONENT_COLORS)
                if self.mode == "pca":
                    terms.append((f"{k + 1}", img, -peak, peak, self.cmaps.diverging[i], P.component_color(k)))
                else:
                    terms.append((f"{k + 1}", img, 0.0, peak, self.cmaps.sequential[i], P.component_color(k)))
            fit = self.A @ self.C[:, f] + self.mean_frame
            r = max(3 * self.params.noise, 0.1)
            terms.append(("residual", self.movie.image(Y[:, f] - fit), -r, r, self.cmaps.residual, P.DIM))
        ops = len(terms) - 1
        avail = imgui.get_content_region_avail().x - ops * 2.4 * em
        size = min(6 * em, avail / max(len(terms), 1))
        for i, (label, img, lo, hi, cmap, color) in enumerate(terms):
            if i:
                operator("=" if i == 1 else "+", size)
            imgui.begin_group()
            P.image_plot(f"##eq{i}", img, size, lo, hi, cmap)
            w = imgui.calc_text_size(label).x
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (size - w) / 2)
            imgui.text_colored(color, label)
            imgui.end_group()
        if self.A is None:
            operator("=", size)
            imgui.text_colored(KEY, "?")

    def movie_image(self) -> tuple[np.ndarray, float, float, int]:
        """The movie panel for the current view: (image, vmin, vmax, colormap)."""
        Y, f = self.movie.Y, self.frame
        view = self.view if self.A is not None else min(self.view, 1)
        if view == 1:
            return self.mean_img, 0.0, float(self.mean_img.max()), self.cmaps.gray
        if view == 0:
            return self.movie.image(Y[:, f]), 0.0, self.vmax, self.cmaps.gray
        fit = self.A @ self.C[:, f] + self.mean_frame
        if view == 2:
            return self.movie.image(fit), 0.0, self.vmax, self.cmaps.gray
        r = max(3 * self.params.noise, 0.1)
        return self.movie.image(Y[:, f] - fit), -r, r, self.cmaps.residual

    def draw_movie(self, side: float) -> None:
        img, vmin, vmax, cmap = self.movie_image()
        P.image_plot("##movie", img, side, vmin, vmax, cmap, on_click=self.select, mark=self.sel)
        if imgui.is_item_hovered():
            imgui.set_tooltip("click or drag: pick a pixel")
        imgui.set_next_item_width(side)
        changed, f = imgui.slider_int("##frame", self.frame, 0, self.movie.frames - 1, f"{fa.ICON_FA_CLOCK} %d")
        if changed:
            self.frame, self.playing = f, False

    def select(self, x: int, y: int) -> None:
        self.sel = (x, y)

    def draw_pixel(self, height: float) -> None:
        p = self.sel_index
        imgui.text_colored(P.HIGHLIGHT, f"{fa.ICON_FA_CROSSHAIRS}  pixel {self.sel[0]}, {self.sel[1]}")
        series = [("recorded", self.movie.Y[p], P.RAW, 1.5)]
        if self.A is not None:
            base = self.mean_frame[p]
            for k in range(self.A.shape[1]):
                w = self.A[p, k]
                if abs(w) > SHOW_WEIGHT:
                    series.append((f"{k + 1}", base + w * self.C[k], P.component_color(k), 2.5))
            series.append(("sum", base + self.A[p] @ self.C, P.WHITE, 1.0))
        P.trace_plot("##pixel", self.t, series, height, cursor=self.frame)

    def draw_parts(self, size: float) -> None:
        """One row per part: its footprint (where), then its trace (when)."""
        if self.A is None:
            return
        for k in range(self.A.shape[1]):
            img = self.movie.image(self.A[:, k])
            m = float(np.abs(img).max()) or 1.0
            i = k % len(P.COMPONENT_COLORS)
            if self.mode == "pca":
                vmin, cmap = -m, self.cmaps.diverging[i]
            else:
                vmin, cmap = 0.0, self.cmaps.sequential[i]
            P.image_plot(f"##fp{k}", img, size, vmin, m, cmap)
            operator("x", size)
            P.trace_plot(f"##tr{k}", self.t, [(f"{k + 1}", self.C[k], P.component_color(k), 2.0)], size,
                         cursor=self.frame, legend=False)
