"""
The NMF guide: a 6 x 6, 24 frame movie followed from its frames to footprints and traces, every value drawn: one
frame becoming one column of Y, why Y can be factored, and HALS one update at a time.

Hovering a pixel or a cell of Y picks that pixel (and frame) in every section.
"""

from typing import Optional

import numpy as np
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui

from imgui_ophys import diagram as D
from imgui_ophys.lessons.nmf.model import TINY_NAMES, HalsWalk, make_tiny
from imgui_ophys.plotting import ACCENT, DIM, HIGHLIGHT, RAW
from imgui_ophys.widgets import KEY

ImVec2 = imgui.ImVec2
K = 3
GUIDE_FPS = 3.0
HALS_STEPS_PER_S = 4.0


def part_color(k: int):
    return D.part_color(k)


class NmfGuide:
    def __init__(self):
        self.movie = make_tiny()
        self.Y = self.movie.Y.astype(float)
        self.side = self.movie.side
        self.T = self.movie.frames
        self.hi = float(self.Y.max())
        self.frame = 3
        self.playing = True
        self.accum = 0.0
        self.sel = (2, 2)  # (row, col) of the picked pixel
        self.hover_frame: Optional[int] = None
        self.walk_seed = 0
        self.walk = HalsWalk(self.Y, K, seed=self.walk_seed)
        self.walk_playing = False
        self.walk_accum = 0.0
        self.truth_error = float(np.linalg.norm(self.Y - self.movie.A @ self.movie.C) / np.linalg.norm(self.Y))
        self.examples = self.pick_examples()

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def pick_examples(self) -> tuple:
        """Four telling pixels: in cell 1 only, in cell 2 only, where they overlap, and background only."""
        A = self.movie.A
        picks = (
            int(np.argmax(A[:, 0] - A[:, 1])),
            int(np.argmax(A[:, 1] - A[:, 0])),
            int(np.argmax(np.minimum(A[:, 0], A[:, 1]))),
            int(np.argmin(A[:, 0] + A[:, 1])),
        )
        return tuple(zip(("cell 1", "cell 2", "overlap", "background"), picks))

    @property
    def t(self) -> int:
        return self.hover_frame if self.hover_frame is not None else self.frame

    @property
    def p(self) -> int:
        return self.sel[0] * self.side + self.sel[1]

    def update(self, dt: float) -> None:
        if self.playing and self.hover_frame is None:
            self.accum += dt
            while self.accum >= 1 / GUIDE_FPS:
                self.accum -= 1 / GUIDE_FPS
                self.frame = (self.frame + 1) % self.T
        if self.walk_playing:
            self.walk_accum += dt
            while self.walk_accum >= 1 / HALS_STEPS_PER_S:
                self.walk_accum -= 1 / HALS_STEPS_PER_S
                self.walk.advance()
            if self.walk.sweep >= 60:
                self.walk_playing = False

    def animating(self) -> bool:
        return self.playing or self.walk_playing

    def frame_image(self, t: int) -> np.ndarray:
        return self.Y[:, t].reshape(self.side, self.side)

    # ------------------------------------------------------------------
    # page
    # ------------------------------------------------------------------

    def frame_controls(self) -> None:
        if imgui.small_button(fa.ICON_FA_PAUSE if self.playing else fa.ICON_FA_PLAY):
            self.playing = not self.playing
        imgui.same_line()
        imgui.set_next_item_width(D.em(12))
        changed, f = imgui.slider_int("##guide_frame", self.frame, 0, self.T - 1, "frame %d")
        if changed:
            self.frame, self.playing = f, False

    def frames_to_matrix(self) -> None:
        self.hover_frame = None
        D.heading(fa.ICON_FA_TABLE_CELLS, "The movie as a matrix", "Unroll each frame row by row into a column. Side by side, the columns make Y: one row per pixel, one column per frame.")
        self.frame_controls()
        em, dl = D.em(), imgui.get_window_draw_list()
        s, T = self.side, self.T
        n = s * s
        cell = 1.3 * em
        colw, rowh = 1.6 * em, 0.45 * em
        yw = 0.62 * em
        tall = n * rowh
        w = D.guide_width()
        x, y = D.canvas(w, tall + 3.0 * em)
        top = y + 1.3 * em
        t, p = self.t, self.p

        hover = D.matrix(dl, x, top, self.frame_image(t), cell, color=D.gray, lo=0, hi=self.hi, cells=[self.sel])
        if hover is not None:
            self.sel = hover
        D.text(dl, x, y, f"frame {t}", HIGHLIGHT)
        cx = x + s * cell + 3.2 * em
        D.arrow(dl, x + s * cell + 0.4 * em, top + s * cell / 2, cx - 0.4 * em, top + s * cell / 2)
        D.text(dl, x + s * cell + 0.3 * em, top + s * cell / 2 + 0.3 * em, "row by row", DIM, 0.7)
        D.matrix(dl, cx, top, self.Y[:, t : t + 1], colw, rowh, color=D.gray, lo=0, hi=self.hi, rows=[p])
        D.text(dl, cx, y, "column", DIM, 0.8)
        # where the picked pixel goes
        r, c = self.sel
        D.arrow(dl, x + (c + 0.5) * cell, top + (r + 0.5) * cell, cx - 0.1 * em, top + (p + 0.5) * rowh, HIGHLIGHT, 1.0)

        yx = cx + colw + 3.2 * em
        D.arrow(dl, cx + colw + 0.4 * em, top + tall / 2, yx - 0.4 * em, top + tall / 2)
        hover = D.matrix(dl, yx, top, self.Y, yw, rowh, color=D.gray, lo=0, hi=self.hi, rows=[p], cols=[t])
        if hover is not None:
            pr, self.hover_frame = hover
            self.sel = divmod(pr, s)
        D.segments(dl, yx, y, [("Y", KEY), (f"  {n} pixels x {T} frames", DIM)], 0.85)
        sx = yx + T * yw + 1.5 * em
        sw = x + w - sx
        if sw > 8 * em:
            D.text(dl, sx, top, f"row {p} = pixel ({r}, {c})", HIGHLIGHT, 0.8)
            D.sparkline(dl, sx, top + 1.2 * em, sw, 4 * em, self.Y[p], RAW, 0, self.hi, cursor=t)
            D.text(dl, sx, top + 5.5 * em, "its trace, frame by frame", DIM, 0.75)
        D.text(dl, yx, top + tall + 0.4 * em, f"Y[{p}, {t}] = {self.Y[p, t]:.0f}", TEXT_, 0.85)

    def why_it_factors(self) -> None:
        D.heading(fa.ICON_FA_LAYER_GROUP, "Footprints and traces", "Each source is a spatial footprint (where) times a temporal trace (when). Every pixel's trace is a weighted sum of the same few traces, so Y = A C.")
        em, dl = D.em(), imgui.get_window_draw_list()
        A, C, s = self.movie.A, self.movie.C, self.side
        w = D.guide_width()
        across = 4 if w > 52 * em else 2
        gap = 1.0 * em
        cw = (w - (across - 1) * gap) / across
        ch = 8.2 * em
        rows = -(-len(self.examples) // across)
        x, y = D.canvas(w, rows * (ch + gap))
        for i, (name, p) in enumerate(self.examples):
            r, c = divmod(i, across)
            bx, by = x + c * (cw + gap), y + r * (ch + gap)
            D.box(dl, bx, by, cw, ch, "", DIM)
            mini = 0.55 * em
            D.matrix(dl, bx + 0.5 * em, by + 0.5 * em, self.frame_image(self.t), mini, color=D.gray, lo=0, hi=self.hi, cells=[divmod(p, s)])
            sx = bx + 0.5 * em + s * mini + 0.6 * em
            D.text(dl, sx, by + 0.4 * em, name, KEY, 0.9)
            # what the pixel recorded, and the sum of its weighted parts on top
            D.sparkline(dl, sx, by + 1.5 * em, bx + cw - 0.5 * em - sx, 2.3 * em, self.Y[p], RAW, 0, self.hi, cursor=self.t)
            D.sparkline(dl, sx, by + 1.5 * em, bx + cw - 0.5 * em - sx, 2.3 * em, A[p] @ C, D.TEXT, 0, self.hi, frame=False, thickness=1.0)
            # its weights: how much of each shared trace this pixel holds
            label_w = 6.2 * em
            bar_w = cw - label_w - 3.6 * em
            for k in range(K):
                ry = by + 4.4 * em + k * 1.2 * em
                D.text(dl, bx + 0.5 * em, ry, TINY_NAMES[k], part_color(k), 0.8)
                x0 = bx + label_w
                dl.add_rect_filled(ImVec2(x0, ry + 0.15 * em), ImVec2(x0 + bar_w, ry + 0.85 * em), D.u32(D.PLOT), 2.0)
                if A[p, k] > 0:
                    dl.add_rect_filled(ImVec2(x0, ry + 0.15 * em), ImVec2(x0 + A[p, k] * bar_w, ry + 0.85 * em), D.u32(part_color(k)), 2.0)
                D.text(dl, x0 + bar_w + 0.4 * em, ry, f"{A[p, k]:.2f}", part_color(k), 0.8)
        D.segments(dl, x, y + rows * (ch + gap) - 0.6 * em, [("gray ", RAW), ("recorded   ", DIM), ("white ", D.TEXT), ("weights x shared traces   ", DIM), ("bars ", KEY), ("the weights (footprint values at that pixel)", DIM)], 0.75)
        imgui.dummy(ImVec2(0, 0.8 * em))
        self.draw_shapes()

    def draw_shapes(self) -> None:
        """Y = A x C drawn to shape, with how many numbers each holds, and one entry worked out."""
        em, dl = D.em(), imgui.get_window_draw_list()
        A, C, s, T = self.movie.A, self.movie.C, self.side, self.T
        n = s * s
        rh, yw, aw, chh = 0.32 * em, 0.42 * em, 0.9 * em, 1.0 * em
        tall = n * rh
        w = D.guide_width()
        x, y = D.canvas(w, max(tall, 3 * 1.3 * em + 9.5 * em) + 4.6 * em)
        top = y + 1.3 * em
        p, t = self.p, self.t
        hover = D.matrix(dl, x, top, self.Y, yw, rh, color=D.gray, lo=0, hi=self.hi, cells=[(p, t)])
        if hover is not None:
            pr, self.hover_frame = hover
            self.sel = divmod(pr, s)
        D.segments(dl, x, y, [("Y", KEY), (f"  {n} x {T}", DIM)], 0.85)
        D.text(dl, x, top + tall + 0.3 * em, f"{n * T} numbers", DIM, 0.8)
        ex = x + T * yw + 0.8 * em
        D.text(dl, ex, top + tall / 2 - 0.7 * em, "~", DIM, 1.4)
        ax = ex + 2.0 * em
        for k in range(K):
            D.matrix(dl, ax + k * aw, top, A[:, k : k + 1], aw, rh, color=D.tinted(part_color(k)), lo=0, hi=1, rows=[p] if k == 0 else [])
        D.segments(dl, ax, y, [("A", KEY), (f"  {n} x {K}", DIM)], 0.85)
        D.text(dl, ax, top + tall + 0.3 * em, "where", DIM, 0.8)
        mx = ax + K * aw + 0.8 * em
        D.text(dl, mx, top + tall / 2 - 0.7 * em, "x", DIM, 1.4)
        cx = mx + 2.0 * em
        cw = min(T * yw * 1.4, x + w - cx)
        D.segments(dl, cx, y, [("C", KEY), (f"  {K} x {T}", DIM)], 0.85)
        for k in range(K):
            D.sparkline(dl, cx, top + k * (chh + 0.3 * em), cw, chh, C[k], part_color(k), 0, float(C.max()), cursor=t)
        D.text(dl, cx, top + K * (chh + 0.3 * em), "when", DIM, 0.8)
        D.text(dl, cx, top + K * (chh + 0.3 * em) + 1.4 * em, f"{n * K} + {K * T} = {n * K + K * T} numbers", KEY, 0.85)
        # footprints as images under the traces
        fy = top + K * (chh + 0.3 * em) + 3.0 * em
        for k in range(K):
            D.matrix(dl, cx + k * (s * 0.6 * em + 0.8 * em), fy, A[:, k].reshape(s, s), 0.6 * em, color=D.tinted(part_color(k)), lo=0, hi=1)
        # one entry, worked out
        ly = top + tall + 1.6 * em
        terms = A[p] * C[:, t]
        parts = [(f"Y[{p}, {t}] = {self.Y[p, t]:.0f}   ~   ", TEXT_)]
        for k in range(K):
            parts += [(f"{A[p, k]:.2f} x {C[k, t]:.0f}", part_color(k)), ("  +  " if k < K - 1 else "  =  ", DIM)]
        parts.append((f"{terms.sum():.0f}", TEXT_))
        D.segments(dl, x, ly, parts, 0.85)
        D.text(dl, x, ly + 1.3 * em, "hover Y to pick an entry", DIM, 0.75)

    def hals(self) -> None:
        D.heading(fa.ICON_FA_ROTATE, "HALS updates", "Hierarchical alternating least squares: fix everything but one trace (or one footprint), solve for it by least squares, set negatives to 0. Repeat for each in turn.")
        em, dl = D.em(), imgui.get_window_draw_list()
        walk = self.walk
        if imgui.button(f"{fa.ICON_FA_FORWARD_STEP}  step"):
            walk.advance()
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_FORWARD_FAST}  sweep"):
            walk.advance_sweep()
        imgui.same_line()
        if imgui.button(fa.ICON_FA_PAUSE if self.walk_playing else f"{fa.ICON_FA_PLAY}  play"):
            self.walk_playing = not self.walk_playing
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_DICE}  new start"):
            self.walk_seed += 1
            self.walk = HalsWalk(self.Y, K, seed=self.walk_seed)
            walk = self.walk
        imgui.same_line()
        imgui.text_disabled(f"sweep {walk.sweep}   error {100 * walk.errors[-1]:.1f}%")

        w = D.guide_width()
        labels = [f"trace {k + 1}" for k in range(K)] + [f"footprint {k + 1}" for k in range(K)] + ["rescale"]
        x, y = D.canvas(w, 2.2 * em)
        nsteps = len(labels)
        last = (walk.pos - 1) % nsteps if walk.pos else -1
        D.chip_row(dl, x, y, w, labels, walk.pos % nsteps, last if walk.pos % nsteps else -1)

        kind, j = walk.last() or ("", -1)
        s, n, T = self.side, self.side * self.side, self.T
        rh, rw = 0.3 * em, 0.4 * em
        rx_w = T * rw + 2 * em
        col_w = (w - rx_w - 1.5 * em) / K
        h = n * rh + 6.0 * em
        x, y = D.canvas(w, h)
        top = y + 1.4 * em
        # what the last update had to explain
        if kind in ("trace", "footprint"):
            R = walk.residual
            m = float(np.abs(R).max()) or 1.0
            D.matrix(dl, x, top, R, rw, rh, color=D.diverging(part_color(j)), lo=-m, hi=m)
            D.segments(dl, x, y, [(f"R{j + 1}", KEY), ("  = Y - other parts", DIM)], 0.8)
            D.arrow(dl, x + T * rw + 0.3 * em, top + n * rh / 2, x + rx_w - 0.2 * em, top + n * rh / 2, part_color(j))
        else:
            D.text(dl, x, y, "rescale: footprints to max 1" if kind else "random start", DIM, 0.8)
        # each part: its footprint and its trace, lit when it was just updated
        fcell = min(1.2 * em, (col_w - em) / s)
        cmax = max(float(walk.C.max()), 1.0)
        for k in range(K):
            px = x + rx_w + 1.5 * em + k * col_w
            lit = j == k
            D.box(dl, px, y, col_w - 0.6 * em, h, "", part_color(k), 0.06 if lit else 0.0, lit)
            D.text(dl, px + 0.5 * em, y + 0.3 * em, f"part {k + 1}", part_color(k))
            a = walk.A[:, k].reshape(s, s)
            D.matrix(dl, px + 0.5 * em, top + 0.3 * em, a, fcell, color=D.tinted(part_color(k)), lo=0, hi=max(float(a.max()), 1e-9))
            if lit and kind == "footprint" and walk.previous is not None:
                prev = walk.previous.reshape(s, s)
                gx = px + 0.5 * em + s * fcell + 0.6 * em
                small = min(0.5 * em, (col_w - s * fcell - 2.5 * em) / s)
                if small > 0.15 * em:
                    D.matrix(dl, gx, top + 0.3 * em, prev, small, color=D.tinted(DIM), lo=0, hi=max(float(prev.max()), 1e-9))
                    D.text(dl, gx, top + 0.3 * em + s * small + 0.2 * em, "before", DIM, 0.7)
            ty = top + 0.3 * em + s * fcell + 0.6 * em
            th = y + h - ty - 0.5 * em
            if lit and kind == "trace" and walk.previous is not None:
                D.sparkline(dl, px + 0.5 * em, ty, col_w - 1.6 * em, th, walk.previous, DIM, 0, cmax)
                D.sparkline(dl, px + 0.5 * em, ty, col_w - 1.6 * em, th, walk.C[k], part_color(k), 0, cmax, frame=False, thickness=2.0)
            else:
                D.sparkline(dl, px + 0.5 * em, ty, col_w - 1.6 * em, th, walk.C[k], part_color(k), 0, cmax)

        # the update rule just used, and the error so far
        x, y = D.canvas(w, 5.2 * em)
        k = j + 1
        if kind == "trace":
            rule = [(f"c{k}", part_color(j)), (" = max(0, ", DIM), (f"a{k}'R{k}", TEXT_), (" / ", DIM), (f"a{k}'a{k}", TEXT_), (")", DIM),
                    ("     the trace that best explains R with this footprint", DIM)]
        elif kind == "footprint":
            rule = [(f"a{k}", part_color(j)), (" = max(0, ", DIM), (f"R{k}c{k}'", TEXT_), (" / ", DIM), (f"c{k}c{k}'", TEXT_), (")", DIM),
                    ("     the footprint that best explains R with this trace", DIM)]
        elif kind == "scale":
            rule = [("a / max(a),  c x max(a)", TEXT_), ("     same product, footprints peak at 1", DIM)]
        else:
            rule = [("press step", KEY), ("     each step solves one part with the others held fixed", DIM)]
        D.segments(dl, x, y, rule, 0.9)
        ew = min(w, 24 * em)
        D.text(dl, x, y + 1.5 * em, "error per sweep", DIM, 0.75)
        errs = np.array(walk.errors)
        D.sparkline(dl, x, y + 2.5 * em, ew, 2.5 * em, errs, ACCENT, 0, float(errs.max()))
        D.text(dl, x + ew + 0.6 * em, y + 2.6 * em, f"{100 * errs[-1]:.1f}%", ACCENT, 0.85)
        D.text(dl, x + ew + 0.6 * em, y + 3.7 * em, f"truth {100 * self.truth_error:.1f}% (noise)", DIM, 0.75)


TEXT_ = D.TEXT
