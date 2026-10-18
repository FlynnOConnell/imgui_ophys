"""
SVD, under Other: a matrix taken apart into rank-1 pieces value by value, the pieces found one at a time, and keeping
the top k.
"""

from functools import lru_cache

import numpy as np
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui

from imgui_ophys import diagram as D
from imgui_ophys import plotting as P
from imgui_ophys.lesson import Lesson
from imgui_ophys.lessons.svd import model as M
from imgui_ophys.widgets import KEY, help_mark, section

ImVec2 = imgui.ImVec2
PEEL_STEPS_PER_S = 3.0
PEEL_CHIPS = ("guess a row v", "u = R v: score each row", "v = R' u: score each column", "subtract s u v'")


@lru_cache(maxsize=1)
def cached():
    return M.svd(M.small_image()), M.svd(M.image())


class SvdLesson(Lesson):
    key = "svd"
    title = "SVD"
    group = "Other"
    summary = "singular value decomposition: any matrix as a sum of rank-1 matrices, largest first"
    sections = (
        (fa.ICON_FA_TABLE_CELLS, "Rank-1 matrices"),
        (fa.ICON_FA_LAYER_GROUP, "Power iteration and deflation"),
        (fa.ICON_FA_SCISSORS, "Truncated SVD"),
    )
    references = (
        ("Strang G (2023). Introduction to Linear Algebra, 6th ed. Wellesley-Cambridge Press.",
         "https://math.mit.edu/~gs/linearalgebra/ila6/indexila6.html"),
        ("Trefethen LN, Bau D (1997). Numerical Linear Algebra. SIAM.",
         "https://doi.org/10.1137/1.9780898719574"),
        ("Eckart C, Young G (1936). The approximation of one matrix by another of lower rank. Psychometrika 1:211-218.",
         "https://doi.org/10.1007/BF02288367"),
    )

    def __init__(self):
        self.peel = M.Peel(M.small_image())
        self.peel_playing = False
        self.peel_accum = 0.0
        self.peel_hover: tuple = ()  # ("row", r) or ("col", c) of the leftover being hovered
        self.cell = (2, 3)  # (row, col) of the times table
        self.rank = 6  # rank of the image in Truncated SVD

    def update(self, dt: float) -> None:
        if self.peel_playing:
            self.peel_accum += dt
            while self.peel_accum >= 1 / PEEL_STEPS_PER_S:
                self.peel_accum -= 1 / PEEL_STEPS_PER_S
                if self.peel.done:
                    self.peel_playing = False
                    break
                self.peel.step()

    def animating(self) -> bool:
        return self.peel_playing

    def controls(self) -> None:
        section(f"{fa.ICON_FA_SCISSORS}  Truncated SVD")
        _, self.rank = imgui.slider_int("rank", self.rank, 1, 32)
        help_mark("Components kept of the 32 x 32 image.")

    # ------------------------------------------------------------------

    def section_0(self) -> None:
        D.heading(fa.ICON_FA_TABLE_CELLS, "Rank-1 matrices",
                  "An outer product: a column u down the side, a row v across the top, every entry u[row] x v[col], all scaled by one singular value s. Each piece of the SVD is one of these.")
        small, _ = cached()
        em, dl = D.em(), imgui.get_window_draw_list()
        n = M.SMALL
        w = D.guide_width()
        s1, u, v = float(small.S[0]), small.U[:, 0], small.Vt[0]
        piece = small.piece(0)
        hi = float(small.Y.max())
        cell = min(1.6 * em, (w - 8 * em) / (2 * n + 2))
        x, y = D.canvas(w, (n + 1) * cell + 4.6 * em)
        top = y + 1.3 * em
        color = P.component_color(0)
        D.segments(dl, x, y, [(f"piece 1 = {s1:.2f}", color), (" x  column u1 x row v1", P.DIM)], 0.85)
        gx, gy = x + 1.2 * cell, top + cell  # the table, with u to its left and v above
        r, c = self.cell
        D.matrix(dl, gx - 1.2 * cell, gy, u[:, None], cell, color=D.tinted(color), lo=0, hi=float(u.max()), fmt="{:.2f}", rows=[r])
        D.matrix(dl, gx, top, v[None, :], cell, color=D.tinted(color), lo=0, hi=float(v.max()), fmt="{:.2f}", cols=[c])
        hover = D.matrix(dl, gx, gy, piece, cell, color=D.gray, lo=0, hi=hi, fmt="{:.2f}", cells=[self.cell])
        if hover is not None:
            self.cell = hover
        D.text(dl, gx - 1.2 * cell, top + 0.2 * cell, "u \\ v", P.DIM, 0.7)
        ty = gy + n * cell + 0.5 * em
        D.segments(dl, x, ty, [(f"cell ({r}, {c}) = ", P.DIM), (f"{s1:.2f}", color), (" x ", P.DIM), (f"{u[r]:.2f}", color), (" x ", P.DIM),
                               (f"{v[c]:.2f}", color), ("  =  ", P.DIM), (f"{piece[r, c]:.2f}", KEY), ("     hover a cell", P.DIM)], 0.85)
        # Y beside it: the one piece is already close
        yx = gx + n * cell + 3.0 * em
        if yx + n * cell <= x + w:
            D.segments(dl, yx, y, [("Y", KEY), ("  for comparison", P.DIM)], 0.85)
            D.matrix(dl, yx, gy, small.Y, cell, color=D.gray, lo=0, hi=hi, fmt="{:.1f}", cells=[self.cell])
            D.segments(dl, yx, ty, [(f"Y[{r}, {c}] = {small.Y[r, c]:.2f}", D.TEXT)], 0.85)
        x, y = D.canvas(w, 1.4 * em)
        D.text(dl, x, y, "Every row of a piece is the same row v, just scaled by u[row]. Y's rows are mostly like that too, so one piece gets close.", P.DIM, 0.8)

    def section_1(self) -> None:
        D.heading(fa.ICON_FA_LAYER_GROUP, "Power iteration and deflation",
                  "Power iteration: guess a row v; u = Y v, scaled to length 1; v = Y'u, scaled to length 1; repeat until v stops changing. That is the largest piece. Deflation: subtract it and repeat on what is left.")
        peel = self.peel
        if imgui.button(f"{fa.ICON_FA_FORWARD_STEP}  step"):
            peel.step()
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_FORWARD_FAST}  finish piece"):
            if not peel.done:
                peel.finish_piece()
        imgui.same_line()
        if imgui.button(fa.ICON_FA_PAUSE if self.peel_playing else f"{fa.ICON_FA_PLAY}  play"):
            self.peel_playing = not self.peel_playing
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_ROTATE_LEFT}  start over"):
            self.peel = peel = M.Peel(M.small_image())
            self.peel_playing = False
        imgui.same_line()
        k = len(peel.pieces) + 1
        imgui.text_disabled("all pieces found" if peel.done else f"piece {k}, iteration {peel.iterations}")

        em, dl = D.em(), imgui.get_window_draw_list()
        w = D.guide_width()
        x, y = D.canvas(w, 2.2 * em)
        lit = {"u": 1, "v": 2, "subtract": 3}[peel.next] if peel.iterations or peel.next != "u" else 0
        D.chip_row(dl, x, y, w, PEEL_CHIPS, lit)

        n = M.SMALL
        R = peel.R
        cell = min(1.45 * em, (w - 16 * em) / (n + 1.4))
        color = P.component_color(len(peel.pieces))
        x, y = D.canvas(w, (n + 1.4) * cell + 5.2 * em)
        top = y + 1.3 * em
        rm = float(np.abs(R).max()) or 1.0  # colors to the leftover's own range, so its pattern shows
        D.segments(dl, x, y, [("R", KEY), ("  what is left of Y (colors rescaled)" if peel.pieces else "  = Y (nothing found yet)", P.DIM)], 0.85)
        hover = D.matrix(dl, x, top, R, cell, color=D.diverging(D.TEXT), lo=-rm, hi=rm, fmt="{:.2f}")
        last = peel.next  # what the next step computes
        ux = x + n * cell + 0.4 * cell
        vy = top + n * cell + 0.4 * cell
        um = float(np.abs(peel.u).max()) or 1.0
        vm = float(np.abs(peel.v).max()) or 1.0
        D.matrix(dl, ux, top, peel.u[:, None], cell, color=D.diverging(color), lo=-um, hi=um, fmt="{:.2f}", cols=[0] if last == "v" else [])
        D.matrix(dl, x, vy, peel.v[None, :], cell, color=D.diverging(color), lo=-vm, hi=vm, fmt="{:.2f}", rows=[0] if last == "u" else [])
        D.text(dl, ux, top - 1.2 * em, "u", color, 0.85)
        D.text(dl, x - 1.0 * em, vy + 0.25 * cell, "v", color, 0.85)
        # what one score is: a row of R against v, or a column against u
        fy = vy + cell + 0.5 * em
        if hover is not None:
            r, c = hover
            if last == "v":
                terms = R[:, c] * peel.u
                label = f"column {c} . u = "
            else:
                terms = R[r] * peel.v
                label = f"row {r} . v = "
            parts = [(label, P.DIM)] + [(f"{a:.2f}x{b:.2f} + ", P.DIM) for a, b in zip((R[:, c] if last == "v" else R[r])[:3], (peel.u if last == "v" else peel.v)[:3])]
            parts += [("... = ", P.DIM), (f"{terms.sum():.3f}", KEY)]
            D.segments(dl, x, fy, parts, 0.8)
        else:
            D.text(dl, x, fy, "hover R: a row's score against v (or a column's against u)", P.DIM, 0.8)

        # the piece as it stands, and how its strength s settles
        px = ux + 2.5 * cell
        pc = min(0.9 * em, (x + w - px) / (n + 2))
        D.segments(dl, px, y, [(f"piece {k} so far", color), (f"   s = {peel.s:.3f}", KEY)], 0.85)
        piece = peel.piece()
        pm = float(np.abs(piece).max()) or 1.0
        D.matrix(dl, px, top, piece, pc, color=D.diverging(color), lo=-pm, hi=pm)
        sy = top + n * pc + 0.6 * em
        if peel.history:
            D.text(dl, px, sy, "s after each iteration", P.DIM, 0.75)
            D.sparkline(dl, px, sy + 1.1 * em, min(x + w - px, 14 * em), 2.4 * em, np.array(peel.history), color)
            D.text(dl, px, sy + 3.8 * em, f"v changed by {peel.change:.1e}", P.DIM, 0.75)

        # Y as the pieces found so far plus what is left
        x, y = D.canvas(w, n * 0.7 * em + 3.2 * em)
        mc = 0.7 * em
        D.text(dl, x, y, "so far", P.DIM, 0.8)
        bx, top = x, y + 1.3 * em
        D.text(dl, bx, top + n * mc / 2 - 0.6 * em, "Y =", D.TEXT, 1.0)
        bx += 2.6 * em
        for j, (s_j, u_j, v_j) in enumerate(peel.pieces[:5]):
            pj = s_j * np.outer(u_j, v_j)
            m = float(np.abs(pj).max()) or 1.0
            cj = P.component_color(j)
            D.matrix(dl, bx, top, pj, mc, color=D.diverging(cj), lo=-m, hi=m)
            D.text(dl, bx, top + n * mc + 0.2 * em, f"s{j + 1} {s_j:.2f}", cj, 0.7)
            bx += n * mc + 0.4 * em
            D.text(dl, bx, top + n * mc / 2 - 0.6 * em, "+", P.DIM, 1.0)
            bx += 1.4 * em
        if len(peel.pieces) > 5:
            D.text(dl, bx, top + n * mc / 2 - 0.6 * em, "... +", P.DIM, 1.0)
            bx += 2.4 * em
        D.matrix(dl, bx, top, R, mc, color=D.diverging(D.TEXT), lo=-rm, hi=rm)
        left = float(np.linalg.norm(R) / np.linalg.norm(peel.Y))
        D.text(dl, bx, top + n * mc + 0.2 * em, f"leftover R: {100 * left:.1f}% of Y", P.DIM, 0.7)

    def section_2(self) -> None:
        D.heading(fa.ICON_FA_SCISSORS, "Truncated SVD",
                  "Keep the first k pieces and drop the rest. No rank-k matrix is closer to the original (Eckart-Young); the error is set by the dropped s values.")
        _, img = cached()
        em, dl = D.em(), imgui.get_window_draw_list()
        k = self.rank
        n = img.Y.shape[0]
        w = D.guide_width()
        cell = min(0.4 * em, (w - 4 * em) / (3 * n))
        side = n * cell
        gap = (w - 3 * side) / 2
        x, y = D.canvas(w, side + 2.6 * em)
        approx = img.rank(k)
        hi = float(img.Y.max())
        res = img.Y - approx
        m = float(np.abs(res).max()) or 1.0
        for i, (a, label, color, lo, top_v) in enumerate(((img.Y, "image, 32 x 32", P.RAW, 0, hi), (approx, f"first {k} pieces", P.ACCENT, 0, hi), (res, "left out", P.DIM, -m, m))):
            bx = x + i * (side + gap)
            D.text(dl, bx, y, label, color, 0.8)
            D.matrix(dl, bx, y + 1.2 * em, a, cell, color=D.gray if lo == 0 else D.diverging(D.TEXT), lo=lo, hi=top_v)
        x, y = D.canvas(w, 7.6 * em)
        D.text(dl, x, y, "s values: kept (blue) and dropped", P.DIM, 0.75)
        bars_w = w * 0.55
        bw = bars_w / n
        base = y + 6.0 * em
        for j in range(n):
            h = img.S[j] / img.S[0] * 4.6 * em
            dl.add_rect_filled(ImVec2(x + j * bw, base - h), ImVec2(x + (j + 1) * bw - 2, base), D.u32(P.ACCENT if j < k else P.DIM, 0.9 if j < k else 0.5))
        ex = x + bars_w + 2 * em
        ew = x + w - ex
        D.text(dl, ex, y, "left out vs k", P.DIM, 0.75)
        errs = np.array([img.error(j) for j in range(1, n + 1)])
        D.sparkline(dl, ex, y + 1.2 * em, ew, 4.8 * em, errs, P.HIGHLIGHT, 0, 1, cursor=k - 1)
        D.segments(dl, x, base + 0.4 * em, [("left out  ", P.DIM), (f"{100 * img.error(k):.1f}%", KEY), ("     store  ", P.DIM),
                                            (f"{k} x ({n} + {n} + 1) = {k * (2 * n + 1)}", KEY), (f"  numbers instead of {n * n}", P.DIM)], 0.85)
