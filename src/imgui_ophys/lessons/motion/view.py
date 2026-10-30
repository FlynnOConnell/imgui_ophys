"""
The Motion correction stage: registration of every frame to a reference image, rigid and non-rigid (piecewise rigid),
and how suite2p builds the reference.
"""

import numpy as np
from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui

from imgui_ophys import diagram as D
from imgui_ophys.lesson import Lesson
from imgui_ophys.lessons.motion import model as M
from imgui_ophys.plotting import ACCENT, DIM, HIGHLIGHT
from imgui_ophys.widgets import KEY, help_mark, section

ImVec2 = imgui.ImVec2
ARROW_GAIN = 1.5  # arrows are drawn this many times longer than the shift
ITERATIONS = 8  # suite2p's refinement passes


class MotionLesson(Lesson):
    key = "motion"
    title = "Motion correction"
    summary = "register every frame to a reference image"
    sections = (
        (fa.ICON_FA_UP_DOWN_LEFT_RIGHT, "Registration"),
        (fa.ICON_FA_CROSSHAIRS, "Rigid registration"),
        (fa.ICON_FA_TABLE_CELLS_LARGE, "Non-rigid registration"),
        (fa.ICON_FA_IMAGE, "Reference image"),
    )
    references = (
        ("Pnevmatikakis EA, Giovannucci A (2017). NoRMCorre: An online algorithm for piecewise rigid motion correction of calcium imaging data. J Neurosci Methods 291:83-94.",
         "https://doi.org/10.1016/j.jneumeth.2017.07.031"),
        ("Pachitariu M, Stringer C, Dipoppa M, et al. (2016). Suite2p: beyond 10,000 neurons with standard two-photon microscopy. bioRxiv.",
         "https://doi.org/10.1101/061507"),
        ("suite2p documentation: Registration (reference image computation, rigid and non-rigid registration).",
         "https://suite2p.readthedocs.io/en/latest/registration/"),
        ("Guizar-Sicairos M, Thurman ST, Fienup JR (2008). Efficient subpixel image registration algorithms. Optics Letters 33:156-158.",
         "https://doi.org/10.1364/OL.33.000156"),
    )

    def __init__(self):
        self.bends = True
        self.max_shift = 3.0
        self.noise = 0.04
        self.seed_frames = 20  # suite2p's default
        self.frame, self.playing, self.accum = 0, True, 0.0
        self.make()

    def make(self) -> None:
        self.motion = M.make_motion(max_shift=self.max_shift, bend=1.0 if self.bends else 0.0, noise=self.noise)
        frames = self.motion.frames
        self.make_reference()
        self.hi = float(np.percentile(frames, 99.5))

    def make_reference(self) -> None:
        frames = self.motion.frames
        self.refs = M.compute_reference(frames, self.seed_frames, ITERATIONS)
        self.ref = self.refs[-1]
        self.rigid, self.rigid_shifts = M.register(frames, self.ref)
        self.patched, self.patch_shifts = M.register_patches(frames, self.ref)
        # the reference sits where most frames are: true motion is shown relative to it
        _, offset = M.register(self.ref[None], M.field())
        self.offset = offset[0]

    def update(self, dt: float) -> None:
        if self.playing:
            self.accum += dt
            while self.accum >= 1 / 4:
                self.accum -= 1 / 4
                self.frame = (self.frame + 1) % len(self.motion.frames)

    def animating(self) -> bool:
        return self.playing

    def controls(self) -> None:
        section(f"{fa.ICON_FA_UP_DOWN_LEFT_RIGHT}  Motion")
        changed, self.bends = imgui.checkbox("tissue bends", self.bends)
        help_mark("Off: the whole field moves together (rigid motion). On: it also bends smoothly, up to 1 px either way, so different parts move differently.")
        if changed:
            self.make()
        section(f"{fa.ICON_FA_PLAY}  Playback")
        if imgui.button(fa.ICON_FA_PAUSE if self.playing else fa.ICON_FA_PLAY):
            self.playing = not self.playing
        imgui.same_line()
        c, f = imgui.slider_int("frame", self.frame, 0, len(self.motion.frames) - 1)
        if c:
            self.frame, self.playing = f, False

    # ------------------------------------------------------------------

    def image(self, dl, x: float, y: float, img: np.ndarray, cell: float, label: str, color=KEY, lo: float = 0.0, hi: float = None) -> None:
        D.text(dl, x, y, label, color, 0.8)
        D.matrix(dl, x, y + 1.2 * D.em(), img, cell, color=D.gray, lo=lo, hi=self.hi if hi is None else hi)

    def difference(self, dl, x: float, y: float, img: np.ndarray, cell: float, label: str, m: float) -> None:
        D.text(dl, x, y, label, DIM, 0.8)
        D.matrix(dl, x, y + 1.2 * D.em(), img - self.ref, cell, color=D.diverging(D.TEXT), lo=-m, hi=m)

    def arrows(self, dl, x: float, y: float, cell: float, ys, xs, shifts: np.ndarray, color) -> None:
        """An arrow at each point (ys[i], xs[j]) in pixels, ARROW_GAIN times its (dy, dx); a dot where it is ~0."""
        side = M.SIDE * cell
        dl.push_clip_rect(ImVec2(x, y), ImVec2(x + side, y + side), True)
        for i, py in enumerate(ys):
            for j, px in enumerate(xs):
                dy, dx = shifts[i, j]
                cx, cy = x + (px + 0.5) * cell, y + (py + 0.5) * cell
                if abs(dy) + abs(dx) > 0.05:
                    D.arrow(dl, cx, cy, cx + dx * cell * ARROW_GAIN, cy + dy * cell * ARROW_GAIN, color, 2.0)
                else:
                    dl.add_circle_filled(ImVec2(cx, cy), 0.15 * D.em(), D.u32(color))
        dl.pop_clip_rect()

    def true_shifts(self, t: int, points: np.ndarray) -> np.ndarray:
        """True displacement of frame t from the reference at (points x points)."""
        return self.motion.true_at(t, points, points) - self.offset

    def layout(self, n: int, extra: float = 4.0):
        em, w = D.em(), D.guide_width()
        cell = min(0.42 * em, (w - extra * em) / (n * M.SIDE))
        side = M.SIDE * cell
        return w, cell, side, (w - n * side) / max(n - 1, 1)

    def section_0(self) -> None:
        D.heading(fa.ICON_FA_UP_DOWN_LEFT_RIGHT, "Registration",
                  "Motion correction registers each frame to a reference image: estimate the frame's displacement from the reference, then shift it back, so a pixel is the same piece of tissue in every frame.")
        em, dl = D.em(), imgui.get_window_draw_list()
        w, cell, side, gap = self.layout(3)
        t = self.frame
        frame = self.motion.frames[t]
        middle = np.array([(M.SIDE - 1) / 2])
        x, y = D.canvas(w, side + 2.8 * em)
        top = y + 1.2 * em
        self.image(dl, x, y, self.ref, cell, "reference image")
        self.image(dl, x + side + gap, y, frame, cell, f"frame {t}", HIGHLIGHT)
        self.arrows(dl, x + side + gap, top, cell, middle, middle, self.true_shifts(t, middle), HIGHLIGHT)
        m = float(np.abs(frame - self.ref).max()) or 1.0
        self.difference(dl, x + 2 * (side + gap), y, frame, cell, "frame minus reference", m)
        dy, dx = self.true_shifts(t, middle)[0, 0]
        D.segments(dl, x + side + gap, top + side + 0.4 * em, [("displacement  ", DIM), (f"{dy:+.1f}, {dx:+.1f} px", HIGHLIGHT)], 0.8)
        D.text(dl, x + 2 * (side + gap), top + side + 0.4 * em, "white brighter, pink dimmer than the reference", DIM, 0.75)

        x, y = D.canvas(w, 5.4 * em)
        true = np.array([self.true_shifts(i, middle)[0, 0] for i in range(len(self.motion.frames))])
        lim = max(float(np.abs(true).max()), 1.0)
        for k, name in enumerate(("dy (down)", "dx (right)")):
            bx = x + k * (w / 2 + 0.5 * em)
            D.text(dl, bx, y, f"{name}, every frame", DIM, 0.8)
            D.sparkline(dl, bx, y + 1.2 * em, w / 2 - 0.5 * em, 3.6 * em, true[:, k], HIGHLIGHT, -lim, lim, cursor=t)

    def section_1(self) -> None:
        D.heading(fa.ICON_FA_CROSSHAIRS, "Rigid registration",
                  "One translation (dy, dx) per frame. Cross-correlate the frame with the reference at every offset (fast with FFTs); the peak's offset from the center is the shift, refined to a fraction of a pixel. suite2p whitens both images first (phase correlation).")
        em, dl = D.em(), imgui.get_window_draw_list()
        w, cell, side, gap = self.layout(3, 8.0)
        t = self.frame
        frame = self.motion.frames[t]
        corr = M.correlation(self.ref, frame)
        py, px = M.find_shift(self.ref, frame)
        x, y = D.canvas(w, side + 4.0 * em)
        self.image(dl, x, y, self.ref, cell, "reference image")
        self.image(dl, x + side + gap, y, frame, cell, f"frame {t}", HIGHLIGHT)
        cx = x + 2 * (side + gap)
        D.text(dl, cx, y, "cross-correlation", KEY, 0.8)
        c = M.SIDE // 2
        m = float(corr.max()) or 1.0
        D.matrix(dl, cx, y + 1.2 * em, np.maximum(corr, 0), cell, color=D.tinted(ACCENT), lo=0, hi=m, cells=[(c + py, c + px)])
        mid = (cx + (c + 0.5) * cell, y + 1.2 * em + (c + 0.5) * cell)
        dl.add_circle(ImVec2(*mid), 0.3 * em, D.u32(DIM), 0, 1.5)
        if py or px:
            D.arrow(dl, mid[0], mid[1], cx + (c + px + 0.5) * cell, y + 1.2 * em + (c + py + 0.5) * cell, HIGHLIGHT, 1.5)
        dy, dx = self.rigid_shifts[t]
        D.segments(dl, cx, y + side + 1.6 * em, [("peak  ", DIM), (f"{py:+d}, {px:+d}", KEY), ("   subpixel  ", DIM), (f"{dy:+.2f}, {dx:+.2f}", ACCENT)], 0.8)
        D.text(dl, cx, y + side + 2.7 * em, "o = zero shift", DIM, 0.75)
        D.text(dl, x, y + side + 1.6 * em, f"shifts searched up to {M.SEARCH} px each way", DIM, 0.75)

        # every frame's shift, found against true; and the frame moved back
        x, y = D.canvas(w, side + 2.8 * em)
        top = y + 1.2 * em
        middle = np.array([(M.SIDE - 1) / 2])
        true = np.array([self.true_shifts(i, middle)[0, 0] for i in range(len(self.motion.frames))])
        lim = max(float(np.abs(true).max()), float(np.abs(self.rigid_shifts).max()), 1.0)
        pw = w - side - gap
        h = (side - 1.2 * em) / 2
        for k, name in enumerate(("dy", "dx")):
            sy = top + k * (h + 1.2 * em)
            D.segments(dl, x, sy - 1.2 * em, [(f"{name}:  ", DIM), ("true  ", HIGHLIGHT), ("found", ACCENT)], 0.8)
            D.sparkline(dl, x, sy, pw, h, true[:, k], HIGHLIGHT, -lim, lim, cursor=t)
            D.sparkline(dl, x, sy, pw, h, self.rigid_shifts[:, k], ACCENT, -lim, lim, frame=False)
        self.image(dl, x + pw + gap, y, self.rigid[t], cell, f"frame {t}, shifted back")
        err = np.abs(self.rigid_shifts - true).mean()
        D.segments(dl, x, top + side + 0.4 * em, [("found minus true, mean  ", DIM), (f"{err:.2f} px", KEY)], 0.8)

    def section_2(self) -> None:
        D.heading(fa.ICON_FA_TABLE_CELLS_LARGE, "Non-rigid registration",
                  "When the tissue bends, one shift cannot fit every part. Split the frame into overlapping patches, register each patch rigidly, interpolate the patch shifts into a shift for every pixel, and resample the frame. NoRMCorre calls this piecewise rigid; suite2p calls the patches blocks.")
        em, dl = D.em(), imgui.get_window_draw_list()
        w, cell, side, gap = self.layout(3)
        t = self.frame
        frame = self.motion.frames[t]
        pc = M.PATCH_CENTERS
        x, y = D.canvas(w, side + 5.8 * em)
        top = y + 1.2 * em
        self.image(dl, x, y, frame, cell, f"frame {t}, true motion", HIGHLIGHT)
        self.arrows(dl, x, top, cell, pc, pc, self.true_shifts(t, pc), HIGHLIGHT)
        # one patch outlined: the one under the mouse, else the first
        hover = D.mouse_in(x, top, side, side)
        i = j = 0
        if hover:
            mp = imgui.get_mouse_pos()
            i = int(np.argmin(np.abs(pc - (mp.y - top) / cell)))
            j = int(np.argmin(np.abs(pc - (mp.x - x) / cell)))
        r0, c0 = M.PATCH_STARTS[i], M.PATCH_STARTS[j]
        dl.add_rect(ImVec2(x + c0 * cell, top + r0 * cell), ImVec2(x + (c0 + M.PATCH) * cell, top + (r0 + M.PATCH) * cell), D.u32(KEY), 0, 2.0)
        if hover:
            dy, dx = self.patch_shifts[t, i, j]
            ty, tx = self.true_shifts(t, pc)[i, j]
            imgui.set_tooltip(f"patch ({i}, {j}): rows {r0}-{r0 + M.PATCH - 1}, columns {c0}-{c0 + M.PATCH - 1}\n"
                              f"found {dy:+.2f}, {dx:+.2f}   true at its center {ty:+.2f}, {tx:+.2f}")
        middle = np.array([(M.SIDE - 1) / 2])
        panels = (
            (self.rigid[t], middle, self.rigid_shifts[t][None, None, :], "rigid", "one shift for the frame"),
            (self.patched[t], pc, self.patch_shifts[t], "piecewise rigid", f"{len(pc) ** 2} patches, {M.PATCH} x {M.PATCH}, overlap {M.PATCH - M.STRIDE}"),
        )
        for k, (img, points, shifts, name, note) in enumerate(panels):
            bx = x + (k + 1) * (side + gap)
            self.image(dl, bx, y, img, cell, f"{name}, found shifts")
            self.arrows(dl, bx, top, cell, points, points, shifts, ACCENT)
            D.text(dl, bx, top + side + 0.4 * em, note, DIM, 0.75)
        D.text(dl, x, top + side + 0.4 * em, f"arrows x{ARROW_GAIN:g}; hover for a patch", DIM, 0.75)
        note = ("the tissue bends: the patches find different shifts" if self.bends
                else "rigid motion: every patch finds the same shift, and both agree")
        D.text(dl, x, top + side + 1.6 * em, note, DIM, 0.8)

        x, y = D.canvas(w, side + 2.6 * em)
        top = y + 1.2 * em
        m = float(np.abs(self.rigid[t] - self.ref).max()) or 1.0
        for k, (img, label) in enumerate(((frame, "recorded"), (self.rigid[t], "rigid"), (self.patched[t], "piecewise rigid"))):
            bx = x + k * (side + gap)
            self.difference(dl, bx, y, img, cell, f"{label} minus reference", m)
            D.segments(dl, bx, top + side + 0.3 * em, [("mean |difference|  ", DIM), (f"{M.mismatch(img, self.ref):.3f}", KEY)], 0.75)

    def section_3(self) -> None:
        D.heading(fa.ICON_FA_IMAGE, "Reference image",
                  f"Every frame is aligned to it, so it should be sharp. suite2p seeds it with the frame most correlated with its 20 most-correlated frames, averages those, then {ITERATIONS} times: register every frame rigidly and average the best-matching ones.")
        em, dl = D.em(), imgui.get_window_draw_list()
        images = [self.motion.frames.mean(axis=0), self.refs[0], self.refs[-1]]
        names = ["mean of raw frames", "seed average", f"after {ITERATIONS} passes"]
        w, cell, side, gap = self.layout(3)
        x, y = D.canvas(w, side + 1.6 * em)
        for i, (img, name) in enumerate(zip(images, names)):
            self.image(dl, x + i * (side + gap), y, img, cell, name, KEY if i == 2 else DIM, hi=float(img.max()))
