# /// script
# requires-python = ">=3.11"
# dependencies = ["manim>=0.22", "numpy", "pillow<13"]
# ///
"""
Demixing as one story: a 6 x 6, 24 frame movie, each frame unrolled into a column of Y, Y as footprints times
traces, and HALS fitting them from a random start one sweep at a time. Built from imgui_ophys's NMF model (the
guide's tiny movie); drawn with storykit.

    uv run -p 3.12 stories/nmf_story.py --out <dir> [--stills-only] [--gif-width 720]
"""

import manim as mn
import numpy as np

import storykit as kit
from storykit import DIM, KEY, PARTS, TEXT, TRACE, Count, Recolor, at, axes_points, chapter, color, gray_rgb, grid, label, path, plain_axes, put, replace, story, tinted_rgb

M = kit.lesson_model("nmf")

TITLE = "Demixing: a movie as footprints x traces"
CHAPTERS = (
    ("", 8.0),
    ("Y = A C: a footprint (where) times a trace (when) per source", 9.0),
    ("HALS: from a random start, one sweep at a time", 12.0),
    ("Fitted and true", 6.0),
)
UNTITLED = {0: "the-movie"}
K = 3
SWEEPS = 12
NAMES = ("cell 1", "cell 2", "background")
CELL = 0.42  # scene units per movie pixel
YW, YH = 0.16, 0.1  # a cell of Y
X_MOVIE, X_Y = -5.4, -1.3
X_A, X_C, C_WIDTH = 1.85, 4.7, 3.4


def error_label(x: float) -> mn.Text:
    return label(f"error {x:.0f}%", KEY, 26)


def column(values: np.ndarray, hi: float, x: float, y: float, c: tuple = None) -> mn.VGroup:
    rgb = gray_rgb(values[:, None], 0, hi) if c is None else tinted_rgb(values[:, None], 0, hi, c)
    return grid(rgb, YW, x, y, cell_h=YH)


def footprint(values: np.ndarray, side: int, c: tuple, cell: float, x: float, y: float) -> mn.VGroup:
    return grid(tinted_rgb(values.reshape(side, side), 0, float(values.max()) or 1.0, c), cell, x, y)


def trace(values: np.ndarray, hi: float, c: tuple, x: float, y: float, width: float, height: float) -> mn.VMobject:
    axes = plain_axes(0, values.size - 1, 0, hi, width, height, visible=False).move_to(at(x, y))
    return path(axes, np.arange(values.size), values, c, 2.0, 0, hi)


def build() -> kit.Story:
    movie = M.make_tiny()
    Y = movie.Y.astype(float)
    side, T = movie.side, movie.frames
    n = side * side
    hi = float(Y.max())
    title = put(label(TITLE, TEXT, 30), 0.0, 3.6)
    titles = kit.titles_for(CHAPTERS)

    # chapter 1: the movie plays; every frame unrolls into a column of Y
    frames = [gray_rgb(movie.image(Y[:, t]), 0, hi) for t in range(T)]
    screen = grid(frames[0], CELL, X_MOVIE, 0.6)
    screen_frame = kit.frame_of(screen, buff=0.04)
    name_movie = label(f"{T} frames, {side} x {side} pixels", DIM, 18).next_to(screen, mn.DOWN, buff=0.15)
    y_center = 0.1
    columns = [column(Y[:, t], hi, X_Y + (t - (T - 1) / 2) * YW, y_center) for t in range(T)]
    flights = [grid(frames[t], CELL, X_MOVIE, 0.6, opacity=0.0) for t in range(T)]
    name_y = mn.VGroup(label("Y", KEY, 22), label(f"{n} pixels x {T} frames", DIM, 18)).arrange(mn.RIGHT, buff=0.3)
    name_y.next_to(columns[0], mn.UP, buff=0.2).align_to(columns[0], mn.LEFT)
    arrow = mn.Arrow(screen.get_right() + at(0.3, 0), columns[0].get_left() + at(-0.3, 0), buff=0, stroke_width=2.0, color=color(DIM))
    count_y = label(f"{n * T} numbers", DIM, 18).next_to(mn.VGroup(*columns), mn.DOWN, buff=0.2).align_to(columns[0], mn.LEFT)

    # chapter 2: A and C, to shape, with the footprints reshaped back into images
    A_true, C_true = movie.A, movie.C
    a_columns = [column(A_true[:, k], 1.0, X_A + (k - 1) * (YW + 0.08), y_center, PARTS[k]) for k in range(K)]
    name_a = mn.VGroup(label("A", KEY, 22), label(f"{n} x {K}", DIM, 18)).arrange(mn.RIGHT, buff=0.3).next_to(a_columns[1], mn.UP, buff=0.2)
    c_hi = float(C_true.max()) * 1.05
    c_height = 0.75
    c_traces = [trace(C_true[k], c_hi, PARTS[k], X_C, 1.1 - k * 1.05, C_WIDTH, c_height) for k in range(K)]
    name_c = mn.VGroup(label("C", KEY, 22), label(f"{K} x {T}", DIM, 18)).arrange(mn.RIGHT, buff=0.3)
    name_c.next_to(c_traces[0], mn.UP, buff=0.35).align_to(c_traces[0], mn.LEFT)
    equals = label("~", DIM, 34).move_to(at(0.5 * (X_Y + (T / 2) * YW + X_A - (YW + 0.08)), y_center))
    times = label("x", DIM, 26).move_to(at(0.5 * (X_A + (YW + 0.08) + X_C - C_WIDTH / 2), y_center))
    small = 0.28
    footprints = [footprint(A_true[:, k], side, PARTS[k], small, 1.0 + 2.0 * k, -2.55) for k in range(K)]
    footprint_flights = [grid(tinted_rgb(A_true[:, k][:, None], 0, 1.0, PARTS[k]), YW, X_A + (k - 1) * (YW + 0.08), y_center, cell_h=YH, opacity=0.0) for k in range(K)]
    part_names = [label(NAMES[k], PARTS[k], 15).next_to(c_traces[k], mn.UP, buff=0.02).align_to(c_traces[k], mn.LEFT) for k in range(K)]
    count_ac = label(f"{n * K} + {K * T} = {n * K + K * T} numbers", KEY, 18).move_to(count_y, aligned_edge=mn.LEFT)

    # chapter 3: HALS from a random start, the fitted footprints and traces morphing sweep by sweep
    walk = M.HalsWalk(Y, K, seed=0)
    snapshots = [(walk.A.copy(), walk.C.copy(), walk.error())]
    for _ in range(SWEEPS):
        walk.advance_sweep()
        snapshots.append((walk.A.copy(), walk.C.copy(), walk.error()))
    order = M.match_to_truth(walk.A, walk.C, A_true)  # the final permutation, so colors stay put through the sweeps
    sim = M.cosine(A_true, walk.A)
    perm = []
    left = list(range(K))
    for j in range(K):
        best = max(left, key=lambda k: sim[j, k])
        left.remove(best)
        perm.append(best)
    fit_x = [-4.2, -1.4, 1.4]
    fit_hi = max(float(C[perm].max()) for _, C, _ in snapshots[1:])
    fit_footprints = [[footprint(A[:, perm[k]], side, PARTS[k], 0.33, fit_x[k], 1.3) for k in range(K)] for A, _, _ in snapshots]
    fit_traces = [[trace(C[perm[k]], fit_hi if i else float(C.max()), PARTS[k], fit_x[k], -0.4, 2.4, 0.9) for k in range(K)] for i, (_, C, _) in enumerate(snapshots)]
    fit_names = [label(f"part {k + 1}", PARTS[k], 18).next_to(fit_footprints[0][k], mn.UP, buff=0.15) for k in range(K)]
    errors = [100 * e for _, _, e in snapshots]
    error_text = error_label(errors[0]).move_to(at(3.9, 0.9), aligned_edge=mn.LEFT)
    sweep_text = label("random start", DIM, 18).move_to(at(3.9, 0.3), aligned_edge=mn.LEFT)
    sweep_texts = [label(f"sweep {i}", DIM, 18).move_to(at(3.9, 0.3), aligned_edge=mn.LEFT) for i in range(1, SWEEPS + 1)]
    truth_error = 100 * M.rel_error(Y, A_true, C_true)
    truth_text = label(f"noise alone: {truth_error:.0f}%", DIM, 16).move_to(at(3.9, -0.3), aligned_edge=mn.LEFT)

    # chapter 4: the true parts under the fitted ones
    true_footprints = [footprint(A_true[:, k], side, PARTS[k], 0.25, fit_x[k], -1.7) for k in range(K)]
    true_traces = [trace(C_true[k], float(C_true.max()) * 1.05, PARTS[k], fit_x[k], -3.1, 2.4, 0.6) for k in range(K)]
    row_names = mn.VGroup(label("fitted", DIM, 16).move_to(at(-6.9, 0.5), aligned_edge=mn.LEFT), label("true (simulated)", DIM, 16).move_to(at(-6.9, -2.4), aligned_edge=mn.LEFT))

    seconds = [d for _, d in CHAPTERS]
    chapters = [
        chapter(seconds[0],
                mn.AnimationGroup(mn.FadeIn(title), mn.FadeIn(screen), mn.FadeIn(screen_frame), mn.FadeIn(name_movie), run_time=0.8),
                mn.AnimationGroup(mn.FadeIn(name_y), mn.FadeIn(arrow), run_time=0.5),
                mn.AnimationGroup(Recolor(screen, frames, rate_func=mn.linear),
                                  mn.LaggedStart(*[mn.Transform(flights[t], columns[t]) for t in range(T)], lag_ratio=1 / T), run_time=5.0),
                mn.FadeIn(count_y, run_time=0.5), lag_ratio=1.0),
        chapter(seconds[1],
                mn.AnimationGroup(mn.FadeOut(title), mn.FadeIn(titles[1]), mn.FadeOut(screen), mn.FadeOut(screen_frame), mn.FadeOut(name_movie),
                                  mn.FadeOut(arrow), run_time=0.8),
                mn.AnimationGroup(mn.FadeIn(equals), *[mn.FadeIn(c) for c in a_columns], mn.FadeIn(name_a), mn.FadeIn(times),
                                  *[mn.Create(c) for c in c_traces], mn.FadeIn(name_c), *[mn.FadeIn(p) for p in part_names], run_time=1.5),
                mn.AnimationGroup(*[mn.Transform(footprint_flights[k], footprints[k]) for k in range(K)], run_time=1.5),
                mn.AnimationGroup(*replace(count_y, count_ac), run_time=0.8), lag_ratio=1.0),
        chapter(seconds[2],
                mn.AnimationGroup(*replace(titles[1], titles[2]), *[mn.FadeOut(m) for m in (*a_columns, *c_traces, *footprint_flights,
                                  *flights, name_y, name_a, name_c, equals, times, count_ac, *part_names)], run_time=0.8),
                mn.AnimationGroup(*[mn.FadeIn(f) for f in fit_footprints[0]], *[mn.Create(t) for t in fit_traces[0]], *[mn.FadeIn(nm) for nm in fit_names],
                                  mn.FadeIn(error_text), mn.FadeIn(sweep_text), mn.FadeIn(truth_text), run_time=1.0),
                mn.AnimationGroup(*[mn.AnimationGroup(*[mn.Transform(fit_footprints[0][k], fit_footprints[i][k]) for k in range(K)],
                                                      *[mn.Transform(fit_traces[0][k], fit_traces[i][k]) for k in range(K)],
                                                      Count(error_text, errors[i - 1], errors[i], error_label, 0),
                                                      *replace(sweep_text if i == 1 else sweep_texts[i - 2], sweep_texts[i - 1]), run_time=0.6)
                                    for i in range(1, SWEEPS + 1)], lag_ratio=1.0), lag_ratio=1.0),
        chapter(seconds[3],
                mn.AnimationGroup(*replace(titles[2], titles[3]), run_time=0.8),
                mn.AnimationGroup(mn.FadeIn(row_names), *[mn.FadeIn(f) for f in true_footprints], *[mn.Create(t) for t in true_traces], run_time=1.5), lag_ratio=1.0),
    ]
    print(f"errors per sweep {np.round(errors, 1)}; noise alone {truth_error:.1f}%; parts matched to truth as {perm}")
    return story([], chapters)


if __name__ == "__main__":
    kit.run("nmf", CHAPTERS, UNTITLED, build, __doc__)
