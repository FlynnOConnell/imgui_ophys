# /// script
# requires-python = ">=3.11"
# dependencies = ["manim>=0.22", "numpy", "pillow<13"]
# ///
"""
SVD as one story: a small image, one rank-1 piece of it as a column times a row, power iteration finding that
piece, deflation finding the rest, and a larger image rebuilt from its first k pieces. Built from imgui_ophys's SVD
model; drawn with storykit.

    uv run -p 3.12 stories/svd_story.py --out <dir> [--stills-only] [--gif-width 720]
"""

import manim as mn
import numpy as np

import storykit as kit
from storykit import ACCENT, DIM, KEY, PARTS, TEXT, TRACE, Count, Recolor, at, chapter, color, diverging_rgb, gray_rgb, grid, image, label, put, replace, story

M = kit.lesson_model("svd")

TITLE = "SVD: an image as a sum of rank-1 pieces"
CHAPTERS = (
    ("", 5.0),
    ("A rank-1 piece: s x column u x row v", 7.0),
    ("Power iteration: u = R v, v = R'u, until v settles", 9.0),
    ("Deflation: subtract the piece, repeat on what is left", 9.0),
    ("Truncated SVD: keep the first k pieces", 12.0),
)
UNTITLED = {0: "the-image"}
CELL = 0.4
RANKS = (1, 2, 4, 8, 16)
N_PIECES = 3


def s_label(x: float) -> mn.Text:
    return label(f"s = {x:.2f}", KEY, 26)


def left_label(x: float) -> mn.Text:
    return label(f"left out {x:.1f}%", KEY, 26)


def piece_grid(piece: np.ndarray, m: float, c: tuple, cell: float, x: float, y: float) -> mn.VGroup:
    return grid(diverging_rgb(piece, m, c), cell, x, y)


def build() -> kit.Story:
    Y = M.small_image()
    n = Y.shape[0]
    small = M.svd(Y)
    hi = float(Y.max())
    title = put(label(TITLE, TEXT, 30), 0.0, 3.6)
    titles = kit.titles_for(CHAPTERS)

    # chapter 1: Y
    x_y, y_c = -4.0, 0.0
    y_grid = grid(gray_rgb(Y, 0, hi), CELL, x_y, y_c)
    y_frame = kit.frame_of(y_grid, buff=0.03)
    name_y = mn.VGroup(label("Y", KEY, 22), label(f"{n} x {n}", DIM, 18)).arrange(mn.RIGHT, buff=0.3).next_to(y_grid, mn.UP, buff=0.2)

    # chapter 2: the first piece as a column times a row
    s1, u1, v1 = float(small.S[0]), small.U[:, 0], small.Vt[0]
    c1 = PARTS[0]
    x_p = 1.6
    piece1 = small.piece(0)
    p_grid = grid(gray_rgb(piece1, 0, hi), CELL, x_p, y_c)
    u_col = grid(diverging_rgb(u1[:, None], float(np.abs(u1).max()), c1), CELL, x_p - (n / 2 + 0.7) * CELL, y_c)
    v_row = grid(diverging_rgb(v1[None, :], float(np.abs(v1).max()), c1), CELL, x_p, y_c + (n / 2 + 0.7) * CELL)
    name_u = label("u", c1, 20).next_to(u_col, mn.DOWN, buff=0.15)
    name_v = label("v", c1, 20).next_to(v_row, mn.LEFT, buff=0.15)
    name_p = mn.VGroup(label("piece 1", c1, 22), label(f"s = {s1:.2f}", KEY, 18)).arrange(mn.RIGHT, buff=0.3).next_to(v_row, mn.UP, buff=0.2)
    approx = label("~", DIM, 34).move_to(at(0.5 * (x_y + x_p) - 0.7, y_c))
    r, c = 2, 5
    lit = mn.Square(side_length=CELL, color=color(KEY), stroke_width=2.0).move_to(p_grid[r * n + c])
    lit_u = mn.Square(side_length=CELL, color=color(KEY), stroke_width=2.0).move_to(u_col[r])
    lit_v = mn.Square(side_length=CELL, color=color(KEY), stroke_width=2.0).move_to(v_row[c])
    worked = mn.VGroup(label(f"{s1:.2f} x {u1[r]:.2f} x {v1[c]:.2f} = {piece1[r, c]:.2f}", TEXT, 18), label(f"Y there: {Y[r, c]:.2f}", DIM, 18)).arrange(mn.DOWN, buff=0.12)
    worked.next_to(p_grid, mn.DOWN, buff=0.3)

    # chapter 3: power iteration, u and v settling and s climbing
    peel = M.Peel(Y)
    states = [(peel.u.copy(), peel.v.copy(), peel.s)]
    while peel.next != "subtract":
        peel.step()
        if peel.next == "u" or peel.next == "subtract":
            states.append((peel.u.copy(), peel.v.copy(), peel.s))
    iterations = len(states) - 1
    u_frames = [diverging_rgb(u[:, None], float(np.abs(u).max()) or 1.0, c1) for u, _, _ in states]
    v_frames = [diverging_rgb(v[None, :], float(np.abs(v).max()) or 1.0, c1) for _, v, _ in states]
    p_frames = [gray_rgb(s * np.outer(u, v), 0, hi) for u, v, s in states]
    s_text = s_label(states[0][2]).next_to(p_grid, mn.DOWN, buff=0.3)
    iteration_text = label(f"{iterations} iterations", DIM, 18).next_to(s_text, mn.DOWN, buff=0.15)

    # chapter 4: deflation, Y as pieces plus what is left
    peel = M.Peel(Y)
    rests, pieces = [], []
    for _ in range(N_PIECES):
        peel.finish_piece()
        s_k, u_k, v_k = peel.pieces[-1]
        pieces.append(s_k * np.outer(u_k, v_k))
        rests.append(peel.R.copy())
    small_cell = 0.22
    x_row = -5.2
    pitch = n * small_cell + 0.9
    pieces_grids = [piece_grid(p, float(np.abs(p).max()), PARTS[k], small_cell, x_row + (k + 1) * pitch, 0.0) for k, p in enumerate(pieces)]
    rest_grids = [piece_grid(R, float(np.abs(R).max()) or 1.0, TEXT, small_cell, x_row + (N_PIECES + 1) * pitch, 0.0) for R in rests]
    y_small = grid(gray_rgb(Y, 0, hi), small_cell, x_row, 0.0)
    signs = mn.VGroup(label("=", DIM, 26).move_to(at(x_row + 0.5 * pitch, 0.0)),
                      *[label("+", DIM, 26).move_to(at(x_row + (k + 1.5) * pitch, 0.0)) for k in range(N_PIECES)])
    piece_names = [label(f"s{k + 1} = {small.S[k]:.2f}", PARTS[k], 16).next_to(g, mn.DOWN, buff=0.12) for k, g in enumerate(pieces_grids)]
    left = [100 * float(np.linalg.norm(R) / np.linalg.norm(Y)) for R in rests]
    rest_names = [label(f"left: {x:.1f}% of Y", DIM, 16).next_to(rest_grids[0], mn.DOWN, buff=0.12) for x in left]
    name_small = label("Y", KEY, 18).next_to(y_small, mn.DOWN, buff=0.12)

    # chapter 5: a 32 x 32 image from its first k pieces
    img = M.svd(M.image())
    big = img.Y
    b_hi = float(big.max())
    side = 2.8
    xs = (-4.4, 0.0, 4.4)
    y_big = 0.3
    big_img = image(gray_rgb(big, 0, b_hi), side, xs[0], y_big)
    approx_imgs = [gray_rgb(img.rank(k), 0, b_hi) for k in RANKS]
    rest_scale = float(np.abs(big - img.rank(RANKS[0])).max())
    rest_imgs = [diverging_rgb(big - img.rank(k), rest_scale, TEXT) for k in RANKS]
    approx_img = image(approx_imgs[0], side, xs[1], y_big)
    rest_img = image(rest_imgs[0], side, xs[2], y_big)
    big_frames = [kit.frame_of(m) for m in (big_img, approx_img, rest_img)]
    big_names = mn.VGroup(label(f"image, {big.shape[0]} x {big.shape[1]}", DIM, 18).next_to(big_img, mn.DOWN, buff=0.15),
                          label("first k pieces", ACCENT, 18).next_to(approx_img, mn.DOWN, buff=0.15),
                          label("left out", DIM, 18).next_to(rest_img, mn.DOWN, buff=0.15))
    k_texts = [label(f"k = {k}", ACCENT, 26).move_to(at(xs[1], -2.3)) for k in RANKS]
    errors = [100 * img.error(k) for k in RANKS]
    left_text = left_label(errors[0]).move_to(at(xs[2], -2.3))
    bars = mn.VGroup()
    n_big = big.shape[0]
    bar_w = 2.6 / n_big
    for j in range(n_big):
        h = max(float(img.S[j] / img.S[0]) * 1.0, 0.01)
        bars.add(mn.Rectangle(width=bar_w * 0.8, height=h, stroke_width=0, fill_color=color(DIM), fill_opacity=0.5)
                 .move_to(at(xs[0] - 1.3 + (j + 0.5) * bar_w, -3.3 + h / 2)))
    bars_kept = [mn.VGroup(*[bars[j].copy().set_fill(color(ACCENT), opacity=0.9 if j < k else 0.0) for j in range(n_big)]) for k in RANKS]
    bars_name = label("s values, largest first", DIM, 16).next_to(bars, mn.UP, buff=0.1)

    seconds = [d for _, d in CHAPTERS]
    chapters = [
        chapter(seconds[0], mn.FadeIn(title), mn.FadeIn(y_grid, lag_ratio=0.01), mn.FadeIn(y_frame), mn.FadeIn(name_y), lag_ratio=0.3),
        chapter(seconds[1],
                mn.AnimationGroup(mn.FadeOut(title), mn.FadeIn(titles[1]), run_time=0.8),
                mn.AnimationGroup(mn.FadeIn(u_col, lag_ratio=0.05), mn.FadeIn(name_u), mn.FadeIn(v_row, lag_ratio=0.05), mn.FadeIn(name_v), mn.FadeIn(name_p), run_time=1.2),
                mn.AnimationGroup(mn.FadeIn(approx), mn.FadeIn(p_grid, lag_ratio=0.01), run_time=1.5),
                mn.AnimationGroup(mn.Create(lit), mn.Create(lit_u), mn.Create(lit_v), mn.FadeIn(worked), run_time=0.8), lag_ratio=1.0),
        chapter(seconds[2],
                mn.AnimationGroup(*replace(titles[1], titles[2]), mn.FadeOut(lit), mn.FadeOut(lit_u), mn.FadeOut(lit_v), mn.FadeOut(worked), mn.FadeOut(name_p),
                                  Recolor(u_col, u_frames[:1]), Recolor(v_row, v_frames[:1]), Recolor(p_grid, p_frames[:1]), run_time=0.8),
                mn.AnimationGroup(mn.FadeIn(s_text), run_time=0.4),
                mn.AnimationGroup(Recolor(u_col, u_frames, rate_func=mn.linear), Recolor(v_row, v_frames, rate_func=mn.linear),
                                  Recolor(p_grid, p_frames, rate_func=mn.linear), Count(s_text, states[0][2], states[-1][2], s_label, 2, rate_func=mn.linear),
                                  run_time=4.0),
                mn.FadeIn(iteration_text, run_time=0.4), lag_ratio=1.0),
        chapter(seconds[3],
                mn.AnimationGroup(*replace(titles[2], titles[3]), mn.FadeOut(s_text), mn.FadeOut(iteration_text), mn.FadeOut(name_u), mn.FadeOut(name_v),
                                  mn.Transform(y_grid, y_small), mn.FadeOut(y_frame), mn.FadeOut(name_y), mn.FadeOut(approx), mn.FadeOut(u_col), mn.FadeOut(v_row),
                                  mn.Transform(p_grid, pieces_grids[0]), mn.FadeIn(name_small), mn.FadeIn(signs[0]), mn.FadeIn(signs[1]), mn.FadeIn(piece_names[0]), run_time=1.5),
                mn.AnimationGroup(mn.FadeIn(rest_grids[0]), mn.FadeIn(rest_names[0]), run_time=0.6),
                mn.AnimationGroup(mn.FadeIn(pieces_grids[1]), mn.FadeIn(piece_names[1]), mn.FadeIn(signs[2]), mn.Transform(rest_grids[0], rest_grids[1]),
                                  *replace(rest_names[0], rest_names[1]), run_time=1.5),
                mn.AnimationGroup(mn.FadeIn(pieces_grids[2]), mn.FadeIn(piece_names[2]), mn.FadeIn(signs[3]), mn.Transform(rest_grids[0], rest_grids[2]),
                                  *replace(rest_names[1], rest_names[2]), run_time=1.5), lag_ratio=1.0),
        chapter(seconds[4],
                mn.AnimationGroup(*replace(titles[3], titles[4]), *[mn.FadeOut(m) for m in (y_grid, p_grid, pieces_grids[1], pieces_grids[2], rest_grids[0], name_small,
                                  signs, *piece_names, rest_names[2])], run_time=0.8),
                mn.AnimationGroup(*[mn.FadeIn(m) for m in (big_img, approx_img, rest_img, *big_frames, big_names, k_texts[0], left_text, bars, bars_name, bars_kept[0])], run_time=1.0),
                mn.AnimationGroup(*[mn.AnimationGroup(mn.Transform(approx_img, image(approx_imgs[i], side, xs[1], y_big)),
                                                      mn.Transform(rest_img, image(rest_imgs[i], side, xs[2], y_big)),
                                                      *replace(k_texts[i - 1], k_texts[i]), mn.Transform(bars_kept[0], bars_kept[i]),
                                                      Count(left_text, errors[i - 1], errors[i], left_label, 1), run_time=1.5)
                                    for i in range(1, len(RANKS))], lag_ratio=1.0), lag_ratio=1.0),
    ]
    print(f"s {small.S[:4].round(2)}; power iteration {iterations} iterations; left after pieces {np.round(left, 1)}; big image left out {np.round(errors, 1)}")
    return story([], chapters)


if __name__ == "__main__":
    kit.run("svd", CHAPTERS, UNTITLED, build, __doc__)
