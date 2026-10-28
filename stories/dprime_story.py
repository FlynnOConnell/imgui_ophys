# /// script
# requires-python = ">=3.11"
# dependencies = ["manim>=0.22", "numpy", "pillow<13"]
# ///
"""
d' as one story: a trace, one spike, its two windows clustered, the number, then the picture morphing as noise and
photobleaching come and go, and a table of examples. Built from imgui_ophys's trace simulator with the spike times
known; drawn with storykit (manim through a Cairo camera, no Scene).

    uv run -p 3.12 stories/dprime_story.py --out <dir> [--stills-only] [--gif-width 720]

Text only, no LaTeX. Needs ffmpeg on PATH.
"""

from dataclasses import dataclass

import manim as mn
import numpy as np

import storykit as kit
from storykit import (ACCENT, DIM, KEY, TEXT, TRACE, TRUTH, Count, axes_points, caption, chapter, color, dots_at, label, path,
                      plain_axes, put, replace, scale_bars, story)

M = kit.lesson_model("dprime")
from imgui_ophys import sim  # noqa: E402

BEFORE, PEAK = ACCENT, KEY  # the frames just before a spike, and at the response peak
WINDOW_FRAMES = 6  # frames in each pool per spike, as in the lesson and the jGCaMP8 paper's six frames
F0_WINDOW = 5.0  # s, running 20th-percentile baseline for dF/F
TARGETS = (0.5, 1.0, 2.0, 3.0, 4.0)  # the closing table: white noise added to a clean cell until d' lands here

TITLE = "d': a measure of how well spikes stand out from noise"
CHAPTERS = (
    ("", 4.0),
    ("", 4.5),
    ("", 4.0),
    ("", 6.0),
    ("", 4.0),
    ("Add noise", 7.0),
    ("Remove noise", 7.0),
    ("Add photobleaching", 7.0),
    ("", 12.0),
)
UNTITLED = {0: "the-trace", 1: "one-spike", 2: "two-clusters", 3: "every-spike", 4: "the-number", 8: "examples"}  # still file names of the chapters without a title
# the cells the story morphs between, in order: (name, brightness in photons / pixel / frame, bleach fraction, F0)
CELLS = (
    ("base", 20.0, 0.2, "running window"),
    ("noisy", 4.0, 0.2, "running window"),
    ("quiet", 100.0, 0.2, "running window"),
    ("bleached", 100.0, 0.35, "whole recording"),
)



def number_label(d: float) -> mn.Text:
    return label(f"d' = {d:.1f}", PEAK, 44)


def swap(titles: list, captions: list, k: int) -> list:
    """Chapter k's title and caption replace chapter k - 1's."""
    return [*replace(titles[k - 1], titles[k]), *replace(captions[k - 1], captions[k])]


# the data: one cell, same spikes, under different noise and bleaching
# ----------------------------------------------------------------------


@dataclass
class Cell:
    name: str
    t: np.ndarray
    dff: np.ndarray
    spikes: np.ndarray
    fs: float
    score: M.Score


def rescore(cell: Cell, dff: np.ndarray) -> M.Score:
    ind = sim.BY_NAME["GCaMP6f"]
    return M.score(dff, cell.spikes, cell.fs, M.peak_frames(ind, cell.fs), WINDOW_FRAMES, M.quiet_before(ind))


def simulate(name: str, brightness: float, bleach: float, f0: str, neuropil: float = 0.3, read_noise: float = 1.5) -> Cell:
    """The story's cell (seed fixed, so the spikes are the same) at this brightness and bleaching, scored on its dF/F."""
    p = sim.TraceParams(duration=40.0, brightness=brightness, bleach=bleach, seed=3, neuropil=neuropil, read_noise=read_noise)
    tr = sim.make_trace(p)
    f = tr.recorded
    dff = sim.to_dff(f, sim.baseline(f, f0, int(F0_WINDOW * p.fs)))
    cell = Cell(name, tr.t, dff, tr.spikes, p.fs, None)
    cell.score = rescore(cell, dff)
    return cell


def mean_line(axes: mn.Axes, x_center: float, value: float, c: tuple) -> mn.Line:
    a, b = axes_points(axes, np.array([x_center - 0.42, x_center + 0.42]), np.array([value, value]))
    return mn.Line(a, b, color=color(c), stroke_width=3.0)


def sd_bar(axes: mn.Axes, x: float, mean: float, sd: float, c: tuple) -> mn.VGroup:
    """A vertical bar from mean - sd to mean + sd with end caps."""
    lo, hi = axes_points(axes, np.array([x, x]), np.array([mean - sd, mean + sd]))
    cap = np.array([0.06, 0, 0])
    return mn.VGroup(mn.Line(lo, hi, color=color(c), stroke_width=2.0),
                     mn.Line(lo - cap, lo + cap, color=color(c), stroke_width=2.0),
                     mn.Line(hi - cap, hi + cap, color=color(c), stroke_width=2.0))


@dataclass
class Pieces:
    """Everything a cell change morphs: the traces, the dots and the pile marks. The number is a Count of its own."""

    paths_trace: dict  # cell name -> the whole trace
    paths_zoom: dict
    zoom_pts: dict  # cell name -> (before points, peak points) on the zoom
    zoom_dots: tuple  # (before dots, peak dots) on the zoom
    pile_pts: dict  # cell name -> ([before points per spike], [peak points per spike])
    dots_before: list  # per spike, the dots now on screen
    dots_peak: list
    means: dict
    sds: dict
    gaps: dict


def morph(duration: float, lead: list, p: Pieces, name: str, n: int, number: mn.Text, d_from: float, d_to: float) -> mn.AnimationGroup:
    """
    A chapter that turns what is on screen (built as 'base') into the cell called name: the lead (the chapter's
    title, which primes the question) first, then the trace, the zoom, the clusters and the d' count all at once.
    """
    change = mn.AnimationGroup(
        mn.Transform(p.paths_trace["base"], p.paths_trace[name]),
        mn.Transform(p.paths_zoom["base"], p.paths_zoom[name]),
        mn.Transform(p.zoom_dots[0], dots_at(p.zoom_pts[name][0], BEFORE, 0.06)),
        mn.Transform(p.zoom_dots[1], dots_at(p.zoom_pts[name][1], PEAK, 0.06)),
        *[mn.Transform(p.dots_before[k], dots_at(p.pile_pts[name][0][k], BEFORE)) for k in range(n)],
        *[mn.Transform(p.dots_peak[k], dots_at(p.pile_pts[name][1][k], PEAK)) for k in range(n)],
        mn.Transform(p.means["base"], p.means[name]), mn.Transform(p.sds["base"], p.sds[name]),
        mn.Transform(p.gaps["base"], p.gaps[name]), Count(number, d_from, d_to, number_label),
        run_time=4.0,
    )
    return chapter(duration, mn.AnimationGroup(*lead, run_time=1.0), change, lag_ratio=1.0)


def example_rows(examples: list, t: np.ndarray, y_top: float, pitch: float) -> list:
    """
    One row per (d', dF/F): the trace on a shared scale across the left, then d' and the fraction of single spikes
    an ideal threshold gets right, Phi(d' / 2). Returns (trace, d' label, % label) per row, top to bottom.
    """
    values = np.concatenate([dff for _, dff in examples])
    r_lo, r_hi = float(np.percentile(values, 0.5)), float(np.percentile(values, 99.5))
    rows = []
    for k, (d, dff) in enumerate(examples):
        y = y_top - k * pitch
        axes = put(plain_axes(0, float(t[-1]), r_lo, r_hi, 9.0, 0.8 * pitch, visible=False), -2.0, y)
        trace = path(axes, t, dff, TRACE, 1.2, r_lo, r_hi)
        d_label = label(f"d' = {d:.1f}", PEAK, 22).move_to(np.array([3.0, y, 0.0]), aligned_edge=mn.LEFT)
        pct_label = label(f"{100 * M.accuracy(d):.0f}%", TEXT, 22).move_to(np.array([5.6, y, 0.0]), aligned_edge=mn.LEFT)
        rows.append((trace, d_label, pct_label))
    return rows


def build() -> tuple:
    """The story and its cells."""
    cells = {name: simulate(name, brightness, bleach, f0) for name, brightness, bleach, f0 in CELLS}
    base = cells["base"]
    fs = 1.0 / (base.t[1] - base.t[0])
    s = base.score
    n_frames = WINDOW_FRAMES
    peak = s.peak
    names = [name for name, *_ in CELLS]

    title = put(label(TITLE, TEXT, 30), 0.0, 3.6)
    chapter_titles = [put(label(name, DIM, 24), 0.0, 3.1) if name else None for name, _ in CHAPTERS]

    # the whole trace, top
    y_lo, y_hi = float(min(c.dff.min() for c in cells.values())), float(max(c.dff.max() for c in cells.values()))
    pad = 0.05 * (y_hi - y_lo)
    y_lo, y_hi = y_lo - pad, y_hi + pad
    axes_trace = put(plain_axes(0, float(base.t[-1]), y_lo, y_hi, 13.0, 1.7, visible=False), 0.0, 1.85)
    bars_trace = scale_bars(axes_trace, float(base.t[-1]), y_lo, y_hi, 5.0, 0.22)
    paths_trace = {name: path(axes_trace, c.t, c.dff, TRACE, 1.4, y_lo, y_hi) for name, c in cells.items()}
    tick_bottom, tick_top = y_lo - 0.08 * (y_hi - y_lo), y_lo - 0.02 * (y_hi - y_lo)
    ticks = mn.VGroup(*[
        mn.Line(*axes_points(axes_trace, np.array([sp, sp]), np.array([tick_bottom, tick_top])), color=color(TRUTH), stroke_width=2.0)
        for sp in base.spikes
    ])
    legend_trace = label("Ground truth spikes (simulated)", TRUTH, 18)
    legend_trace.next_to(ticks, mn.DOWN, buff=0.06).align_to(axes_trace, mn.LEFT)

    # the zoom on one spike, bottom left; piles to its right on the same y scale
    amplitudes = s.amplitudes
    chosen = int(np.argmin(np.abs(amplitudes - np.median(amplitudes))))
    center = int(s.centers[chosen])
    half = 4 * n_frames
    lo_i, hi_i = max(0, center - half), min(base.dff.size, center + peak + half)
    pools = np.concatenate([c.score.top.ravel() for c in cells.values()] + [c.score.bottom.ravel() for c in cells.values()])
    z_lo, z_hi = float(np.percentile(pools, 0.5)), float(np.percentile(pools, 99.5))
    z_pad = 0.12 * (z_hi - z_lo)
    z_lo, z_hi = z_lo - z_pad, z_hi + z_pad
    span = (hi_i - lo_i) / fs
    axes_zoom = put(plain_axes(0, span, z_lo, z_hi, 4.6, 2.3, visible=False), -4.3, -1.35)
    bars_zoom = scale_bars(axes_zoom, span, z_lo, z_hi, 0.5, 0.0, with_dff=False)
    t_zoom = (np.arange(lo_i, hi_i) - lo_i) / fs
    paths_zoom = {name: path(axes_zoom, t_zoom, c.dff[lo_i:hi_i], TRACE, 1.6, z_lo, z_hi) for name, c in cells.items()}
    # the boxed piece of the trace flies down and grows into the zoom; afterwards it is the zoom trace on screen
    zoom_target = paths_zoom["base"]
    paths_zoom["base"] = path(axes_trace, base.t[lo_i:hi_i], base.dff[lo_i:hi_i], TRACE, 1.4, y_lo, y_hi)
    paths_zoom["base"].set_stroke(opacity=0.0)  # a Transform source is drawn from t = 0; the flight reveals it
    frame_zoom = mn.Rectangle(width=4.6, height=2.3, color=color(DIM), stroke_width=1.5).move_to(axes_zoom)
    zoom_name = label(f"spike +/- {span / 2:.1f} s", DIM, 18).next_to(axes_zoom, mn.DOWN, buff=0.1).align_to(axes_zoom, mn.LEFT)
    on_trace = mn.Rectangle(width=max(span / float(base.t[-1]) * 13.0, 0.12), height=1.7, color=color(DIM), stroke_width=1.5)
    on_trace.move_to(axes_points(axes_trace, np.array([0.5 * (base.t[lo_i] + base.t[hi_i - 1])]), np.array([0.5 * (y_lo + y_hi)]))[0])
    box_flying = on_trace.copy()
    before_i = np.arange(center - n_frames, center) - lo_i
    peak_i = np.arange(center + peak, center + peak + n_frames) - lo_i
    zoom_pts = {name: (axes_points(axes_zoom, before_i / fs, np.clip(c.dff[lo_i:hi_i][before_i], z_lo, z_hi)),
                       axes_points(axes_zoom, peak_i / fs, np.clip(c.dff[lo_i:hi_i][peak_i], z_lo, z_hi))) for name, c in cells.items()}
    dots_before_zoom = dots_at(zoom_pts["base"][0], BEFORE, 0.06)
    dots_peak_zoom = dots_at(zoom_pts["base"][1], PEAK, 0.06)
    names_zoom = mn.VGroup(label("6 frames before the spike", BEFORE, 18), label("6 at the peak", PEAK, 18)).arrange(mn.RIGHT, buff=0.4)
    names_zoom.next_to(axes_zoom, mn.UP, buff=0.1).align_to(axes_zoom, mn.LEFT)

    axes_piles = put(plain_axes(0, 2.0, z_lo, z_hi, 2.0, 2.3), -0.6, -1.35)
    x_before, x_peak = 0.5, 1.5
    rng = np.random.default_rng(0)
    jitter = [rng.uniform(-0.28, 0.28, n_frames) for _ in range(s.n)]
    pile_pts = {}
    for name, cell in cells.items():
        sc = cell.score
        pile_pts[name] = (
            [axes_points(axes_piles, x_before + jitter[k], np.clip(sc.bottom[k], z_lo, z_hi)) for k in range(sc.n)],
            [axes_points(axes_piles, x_peak + jitter[k], np.clip(sc.top[k], z_lo, z_hi)) for k in range(sc.n)],
        )
    dots_before = [dots_at(pts, BEFORE, opacity=0.0) for pts in pile_pts["base"][0]]
    dots_peak = [dots_at(pts, PEAK, opacity=0.0) for pts in pile_pts["base"][1]]
    others = [k for k in range(s.n) if k != chosen]
    for group, zoomed in ((dots_before[chosen], dots_before_zoom), (dots_peak[chosen], dots_peak_zoom)):
        group.set_fill(opacity=0.9)  # these fade in with the zoom's dots, on top of them, then fly to the piles
        for dot, source in zip(group, zoomed):
            dot.move_to(source)
    for k in others:
        c_k = int(s.centers[k])
        dots_before[k].move_to(axes_points(axes_trace, np.array([base.t[c_k - n_frames // 2]]), np.array([base.dff[c_k - n_frames // 2]]))[0])
        dots_peak[k].move_to(axes_points(axes_trace, np.array([base.t[c_k + peak + n_frames // 2]]), np.array([base.dff[c_k + peak + n_frames // 2]]))[0])
    bands = mn.VGroup()
    for k in others:
        c_k = int(s.centers[k])
        for start, c in ((c_k - n_frames, BEFORE), (c_k + peak, PEAK)):
            a = axes_points(axes_trace, np.array([base.t[start], base.t[start + n_frames - 1] + 1 / fs]), np.array([y_lo, y_hi]))
            bands.add(mn.Rectangle(width=max(a[1, 0] - a[0, 0], 0.03), height=1.7, stroke_width=0, fill_color=color(c), fill_opacity=0.35)
                      .move_to(0.5 * (a[0] + a[1])))
    names_piles = label("clustered", DIM, 18).next_to(axes_piles, mn.UP, buff=0.1)
    count = label(f"{s.n} spikes x {n_frames} frames", DIM, 18).next_to(axes_piles, mn.DOWN, buff=0.1)

    x_right = 1.1
    means, sds, gaps = {}, {}, {}
    for name, cell in cells.items():
        sc = cell.score
        top, bottom = sc.top.ravel(), sc.bottom.ravel()
        pooled = float(np.sqrt(0.5 * (top.var() + bottom.var())))
        means[name] = mn.VGroup(mean_line(axes_piles, x_before, float(bottom.mean()), BEFORE), mean_line(axes_piles, x_peak, float(top.mean()), PEAK))
        sds[name] = mn.VGroup(sd_bar(axes_piles, x_before - 0.46, float(bottom.mean()), pooled, BEFORE),
                              sd_bar(axes_piles, x_peak + 0.46, float(top.mean()), pooled, PEAK))
        a, b = axes_points(axes_piles, np.array([1.0, 1.0]), np.array([float(bottom.mean()), float(top.mean())]))
        gaps[name] = mn.DoubleArrow(a, b, buff=0, stroke_width=2.0, tip_length=0.12, color=color(TEXT))
    number = number_label(s.d).move_to(np.array([x_right, -1.0, 0.0]), aligned_edge=mn.LEFT)
    pieces = Pieces(paths_trace, paths_zoom, zoom_pts, (dots_before_zoom, dots_peak_zoom), pile_pts, dots_before, dots_peak, means, sds, gaps)
    clean = simulate("clean", 5000.0, 0.0, "running window", neuropil=0.0, read_noise=0.0)
    examples = M.noise_rows(clean.dff, clean.spikes, clean.fs, peak, n_frames, M.quiet_before(sim.BY_NAME["GCaMP6f"]), TARGETS)
    rows = example_rows(examples, base.t, -0.45, 0.58)
    header = mn.VGroup(label("d'", DIM, 18).move_to(np.array([3.0, 0.1, 0.0]), aligned_edge=mn.LEFT),
                       label("% correct", DIM, 18).move_to(np.array([5.6, 0.1, 0.0]), aligned_edge=mn.LEFT))
    trace_again = path(axes_trace, base.t, base.dff, TRACE, 1.4, y_lo, y_hi)
    lower = [on_trace, bands, bars_zoom, zoom_name, names_zoom, dots_before_zoom, dots_peak_zoom, *dots_before, *dots_peak,
             axes_piles, names_piles, count, means["base"], sds["base"], gaps["base"], number, paths_zoom["base"]]

    captions = [
        None,
        caption("Take one spike. Keep the 6 frames just before it (blue) and the 6 frames at the peak of its response (yellow)."),
        caption("Stack those frames by value: a blue cluster and a yellow cluster. d' is how far the yellow cluster sits above the blue one, in units of how wide the clusters are."),
        caption(f"Do the same for every isolated spike and pool the frames: {s.n} spikes here, {n_frames} frames each. The result is one number for the cell."),
        None,
        None,
        None,
        None,
        None,
    ]

    seconds = [d for _, d in CHAPTERS]
    chapters = [
        chapter(seconds[0], mn.FadeIn(title), mn.Create(paths_trace["base"]), mn.FadeIn(bars_trace),
                mn.FadeIn(ticks), mn.FadeIn(legend_trace), lag_ratio=0.15),
        chapter(seconds[1],
                mn.AnimationGroup(mn.FadeOut(title), *swap(chapter_titles, captions, 1), mn.FadeIn(on_trace), lag_ratio=0.1),
                mn.AnimationGroup(mn.FadeIn(box_flying, run_time=0.2),
                                  mn.AnimationGroup(mn.Transform(box_flying, frame_zoom), mn.Transform(paths_zoom["base"], zoom_target), run_time=1.6),
                                  mn.FadeOut(box_flying, run_time=0.3), lag_ratio=1.0),
                mn.AnimationGroup(mn.FadeIn(bars_zoom), mn.FadeIn(zoom_name),
                                  mn.AnimationGroup(mn.FadeIn(dots_before_zoom, scale=1.5), mn.FadeIn(dots_before[chosen], scale=1.5)),
                                  mn.AnimationGroup(mn.FadeIn(dots_peak_zoom, scale=1.5), mn.FadeIn(dots_peak[chosen], scale=1.5)),
                                  mn.FadeIn(names_zoom), lag_ratio=0.12),
                lag_ratio=1.0),
        chapter(seconds[2], *swap(chapter_titles, captions, 2), mn.Create(axes_piles), mn.FadeIn(names_piles),
                mn.Transform(dots_before[chosen], dots_at(pile_pts["base"][0][chosen], BEFORE)),
                mn.Transform(dots_peak[chosen], dots_at(pile_pts["base"][1][chosen], PEAK)), lag_ratio=0.15),
        chapter(seconds[3], *swap(chapter_titles, captions, 3), mn.FadeIn(bands),
                mn.LaggedStart(*[mn.Transform(dots_before[k], dots_at(pile_pts["base"][0][k], BEFORE)) for k in others], lag_ratio=0.12),
                mn.LaggedStart(*[mn.Transform(dots_peak[k], dots_at(pile_pts["base"][1][k], PEAK)) for k in others], lag_ratio=0.12),
                mn.FadeIn(count), lag_ratio=0.1),
        chapter(seconds[4], *swap(chapter_titles, captions, 4), mn.Create(means["base"]), mn.FadeIn(gaps["base"]), mn.Create(sds["base"]),
                mn.FadeIn(number), lag_ratio=0.12),
        morph(seconds[5], swap(chapter_titles, captions, 5), pieces, "noisy", s.n, number, s.d, cells["noisy"].score.d),
        morph(seconds[6], swap(chapter_titles, captions, 6), pieces, "quiet", s.n, number, cells["noisy"].score.d, cells["quiet"].score.d),
        morph(seconds[7], swap(chapter_titles, captions, 7), pieces, "bleached", s.n, number, cells["quiet"].score.d, cells["bleached"].score.d),
        chapter(seconds[8],
                mn.AnimationGroup(*swap(chapter_titles, captions, 8), *[mn.FadeOut(m) for m in lower],
                                  mn.Transform(paths_trace["base"], trace_again), run_time=1.5),
                mn.AnimationGroup(mn.FadeIn(header),
                                  mn.LaggedStart(*[mn.AnimationGroup(mn.Create(trace), mn.FadeIn(d_label), mn.FadeIn(pct_label), lag_ratio=0.3)
                                                   for trace, d_label, pct_label in rows], lag_ratio=0.6),
                                  lag_ratio=0.3),
                lag_ratio=1.0),
    ]
    cells["clean"] = clean
    return story([], chapters), cells


# ----------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------




def build_story() -> kit.Story:
    s, cells = build()
    for c in cells.values():
        print(f"{c.name:9s}: d' = {c.score.d:.2f} over {c.score.n} spikes, accuracy {100 * M.accuracy(c.score.d):.0f}%")
    return s


if __name__ == "__main__":
    kit.run("dprime", CHAPTERS, UNTITLED, build_story, __doc__)
